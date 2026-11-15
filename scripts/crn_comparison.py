"""C1 — pair the policies against a shared random stream, so the null is a
measurement instead of an upper bound.

`policy_comparison.py` draws an independent seed per policy, so the reported
difference between two cutoff policies carries the full Monte Carlo variance of
both.  At 8 replicates that gave standard errors of 0.010-0.016 Hz, and the
headline comparison -- an accumulated-decay (Werner-parameter) gate against a
global age gate -- came out at -0.4%, well inside the band.  A difference
"consistent with zero" at that resolution excludes a ~12% gap and establishes
nothing narrower.  It is a statement about the sample size.

Almost all of that variance is shared: both policies run the same chain, the
same link-generation process, the same decoherence.  Common random numbers
remove it, provided the coupling survives the policies consuming *different
numbers* of draws -- which they do, because a stricter cutoff triggers more
rebuilds.  Reusing one `default_rng` per replicate does not survive that; the
streams desynchronise at the first divergent rebuild and the pairing is lost
exactly where the policies start to differ.

The coupling used here is by source.  Every elementary link owns its own
stream, and the k-th generation attempt on link j draws the k-th variate of
stream j no matter which policy asked for it or when.  A policy that rebuilds
link j more often simply consumes more of stream j; it never reads another
link's numbers.  The construction is exact, it is the standard random-number-
reuse form of CRN, and it makes the paired difference a measurement of the
policy rather than of the seed.

The engine below is `simulate_chain_fast` with its `_geom_slots(rng, p)` calls
replaced by draws from that pool, and nothing else changed.  `validate()`
checks it against the original engine, which it must reproduce in distribution.

Also folded in (the literature map's item 5.10, which costs one line and makes
a null informative): every run reports the delivered Werner parameter and the
delivery-time distribution, not just their means, so a policy that trades rate
against fidelity can be seen doing it.
"""

from __future__ import annotations

import json
import math
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from repeater_chain import (  # noqa: E402
    ChainConfig, simulate_chain_fast, secret_key_fraction, werner_to_fidelity,
    plob_bound_hz, _fidelity_lifetime,
)

RESULTS = os.path.join(_HERE, "..", "results")


class DrawPool:
    """One geometric stream per elementary link, extended on demand.

    `take(j)` returns the next number of attempts-until-success for elementary
    link j.  The value depends only on (sample, j, how many times link j has
    already been generated) -- never on the policy, on the order in which
    links were touched, or on how many draws other links consumed.
    """

    __slots__ = ("_p", "_rngs", "_cache", "_used")

    def __init__(self, n_links: int, p: float, seed: int) -> None:
        self._p = p
        ss = np.random.SeedSequence(seed)
        self._rngs = [np.random.default_rng(s) for s in ss.spawn(n_links)]
        self._cache: list[list[int]] = [[] for _ in range(n_links)]
        self._used = [0] * n_links

    def reset(self) -> None:
        """Rewind every stream; the cached values are replayed identically."""
        self._used = [0] * len(self._used)

    def take(self, j: int) -> int:
        k = self._used[j]
        cache = self._cache[j]
        while len(cache) <= k:
            cache.append(1 if self._p >= 1.0
                         else int(self._rngs[j].geometric(self._p)))
        self._used[j] = k + 1
        return cache[k]


def simulate_chain_crn(cfg: ChainConfig, pool: DrawPool,
                       max_slots: int = 10**9) -> tuple[float, float]:
    """One sample of the swap-asap chain, driven by `pool`.

    Returns (delivery time in slots, delivered Werner parameter).  Identical
    dynamics to `simulate_chain_fast`; only the source of randomness differs.
    """
    p = cfg.p_gen
    dt = cfg.attempt_time_s
    n = cfg.n_segments
    base_cutoff = (cfg.cutoff_s / dt) if cfg.cutoff_s is not None else None

    def cutoff_for(span_: int) -> float | None:
        if base_cutoff is None:
            return None
        return base_cutoff * float(span_) ** cfg.cutoff_gamma

    birth = np.array([float(pool.take(j)) for j in range(n)])
    wpar = np.full(n, cfg.w_init)
    span = np.ones(n, dtype=np.int64)
    live = list(range(n))

    while len(live) > 1:
        best, best_t = None, np.inf
        for k in range(len(live) - 1):
            a, b = live[k], live[k + 1]
            t_swap = max(birth[a], birth[b])
            if t_swap < best_t:
                best, best_t = k, t_swap
        a, b = live[best], live[best + 1]
        older, younger = (a, b) if birth[a] <= birth[b] else (b, a)
        wait = birth[younger] - birth[older]

        if cfg.cutoff_mode == "fidelity":
            cutoff = (_fidelity_lifetime(wpar[older], cfg, dt)
                      if cfg.w_min is not None else None)
        else:
            cutoff = cutoff_for(int(span[older]))

        if cutoff is not None and wait > cutoff:
            expiry = birth[older] + cutoff
            n_rebuild = int(span[older])
            # `older` is the leftmost elementary index of its merged block, so
            # the block covers elementary links [older, older + n_rebuild).
            # Each of them is regenerated from its own stream.
            restart = expiry + max(pool.take(older + off)
                                   for off in range(n_rebuild))
            birth[older] = restart
            wpar[older] = cfg.w_init ** n_rebuild * cfg.w_swap ** (n_rebuild - 1)
            if birth[older] >= max_slots:
                # Censored: report the cap rather than the (unbounded) truth.
                return float(max_slots), float(wpar[older])
            continue

        wa = wpar[a] * math.exp(-(best_t - birth[a]) * dt / cfg.t_coh_s)
        wb = wpar[b] * math.exp(-(best_t - birth[b]) * dt / cfg.t_coh_s)
        wpar[a] = wa * wb * cfg.w_swap
        birth[a] = best_t
        span[a] = span[a] + span[b]
        live.pop(best + 1)

    i = live[0]
    return float(birth[i]), float(wpar[i])


def run_paired(cfgs: dict, n_samples: int, seed: int,
               max_slots: int = 10**9) -> dict:
    """Every policy in `cfgs` on the SAME per-link streams, sample by sample.

    `max_slots` censors a sample that has not delivered by then.  It is needed
    because part of the tuning grid is structurally degenerate rather than
    merely bad, and without a cap those points do not finish.

    The degenerate corner is the fidelity gate, not the span-indexed one.  A
    link's remaining lifetime under that gate is (T_coh/dt) log(w_now/w_min),
    so as the threshold approaches the Werner parameter of a freshly made link
    the lifetime goes to zero: the link is discarded about as fast as it can be
    built, and the chain makes no progress.  At 8 segments with w_init = 0.99
    the point w_min = 0.95 already leaves 22% of samples undelivered at a cap of
    400x the no-cutoff mean.  (The span-indexed family was the suspect before
    this was measured -- the guess being that c0*span^gamma falls below the mean
    rebuild time at negative gamma -- and the measurement refuted it: at
    8 segments, c0 = 0.03 s and gamma = -1 the chain delivers normally, with no
    censored samples at all.)

    Censored samples are recorded at the cap, which makes the reported rate an
    upper bound on a policy that in fact delivers far less, and the censored
    fraction is returned alongside so that a number resting on censoring is
    never quoted as though it were not.
    """
    names = list(cfgs)
    any_cfg = cfgs[names[0]]
    dt = any_cfg.attempt_time_s
    n = any_cfg.n_segments
    times = {k: np.empty(n_samples) for k in names}
    werners = {k: np.empty(n_samples) for k in names}
    censored = {k: 0 for k in names}
    for s in range(n_samples):
        pool = DrawPool(n, any_cfg.p_gen, seed * 1_000_003 + s)
        for k in names:
            pool.reset()
            t, w = simulate_chain_crn(cfgs[k], pool, max_slots=max_slots)
            if t >= max_slots:
                censored[k] += 1
            times[k][s] = t * dt
            werners[k][s] = w
    out = {}
    for k in names:
        mt, mw = float(times[k].mean()), float(werners[k].mean())
        out[k] = {"mean_delivery_time_s": mt, "werner": mw,
                  "fidelity": werner_to_fidelity(mw),
                  "rate_hz": 1.0 / mt,
                  "secret_key_hz": secret_key_fraction(mw) / mt,
                  "werner_p10": float(np.percentile(werners[k], 10)),
                  "werner_p50": float(np.percentile(werners[k], 50)),
                  "werner_p90": float(np.percentile(werners[k], 90)),
                  "time_p50": float(np.percentile(times[k], 50)),
                  "time_p90": float(np.percentile(times[k], 90)),
                  "censored": censored[k],
                  "censored_fraction": censored[k] / n_samples,
                  "_times": times[k], "_werners": werners[k]}
    return out


def validate(cfg: ChainConfig, n_samples: int = 4000) -> dict:
    """The CRN engine must reproduce the original engine in distribution."""
    ref = simulate_chain_fast(cfg, n_samples=n_samples, seed=12345)
    ts, ws = [], []
    dt = cfg.attempt_time_s
    for s in range(n_samples):
        pool = DrawPool(cfg.n_segments, cfg.p_gen, 777_000 + s)
        t, w = simulate_chain_crn(cfg, pool)
        ts.append(t * dt)
        ws.append(w)
    ts, ws = np.array(ts), np.array(ws)
    mt, sem_t = ts.mean(), ts.std(ddof=1) / math.sqrt(n_samples)
    ref_t = ref["mean_delivery_time_s"]
    out = {"crn_mean_time_s": float(mt), "ref_mean_time_s": float(ref_t),
           "rel_diff_time": float(abs(mt - ref_t) / ref_t),
           "sigma_time": float(abs(mt - ref_t) / sem_t),
           "crn_werner": float(ws.mean()), "ref_werner": ref["werner"],
           "rel_diff_werner": float(abs(ws.mean() - ref["werner"]) / ref["werner"])}
    return out


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired difference of secret-key rate, sample by sample.

    The per-sample key rate is not linear in the sample (the rate is 1/E[T],
    not E[1/T]), so the paired quantity is formed by a jackknife over samples:
    each pseudo-value is the key rate with one sample removed, and the paired
    difference inherits the pairing.
    """
    n = len(a)
    assert len(b) == n
    return {"n": n, "corr": float(np.corrcoef(a, b)[0, 1])}


def key_rate(times: np.ndarray, werners: np.ndarray) -> float:
    return secret_key_fraction(float(werners.mean())) / float(times.mean())


def jackknife_paired(res_a: dict, res_b: dict, n_blocks: int = 40) -> dict:
    """Block jackknife on the paired difference of the secret-key rate."""
    ta, wa = res_a["_times"], res_a["_werners"]
    tb, wb = res_b["_times"], res_b["_werners"]
    n = len(ta)
    full = key_rate(ta, wa) - key_rate(tb, wb)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    ps = []
    for k in range(n_blocks):
        m = np.ones(n, bool)
        m[edges[k]:edges[k + 1]] = False
        ps.append(key_rate(ta[m], wa[m]) - key_rate(tb[m], wb[m]))
    ps = np.array(ps)
    var = (n_blocks - 1) / n_blocks * ((ps - ps.mean()) ** 2).sum()
    return {"difference_hz": float(full), "sem": float(math.sqrt(var)),
            "corr_time": float(np.corrcoef(ta, tb)[0, 1]),
            "corr_werner": float(np.corrcoef(wa, wb)[0, 1])}
