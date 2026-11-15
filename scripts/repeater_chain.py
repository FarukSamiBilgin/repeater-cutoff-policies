"""Monte-Carlo simulator for entanglement distribution over a repeater chain.

Context (motivating literature; the manuscript in paper/ carries the
verified references)
----------------------------------------------------------------------
OIST Networked Quantum Devices Unit (David Elkouss) studies *policies* for
entanglement distribution and cross-validates network simulators
(arXiv:2506.09144, arXiv:2504.01290).  USTC (Bao Xiao-Hui / Zhang Qiang / Pan)
published memory-memory entanglement over 420 km that crosses the repeaterless
PLOB bound beyond ~230 km (PRL, Aug 2026).  This module puts both questions in
one place: given a swapping policy and a memory cutoff, when does a chain beat
the repeaterless bound, and at what fidelity?

Model
-----
A linear chain of `n_segments` elementary links, each of length L0 = L/n.
Each link is heralded at its midpoint, so a photon travels L0/2 through fibre
on each side.  One *attempt* takes t0 = L0 / c_fiber seconds (the herald has to
come back), and succeeds with probability

    p_gen = p0 * eta ** photons_per_herald,      eta = exp(-L0 / (2 * L_att))

`photons_per_herald = 1` is single-photon (single-excitation) heralding, which
is what the USTC atomic-ensemble experiment uses: success scales as eta, at the
cost of a phase-sensitive interferometer.  `= 2` is two-photon heralding, which
scales as eta^2 but is phase-insensitive.

Stored entanglement dephases exponentially with memory coherence time
`t_coh`.  A Werner state of fidelity F held for time t becomes

    w(t) = w(0) * exp(-t / t_coh),     F = (1 + 3w) / 4

and entanglement swapping multiplies the Werner parameters of the two halves
and applies a swap-gate quality factor `w_swap`.

Policies
--------
`swap_asap`   : swap as soon as both adjacent links exist.
`nested`      : swap in a balanced binary tree (needs n_segments a power of 2).
Both accept a `cutoff` (seconds): a stored link older than the cutoff is
discarded and regenerated, trading rate for fidelity.  This is the knob the
NetQ policy papers study.

The cutoff here is not a single number but a *family*:

    cutoff(span) = cutoff_s * span ** cutoff_gamma

`cutoff_gamma = 0` is the usual global cutoff, the baseline every published
sweep uses.  Positive gamma holds longer links for longer, on the argument that
a swapped link embodies more accumulated work and is more expensive to throw
away.  Negative gamma discards them sooner, on the argument that a swapped link
is already the most decohered object in the chain, so holding it is what
destroys the fidelity.  Which argument wins is not obvious from either side,
and `scan_level_cutoffs` below settles it numerically.

`cutoff_mode` chooses what the threshold is applied to.  In "age" mode a link
is discarded once it is older than cutoff(span) — this is the standard policy.
In "fidelity" mode it is discarded once its Werner parameter falls below
`w_min`.  For an elementary link the two are the same rule written differently,
since a fresh link always starts at `w_init` and decays deterministically.  For
a *merged* link they are genuinely different: its Werner parameter depends on
how long each half waited before the swap, so two links of the same age can
carry very different fidelity.  An age rule cannot see that; a fidelity rule
keeps the good one and drops the bad one.

Everything here is a discrete-event Monte Carlo over attempt slots; no external
simulator is required, which is exactly what makes it usable as an independent
cross-check against NetSquid / SeQUeNCe results.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict

import numpy as np

C_FIBER_KM_S = 2.0e5  # speed of light in fibre, km/s
L_ATT_KM = 22.0  # telecom-band attenuation length (0.2 dB/km)


# --------------------------------------------------------------------------
# Werner-state helpers
# --------------------------------------------------------------------------
def fidelity_to_werner(fidelity: float) -> float:
    """Werner parameter w from singlet fidelity F."""
    return (4.0 * fidelity - 1.0) / 3.0


def werner_to_fidelity(w: float) -> float:
    return (1.0 + 3.0 * w) / 4.0


def qber_from_werner(w: float) -> tuple[float, float]:
    """(e_x, e_z) for a Werner state with parameter w."""
    e = (1.0 - w) / 2.0
    return e, e


def _binary_entropy(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def secret_key_fraction(w: float) -> float:
    """Asymptotic BB84 key per distributed ebit, r = 1 - h(e_x) - h(e_z)."""
    e_x, e_z = qber_from_werner(w)
    return max(0.0, 1.0 - _binary_entropy(e_x) - _binary_entropy(e_z))


# --------------------------------------------------------------------------
# Repeaterless benchmark
# --------------------------------------------------------------------------
def plob_bound_per_use(length_km: float, l_att_km: float = L_ATT_KM) -> float:
    """PLOB repeaterless capacity in ebits per channel use: -log2(1 - eta)."""
    eta = math.exp(-length_km / l_att_km)
    return -math.log2(max(1.0 - eta, 1e-300))


def plob_bound_hz(length_km: float, l_att_km: float = L_ATT_KM,
                  c_fiber: float = C_FIBER_KM_S) -> float:
    """PLOB bound as a rate, assuming one channel use per round trip.

    A direct-transmission link over `length_km` can attempt at most
    c_fiber / length_km times per second if each use is confirmed, so the
    repeaterless ceiling in Hz is that clock times the per-use capacity.
    This is the comparison the USTC 420 km paper makes.
    """
    clock = c_fiber / length_km
    return clock * plob_bound_per_use(length_km, l_att_km)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class ChainConfig:
    total_length_km: float = 400.0
    n_segments: int = 4
    p0: float = 0.1                  # source/detector efficiency prefactor
    photons_per_herald: int = 1      # 1 = single-photon heralding, 2 = two-photon
    t_coh_s: float = 0.5             # memory coherence time
    w_init: float = 0.98             # Werner parameter of a fresh elementary link
    w_swap: float = 0.98             # swap-gate quality
    cutoff_s: float | None = None    # discard a stored link older than this
    cutoff_gamma: float = 0.0        # span-dependence of the cutoff, see below
    cutoff_mode: str = "age"         # "age" or "fidelity"
    w_min: float | None = None       # fidelity-mode threshold on the Werner parameter
    policy: str = "swap_asap"        # or "nested"
    l_att_km: float = L_ATT_KM
    c_fiber: float = C_FIBER_KM_S

    @property
    def segment_km(self) -> float:
        return self.total_length_km / self.n_segments

    @property
    def attempt_time_s(self) -> float:
        """One attempt = photons out to the midpoint and herald back."""
        return self.segment_km / self.c_fiber

    @property
    def p_gen(self) -> float:
        eta = math.exp(-self.segment_km / (2.0 * self.l_att_km))
        return min(1.0, self.p0 * eta**self.photons_per_herald)


# --------------------------------------------------------------------------
# Analytic reference (no cutoff, no decoherence) — used as a self-check
# --------------------------------------------------------------------------
def expected_attempts_two_links(p: float) -> float:
    """E[max(G1, G2)] for two independent geometric(p) variables, in attempts."""
    return 2.0 / p - 1.0 / (1.0 - (1.0 - p) ** 2)


def analytic_rate_two_segments(cfg: ChainConfig) -> float:
    """End-to-end rate (Hz) for a 2-segment swap-asap chain without cutoff."""
    attempts = expected_attempts_two_links(cfg.p_gen)
    return 1.0 / (attempts * cfg.attempt_time_s)


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def simulate_chain(cfg: ChainConfig, n_samples: int = 2000,
                   seed: int = 0) -> dict:
    """Slot-stepping reference implementation.

    Every attempt is simulated explicitly, which makes the model transparent at
    the cost of being unusable once p_gen gets small.  The state is a list of
    live links, each covering a contiguous run of elementary segments; a link
    whose age exceeds the cutoff is discarded, and if it was a *merged* link
    every elementary segment underneath it goes back to generating.  The
    event-driven `simulate_chain_fast` implements this same model.
    """
    rng = np.random.default_rng(seed)
    p = cfg.p_gen
    dt = cfg.attempt_time_s
    n = cfg.n_segments
    base_cutoff_slots = (cfg.cutoff_s / dt) if cfg.cutoff_s is not None else None

    def cutoff_slots_for(span_: int) -> float | None:
        if base_cutoff_slots is None:
            return None
        return base_cutoff_slots * float(span_) ** cfg.cutoff_gamma

    if cfg.policy == "nested" and (n & (n - 1)) != 0:
        raise ValueError("nested policy needs n_segments to be a power of 2")

    times = np.empty(n_samples)
    werners = np.empty(n_samples)

    for s in range(n_samples):
        slot = 0
        # each live link: [span, ready_slot or -1, werner]
        links = [[1, -1, 0.0] for _ in range(n)]

        while True:
            slot += 1
            # 1. attempt generation on every empty elementary link
            for lk in links:
                if lk[1] < 0 and lk[0] == 1 and rng.random() < p:
                    lk[1] = slot
                    lk[2] = cfg.w_init

            # 2. cutoff: any stored link older than the cutoff is discarded,
            #    and a merged one falls apart into its elementary segments
            if base_cutoff_slots is not None or cfg.cutoff_mode == "fidelity":
                rebuilt = []
                for lk in links:
                    if cfg.cutoff_mode == "fidelity":
                        cut = _fidelity_lifetime(lk[2], cfg, dt)
                    else:
                        cut = cutoff_slots_for(lk[0])
                    if lk[1] >= 0 and cut is not None and (slot - lk[1]) > cut:
                        rebuilt.extend([1, -1, 0.0] for _ in range(lk[0]))
                    else:
                        rebuilt.append(lk)
                links = rebuilt

            # 3. swap wherever two adjacent links are both ready
            merged = True
            while merged and len(links) > 1:
                merged = False
                for k in range(len(links) - 1):
                    a, b = links[k], links[k + 1]
                    if a[1] < 0 or b[1] < 0:
                        continue
                    if cfg.policy == "nested" and a[0] != b[0]:
                        continue
                    wa = a[2] * math.exp(-(slot - a[1]) * dt / cfg.t_coh_s)
                    wb = b[2] * math.exp(-(slot - b[1]) * dt / cfg.t_coh_s)
                    links[k] = [a[0] + b[0], slot, wa * wb * cfg.w_swap]
                    links.pop(k + 1)
                    merged = True
                    break

            if len(links) == 1 and links[0][1] >= 0:
                lk = links[0]
                held = (slot - lk[1]) * dt
                times[s] = slot * dt
                werners[s] = lk[2] * math.exp(-held / cfg.t_coh_s)
                break

    mean_time = float(times.mean())
    mean_w = float(werners.mean())
    rate_hz = 1.0 / mean_time
    fid = werner_to_fidelity(mean_w)
    key_hz = rate_hz * secret_key_fraction(mean_w)
    return {
        "rate_hz": rate_hz,
        "mean_delivery_time_s": mean_time,
        "werner": mean_w,
        "fidelity": fid,
        "secret_key_hz": key_hz,
        "secret_key_fraction": secret_key_fraction(mean_w),
        "p_gen": p,
        "attempt_time_s": dt,
        "plob_hz": plob_bound_hz(cfg.total_length_km, cfg.l_att_km, cfg.c_fiber),
        "n_samples": n_samples,
        "engine": "slot-stepping",
    }


def selfcheck(verbose: bool = True) -> bool:
    """Cross-check the Monte Carlo against the analytic 2-segment formula."""
    cfg = ChainConfig(total_length_km=100.0, n_segments=2, p0=0.5,
                      t_coh_s=1e9, w_init=1.0, w_swap=1.0)
    mc = simulate_chain(cfg, n_samples=20000, seed=1)
    ana = analytic_rate_two_segments(cfg)
    rel = abs(mc["rate_hz"] - ana) / ana
    if verbose:
        print(f"  selfcheck 2-segment rate: MC={mc['rate_hz']:.4g} Hz  "
              f"analytic={ana:.4g} Hz  rel.err={rel:.3%}")
    return rel < 0.03


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--length", type=float, default=400.0, help="total km")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--p0", type=float, default=0.1)
    ap.add_argument("--photons", type=int, default=1, choices=[1, 2])
    ap.add_argument("--tcoh", type=float, default=0.5, help="memory T_coh in s")
    ap.add_argument("--cutoff", type=float, default=None, help="cutoff in s")
    ap.add_argument("--policy", default="swap_asap",
                    choices=["swap_asap", "nested"])
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = ChainConfig(total_length_km=args.length, n_segments=args.segments,
                      p0=args.p0, photons_per_herald=args.photons,
                      t_coh_s=args.tcoh, cutoff_s=args.cutoff,
                      policy=args.policy)
    res = simulate_chain(cfg, n_samples=args.samples, seed=args.seed)
    if args.json:
        print(json.dumps({"config": asdict(cfg), "result": res}, indent=2))
        return
    print(f"chain: {cfg.total_length_km:.0f} km / {cfg.n_segments} segments "
          f"({cfg.segment_km:.0f} km each), policy={cfg.policy}")
    print(f"  p_gen per attempt : {res['p_gen']:.3e}")
    print(f"  attempt time      : {res['attempt_time_s']*1e3:.3f} ms")
    print(f"  end-to-end rate   : {res['rate_hz']:.4g} Hz")
    print(f"  fidelity          : {res['fidelity']:.4f}")
    print(f"  secret key rate   : {res['secret_key_hz']:.4g} Hz")
    print(f"  PLOB ceiling      : {res['plob_hz']:.4g} Hz  "
          f"-> {'BEATS' if res['secret_key_hz'] > res['plob_hz'] else 'below'} "
          f"repeaterless bound")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# Event-driven implementation (fast path)
# --------------------------------------------------------------------------
# The slot-by-slot `simulate_chain` above is the reference: it literally steps
# through every attempt, which makes it obviously correct but unusable once
# p_gen falls below ~1e-3.  The version below models exactly the same process
# but jumps straight to events, sampling elementary-link successes from the
# geometric waiting-time distribution instead of stepping through failures.
#
# Keeping both and checking that they agree is the same exercise as
# arXiv:2504.01290 (cross-validating quantum network simulators), one level
# down: two independent implementations of one model, compared on the overlap
# where both are affordable.

def _fidelity_lifetime(w_now: float, cfg: ChainConfig, dt: float) -> float:
    """Slots a link with Werner parameter `w_now` may be held before w < w_min."""
    if cfg.w_min is None or w_now <= cfg.w_min:
        return 0.0
    return (cfg.t_coh_s / dt) * math.log(w_now / cfg.w_min)


def _geom_slots(rng: np.random.Generator, p: float) -> int:
    """Attempts until the first success, inclusive (support 1, 2, ...)."""
    if p >= 1.0:
        return 1
    return int(rng.geometric(p))


def simulate_chain_fast(cfg: ChainConfig, n_samples: int = 2000,
                        seed: int = 0, max_slots: int = 10**9) -> dict:
    """Event-driven swap-asap simulation. Same model as `simulate_chain`.

    A stored link older than the cutoff is discarded.  When a *merged* link is
    discarded, every elementary link it was built from has to be regenerated,
    which is what makes an aggressive cutoff expensive deeper in the chain.
    """
    if cfg.policy != "swap_asap":
        raise ValueError("the fast path implements swap_asap only")
    rng = np.random.default_rng(seed)
    p = cfg.p_gen
    dt = cfg.attempt_time_s
    n = cfg.n_segments
    base_cutoff = (cfg.cutoff_s / dt) if cfg.cutoff_s is not None else None

    def cutoff_for(span_: int) -> float | None:
        if base_cutoff is None:
            return None
        return base_cutoff * float(span_) ** cfg.cutoff_gamma

    times = np.empty(n_samples)
    werners = np.empty(n_samples)

    for s in range(n_samples):
        # one entry per live link: [birth_slot, werner, span]
        birth = np.array([float(_geom_slots(rng, p)) for _ in range(n)])
        wpar = np.full(n, cfg.w_init)
        span = np.ones(n, dtype=np.int64)
        live = list(range(n))

        while len(live) > 1:
            # swap-asap: the pair that can swap earliest goes first
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
                # the older link expires before its partner shows up
                expiry = birth[older] + cutoff
                n_rebuild = int(span[older])
                # every elementary link underneath it starts over
                restart = expiry + max(_geom_slots(rng, p)
                                       for _ in range(n_rebuild))
                birth[older] = restart
                wpar[older] = cfg.w_init ** n_rebuild * cfg.w_swap ** (n_rebuild - 1)
                if birth[older] > max_slots:
                    break
                continue

            # both survive: decohere to the swap time and merge
            wa = wpar[a] * math.exp(-(best_t - birth[a]) * dt / cfg.t_coh_s)
            wb = wpar[b] * math.exp(-(best_t - birth[b]) * dt / cfg.t_coh_s)
            wpar[a] = wa * wb * cfg.w_swap
            birth[a] = best_t
            span[a] = span[a] + span[b]
            live.pop(best + 1)

        i = live[0]
        times[s] = birth[i] * dt
        werners[s] = wpar[i]

    mean_time = float(times.mean())
    mean_w = float(werners.mean())
    rate_hz = 1.0 / mean_time
    return {
        "rate_hz": rate_hz,
        "mean_delivery_time_s": mean_time,
        "werner": mean_w,
        "fidelity": werner_to_fidelity(mean_w),
        "secret_key_hz": rate_hz * secret_key_fraction(mean_w),
        "secret_key_fraction": secret_key_fraction(mean_w),
        "p_gen": p,
        "attempt_time_s": dt,
        "plob_hz": plob_bound_hz(cfg.total_length_km, cfg.l_att_km, cfg.c_fiber),
        "n_samples": n_samples,
        "engine": "event-driven",
    }


def cross_validate(n_samples: int = 4000, verbose: bool = True) -> dict:
    """Compare the two engines where both are affordable.

    This is the cross-validation exercise, run on this model: a slot-stepping
    reference and an event-driven implementation should agree on rate and
    fidelity to within Monte Carlo error.
    """
    cases = [
        ("no cutoff, 2 seg", ChainConfig(total_length_km=200.0, n_segments=2,
                                         p0=0.2, t_coh_s=1.0)),
        ("no cutoff, 4 seg", ChainConfig(total_length_km=300.0, n_segments=4,
                                         p0=0.2, t_coh_s=1.0)),
        ("cutoff, 4 seg", ChainConfig(total_length_km=300.0, n_segments=4,
                                      p0=0.2, t_coh_s=1.0, cutoff_s=0.02)),
    ]
    rows = []
    for name, cfg in cases:
        ref = simulate_chain(cfg, n_samples=n_samples, seed=5)
        fast = simulate_chain_fast(cfg, n_samples=n_samples, seed=5)
        d_rate = abs(fast["rate_hz"] - ref["rate_hz"]) / ref["rate_hz"]
        d_fid = abs(fast["fidelity"] - ref["fidelity"])
        rows.append({"case": name, "rate_ref": ref["rate_hz"],
                     "rate_fast": fast["rate_hz"], "rel_rate_diff": d_rate,
                     "fid_ref": ref["fidelity"], "fid_fast": fast["fidelity"],
                     "abs_fid_diff": d_fid})
        if verbose:
            print(f"  {name:<18} rate {ref['rate_hz']:8.4g} vs "
                  f"{fast['rate_hz']:8.4g} ({d_rate:6.2%})   "
                  f"F {ref['fidelity']:.4f} vs {fast['fidelity']:.4f} "
                  f"({d_fid:+.4f})")
    return {"cases": rows}


def scan_level_cutoffs(cfg_base: ChainConfig, c0_grid, gamma_grid,
                       n_samples: int = 4000, seed: int = 21,
                       verbose: bool = True) -> dict:
    """Optimise over the two-parameter cutoff family, honestly.

    Comparing gammas at a fixed c0 would be rigged: each gamma has its own best
    c0, because changing gamma rescales the cutoff of every merged link.  So we
    optimise c0 *within* each gamma slice and compare the slice maxima.  The
    gamma = 0 slice is the baseline — the global cutoff every published sweep
    uses — and the question is whether any other slice beats it by more than
    Monte Carlo error.
    """
    grid = np.zeros((len(gamma_grid), len(c0_grid)))
    for gi, gamma in enumerate(gamma_grid):
        for ci, c0 in enumerate(c0_grid):
            cfg = ChainConfig(**{**asdict(cfg_base),
                                 "cutoff_s": c0, "cutoff_gamma": gamma})
            grid[gi, ci] = simulate_chain_fast(
                cfg, n_samples=n_samples, seed=seed)["secret_key_hz"]
        best_ci = int(np.argmax(grid[gi]))
        if verbose:
            print(f"    gamma={gamma:>5}: best key={grid[gi, best_ci]:7.4g} Hz "
                  f"at c0={c0_grid[best_ci]:g} s", flush=True)

    slice_best = grid.max(axis=1)
    base_gi = int(np.argmin(np.abs(np.array(gamma_grid))))
    best_gi = int(np.argmax(slice_best))
    baseline = float(slice_best[base_gi])
    best = float(slice_best[best_gi])
    return {
        "c0_grid": list(c0_grid),
        "gamma_grid": list(gamma_grid),
        "key_hz_grid": grid.tolist(),
        "baseline_gamma": float(gamma_grid[base_gi]),
        "baseline_key_hz": baseline,
        "best_gamma": float(gamma_grid[best_gi]),
        "best_c0": float(c0_grid[int(np.argmax(grid[best_gi]))]),
        "best_key_hz": best,
        "relative_gain": (best - baseline) / baseline if baseline > 0 else None,
        "n_samples": n_samples,
    }
