"""V2 — validate the generation layer where the results actually live.

The chain simulator's only analytic check so far is the two-segment, no-cutoff,
p_s = 1 identity E[max(G1,G2)] = 2/p - 1/(1-(1-p)^2).  Every reported result
lives in the with-cutoff regime instead, where that identity says nothing.  Two
exact references cover that regime and both are free:

**Praxmeyer (2013), Eq. (20).**  N segments, a *global* cutoff: all N links must
be up within tau slots of the first success, or everything is discarded and the
round restarts.  Writing q = 1 - p, the expected number of slots is

    <K(N,tau)> = [ 1 - (1-q^tau)^N + (1-q^N)( tau - sum_{j=1}^{tau-1}(1-q^j)^N ) ]
                 / [ (1-q^(tau+1))^N - q^N (1-q^tau)^N ]

with the two limits she proves: tau = 0 gives p^-N (all N must succeed in the
same slot) and tau -> infinity gives sum_k (-1)^(k+1) C(N,k)/(1-q^k), the
expected maximum of N geometrics.  Both limits are re-derived and checked below
before the formula is used as a target.

Her rule is NOT our per-link discard -- it is an end-to-end restart -- so this
validates the generation layer under a *different* policy, not our policy.  That
is the honest scope: it fixes the geometry, the cutoff bookkeeping and the
slot accounting, which is where an off-by-one would live.

**Khatri, Matyas, Siddiqui & Dowling (2019), Theorem 1.**  For M links and any
finite cutoff n*,

    sum_{k=1}^{M} C(M,k)(-1)^(k+1)/(1-(1-p)^k)  <=  E[N(M,n*)]  <=  1/p^M

-- the no-cutoff value below, the all-in-one-slot value above.  Free, needs no
implementation, and holds for every cutoff we sweep, so it is a cheap
regression test rather than a single-point check.
"""

from __future__ import annotations

import json
import math
import os
import sys
from itertools import combinations  # noqa: F401  (documentation of the binomials)

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

RESULTS = os.path.join(_HERE, "..", "results")


def praxmeyer_slots(n: int, tau: int, p: float) -> float:
    """Eq. (20): expected slots for n links under a global tau-cutoff."""
    q = 1.0 - p
    num = (1.0 - (1.0 - q ** tau) ** n
           + (1.0 - q ** n) * (tau - sum((1.0 - q ** j) ** n
                                         for j in range(1, tau))))
    den = (1.0 - q ** (tau + 1)) ** n - q ** n * (1.0 - q ** tau) ** n
    return num / den


def expected_max_geometric(n: int, p: float) -> float:
    """sum_k (-1)^(k+1) C(n,k)/(1-q^k) -- E[max of n geometric(p)]."""
    q = 1.0 - p
    return sum((-1.0) ** (k + 1) * math.comb(n, k) / (1.0 - q ** k)
               for k in range(1, n + 1))


def simulate_global_restart(n: int, tau: int, p: float, n_samples: int,
                            seed: int, max_slots: int = 2_000_000
                            ) -> tuple[float, float]:
    """Slot-by-slot simulation of Praxmeyer's rule, all samples in lockstep.

    Each slot every down link attempts independently with probability p.  The
    round succeeds the moment all n links are simultaneously up.  If instead the
    oldest live link reaches age tau without that happening, every live link is
    discarded and the round continues from the next slot.

    Samples advance together rather than one at a time, so the cost is the
    longest single round rather than the sum over rounds.
    """
    rng = np.random.default_rng(seed)
    alive = np.zeros((n_samples, n), dtype=bool)
    birth = np.zeros((n_samples, n), dtype=np.int64)
    done = np.zeros(n_samples, dtype=bool)
    result = np.zeros(n_samples, dtype=np.int64)
    t = 0
    while not done.all():
        t += 1
        if t > max_slots:
            raise RuntimeError(f"n={n} tau={tau} p={p} exceeded {max_slots} slots")
        act = ~done
        hit = (rng.random((n_samples, n)) < p) & ~alive & act[:, None]
        alive |= hit
        birth[hit] = t
        finished = act & alive.all(axis=1)
        result[finished] = t
        done |= finished
        act = ~done
        any_alive = act & alive.any(axis=1)
        if any_alive.any():
            oldest = np.where(alive, birth, np.iinfo(np.int64).max).min(axis=1)
            expire = any_alive & ((t - oldest) >= tau)
            if expire.any():
                alive[expire] = False
    vals = result.astype(float)
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(n_samples))


def khatri_bracket(m: int, p: float) -> tuple[float, float]:
    return expected_max_geometric(m, p), 1.0 / p ** m


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    out = {"praxmeyer_limits": {}, "praxmeyer_points": [], "khatri": []}

    print("Praxmeyer Eq. (20) -- the two limits she proves, checked first")
    ok_all = True
    for n in (2, 4, 8):
        for p in (0.05, 0.2, 0.5):
            got0 = praxmeyer_slots(n, 0, p)
            want0 = p ** (-n)
            got_inf = praxmeyer_slots(n, 4000, p)
            want_inf = expected_max_geometric(n, p)
            e0 = abs(got0 - want0) / want0
            ei = abs(got_inf - want_inf) / want_inf
            ok = e0 < 1e-9 and ei < 1e-9
            ok_all &= ok
            out["praxmeyer_limits"][f"n{n}_p{p}"] = {
                "tau0_rel_err": e0, "tau_inf_rel_err": ei, "ok": bool(ok)}
            print(f"   n={n} p={p:<5}  tau=0: {got0:12.4f} vs {want0:12.4f} "
                  f"({e0:.1e})   tau->inf: {got_inf:8.4f} vs {want_inf:8.4f} "
                  f"({ei:.1e})   {'ok' if ok else 'FAIL'}")
    print(f"   -> limits {'all reproduced' if ok_all else 'FAILED'}\n")

    print("simulation against Eq. (20), global-restart cutoff")
    print("   (points whose expected round exceeds the slot budget are listed "
          "as formula-only)")
    print(f"   {'n':>2} {'p':>5} {'tau':>4} {'formula':>12} {'simulated':>12} "
          f"{'sigma':>7}")
    worst, skipped = 0.0, 0
    BUDGET = 4000.0          # expected slots per round we are willing to walk
    for n in (2, 4, 8):
        for p in (0.1, 0.3):
            for tau in (1, 3, 10, 30):
                f = praxmeyer_slots(n, tau, p)
                if f > BUDGET:
                    skipped += 1
                    out["praxmeyer_points"].append(
                        {"n": n, "p": p, "tau": tau, "formula": f,
                         "simulated": None, "skipped": "over slot budget"})
                    print(f"   {n:>2} {p:>5} {tau:>4} {f:>12.4f} "
                          f"{'formula only':>12}")
                    continue
                ns = 20000 if f < 200 else 4000
                m, sem = simulate_global_restart(n, tau, p, ns,
                                                 4242 + n * 97 + tau)
                z = abs(m - f) / sem if sem > 0 else 0.0
                worst = max(worst, z)
                out["praxmeyer_points"].append(
                    {"n": n, "p": p, "tau": tau, "formula": f,
                     "simulated": m, "sem": sem, "sigma": z, "n_samples": ns})
                print(f"   {n:>2} {p:>5} {tau:>4} {f:>12.4f} "
                      f"{m:>9.4f}+/-{sem:<6.4f} {z:>6.2f}")
    n_done = len([q for q in out["praxmeyer_points"] if q.get("simulated")])
    print(f"   -> worst deviation {worst:.2f} sigma over {n_done} simulated "
          f"points ({skipped} formula-only)\n")
    out["praxmeyer_worst_sigma"] = worst
    out["praxmeyer_n_simulated"] = n_done

    print("Khatri et al. Theorem 1 bracket -- must hold for every finite cutoff")
    all_in = True
    for m_seg in (2, 4, 8):
        for p in (0.1, 0.3):
            lo, hi = khatri_bracket(m_seg, p)
            vals = []
            for tau in (1, 3, 10, 30, 100):
                if praxmeyer_slots(m_seg, tau, p) > 4000.0:
                    continue
                mean, sem = simulate_global_restart(m_seg, tau, p, 20000,
                                                    9090 + m_seg * 31 + tau)
                # The bracket is a statement about the exact expectation, so a
                # Monte Carlo estimate has to be compared to it WITH its error
                # bar. The lower end is the tau -> infinity limit itself, which
                # the sweep approaches from above, so at large tau the estimate
                # sits within noise of the bound by construction; calling that
                # a violation would be reading the SEM as a discrepancy.
                ok = (mean > lo - 2.0 * sem) and (mean < hi + 2.0 * sem)
                vals.append((tau, mean, sem, ok))
            inside = bool(vals) and all(v[3] for v in vals)
            all_in &= inside
            out["khatri"].append({"m": m_seg, "p": p, "lower": lo, "upper": hi,
                                  "values": [{"tau": t, "mean": v,
                                              "sem": e, "inside": i}
                                             for t, v, e, i in vals],
                                  "all_inside": inside})
            span = f"[{lo:.3f}, {hi:.3f}]"
            got = ", ".join(f"{v:.3f}+/-{e:.3f}" for _, v, e, _ in vals)
            print(f"   M={m_seg} p={p:<5} bracket {span:<26} "
                  f"cutoff sweep {got}  -> {'inside' if inside else 'VIOLATED'}")
    out["khatri_all_inside"] = bool(all_in)

    print(f"\n  VERDICT: Praxmeyer worst {worst:.2f} sigma; "
          f"Khatri bracket {'holds everywhere' if all_in else 'VIOLATED'}")
    with open(os.path.join(RESULTS, "with_cutoff_validation.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote with_cutoff_validation.json")


if __name__ == "__main__":
    main()
