"""Does a richer cutoff policy actually buy anything? With error bars.

Two candidate generalisations of the standard global age cutoff were added to
the model:

  span-dependent age    t_cut(span) = c0 * span^gamma
  fidelity threshold    discard when the Werner parameter drops below w_min

Both are strictly larger policy classes than the baseline, so both *can* only
match or beat it once their own free parameter is optimised.  The question is
whether the gain is real or Monte Carlo noise, and answering that needs
replicates, not a single run.

Protocol: optimise each policy's free parameter on a coarse grid, then run the
winner of each at `n_replicates` independent seeds and compare means with
standard errors.  A difference smaller than two combined standard errors is
reported as no gain, because that is what it is.
"""

from __future__ import annotations

import json
import sys

import os

# One BLAS thread per process: these matrices are small enough that threading
# them buys nothing, and several of these scans running at once oversubscribe
# the machine badly — the same scan went from an hour to 90 seconds once the
# thread count was pinned.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")



import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from repeater_chain import ChainConfig, simulate_chain_fast  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")


def _key(cfg: ChainConfig, n_samples: int, seed: int) -> float:
    return simulate_chain_fast(cfg, n_samples=n_samples, seed=seed)["secret_key_hz"]


def replicate(cfg: ChainConfig, n_samples: int, n_replicates: int,
              seed0: int = 100) -> tuple[float, float, list[float]]:
    vals = [_key(cfg, n_samples, seed0 + r) for r in range(n_replicates)]
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))), vals


def optimise_age(base: dict, c0_grid, gamma: float, n_samples: int,
                 seed: int = 7) -> tuple[float, float]:
    best_c0, best_key = None, -1.0
    for c0 in c0_grid:
        k = _key(ChainConfig(cutoff_s=c0, cutoff_gamma=gamma, **base),
                 n_samples, seed)
        if k > best_key:
            best_c0, best_key = c0, k
    return best_c0, best_key


def optimise_fidelity(base: dict, w_grid, n_samples: int,
                      seed: int = 7) -> tuple[float, float]:
    best_w, best_key = None, -1.0
    for w in w_grid:
        k = _key(ChainConfig(cutoff_mode="fidelity", w_min=w, cutoff_s=None,
                             **base), n_samples, seed)
        if k > best_key:
            best_w, best_key = w, k
    return best_w, best_key


def compare(name: str, base: dict, c0_grid, w_grid, gamma_grid,
            n_samples: int = 1500, n_replicates: int = 8) -> dict:
    print(f"\n{name}", flush=True)
    print("  tuning each policy ...", flush=True)

    c0_star, _ = optimise_age(base, c0_grid, 0.0, n_samples)
    gamma_star, c0_g_star, best_g_key = 0.0, c0_star, -1.0
    for g in gamma_grid:
        c0_g, k = optimise_age(base, c0_grid, g, n_samples)
        if k > best_g_key:
            gamma_star, c0_g_star, best_g_key = g, c0_g, k
    w_star, _ = optimise_fidelity(base, w_grid, n_samples)
    print(f"    global age    : c0={c0_star:g} s", flush=True)
    print(f"    span-dependent: c0={c0_g_star:g} s, gamma={gamma_star:g}",
          flush=True)
    print(f"    fidelity      : w_min={w_star:g}", flush=True)

    print(f"  {n_replicates} replicates each ...", flush=True)
    policies = {
        "global_age": ChainConfig(cutoff_s=c0_star, cutoff_gamma=0.0, **base),
        "span_dependent_age": ChainConfig(cutoff_s=c0_g_star,
                                          cutoff_gamma=gamma_star, **base),
        "fidelity_threshold": ChainConfig(cutoff_mode="fidelity", w_min=w_star,
                                          cutoff_s=None, **base),
    }
    out = {"case": name, "tuned": {"c0": c0_star, "gamma": gamma_star,
                                   "c0_gamma": c0_g_star, "w_min": w_star},
           "policies": {}}
    for pname, cfg in policies.items():
        mean, sem, vals = replicate(cfg, n_samples, n_replicates)
        out["policies"][pname] = {"mean_key_hz": mean, "sem": sem,
                                  "replicates": vals}
        print(f"    {pname:<20} {mean:7.4g} +/- {sem:.4g} Hz", flush=True)

    baseline = out["policies"]["global_age"]
    verdicts = {}
    for pname, res in out["policies"].items():
        if pname == "global_age":
            continue
        diff = res["mean_key_hz"] - baseline["mean_key_hz"]
        combined = 2.0 * np.hypot(res["sem"], baseline["sem"])
        verdicts[pname] = {
            "difference_hz": diff,
            "relative": diff / baseline["mean_key_hz"],
            "two_sigma_band": float(combined),
            "significant": bool(abs(diff) > combined),
        }
        tag = "SIGNIFICANT" if abs(diff) > combined else "within noise"
        print(f"    vs baseline: {diff:+.4g} Hz "
              f"({diff / baseline['mean_key_hz']:+.1%}), "
              f"2sigma={combined:.4g} -> {tag}", flush=True)
    out["verdicts"] = verdicts
    return out


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    c0_grid = [0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.006]
    w_grid = [0.95, 0.93, 0.91, 0.90, 0.88, 0.86, 0.83, 0.80]
    gamma_grid = [-1.0, -0.5, 0.5, 1.0]

    cases = []
    out_path = os.path.join(RESULTS, "policy_comparison.json")

    def save():
        with open(out_path, "w") as fh:
            json.dump({"cases": cases}, fh, indent=2)

    cases.append(compare(
        "4 segments, 400 km, T_coh = 0.5 s",
        dict(total_length_km=400.0, n_segments=4, p0=0.1,
             photons_per_herald=1, t_coh_s=0.5, w_init=0.99, w_swap=0.99),
        c0_grid, w_grid, gamma_grid))
    save()
    cases.append(compare(
        "8 segments, 600 km, T_coh = 0.5 s",
        dict(total_length_km=600.0, n_segments=8, p0=0.1,
             photons_per_herald=1, t_coh_s=0.5, w_init=0.99, w_swap=0.99),
        c0_grid, w_grid, gamma_grid, n_samples=1000, n_replicates=6))
    save()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
