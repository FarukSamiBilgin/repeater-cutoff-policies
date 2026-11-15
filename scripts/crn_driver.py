"""C1-run — the paired policy comparison, and what it does to the two nulls.

Two results from `policy_comparison.py` were reported as "within noise":

  * a span-indexed age cutoff t_cut(span) = c0 * span^gamma against a uniform
    one -- best gamma came out 0, with the tuned gamma = -0.5 point 2.1% worse;
  * an accumulated-decay (Werner-parameter) gate against a uniform age gate --
    -0.4%, inside a two-sigma band of +/-1.7%.

Neither was a measurement.  The second in particular excludes an 11.7% gap and
nothing narrower, which is a statement about the sample size.  This run repeats
both under common random numbers (see `crn_comparison.py` for the coupling) and
with the ensemble raised, so that the paired difference and its error bar say
something about the policies.

Reported per policy, per the literature map's item 5.10: not just the secret-key
rate but the delivered Werner parameter and the delivery-time distribution, so a
policy that trades rate against fidelity is visible doing it rather than hidden
inside a single scalar.
"""

from __future__ import annotations

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from repeater_chain import ChainConfig  # noqa: E402
from crn_comparison import (  # noqa: E402
    DrawPool, simulate_chain_crn, run_paired, validate, jackknife_paired,
    key_rate,
)

RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
N_SAMPLES = 40000
N_TUNE = 8000


def tune(base: dict, key: str, grid, n_samples: int = N_TUNE,
         seed: int = 5150, max_slots: int = 10**9,
         **fixed) -> tuple[float, float]:
    """Pick a policy parameter on a shared stream, so the grid is paired too."""
    cfgs = {}
    for v in grid:
        kw = dict(fixed)
        kw[key] = v
        cfgs[f"{v}"] = ChainConfig(**kw, **base)
    res = run_paired(cfgs, n_samples, seed, max_slots=max_slots)
    best_v, best_k = None, -1.0
    for v in grid:
        k = res[f"{v}"]["secret_key_hz"]
        if k > best_k:
            best_v, best_k = v, k
    return float(best_v), float(best_k)


def gamma_landscape(base: dict, c0: float, gammas, n_samples: int,
                    seed: int, max_slots: int) -> dict:
    cfgs = {f"g{g}": ChainConfig(cutoff_s=c0, cutoff_gamma=g, **base)
            for g in gammas}
    res = run_paired(cfgs, n_samples, seed, max_slots=max_slots)
    ref = res["g0.0"]
    rows = []
    for g in gammas:
        r = res[f"g{g}"]
        jk = jackknife_paired(r, ref)
        rows.append({"gamma": float(g), "secret_key_hz": r["secret_key_hz"],
                     "werner": r["werner"],
                     "mean_delivery_time_s": r["mean_delivery_time_s"],
                     "diff_vs_uniform_hz": jk["difference_hz"],
                     "sem": jk["sem"],
                     "relative": jk["difference_hz"] / ref["secret_key_hz"],
                     "sigma": (abs(jk["difference_hz"]) / jk["sem"]
                               if jk["sem"] > 0 else float("inf")),
                     "corr_time": jk["corr_time"],
                     "censored_fraction": r["censored_fraction"]})
        print(f"    gamma={g:>5}  key={r['secret_key_hz']:8.5f} Hz  "
              f"diff={jk['difference_hz']:+9.6f} +/- {jk['sem']:.6f}  "
              f"({rows[-1]['relative']:+.2%}, {rows[-1]['sigma']:.1f} sigma)"
              + (f"   [CENSORED {r['censored_fraction']:.0%} -- degenerate]"
                 if r["censored_fraction"] > 0 else ""),
              flush=True)
    return {"c0": c0, "rows": rows}


def case(name: str, base: dict, c0_grid, w_grid, gamma_grid,
         n_samples: int = N_SAMPLES, n_tune: int = N_TUNE,
         cap_factor: float = 400.0) -> dict:
    print(f"\n=== {name}")
    # Cap a single delivery at cap_factor times the no-cutoff mean, which is
    # far beyond anything a working policy needs and short enough that a
    # degenerate one is reported rather than waited for. Set before tuning,
    # because a degenerate grid point would otherwise hang the tuner.
    ref_cfg = ChainConfig(cutoff_s=None, **base)
    ref_slots = (run_paired({"r": ref_cfg}, 400, 11)["r"]["mean_delivery_time_s"]
                 / ref_cfg.attempt_time_s)
    max_slots = int(cap_factor * ref_slots)
    print(f"    delivery cap: {max_slots} slots "
          f"({cap_factor:g}x the no-cutoff mean of {ref_slots:.0f})")

    print("  tuning on a shared stream ...", flush=True)
    c0_star, _ = tune(base, "cutoff_s", c0_grid, n_samples=n_tune,
                      max_slots=max_slots, cutoff_gamma=0.0)
    w_star, _ = tune(base, "w_min", w_grid, n_samples=n_tune,
                     max_slots=max_slots, cutoff_mode="fidelity",
                     cutoff_s=None)
    print(f"    uniform age  : c0    = {c0_star:g} s")
    print(f"    fidelity gate: w_min = {w_star:g}")

    print(f"  gamma landscape, paired, {n_samples} samples:", flush=True)
    land = gamma_landscape(base, c0_star, gamma_grid, n_samples, 606061,
                           max_slots)

    print(f"  head-to-head, paired, {n_samples} samples:", flush=True)
    cfgs = {
        "uniform_age": ChainConfig(cutoff_s=c0_star, cutoff_gamma=0.0, **base),
        "fidelity_gate": ChainConfig(cutoff_mode="fidelity", w_min=w_star,
                                     cutoff_s=None, **base),
        "no_cutoff": ChainConfig(cutoff_s=None, **base),
    }
    res = run_paired(cfgs, n_samples, 707071, max_slots=max_slots)
    ref = res["uniform_age"]
    summary, comparisons = {}, {}
    for k, r in res.items():
        summary[k] = {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
        print(f"    {k:<14} key={r['secret_key_hz']:8.5f} Hz  "
              f"w={r['werner']:.4f} (p10 {r['werner_p10']:.4f}, "
              f"p90 {r['werner_p90']:.4f})  E[T]={r['mean_delivery_time_s']:.4f} s "
              f"(median {r['time_p50']:.4f}, p90 {r['time_p90']:.4f})")
    for k in ("fidelity_gate", "no_cutoff"):
        jk = jackknife_paired(res[k], ref)
        rel = jk["difference_hz"] / ref["secret_key_hz"]
        sig = abs(jk["difference_hz"]) / jk["sem"] if jk["sem"] > 0 else float("inf")
        comparisons[k] = {"difference_hz": jk["difference_hz"], "sem": jk["sem"],
                          "relative": rel, "sigma": sig,
                          "corr_time": jk["corr_time"],
                          "corr_werner": jk["corr_werner"],
                          "significant": bool(sig > 2.0)}
        print(f"    {k} vs uniform_age: {jk['difference_hz']:+.6f} "
              f"+/- {jk['sem']:.6f} Hz ({rel:+.2%}), {sig:.1f} sigma, "
              f"pair corr(T) = {jk['corr_time']:.3f} "
              f"-> {'MEASURED DIFFERENCE' if sig > 2 else 'no difference'}")
    return {"case": name, "tuned": {"c0": c0_star, "w_min": w_star},
            "max_slots": max_slots, "gamma_landscape": land,
            "policies": summary, "comparisons": comparisons,
            "n_samples": n_samples}


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    base4 = dict(total_length_km=400.0, n_segments=4, p0=0.1,
                 photons_per_herald=1, t_coh_s=0.5, w_init=0.99, w_swap=0.99)
    base8 = dict(total_length_km=600.0, n_segments=8, p0=0.1,
                 photons_per_herald=1, t_coh_s=0.5, w_init=0.99, w_swap=0.99)

    print("engine check: the CRN engine must reproduce the original in "
          "distribution")
    v = validate(ChainConfig(cutoff_s=0.03, cutoff_gamma=0.0, **base4), 6000)
    print(f"   mean delivery time {v['crn_mean_time_s']:.6f} s vs "
          f"{v['ref_mean_time_s']:.6f} s  ({v['sigma_time']:.2f} sigma, "
          f"{v['rel_diff_time']:.2%})")
    print(f"   Werner {v['crn_werner']:.6f} vs {v['ref_werner']:.6f}  "
          f"({v['rel_diff_werner']:.2%})")

    c0_grid = [0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.006]
    w_grid = [0.95, 0.93, 0.91, 0.90, 0.88, 0.86, 0.83, 0.80]
    gamma_grid = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]

    # Written after every case. The 8-segment geometry costs an order of
    # magnitude more per sample than the 4-segment one, and losing a finished
    # case because a later one was interrupted is a waste of a finished case.
    out_path = os.path.join(RESULTS, "crn_comparison.json")
    cases = []

    def save():
        with open(out_path, "w") as fh:
            json.dump({"engine_check": v, "cases": cases}, fh, indent=2)

    cases.append(case("4 segments, 400 km, T_coh = 0.5 s", base4,
                      c0_grid, w_grid, gamma_grid))
    save()
    cases.append(case("8 segments, 600 km, T_coh = 0.5 s", base8,
                      c0_grid, w_grid, gamma_grid, n_samples=8000,
                      n_tune=2500))
    save()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, c in zip(axes, cases):
        rows = c["gamma_landscape"]["rows"]
        g = [r["gamma"] for r in rows]
        d = [100 * r["relative"] for r in rows]
        e = [100 * r["sem"] / c["policies"]["uniform_age"]["secret_key_hz"]
             for r in rows]
        ax.errorbar(g, d, yerr=e, fmt="o-", ms=4, capsize=3, color="k")
        ax.axhline(0.0, color="0.6", ls=":")
        ax.set_xlabel(r"$\gamma$  in  $t_{\rm cut}(\mathrm{span})=c_0\,$span$^\gamma$")
        ax.set_ylabel("secret-key rate vs uniform (%)")
        ax.set_title(c["case"], fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Span-indexed cutoff against a uniform one, paired samples",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIGURES, "crn_gamma_landscape.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {path} and crn_comparison.json")


if __name__ == "__main__":
    main()
