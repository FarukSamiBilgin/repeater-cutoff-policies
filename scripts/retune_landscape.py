"""R1 — re-tune the cutoff scale at every span exponent, and measure the frontier.

Two defects in the first version of the span-exponent scan, both found by an
adversarial read of the driver rather than of the numbers:

**The landscape was a fixed-scale slice.**  `gamma_landscape()` tuned c0 once at
gamma = 0 and then held it while sweeping gamma.  Since t_cut = c0 * span^gamma,
moving gamma at fixed c0 changes the cutoff's overall magnitude as well as its
span-dependence -- at 8 segments, gamma = +1 hands a span-8 link 320 slots while
gamma = -1 hands it 5.  Most of the reported landscape was therefore the scale
being wrong, not the exponent being wrong, and the asymmetry that was read off it
does not belong to the exponent at all.  Here c0 is re-tuned at every gamma, so
each point is that exponent's best achievable policy.

**The head-to-head compared two policies at different operating points.**  The
fidelity gate delivered a better state and took longer; that is the shape of any
rate-fidelity trade-off, not a mechanism.  The question is whether the two
families trace *different frontiers*, which needs both swept and compared at
matched delivery time.  That is what `frontier()` produces.

Also refines the c0 grid: the original had no point between 0.015 and 0.010, and
the 8-segment optimum appears to fall in the gap, which would leave the published
baseline below its own optimum and the headline understated.
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
from crn_comparison import run_paired, jackknife_paired  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

C0_GRID = [0.080, 0.060, 0.050, 0.040, 0.035, 0.030, 0.025, 0.020, 0.017,
           0.015, 0.013, 0.012, 0.010, 0.008, 0.006, 0.004]
W_GRID = [0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.905, 0.90, 0.895, 0.89,
          0.88, 0.86, 0.83, 0.80, 0.75]
GAMMAS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]


def cap_for(base: dict, factor: float = 400.0) -> int:
    ref = ChainConfig(cutoff_s=None, **base)
    slots = (run_paired({"r": ref}, 400, 11)["r"]["mean_delivery_time_s"]
             / ref.attempt_time_s)
    return int(factor * slots)


def tune_c0(base: dict, gamma: float, n: int, seed: int, cap: int):
    cfgs = {f"{c}": ChainConfig(cutoff_s=c, cutoff_gamma=gamma, **base)
            for c in C0_GRID}
    res = run_paired(cfgs, n, seed, max_slots=cap)
    best = max(C0_GRID, key=lambda c: res[f"{c}"]["secret_key_hz"])
    return float(best), res


def retuned_landscape(name: str, base: dict, n_tune: int, n_meas: int,
                      cap: int) -> dict:
    print(f"\n=== {name}: span exponent with c0 re-tuned at every gamma")
    tuned = {}
    for g in GAMMAS:
        c0, _ = tune_c0(base, g, n_tune, 4242, cap)
        tuned[g] = c0
        print(f"    gamma={g:>5}  c0* = {c0:g} s", flush=True)

    cfgs = {f"g{g}": ChainConfig(cutoff_s=tuned[g], cutoff_gamma=g, **base)
            for g in GAMMAS}
    res = run_paired(cfgs, n_meas, 515151, max_slots=cap)
    ref = res["g0.0"]
    rows = []
    for g in GAMMAS:
        r = res[f"g{g}"]
        jk = jackknife_paired(r, ref)
        rel = jk["difference_hz"] / ref["secret_key_hz"]
        sig = abs(jk["difference_hz"]) / jk["sem"] if jk["sem"] > 0 else 0.0
        rows.append({"gamma": float(g), "c0_star": tuned[g],
                     "secret_key_hz": r["secret_key_hz"],
                     "relative": rel, "sem_rel": jk["sem"] / ref["secret_key_hz"],
                     "sigma": sig, "censored_fraction": r["censored_fraction"]})
        print(f"    gamma={g:>5}  c0*={tuned[g]:<6g} key={r['secret_key_hz']:8.5f}  "
              f"{rel:+7.2%} +/- {rows[-1]['sem_rel']:.2%}  ({sig:5.1f} sigma)",
              flush=True)
    return {"tuned_c0": {str(k): v for k, v in tuned.items()}, "rows": rows}


def frontier(name: str, base: dict, n: int, cap: int) -> dict:
    """Both families swept, so the comparison is of curves, not of two points."""
    print(f"\n=== {name}: rate-fidelity frontier, both families swept")
    cfgs = {}
    for c in C0_GRID:
        cfgs[f"age:{c}"] = ChainConfig(cutoff_s=c, cutoff_gamma=0.0, **base)
    for w in W_GRID:
        cfgs[f"fid:{w}"] = ChainConfig(cutoff_mode="fidelity", w_min=w,
                                       cutoff_s=None, **base)
    res = run_paired(cfgs, n, 626262, max_slots=cap)
    out = {"age": [], "fidelity": []}
    for k, r in res.items():
        fam, val = k.split(":")
        out["age" if fam == "age" else "fidelity"].append(
            {"param": float(val), "mean_delivery_time_s": r["mean_delivery_time_s"],
             "werner": r["werner"], "secret_key_hz": r["secret_key_hz"],
             "censored_fraction": r["censored_fraction"]})
    for fam in ("age", "fidelity"):
        out[fam].sort(key=lambda d: d["mean_delivery_time_s"])
        best = max(out[fam], key=lambda d: d["secret_key_hz"])
        print(f"    {fam:<9} best: param={best['param']:g}  "
              f"key={best['secret_key_hz']:.5f}  E[T]={best['mean_delivery_time_s']:.4f}  "
              f"w={best['werner']:.4f}")

    # matched-time comparison: for the fidelity optimum, find the age policy with
    # the closest E[T] and compare the state it delivers.
    fbest = max(out["fidelity"], key=lambda d: d["secret_key_hz"])
    amatch = min(out["age"],
                 key=lambda d: abs(d["mean_delivery_time_s"]
                                   - fbest["mean_delivery_time_s"]))
    print(f"    matched E[T]: age c0={amatch['param']:g} at "
          f"{amatch['mean_delivery_time_s']:.4f} s -> w={amatch['werner']:.4f}, "
          f"key={amatch['secret_key_hz']:.5f}")
    print(f"                  fid w_min={fbest['param']:g} at "
          f"{fbest['mean_delivery_time_s']:.4f} s -> w={fbest['werner']:.4f}, "
          f"key={fbest['secret_key_hz']:.5f}")
    dom = (amatch["werner"] > fbest["werner"]
           and amatch["secret_key_hz"] > fbest["secret_key_hz"])
    print(f"    -> at matched delivery time the age gate is "
          f"{'DOMINANT on both axes' if dom else 'not dominant'}")
    out["matched"] = {"age": amatch, "fidelity": fbest, "age_dominates": bool(dom)}
    return out


def main() -> None:
    base4 = dict(total_length_km=400.0, n_segments=4, p0=0.1,
                 photons_per_herald=1, t_coh_s=0.5, w_init=0.99, w_swap=0.99)
    base8 = dict(total_length_km=600.0, n_segments=8, p0=0.1,
                 photons_per_herald=1, t_coh_s=0.5, w_init=0.99, w_swap=0.99)
    out = {}
    path = os.path.join(RESULTS, "retuned_landscape.json")

    def save():
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)

    for label, base, nt, nm in (("4seg", base4, 8000, 40000),
                                ("8seg", base8, 6000, 20000)):
        cap = cap_for(base)
        print(f"\n######## {label}: delivery cap {cap} slots")
        out[label] = {"cap": cap,
                      "landscape": retuned_landscape(label, base, nt, nm, cap)}
        save()
        out[label]["frontier"] = frontier(label, base, nm, cap)
        save()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for col, label in enumerate(("4seg", "8seg")):
        rows = out[label]["landscape"]["rows"]
        ax = axes[0, col]
        g = [r["gamma"] for r in rows]
        d = [100 * r["relative"] for r in rows]
        e = [100 * r["sem_rel"] for r in rows]
        ax.errorbar(g, d, yerr=e, fmt="o-", ms=4, capsize=3, color="k")
        ax.axhline(0.0, color="0.6", ls=":")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("key rate vs uniform (%)")
        ax.set_title(f"{label}: $c_0$ re-tuned at every $\\gamma$", fontsize=9)
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        fr = out[label]["frontier"]
        for fam, mk, c in (("age", "o", "tab:blue"), ("fidelity", "s", "tab:red")):
            t = [p["mean_delivery_time_s"] for p in fr[fam]]
            w = [p["werner"] for p in fr[fam]]
            ax.plot(t, w, mk + "-", ms=4, color=c, label=fam)
        ax.set_xlabel(r"$E[T]$ (s)")
        ax.set_ylabel(r"delivered $\bar w$")
        ax.set_title(f"{label}: frontier, both families swept", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Span exponent re-tuned, and the rate-fidelity frontier",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "retuned_landscape.png"), dpi=140)
    plt.close(fig)
    print(f"\nwrote {path} and retuned_landscape.png")


if __name__ == "__main__":
    main()
