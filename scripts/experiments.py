"""Two experiments on the repeater-chain model.

E1  Cutoff-policy Pareto front
    Sweep the memory cutoff and trace the rate/fidelity trade-off, plus the
    secret-key rate that results.  This is the question the NetQ policy work
    asks: a cutoff throws away stale links, buying fidelity with rate, and the
    key rate is not monotone in either.

E2  Crossing the repeaterless bound
    Sweep total distance for several segment counts and find where the chain's
    secret-key rate overtakes the PLOB ceiling — the claim USTC makes at
    ~230 km with atomic-ensemble memories.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repeater_chain import (  # noqa: E402
    ChainConfig, simulate_chain, simulate_chain_fast, plob_bound_hz,
    selfcheck, cross_validate,
)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")


def e1_cutoff_pareto(samples: int = 3000) -> dict:
    """Rate / fidelity / key-rate vs memory cutoff, at fixed geometry."""
    # A memory-limited regime on purpose: without a cutoff the chain delivers
    # entanglement but at a fidelity below the QKD threshold, so the key rate
    # is exactly zero. The cutoff is what makes key distribution possible.
    base = dict(total_length_km=400.0, n_segments=4, p0=0.1,
                photons_per_herald=1, t_coh_s=0.5, w_init=0.99, w_swap=0.99)
    # cutoff in units of the memory coherence time
    ratios = np.array([np.inf, 0.5, 0.25, 0.12, 0.06, 0.03, 0.02, 0.015,
                       0.01, 0.007, 0.005, 0.003])
    rows = []
    for r in ratios:
        cutoff = None if np.isinf(r) else r * base["t_coh_s"]
        cfg = ChainConfig(cutoff_s=cutoff, **base)
        res = simulate_chain_fast(cfg, n_samples=samples, seed=7)
        rows.append({"cutoff_over_tcoh": float(r), "cutoff_s": cutoff, **res})
        print(f"  cutoff/T_coh={r:>6}: rate={res['rate_hz']:8.3g} Hz  "
              f"F={res['fidelity']:.4f}  key={res['secret_key_hz']:8.3g} Hz")
    finite = [r for r in rows if r["cutoff_s"] is not None]
    best = max(finite, key=lambda r: r["secret_key_hz"])
    no_cut = [r for r in rows if r["cutoff_s"] is None][0]
    print(f"  optimum: cutoff={best['cutoff_over_tcoh']}*T_coh -> "
          f"{best['secret_key_hz']:.4g} Hz key "
          f"(no cutoff: {no_cut['secret_key_hz']:.4g} Hz)")
    return {"base": base, "rows": rows,
            "optimum": {"cutoff_over_tcoh": best["cutoff_over_tcoh"],
                        "secret_key_hz": best["secret_key_hz"],
                        "fidelity": best["fidelity"],
                        "rate_hz": best["rate_hz"]},
            "no_cutoff": {"secret_key_hz": no_cut["secret_key_hz"],
                          "fidelity": no_cut["fidelity"],
                          "rate_hz": no_cut["rate_hz"]}}


def e2_plob_crossing(samples: int = 1200) -> dict:
    """Secret-key rate vs distance for several segment counts, vs PLOB."""
    distances = np.array([25, 50, 75, 100, 150, 200, 300, 400, 500, 600],
                         dtype=float)
    seg_counts = [2, 4, 8]
    out = {"distances_km": distances.tolist(), "series": {}}
    for n_seg in seg_counts:
        key_hz, rate_hz, fids = [], [], []
        for d in distances:
            cfg = ChainConfig(total_length_km=float(d), n_segments=n_seg,
                              p0=0.1, photons_per_herald=1, t_coh_s=2.0,
                              w_init=0.99, w_swap=0.99,
                              cutoff_s=0.25 * 2.0)
            res = simulate_chain_fast(cfg, n_samples=samples, seed=11)
            key_hz.append(res["secret_key_hz"])
            rate_hz.append(res["rate_hz"])
            fids.append(res["fidelity"])
            print(f"  n_seg={n_seg} L={d:5.0f} km: rate={res['rate_hz']:8.3g} Hz  "
                  f"F={res['fidelity']:.4f}  key={res['secret_key_hz']:8.3g} Hz")
        out["series"][str(n_seg)] = {
            "secret_key_hz": key_hz, "rate_hz": rate_hz, "fidelity": fids,
        }
    out["plob_hz"] = [plob_bound_hz(float(d)) for d in distances]

    # first distance where each series overtakes PLOB
    crossings = {}
    for n_seg, s in out["series"].items():
        cross = None
        for d, k, p in zip(distances, s["secret_key_hz"], out["plob_hz"]):
            if k > p:
                cross = float(d)
                break
        crossings[n_seg] = cross
    out["first_distance_above_plob_km"] = crossings
    return out


def e3_efficiency_needed(target_km: float = 230.0, n_seg: int = 2,
                         samples: int = 900) -> dict:
    """Which link efficiency p0 puts the PLOB crossing at `target_km`?

    USTC report their atomic-ensemble chain overtaking the repeaterless bound
    beyond ~230 km.  In this model the crossing distance is set almost entirely
    by p0 (source * detector * heralding efficiency), so scanning p0 and reading
    off the crossing turns their reported distance into an implied efficiency —
    a number that can be checked against the experiment.
    """
    p0_grid = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 5e-3, 2e-3, 1e-3]
    distances = np.array([50, 75, 100, 150, 200, 230, 300, 400, 500, 600],
                         dtype=float)
    plob = np.array([plob_bound_hz(float(d)) for d in distances])
    rows = []
    for p0 in p0_grid:
        keys = []
        for d in distances:
            cfg = ChainConfig(total_length_km=float(d), n_segments=n_seg,
                              p0=p0, photons_per_herald=1, t_coh_s=2.0,
                              w_init=0.99, w_swap=0.99, cutoff_s=0.5)
            keys.append(simulate_chain_fast(cfg, n_samples=samples,
                                            seed=13)["secret_key_hz"])
        keys = np.array(keys)
        above = distances[keys > plob]
        cross = float(above.min()) if above.size else None
        rows.append({"p0": p0, "crossing_km": cross,
                     "secret_key_hz": keys.tolist()})
        print(f"  p0={p0:<7g} crossing at {cross} km", flush=True)
    # p0 whose crossing is closest to the target
    scored = [r for r in rows if r["crossing_km"] is not None]
    best = min(scored, key=lambda r: abs(r["crossing_km"] - target_km)) \
        if scored else None
    return {"target_km": target_km, "n_segments": n_seg,
            "distances_km": distances.tolist(), "plob_hz": plob.tolist(),
            "rows": rows, "implied_p0": best["p0"] if best else None}


def plot_e1(data: dict, path: str) -> None:
    rows = [r for r in data["rows"] if not np.isinf(r["cutoff_over_tcoh"])]
    inf_row = [r for r in data["rows"] if np.isinf(r["cutoff_over_tcoh"])][0]
    x = [r["cutoff_over_tcoh"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    ax.semilogx(x, [r["rate_hz"] for r in rows], "o-", label="delivery rate")
    ax.axhline(inf_row["rate_hz"], ls=":", color="0.5", label="no cutoff")
    ax.set_xlabel(r"cutoff / $T_{\rm coh}$")
    ax.set_ylabel("end-to-end rate (Hz)")
    ax2 = ax.twinx()
    ax2.semilogx(x, [r["fidelity"] for r in rows], "s--", color="tab:red",
                 label="fidelity")
    ax2.axhline(inf_row["fidelity"], ls=":", color="tab:red", alpha=0.5)
    ax2.set_ylabel("fidelity", color="tab:red")
    ax.set_title("Cutoff buys fidelity with rate")
    ax.legend(loc="center left", fontsize=8)

    ax = axes[1]
    ax.semilogx(x, [r["secret_key_hz"] for r in rows], "o-", color="tab:green")
    ax.axhline(inf_row["secret_key_hz"], ls=":", color="0.5",
               label=f"no cutoff ({inf_row['secret_key_hz']:.2g} Hz)")
    ax.set_xlabel(r"cutoff / $T_{\rm coh}$")
    ax.set_ylabel("secret key rate (Hz)")
    ax.set_title("Key rate is non-monotone in the cutoff")
    ax.legend(fontsize=8)

    b = data["base"]
    fig.suptitle(f"{b['total_length_km']:.0f} km, {b['n_segments']} segments, "
                 f"$T_{{\\rm coh}}$={b['t_coh_s']} s, single-photon heralding",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_e2(data: dict, path: str) -> None:
    d = np.array(data["distances_km"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for n_seg, s in sorted(data["series"].items(), key=lambda kv: int(kv[0])):
        k = np.array(s["secret_key_hz"])
        ax.semilogy(d, np.maximum(k, 1e-12), "o-", label=f"{n_seg} segments")
    ax.semilogy(d, data["plob_hz"], "k--", lw=2, label="PLOB (repeaterless)")
    ax.set_xlabel("total distance (km)")
    ax.set_ylabel("secret key rate (Hz)")
    ax.set_ylim(1e-9, 1e4)
    ax.set_title("Where a memory chain overtakes the repeaterless bound")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    print("cross-check against the analytic 2-segment formula:")
    ok = selfcheck()
    print(f"  {'PASS' if ok else 'FAIL'}\n")

    print("cross-validation of the two engines:")
    xval = cross_validate(n_samples=3000)
    worst = max(r["rel_rate_diff"] for r in xval["cases"])
    print(f"  worst relative rate difference: {worst:.2%}\n")

    print("E1 — cutoff policy sweep:")
    e1 = e1_cutoff_pareto()
    plot_e1(e1, os.path.join(FIGURES, "e1_cutoff_pareto.png"))

    print("\nE2 — distance sweep vs PLOB:")
    e2 = e2_plob_crossing()
    plot_e2(e2, os.path.join(FIGURES, "e2_plob_crossing.png"))

    print("\nfirst distance above PLOB (km):",
          e2["first_distance_above_plob_km"])

    print("\nE3 — link efficiency implied by a 230 km crossing:")
    e3 = e3_efficiency_needed()
    print(f"  implied p0 ~ {e3['implied_p0']}")

    with open(os.path.join(RESULTS, "results.json"), "w") as fh:
        json.dump({"selfcheck_passed": ok, "cross_validation": xval,
                   "e1": e1, "e2": e2, "e3": e3}, fh, indent=2, default=str)
    print(f"\nwrote {RESULTS}/results.json and two figures")


if __name__ == "__main__":
    main()
