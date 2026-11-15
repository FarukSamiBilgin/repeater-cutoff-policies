"""Is a single global cutoff the right policy?

Published cutoff studies tune one number: discard any stored link older than
t_cut.  But a chain holds links of very different provenance — a fresh
elementary link and a link that already survived two swaps are not the same
object, and there is no reason one threshold should serve both.

This scans the two-parameter family cutoff(span) = c0 * span^gamma, optimising
c0 inside each gamma slice so the comparison is fair, and asks whether any
gamma beats the gamma = 0 baseline by more than Monte Carlo error.  It runs at
4 and 8 segments, because 8 segments has spans 1, 2, 4, 8 and therefore more
room for the policy to differentiate.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from repeater_chain import ChainConfig, scan_level_cutoffs  # noqa: E402

RESULTS = os.path.join(_HERE, "..", "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

C0_GRID = [0.2, 0.1, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.006, 0.003]
GAMMA_GRID = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


def run_case(n_segments: int, length_km: float, t_coh_s: float,
             n_samples: int = 4000) -> dict:
    base = ChainConfig(total_length_km=length_km, n_segments=n_segments,
                       p0=0.1, photons_per_herald=1, t_coh_s=t_coh_s,
                       w_init=0.99, w_swap=0.99, policy="swap_asap")
    print(f"  {n_segments} segments, {length_km:.0f} km, "
          f"T_coh={t_coh_s} s:", flush=True)
    res = scan_level_cutoffs(base, C0_GRID, GAMMA_GRID, n_samples=n_samples)
    gain = res["relative_gain"]
    print(f"    baseline (gamma=0): {res['baseline_key_hz']:.4g} Hz")
    print(f"    best gamma={res['best_gamma']:g} at c0={res['best_c0']:g} s: "
          f"{res['best_key_hz']:.4g} Hz  ->  "
          f"{gain:+.1%}" if gain is not None else "")
    return res


def plot_heatmaps(data: dict, path: str) -> None:
    cases = list(data["cases"].items())
    fig, axes = plt.subplots(1, len(cases), figsize=(5.6 * len(cases), 4.4))
    if len(cases) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, cases):
        grid = np.array(res["key_hz_grid"])
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(res["c0_grid"])))
        ax.set_xticklabels([f"{c:g}" for c in res["c0_grid"]], rotation=45,
                           fontsize=7)
        ax.set_yticks(range(len(res["gamma_grid"])))
        ax.set_yticklabels([f"{g:g}" for g in res["gamma_grid"]], fontsize=7)
        ax.set_xlabel("$c_0$ (s)")
        ax.set_ylabel(r"$\gamma$")
        gi = res["gamma_grid"].index(res["best_gamma"])
        ci = res["c0_grid"].index(res["best_c0"])
        ax.plot(ci, gi, "r*", ms=14, mec="w")
        bgi = res["gamma_grid"].index(res["baseline_gamma"])
        bci = int(np.argmax(grid[bgi]))
        ax.plot(bci, bgi, "wo", ms=8, mec="k")
        ax.set_title(f"{name}\nbest {res['best_key_hz']:.3g} Hz vs "
                     f"baseline {res['baseline_key_hz']:.3g} Hz "
                     f"({res['relative_gain']:+.0%})", fontsize=9)
        fig.colorbar(im, ax=ax, label="secret key rate (Hz)")
    fig.suptitle(r"Cutoff family $t_{\rm cut}(\rm span) = c_0\,\rm span^{\gamma}$"
                 "   (white circle = global-cutoff baseline, red star = best)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    cases = {}
    cases["4 seg, 400 km"] = run_case(4, 400.0, 0.5)
    cases["8 seg, 600 km"] = run_case(8, 600.0, 0.5)
    data = {"c0_grid": C0_GRID, "gamma_grid": GAMMA_GRID, "cases": cases}
    plot_heatmaps(data, os.path.join(FIGURES, "e4_level_cutoff.png"))
    with open(os.path.join(RESULTS, "level_policy.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nwrote {RESULTS}/level_policy.json and the heatmap")


if __name__ == "__main__":
    main()
