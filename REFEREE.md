# Adversarial referee pass

Before the manuscript in `paper/` reached its current form, an independent
adversarial referee was given the code and every data artifact, instructed to
audit each number against the JSON files and to hunt for overclaims. The
findings are reproduced below at full severity, including where they overturned
the draft's headline claims. **The manuscript in `paper/` is the revised
version**: every item below was either corrected in the text or is stated there
as an open limitation.

Notes for the reader of this public repository:

- Section numbering ("S-1", "R-1", ...) is the docket's own. "The paper" refers
  to the *pre-referee draft*; tables and numbers quoted as "published" are the
  draft's, not the current manuscript's.
- References to `NOTES.md` and `research/literature/...` point to the author's
  private working notes (unreviewed literature maps and lab-notebook entries),
  which are deliberately not part of this repository; the manuscript cites the
  primary sources directly.
- Original date of the referee pass: 2026-09-12.

## 2. `repeater-cutoffs/` — the landscape claims are an artifact of fixed c₀

**R-1. The γ landscape holds c₀ at its γ=0 optimum while sweeping γ.** Since
t_cut = c₀·span^γ, changing γ at fixed c₀ changes the cutoff's overall scale as
well as its span-dependence. Re-tuning c₀ at every γ:

| γ | 8 seg, published | 8 seg, re-tuned |
|---|---|---|
| −1 | −38.02% | −11.77% |
| −0.5 | −11.90% | −1.60% |
| +0.25 | **−22.99% (36.1σ)** | **−0.54% (0.8σ, null)** |
| +0.5 | −81.02% | −4.67% |
| +1 | −100% | −17.64% |

The abstract's "strongly asymmetric" and §IV.A's entire "granting longer cutoffs
to longer-span links is the direction that fails" paragraph do not survive. The
*conclusion* (γ = 0 optimal) does.

**R-2. The mechanism for the 8-segment failure is backwards.** The paper compares
two policies at different operating points and reads the trade as a mechanism. At
matched delivery time:

| 8 seg | E[T] | w̄ | key |
|---|---|---|---|
| age, c₀ = 0.013 | 0.1396 | **0.8365** | **1.3130** |
| fidelity, w_min = 0.90 | 0.1404 | 0.8316 | 1.1854 |

At the same E[T] the age gate delivers the *better* state. The fidelity gate is
dominated on both axes, not trading.

**R-3. "Excludes a 12% gap ... a seventeen-fold tightening" — wrong by ~7×.** The
unpaired two-sigma band was ±1.74%, so it excluded gaps above 1.7%, not 12%. The
11.7% is Li et al.'s *measured* MAX-TIME-vs-FIDELITY gap, transcribed from the
literature map as if it were our error bar. Correct tightening: **2.45-fold**.

**R-4. "Standard errors fall by an order of magnitude" — the pairing buys 1.34×.**
2.45× raw, but at matched n it is 1.34×; the rest is the 3.3× ensemble. The limit
is `corr_werner` = 0.145 / 0.043 — the pairing barely couples the Werner channel,
which is unreported.

**R-5. "Both were reported as no gain" is false.** `policy_comparison.json` has
`verdicts.span_dependent_age.significant = true`; only the fidelity row was a null.

**R-6. The 153.6σ in Table I is a tautology** — the key rate there is exactly 0
(clipped), so the σ measures the baseline's distance from zero.

The 8-segment baseline is also mis-tuned: the c₀ grid steps over the optimum
(0.013 beats 0.015 by 2.19%), so the headline −8.11% is *understated* — against a
properly tuned baseline it is −9.5%.

**Sound, and worth recording:** the referee verified the CRN construction proves
out — `older` is always the leftmost elementary index of its block under every
merge order, checked by invariant assertion over 6000 samples. The censoring
claim holds (all 20 fractions exactly 0). And a geometry control the paper did not
run *strengthens* it: at fixed 400 km, 4→8 segments moves the gap from −0.2% to
−16.2%, while at fixed 4 segments 400→600 km moves it to +3.6% — so segment count
is the driver and length pushes the other way.

---

## Disposition

Nothing here is being filed as "reviewer disagreement". Every item above was
checked against the artifacts, and the ones that overturn a headline were
re-derived independently before being accepted. The four documents as committed
overstate their results; the revision has to start from that.
