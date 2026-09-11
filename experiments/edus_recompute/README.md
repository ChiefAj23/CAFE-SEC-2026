# EDUS weight sensitivity and the accuracy-ceiling bound

## Why this experiment exists

EDUS aggregates six deployment-relevant terms into a single ranking score, and its weights are
derived from four stated principles rather than fitted. That design buys interpretability, but it
raises the obvious question about any conclusion drawn from the ranking: **is the result a property
of the deployment data, or a property of the weights?**

A metric proposing to rank deployment configurations has to answer that by measurement before its
rankings mean anything. Two aspects of the derivation are worth testing directly.

**1. The memory split has an uncertainty band.** Eq. 4 divides the memory budget by the execution
exposure of each stage, yielding w_cls_size ≈ 0.434 and w_det_size ≈ 0.066, with a band of
[0.417, 0.444] and [0.056, 0.083] over a plausible range of scene densities. The paper adopts the
rounded in-band values 0.42 and 0.08. Whether that rounding, or the exact unrounded derivation,
changes the induced ranking is an empirical question.

**2. The memory split prices execution exposure, not per-face duplication.** Under sequential
inference at batch size 1, model weights are loaded once and are not duplicated per face, so
classifier *memory* does not scale with the face count — classifier *computation and latency* do.
The paper's Principle 4 is therefore stated in terms of execution exposure, and the face-count
factor on computation is captured by the measured end-to-end latency. Since w_cls_size is the
single largest term in the score, it is worth measuring what happens if the per-face factor is
relocated onto the latency budget instead.

## What it does

`recompute_edus.py` re-scores the existing 88-configuration sweep on both platforms under five
weight vectors, holding the three top-level budgets fixed (memory 0.50, detector latency 0.35,
behavioral correctness 0.15) so that only the split *within* a budget varies:

| Variant | w(det size) | w(cls size) | w(det lat) | w(cls lat) |
|---|---|---|---|---|
| `calibrated` | 0.10 | 0.40 | 0.35 | — |
| `in_band` | 0.08 | 0.42 | 0.35 | — |
| `exact_derivation` | 0.0658 | 0.4342 | 0.35 | — |
| `equal_memory_split` | 0.25 | 0.25 | 0.35 | — |
| `latency_scaled` | 0.25 | 0.25 | 0.046 | 0.304 |

`in_band` is the split the paper adopts. The last variant is the substantive alternative: it drops
the execution-exposure memory argument entirely *and* relocates the per-face factor onto the
latency budget, where the scaling physically occurs.

Each variant is compared against the adopted ranking by rank-1 identity, top-10 overlap, and
Spearman ρ, plus the rank movement of the two external classifiers the paper singles out
(POSTERv2, demoted on footprint; LANMSFF, promoted on it). A literal weighted z-score form of
Eq. 5 is also evaluated, since the released implementation uses per-term min-max normalization — a
per-term affine rescaling of the z-score, which changes term magnitudes while preserving each
term's internal ordering.

The script first reproduces the reported EDUS column from the raw metric columns and asserts the
match, so any observed rank movement is attributable to the weights alone rather than to an
implementation difference.

`accuracy_headroom.py` answers a different question: can any accuracy correction rescue a large
classifier? It assigns each classifier a perfect behavioral score — an upper bound no correction
can exceed — and recomputes the ranking.

```bash
python experiments/edus_recompute/recompute_edus.py
python experiments/edus_recompute/accuracy_headroom.py
```

Outputs land in `results/validation/`: `edus_variant_summary.csv` (one row per variant × platform),
`edus_variant_top10.csv` (the top-10 table for each), `edus_recompute_report.json`, and
`accuracy_headroom.{csv,json}`.

## What it establishes

**The reported column reproduces exactly** — maximum absolute error 5.0 × 10⁻⁵ on both platforms,
which is the rounding of the 4-decimal CSV.

**The size-weight conclusions are robust.** Using the adopted in-band weights, the exact unrounded
derivation, or discarding the execution-exposure argument entirely and splitting the memory budget
evenly all preserve rank-1 on both desktop and Jetson, with 9–10/10 top-10 overlap and Spearman ρ
between 0.990 and 0.999. Cross-platform top-10 agreement holds at 8/10 throughout. The literal
z-score form also preserves rank-1. **No conclusion in the deployment analysis depends on the
particular memory split or on the rounding of the adopted weights.**

**Relocating the per-face factor onto latency produces a genuinely different ranking.** LANMSFF
takes rank-1 on both platforms, Spearman ρ falls to ≈0.70, and the previously top-ranked
configuration drops to 13th on desktop and 43rd on Jetson. The mechanism is direct: LANMSFF is both
the smallest classifier (1.224 MB against 6.792 MB) and the fastest per face (0.85 ms against
1.54–13.66 ms on desktop), so once per-face latency carries 0.304 of the score it dominates — at a
cost of 8–10 points of end-to-end accuracy, which the 0.15 behavioral budget prices only lightly.

**Accuracy cannot buy back footprint.** Under a perfect behavioral score, POSTERv2 still ranks
71/88, and 66/88 under weights that drop the execution-exposure argument; PCNN still ranks 64/88.
Every classifier under 7 MB reaches rank 1. The ranking of large classifiers is determined entirely
by their memory and latency envelope, which is the bound the paper reports in Sec. V-H.

## How to read this

EDUS is not a single ranking but a *family* of rankings selected by which constraint the deployment
prices. The adopted weights price the memory envelope, which is the binding constraint on a 7.4 GB
shared-RAM edge node. The latency-scaled weights price throughput under crowd density, which is
the binding constraint when scene occupancy is high. These are different deployment regimes and
they select different configurations for defensible reasons; the metric surfacing that trade-off is
the intended behavior rather than an instability in it.

The measurement also sharpens the paper's claim about LANMSFF. EDUS promotes it to a competitive
position that a size-blind ranking would miss, and under per-face latency weighting it is rank-1
outright, with the accuracy cost left explicit for the application owner to adjudicate.

Both results are portable: they are recomputations over the recorded metric columns and involve no
new timing, so they reproduce exactly on any machine.
