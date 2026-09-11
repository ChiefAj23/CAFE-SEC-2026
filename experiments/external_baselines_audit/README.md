# External-classifier preprocessing audit

## Why this experiment exists

The four external FER classifiers are evaluated through a shared export and runtime path so they
can be compared on one deployment-relevant footing. That shared path is also a shared assumption:
it presumes every model accepts the same input contract. Exported inference graphs generally do not
carry their normalization in-graph, so the contract lives in each model's original training
codebase, and a single preprocessing rule applied to all four will violate a different one for each.

This audit determines each model's actual contract by measurement, so the comparison in the paper
evaluates deployability rather than how closely one shared rule happens to match each model.

Two signals motivated a direct test. ImpMobNetv3 initially scored **23.45%** top-1 on FER2013,
while FER2013's majority class accounts for **24.5%** of the held-out split — a score below the
trivial majority-class predictor indicates a preprocessing mismatch rather than a weak model. And
POSTERv2 appeared *less* accurate than the in-house classifier at component level while achieving
high end-to-end accuracy at pipeline level, a combination that points at the measurement path.

## What it does

Two scripts, holding model, weights, and test data fixed and varying only preprocessing:

- `audit_preprocessing.py` — deep dive on ImpMobNetv3. Sweeps normalization variants, reports the
  predicted-class histogram per variant, and separately computes a **label-permutation upper
  bound**: the accuracy recoverable by optimally relabeling the model's outputs, which distinguishes
  a class-index mismatch from a preprocessing mismatch.
- `audit_all_baselines.py` — the same sweep across all four external classifiers, with variants
  adapted per model (LANMSFF takes 64×64 single-channel NHWC input; the others take 224×224 RGB
  NCHW).
- `audit_norm_and_order.py` — sweeps input normalization *jointly* with output class ordering,
  because a model can disagree on either independently and a one-at-a-time sweep cannot separate
  them. FER2013 has two orderings in common use that differ only in the final three positions, so a
  model trained under one and scored under the other loses accuracy on three of seven classes while
  still appearing to function.

All three check whether the ONNX graph carries a normalization prologue. A graph that begins
directly at a convolution expects externally normalized input. The prediction histogram is the
diagnostic that separates the two failure modes: a genuinely weak model spreads its predictions
across classes, whereas a model receiving out-of-distribution input collapses onto one.

```bash
python experiments/external_baselines_audit/audit_preprocessing.py
python experiments/external_baselines_audit/audit_all_baselines.py
python experiments/external_baselines_audit/audit_norm_and_order.py
```

Outputs land in `results/validation/` as `mobilenetv3_audit.{csv,json}`,
`external_baseline_audit.{csv,json}`, and `external_norm_order_audit.{csv,json}`.

## What it establishes

Evaluated on the 3589-image FER2013 held-out split:

| Model | Raw `[0,1]` input | + ImageNet norm | Δ | Contract determined |
|---|---|---|---|---|
| ImpMobNetv3 | 23.40% | **61.66%** | +38.3 | ImageNet mean/std |
| POSTERv2 | 59.10% | **73.61%** | +14.5 | ImageNet mean/std |
| PCNN | 52.58% | **70.63%** | +18.1 | ImageNet mean/std |
| LANMSFF | 39.15% | 26.61% | — | raw `[0,1]`, as published |

**Three of the four expect ImageNet mean/std normalization.** None of the four graphs carries a
normalization prologue; three were trained with ImageNet normalization and collapse without it. The
collapse is visible in the histograms: ImpMobNetv3 on raw input assigns 80.7% of all images to a
single class and never predicts 2 of the 7 classes at all, whereas under its own contract its
dominant-class share falls to 24.3%, the true class prior. LANMSFF, operating on 64×64 grayscale,
was trained on plain `[0,1]` inputs and degrades under normalization — confirming the contract is
model-specific and that no blanket transform is correct for all four.

Two alternative explanations are ruled out. **Channel order** is irrelevant, since FER2013 is
grayscale and the BGR and RGB variants agree bit-for-bit. **Class-index mismatch** is excluded by
the permutation bound: the best possible relabeling of ImpMobNetv3's outputs reaches only 21.79%,
*below* its raw-input figure, so no assignment of output indices to class names recovers the
missing accuracy.

The joint normalization-and-ordering sweep separates the two conventions per model and is the basis
for the per-model protocol the paper describes in Sec. V-D. The figures the paper reports in
Table VII are the values each model reaches under its own determined contract.

Under those contracts, POSTERv2 at 73.61% is the most accurate classifier evaluated and PCNN at
70.63% is competitive, both against the in-house classifier's 72.19%.

## How to read this

The point is what the determination does to the argument, not the individual accuracy numbers. If
POSTERv2 were both larger and less accurate than the in-house pipeline, ranking it poorly would
demonstrate nothing a size-blind comparison would not also show. Under its own contract POSTERv2 is
the most accurate model under evaluation *and* the least deployable — 180.6 MB against a 7.4 GB
shared envelope that must also hold the OS, frame buffers, and the detector. That is a genuine
accuracy-versus-footprint conflict, and it is the case a deployment metric exists to adjudicate.
The companion analysis in [`../edus_recompute/accuracy_headroom.py`](../edus_recompute/accuracy_headroom.py)
closes the argument by showing the ranking is invariant to accuracy for models of this size.

Two methodological points generalize. A shared evaluation path needs a per-model sanity floor:
comparing each model against its dataset's majority-class rate catches a contract mismatch
immediately, at negligible cost. And exported inference graphs should carry their normalization
in-graph wherever possible, because when the contract lives in a separate codebase a shared harness
will eventually violate it for some model, and the violation surfaces as a plausible-looking low
accuracy rather than as an error.

These results are portable. Everything here is CPU-only inference over a fixed test split, so the
accuracies reproduce exactly given the same weights and data.
