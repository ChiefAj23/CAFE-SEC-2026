# End-to-end pipeline evaluation harness

## What this is

The end-to-end evaluation is where the paper's deployment claims live. Component-level accuracy on
pre-cropped faces is exactly the measurement the paper argues is insufficient; the pipeline-level
numbers are what justify the deployment conclusions and drive the EDUS ranking.

This directory holds the harness that produces those numbers, and the calibration script that fixes
its two free parameters.

## What it does

`run_e2e_sweep.py` runs the full detector × classifier grid over the annotated stress set. For each
scene, the detector runs once, detections are matched to ground truth greedily at IoU ≥ 0.50, each
match is cropped with the scale-normalized contextual padding of Eq. 1, and the classifier runs
once per matched face. Results are aggregated into the columns the summary CSVs carry. End-to-end
accuracy divides by all ground-truth faces, so a missed detection counts as an error; detected
accuracy divides by matched faces only, isolating classifier performance.

`calibrate_lambda.py` fixes the padding coefficient λ and the coarse-group assignment for
`surprise` by solving for them rather than assuming them. Because the summary pins every
configuration's accuracy to four decimals, the parameters can be recovered: the deterministic FP32
detector runs once, then λ and the surprise assignment are swept over the classifier stage and
scored against the all-FP32 row. The FP32 pairing is used because it involves no TensorRT
compilation and so is reproducible across hardware up to floating-point noise.

```bash
python calibrate_lambda.py                       # recover lambda and the coarse mapping
python run_e2e_sweep.py --lam 0.25 --surprise-to neutral \
    --out results/validation/e2e_sweep.csv
```

Ground-truth label semantics were established by inspection: the stress set encodes its three
coarse classes as 0 (happy), 1 (sad), and 6 (neutral), verified by tiling sample crops per class.

## Recovered parameters

| Parameter | Value | Evidence |
|---|---|---|
| Padding coefficient λ | **0.25** | Grid over λ ∈ [0, 0.30]; the best cell fits the all-FP32 row to a combined absolute error of 0.0175, roughly two faces out of 132. The next-best λ is three times worse. |
| Coarse group for `surprise` | **neutral** | Lowest error at every λ tested. |
| `n_det_faces` semantics | raw detections, not matched | At confidence 0.20–0.25 the detector produces 6.75 raw detections per scene, matching the summary column exactly; matched detections are 6.30. |

λ = 0.25 expands each crop by 25% of the box dimensions on every side. The calibration shows the
choice is consequential — accuracy varies by roughly 7 points across the plausible range, with a
clear optimum — which is why the paper treats λ as a deployment hyperparameter rather than a
universal constant. It is held fixed across all configurations so that observed differences reflect
detector, classifier, and compression effects rather than crop-boundary variation.

The recovered value is corroborated by the 56 in-house configurations of the grid rather than by
the single row it was fitted on: across those configurations, where every pipeline component is
under our control, the harness agrees with the summary to a mean absolute error of 0.031, about
four faces on a 132-face set.

## Scope

Detector mAP50-95 is carried over per detector from the summary rather than recomputed here, as the
WIDER FACE split is not distributed with this repository.

The two INT8 detector variants are calibrated on the twenty stress-set scenes rather than the WIDER
FACE validation split used for the reported numbers. The scenes are drawn from the WIDER FACE test
split and so are in-domain, but the calibration set is far smaller and the resulting mAP is
correspondingly lower. This deviation is recorded in each run's `.meta.json`.

Latency is hardware-specific. Runs on hardware other than the platforms in `docs/ENVIRONMENT.md`
are internally consistent but not comparable to the reported latency columns; the accuracy columns
and the ratios are the portable quantities. Each run writes a `.meta.json` recording the parameters
used and any deviation from the reported protocol, so results stay interpretable after the fact.

Detector latency is taken from the framework-reported preprocess, inference, and postprocess total,
which excludes image decode from disk. A wall-clock timer around the prediction call includes a
fixed per-frame I/O cost that dominates a nano-scale detector, so the two bases are not
interchangeable.

External classifiers are evaluated under the input normalization and output class ordering each was
trained with, as determined in
[`experiments/external_baselines_audit/`](../external_baselines_audit/). A single shared
preprocessing rule does not hold across the four, so the harness applies each model's own contract.
