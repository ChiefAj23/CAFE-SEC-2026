# CAFE: Compressed Architecture for Edge-Deployed Facial Expression Recognition in Ambient Invisible Intelligence

Reference implementation, model artifacts, and measurement data for the CAFE paper
(submitted to **SEC 2026**).

CAFE is a two-stage, fully decoupled YOLOv12n pipeline — a face **detector** followed by an
independent emotion **classifier** — compressed for real-time inference on constrained edge
hardware. The repository contains the compression pipeline, the compressed weights, the
end-to-end evaluation harness, and the full 88-configuration measurement sweep on both a
desktop GPU and an NVIDIA Jetson Orin Nano.

> **Status:** under review. This repository is private until a decision is returned.

---

## Why this exists

FER research is almost universally evaluated on idealized, pre-cropped faces. That assumption
hides the two costs that decide whether a model can actually be deployed in an Ambient Invisible
Intelligence (AII) setting: you must *find* the faces first, and the whole pipeline must fit inside
an edge node's memory and per-frame latency budget. CAFE closes that gap with three contributions:

| Contribution | What it does |
|---|---|
| **Topological model surgery** | Makes YOLOv12n structurally prunable. Standard pruners fail on its `C3K2` routing and `A2C2f` area-attention QKV constraints; surgery replaces opaque `C3K2` blocks with a traceable twin (`C3K2_v2`) so a dependency-graph tracer produces a valid pruning graph. |
| **Cross-hardware-class compression study** | 88 detector × classifier configurations swept across pruning ratios (P0/P10/P15) and precisions (FP32/FP16/INT8), measured on *both* a desktop GPU and a Jetson Orin Nano. |
| **EDUS** | The Edge Deployment Unified Score — a weighted z-score that ranks configurations under hard memory and latency constraints instead of accuracy alone. |

## Headline results

Fully compressed **P10-FP16** pipeline vs. the dense **P0-FP32** baseline, on a Jetson Orin Nano
(15 W envelope), measured on the in-the-wild multi-face stress set:

| Metric | P0-FP32 | P10-FP16 | Change |
|---|---|---|---|
| Combined model size | 17.020 MB | 8.025 MB | **2.12× smaller** |
| End-to-end latency | 228.414 ms | 68.565 ms | **3.33× faster** (~4.4 → ~15 FPS) |
| Detector latency | 49.740 ms | 23.548 ms | 2.11× |
| Classifier latency | 28.137 ms | 7.241 ms | 3.89× |
| Detector mAP50-95 | 0.5038 | 0.4895 | 97.2% retained |
| End-to-end accuracy | 0.5076 | 0.4347 | −7.3 pts |

On FER2013 the P10-FP16 classifier retains **96.7%** of the P0-FP32 top-1 accuracy. On the desktop
GPU the same compressed artifacts give 3.56× (detector) and 2.13× (classifier) speedups —
the gains transfer across hardware classes.

Two findings worth flagging:

- **INT8 is slower than FP16 at nano scale.** P10-INT8 runs at 2.66 ms vs. 2.54 ms for P10-FP16.
  With only 8–64 channels per layer, per-layer dequantization overhead exceeds the arithmetic
  savings. Profiling only on desktop hardware would hide this.
- **FP32 pruning alone buys almost no latency on a premium GPU** (9.04 → 8.90 ms). FP16 TensorRT
  compilation is the operative compression step; pruning contributes footprint reduction.

---

## Repository layout

The directory layout mirrors the original working tree, so the notebook resolves every path from a
single `PROJECT_ROOT` with no further edits.

```
CAFE-SEC-2026/
├── notebooks/
│   └── 00_implementation_final.ipynb   Full measurement harness (see below)
├── runs/                               Unpruned P0 baselines
│   ├── classify/fer2013_s0/weights/best.pt          Stage 2 classifier, P0
│   └── detect/face_detector_yolov12n_finetune_s0/   Stage 1 detector, P0
│       └── weights/best.pt
├── ten_percent_fair/                   P10 — 10% structured channel pruning
│   ├── yolo12_emotion_pruned_bncal_finetuned_truepruned.pt      classifier
│   └── runs/finetune_pruned_bncal_prunedgraph_det_01/weights/   detector
│       └── best.pt
├── 15Percent/                          P15 — 15% structured channel pruning
│   ├── yolo12_emotion_pruned_bncal_finetuned_truepruned.pt      classifier
│   └── runs/finetune_pruned_bncal_prunedgraph_det_01/weights/   detector
│       └── best.pt
├── artifacts/                          External-baseline ONNX exports (opset 20)
│   ├── mobilenetv3_best_opset20_dynB.onnx (+ .onnx.data)
│   ├── two_path_massatt_pwfs_opset20.onnx            (LANMSFF)
│   └── onnx_exports/                                 (RT-DETR — see note)
├── data/
│   └── Stress_test/                    In-the-wild multi-face stress set
│       ├── emotion_raw_images/         20 unconstrained scenes
│       └── labels_out/                 132 hand-annotated face instances
├── results/
│   ├── e2e_summary_desktop_with_EDUS.csv    88 configurations, desktop GPU
│   └── e2e_summary_jetson_with_EDUS.csv     88 configurations, Jetson Orin Nano
├── paper/CAFE_SEC_2026.pdf
└── docs/                               Data, environment, and baseline notes
```

### Naming convention

`P0` = unpruned baseline, `P10` = 10% structured channel pruning, `P15` = 15%. Precision is
`FP32` (native PyTorch `.pt`), `FP16`, or `INT8` (both compiled to TensorRT `.engine`). A full
configuration is written `P10-FP16`. The legacy directory names `ten_percent_fair/` and
`15Percent/` correspond to P10 and P15 respectively; they are preserved so the paths in the
notebook match the tree the results were produced on.

### Model inventory

| Model | Stage | Input | Dataset | Precisions |
|---|---|---|---|---|
| YOLOv12n P0 / P10 / P15 | Detection | 640×640 | WIDER FACE | FP32, FP16, INT8 |
| YOLOv12n P0 / P10 / P15 | Classification | 224×224 | FER2013 | FP32, FP16, INT8 |
| RT-DETR (ResNet-50) | Detection | 640×640 | WIDER FACE | FP32 |
| LANMSFF | Classification | 64×64 gray | FER2013 | FP32 |
| ImpMobNetv3 | Classification | 224×224 | FER2013 | FP32 |
| POSTERv2 | Classification | 224×224 | FER2013 | FP32 |
| PCNN | Classification | 224×224 | FER2013 | FP32 |

---

## Setup

```bash
git clone git@github.com:ChiefAj23/CAFE-SEC-2026.git
cd CAFE-SEC-2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU and a TensorRT installation are required — the harness compiles and benchmarks
TensorRT engines. See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for the exact desktop and
Jetson software stacks used for the reported numbers.

Then fetch the datasets described in [`docs/DATA.md`](docs/DATA.md) (FER2013 and WIDER FACE are
not redistributed here) and open the notebook:

```bash
jupyter lab notebooks/00_implementation_final.ipynb
```

Run the configuration cell first. It sets `PROJECT_ROOT` to the repository root; override with
`export CAFE_ROOT=/path/to/assets` if the weights and datasets live elsewhere.

## What the notebook does

Cells run top to bottom; each stage prints a summary table.

| Cell | Stage |
|---|---|
| 0 | Configuration — resolves `PROJECT_ROOT` |
| 1–2 | Stage 2 classifier: weight check, footprint/params/GFLOPs profile |
| 3 | Classifier INT8 TensorRT export (calibrated on the FER2013 held-out split) |
| 4 | Classifier INT8 accuracy + per-phase latency on the FER2013 test split |
| 5 | Classifier FP32/FP16 sweep and the merged precision comparison |
| 6–7 | Stage 1 detector: weight/dataset check, footprint profile |
| 8 | Detector INT8 engine latency on the WIDER FACE test split |
| 9 | Detector FP32/FP16 latency **and** mAP50-95 via `model.val()`, merged |
| 10 | External-baseline ONNX profiling — params, footprint, GFLOPs from the graph |
| 11 | External-baseline ONNX Runtime latency and accuracy |
| 12 | RT-DETR mAP50-95 on WIDER FACE via a self-contained COCO-style evaluator |

The 88-configuration end-to-end sweep that produces `results/*.csv` — including the EDUS scores
and rankings — was executed on the desktop GPU and the Jetson separately; those CSVs are the
measurement record behind Tables VII and VIII of the paper.

### Reading the results CSVs

One row per detector × classifier combination (88 rows each):

`detector`, `classifier`, `Size_detector`, `Size_classifier` (MB), `mAP50-95`, `det_lat_ms`,
`cls_lat_ms`, `e2e_lat_ms`, `n_det_faces`, `n_gt_faces`, `e2e_accuracy`, `cls_accuracy`,
`f1_pipeline`, `f1_classifier`, `EDUS`, `EDUS_rank`.

`e2e_lat_ms` accumulates the classifier over every detected face (6.6 faces/image average), so it
is scene-density dependent by construction. `e2e_accuracy` divides by all **ground-truth** faces,
meaning a missed detection is counted as an error regardless of classifier strength;
`cls_accuracy` divides by detected faces only, isolating classifier performance.

## EDUS

EDUS aggregates six deployment-relevant terms as a weighted z-score:

```
EDUS = 0.10·z⁻(Det_Size) + 0.40·z⁻(Cls_Size) + 0.35·z⁻(Det_lat)
     + 0.05·z⁺(mAP50-95) + 0.05·z⁺(Detected Acc.) + 0.05·z⁺(End2End Acc.)
```

`z⁻` normalizes negatively (smaller is better), `z⁺` positively. The weights are not free
parameters — they are derived from four stated principles: hardware-envelope terms take
α_HW = 0.85 against α_BC = 0.15 for behavioral correctness; memory outweighs latency within the
hardware budget; and the classifier size weight is scaled by the average per-image face count
(n̄ = 6.6), since the detector runs once per frame but the classifier runs once per face.

The ranking is robust: all twelve single-weight ±50% perturbations preserve rank-1 with ≥9/10
top-10 overlap, and tier ratios spanning α_HW ∈ [0.70, 0.90] preserve rank-1 throughout
(Table VI). The top-10 Jetson configurations all appear within the top-13 on desktop.

---

## Data and external baselines

Not included in this repository, with instructions in `docs/`:

- **WIDER FACE** (2.2 GB) and **FER2013** — public datasets, not redistributed. See
  [`docs/DATA.md`](docs/DATA.md).
- **RT-DETR (266 MB), POSTERv2 (181 MB), PCNN (104 MB)** ONNX exports — third-party baseline
  weights above GitHub's file-size limit. See
  [`docs/EXTERNAL_BASELINES.md`](docs/EXTERNAL_BASELINES.md) for the source of each and the export
  settings used. The smaller ImpMobNetv3 and LANMSFF exports *are* included.

## Reproducibility notes

All reported numbers were measured on the two platforms documented in
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md), at batch size 1 to model single-frame AII inference.
TensorRT engines are hardware- and version-specific: they are **not** portable and must be compiled
on the target device. Expect engine-backed latencies to differ on any other GPU or TensorRT
release; the PyTorch FP32 paths are the portable reference.

## Citation

```bibtex
@inproceedings{cafe2026,
  title     = {CAFE: Compressed Architecture for Edge-Deployed Facial Expression
               Recognition in Ambient Invisible Intelligence},
  booktitle = {ACM/IEEE Symposium on Edge Computing (SEC)},
  year      = {2026},
  note      = {Under review}
}
```

## Authors

Mohmmaed Mahdi, Abhijeet Solanki, Dr. Syed Rafay Hasan, @Tennessee Technological University

<!-- Add remaining co-authors here before the repository is made public. -->

## License

No license is granted while the paper is under review. Licensing will be decided on acceptance.
