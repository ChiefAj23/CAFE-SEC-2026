# CAFE

**Compressed Architecture for Edge-Deployed Facial Expression Recognition in Ambient Invisible Intelligence**

[![Venue](https://img.shields.io/badge/ACM%2FIEEE%20SEC-2026-1f6feb)](https://acm-ieee-sec.org/2026/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue)](requirements.txt)

Artifact for the SEC 2026 paper: the reference implementation, the compressed model weights, the
annotated stress set, the end-to-end evaluation harness, the EDUS implementation, and the full
88-configuration measurement matrices for both hardware platforms.

CAFE is a two-stage, fully decoupled YOLOv12n pipeline — a face **detector** followed by an
independent emotion **classifier** — compressed to run in near real time on a 15 W NVIDIA Jetson
Orin Nano.

📄 The paper appears in the SEC 2026 proceedings. The published version will be available
through IEEE Xplore; it is not redistributed here.

---

## Contributions

| | |
|---|---|
| **Topological model surgery** | Makes YOLOv12n structurally prunable. Standard pruners fail on its `C3K2` routing metadata and `A2C2f` area-attention QKV constraints. Surgery replaces each opaque `C3K2` block with a traceable twin (`C3K2_v2`) so a dependency-graph tracer produces a valid pruning graph. The twin is verified lossless before any pruning. |
| **Cross-hardware-class compression study** | 88 detector × classifier configurations across pruning ratios (P0/P10/P15) and precisions (FP32/FP16/INT8), measured on *both* a desktop GPU and a Jetson Orin Nano. |
| **EDUS** | The Edge Deployment Unified Score, which ranks configurations against memory and latency envelopes instead of accuracy alone. |

## Headline result

Fully compressed **P10-FP16** pipeline against the dense **P0-FP32** baseline, on a Jetson Orin
Nano at 15 W, measured on the in-the-wild multi-face stress set (paper Table XI):

| Metric | P0-FP32 | P10-FP16 | Change |
|---|---|---|---|
| Combined model size | 17.020 MB | 8.025 MB | **2.12× smaller** |
| End-to-end latency | 228.414 ms | 68.565 ms | **3.33× faster** (≈4.4 → ≈15 FPS) |
| Detector latency | 49.740 ms | 23.548 ms | 2.11× |
| Classifier latency | 28.137 ms | 7.241 ms | 3.89× |
| Detector mAP50-95 | 0.5038 | 0.4895 | 97.2% retained |
| End-to-end accuracy | 0.5076 | 0.4347 | −7.3 points |

At the component level the compressed detector retains **91.3%** of baseline mAP50-95 on the
held-out WIDER FACE split and the compressed classifier retains **96.7%** of baseline top-1
accuracy on FER2013. The same compressed artifacts transfer across hardware classes: the top-10
Jetson configurations all appear within the top 14 on the desktop GPU.

Two findings worth flagging:

- **INT8 is slower than FP16 at nano scale.** P10-INT8 runs at 2.66 ms against 2.54 ms for
  P10-FP16. With only 8–64 channels per layer, per-layer dequantization overhead exceeds the
  arithmetic saving. Profiling only on desktop hardware hides this.
- **Compilation, not pruning, delivers the latency.** A controlled ablation with the unpruned
  P0-FP16 control (Table IX) attributes 3.418× of the classifier speedup to TensorRT compilation
  with FP16 execution and 0.993× to pruning. Pruning delivers footprint; the surgery delivers the
  feasibility of pruning this architecture at all.

---

## Quick start

```bash
git clone https://github.com/ChiefAj23/CAFE-SEC-2026.git
cd CAFE-SEC-2026
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Two experiments need nothing but a CPU and run in seconds — start there to confirm the checkout:

```bash
python experiments/edus_recompute/recompute_edus.py      # reproduces the published EDUS column
python experiments/edus_recompute/accuracy_headroom.py   # the accuracy-ceiling bound of Sec. V-H
```

The compression sweep itself needs a CUDA GPU and TensorRT 10.x, plus the datasets described in
[`docs/DATA.md`](docs/DATA.md). See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for the exact
desktop and Jetson stacks the reported numbers were measured on, and
[`experiments/README.md`](experiments/README.md) for per-experiment hardware requirements and run
order.

```bash
jupyter lab notebooks/00_implementation_final.ipynb
```

Run the configuration cell first. It resolves `PROJECT_ROOT` from the repository root; override
with `export CAFE_ROOT=/path/to/assets` if the weights and datasets live elsewhere.

## Where each paper result comes from

| Paper | Result | Produced by | Recorded in |
|---|---|---|---|
| Table IV | Routing-metadata consumption | [`experiments/surgery_equivalence/`](experiments/surgery_equivalence/) | `results/validation/metadata_consumption.json` |
| Table VI–VII | Per-stage compression sweep | [`notebooks/`](notebooks/) | — |
| Table VIII | EDUS weight sensitivity | [`experiments/camera_ready/sensitivity_camera_ready.py`](experiments/camera_ready/sensitivity_camera_ready.py) | `results/camera_ready/edus_sensitivity_camera_ready.csv` |
| Table IX | Speedup attribution | [`experiments/precision_ablation/`](experiments/precision_ablation/) | `results/validation/precision_ablation.{csv,json}` |
| Table X | Board-level runtime memory | [`experiments/jetson_memory/`](experiments/jetson_memory/) | `results/validation/jetson_{memory,ort_memory}.json` |
| Table XII | End-to-end ranking, both platforms | [`experiments/camera_ready/build_table8_camera_ready.py`](experiments/camera_ready/build_table8_camera_ready.py) | `results/camera_ready/table8_camera_ready_*.csv` |
| Sec. III-C | Padding coefficient λ = 0.25 | [`experiments/e2e_harness/calibrate_lambda.py`](experiments/e2e_harness/calibrate_lambda.py) | `results/validation/lambda_calibration.json` |
| Sec. V-D | External-classifier normalization | [`experiments/external_baselines_audit/`](experiments/external_baselines_audit/) | `results/validation/external_norm_order_audit.{csv,json}` |
| Sec. V-F | Stress-set bootstrap | [`experiments/camera_ready/bootstrap_stress_set.py`](experiments/camera_ready/bootstrap_stress_set.py) | `results/camera_ready/bootstrap_report.json` |
| Sec. V-H | Accuracy-ceiling bound | [`experiments/edus_recompute/accuracy_headroom.py`](experiments/edus_recompute/accuracy_headroom.py) | `results/validation/accuracy_headroom.{csv,json}` |

## Repository layout

```
CAFE-SEC-2026/
├── notebooks/00_implementation_final.ipynb   Compression + per-stage measurement harness
├── weights/                                  Compressed model artifacts
│   ├── P0/{detector,classifier}.pt           Unpruned baseline
│   ├── P10/{detector,classifier}.pt          10% structured channel pruning
│   └── P15/{detector,classifier}.pt          15% structured channel pruning
├── artifacts/                                External-baseline ONNX exports (opset 20)
├── data/Stress_test/                         In-the-wild multi-face stress set
│   ├── emotion_raw_images/                   20 unconstrained scenes
│   └── labels_out/                           132 annotated face instances
├── experiments/                              Validation experiments, one directory each
├── results/
│   ├── e2e_summary_{desktop,jetson}_with_EDUS.csv    88 configurations per platform
│   ├── camera_ready/                         Camera-ready tables and the bootstrap
│   └── validation/                           Per-experiment measurement records
└── docs/                                     Data, environment, and baseline notes
```

### Naming convention

`P0` is the unpruned baseline, `P10` is 10% structured channel pruning, `P15` is 15%. Precision is
`FP32` (native PyTorch `.pt`), `FP16`, or `INT8` (both compiled to TensorRT `.engine`). A full
configuration is written `P10-FP16`.

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

## EDUS

EDUS aggregates six deployment-relevant terms, each min-max normalized across the 88
configurations of a platform:

```
EDUS = 0.08·z⁻(Det_Size) + 0.42·z⁻(Cls_Size) + 0.35·z⁻(Det_lat)
     + 0.05·z⁺(mAP50-95) + 0.05·z⁺(Detected Acc.) + 0.05·z⁺(End2End Acc.)
```

`z⁻` normalizes so that smaller is better, `z⁺` so that larger is better. The weights are not free
parameters. They follow four stated principles: hardware-envelope terms take α_HW = 0.85 against
α_BC = 0.15 for behavioral correctness; memory takes a slight majority over latency within the
hardware budget; and the memory budget is split between the two size terms by the execution
exposure of each stage, since the detector runs once per frame and the classifier once per detected
face (n̄ = 6.6 faces per scene).

The ranking is robust to those choices. All twelve single-weight ±50% perturbations preserve rank-1
with ≥9/10 top-10 overlap, and tier ratios spanning α_HW ∈ [0.70, 0.90] preserve rank-1 throughout
(Table VIII). EDUS is a ranking instrument for a given inventory rather than an absolute score:
adding or removing a configuration rescales every term.

### Reading the results CSVs

One row per detector × classifier combination, 88 rows per platform:

`detector`, `classifier`, `Size_detector`, `Size_classifier` (MB), `mAP50-95`, `det_lat_ms`,
`cls_lat_ms`, `e2e_lat_ms`, `n_det_faces`, `n_gt_faces`, `e2e_accuracy`, `cls_accuracy`,
`f1_pipeline`, `f1_classifier`, `EDUS`, `EDUS_rank`.

`e2e_lat_ms` accumulates the classifier over every detected face, so it is scene-density dependent
by construction. `e2e_accuracy` divides by all **ground-truth** faces, so a missed detection counts
as an error regardless of classifier strength; `cls_accuracy` divides by detected faces only,
isolating classifier performance.

## Not included

- **WIDER FACE** and **FER2013** are public datasets and are not redistributed here. See
  [`docs/DATA.md`](docs/DATA.md) for the expected layout.
- **RT-DETR (266 MB), POSTERv2 (181 MB), and PCNN (104 MB)** ONNX exports exceed GitHub's 100 MB
  per-file limit. [`docs/EXTERNAL_BASELINES.md`](docs/EXTERNAL_BASELINES.md) gives the source and
  export settings for each. The smaller ImpMobNetv3 and LANMSFF exports *are* included. Scripts
  that need a missing file print `SKIP` and continue.

## Reproducibility notes

All reported numbers were measured at batch size 1 to model single-frame AII inference, on the two
platforms documented in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md). TensorRT engines are tied to
the GPU, driver, and TensorRT version that built them: they are **not portable** and must be
compiled on the target device. Expect engine-backed latencies to differ on other hardware; the
PyTorch FP32 paths are the portable reference. Each experiment's README states which of its results
are hardware-dependent and which are portable.

## Citation

```bibtex
@inproceedings{mahdi2026cafe,
  author    = {Mahdi, Mohammad Mahruf and Solanki, Abhijeet and Hasan, Syed Rafay
               and Guo, Terry and Rizvi, Syed Ali Asad},
  title     = {{CAFE}: Compressed Architecture for Edge-Deployed Facial Expression
               Recognition in Ambient Invisible Intelligence},
  booktitle = {Proceedings of the ACM/IEEE Symposium on Edge Computing (SEC)},
  year      = {2026}
}
```

## Authors

Mohammad Mahruf Mahdi, Abhijeet Solanki, Syed Rafay Hasan, Terry Guo, and Syed Ali Asad Rizvi.

Department of Electrical and Computer Engineering, and Center for Manufacturing Research,
Tennessee Technological University, Cookeville, TN, USA.

Repository maintained by [Abhijeet Solanki](https://github.com/ChiefAj23).

## License

Released under the [MIT License](LICENSE). Third-party components carry their own terms — see the
third-party notice in [`LICENSE`](LICENSE) and [`docs/EXTERNAL_BASELINES.md`](docs/EXTERNAL_BASELINES.md).
