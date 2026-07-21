# External baselines

Five external architectures bound the comparison space: one alternative detector (RT-DETR) and
four FER classifiers spanning a 147× size range, from 1.224 MB (LANMSFF) to 180.560 MB (POSTERv2).

All are exported to ONNX at opset 20 and evaluated **through the CAFE deployment pipeline**, which
is the point: this measures deployment portability under one controlled export and runtime path,
not peak accuracy as reported in the original publications. Several of these models were tuned for
RAF-DB or AffectNet with bespoke preprocessing; re-evaluating them here on FER2013 under a unified
edge export deliberately trades reproduced peak accuracy for a hardware-agnostic comparison.

## What is and is not in this repository

| Baseline | ONNX file | Size | In repo? |
|---|---|---|---|
| RT-DETR (ResNet-50) | `artifacts/onnx_exports/face_det_rtdetr_train_clean2_op20_b1_640_static_shared_baseline.onnx` | 265.8 MB | **No** |
| POSTERv2 | `artifacts/poster_v2_fer2013_dynB.onnx` | 180.7 MB | **No** |
| PCNN | `artifacts/pcnn_best_opset20_dynB.onnx` | 103.6 MB | **No** |
| ImpMobNetv3 | `artifacts/mobilenetv3_best_opset20_dynB.onnx` + `.onnx.data` | 5.7 MB | Yes |
| LANMSFF | `artifacts/two_path_massatt_pwfs_opset20.onnx` | 1.3 MB | Yes |

The three excluded files exceed GitHub's 100 MB per-file hard limit. They are also third-party
model weights that should not be re-hosted here. Cells 10–12 of the notebook skip any baseline
whose file is missing and print `NOT FOUND` rather than failing, so the rest of the harness runs
without them.

Drop the three files into the paths above to reproduce the external-baseline rows.

## How each was produced

**RT-DETR** — pretrained on COCO, fine-tuned on WIDER FACE for 200 epochs at 640×640 using MuSGD
(learning rate 0.01, momentum 0.9, batch size 8) with the class count overridden to one. Serves as
an unconstrained transformer-based detector reference, establishing the upper bound on detection
performance prior to compression: mAP50-95 = 0.3928 against the P0-FP32 YOLOv12n baseline's 0.3888,
but at 87.93 ms total latency versus 9.04 ms, and 253.4 MB — disqualifying for AII edge deployment
regardless of accuracy.

**POSTERv2, ImpMobNetv3, PCNN** — trained on FER2013 with their authors' prescribed optimizers, to
keep comparative fairness: Adam for ImpMobNetv3, SAM for POSTERv2, and auxiliary loss for PCNN.

**LANMSFF** — included in its pretrained form, operating on 64×64 grayscale crops.

## Result

All seven YOLOv12n configurations substantially outperform the external classifiers under this
unified pipeline (top-1: POSTERv2 59.25%, PCNN 53.80%, LANMSFF 38.30%, ImpMobNetv3 23.45%, versus
72.19% for the P0-FP32 CAFE classifier).

EDUS demotes POSTERv2 to ranks 71–77 despite its leading end-to-end accuracy (0.595–0.637), because
its 180.6 MB footprint collapses the classifier-size term — weighted 0.40 to reflect shared-memory
pressure on the Jetson's 7.4 GB envelope. It demotes PCNN to 64–70 on the strength of dual failure
in both size (103.5 MB) and accuracy (0.103–0.156). Conversely it promotes LANMSFF to ranks 12–44
on its 1.224 MB footprint, surfacing a competitive deployment candidate that a size-blind ranking
would miss — with the accuracy cost (0.374–0.401) left for the application owner to adjudicate.
