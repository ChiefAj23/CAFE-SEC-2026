"""Recover the undocumented pipeline parameters from the published summary.

Two parameters of the end-to-end evaluation are not stated numerically in the
paper: the padding coefficient lambda (Sec. III-C promises its value in
Sec. V-B, which does not contain it) and the coarse group assigned to the
'surprise' class in the 7-to-3 emotion mapping. Both are recoverable, because
the published summary CSV pins the accuracy of every configuration to four
decimal places.

This script runs the deterministic FP32 detector once, then sweeps lambda and
the surprise assignment over the classifier stage only, comparing the
resulting (e2e_accuracy, cls_accuracy) for the all-FP32 baseline configuration
against the published desktop row. The FP32 pair is used because it involves
no TensorRT compilation, so its detections are reproducible across GPUs up to
floating-point noise.

Usage (run on a CUDA machine with the repo weights present):
    python calibrate_lambda.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_e2e_sweep as h

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results" / "validation" / "lambda_calibration.json"

LAMBDAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SURPRISE_TO = ["neutral", "happy", "sad"]

DET = "YOLO-FP32-Base"
CLS = "YOLO-FP32-Base"


def main() -> None:
    pub = pd.read_csv(REPO_ROOT / "results" / "e2e_summary_desktop_with_EDUS.csv")
    row = pub[(pub.detector == DET) & (pub.classifier == CLS)].iloc[0]
    target = {"e2e": float(row.e2e_accuracy), "cls": float(row.cls_accuracy),
              "n_det": float(row.n_det_faces)}
    print(f"published desktop {DET}+{CLS}: e2e {target['e2e']}  "
          f"cls {target['cls']}  n_det {target['n_det']}")

    scenes = h.load_scenes()
    n_gt = sum(len(s["gt_groups"]) for s in scenes)

    drun, _ = h.build_detector(DET)
    per_scene = []
    for s in scenes:
        dets, _ = drun(s["path"])
        pairs = h.greedy_match(dets, s["gt_boxes"])
        per_scene.append(h.pairs_with_boxes(dets, pairs))
    n_det = sum(len(p) for p in per_scene) / len(scenes)
    print(f"our matched faces/scene: {n_det:.2f} (published {target['n_det']})")

    crun, _ = h.build_classifier(CLS, corrected_norm=False)

    # Classify every matched crop once per lambda; mappings reuse predictions.
    results = []
    for lam in LAMBDAS:
        preds = []  # (pred7, gt_group)
        for s, pairs in zip(scenes, per_scene):
            for det_box, gi in pairs:
                crop = h.pad_crop(s["img"], det_box, lam)
                if crop.size == 0:
                    continue
                pred7, _ = crun(crop)
                preds.append((pred7, s["gt_groups"][gi]))
        for sur in SURPRISE_TO:
            m = dict(h.BASE_MAP, surprise=sur)
            correct = sum(1 for p7, g in preds if m[p7] == g)
            e2e = correct / n_gt
            cls = correct / len(preds) if preds else 0.0
            err = abs(e2e - target["e2e"]) + abs(cls - target["cls"])
            results.append({"lam": lam, "surprise_to": sur,
                            "e2e_accuracy": round(e2e, 4),
                            "cls_accuracy": round(cls, 4),
                            "abs_err_sum": round(err, 4)})
            print(f"  lam={lam:4.2f} surprise->{sur:8s} "
                  f"e2e {e2e:.4f}  cls {cls:.4f}  err {err:.4f}")

    best = min(results, key=lambda r: r["abs_err_sum"])
    print(f"\nbest: lam={best['lam']} surprise->{best['surprise_to']} "
          f"(err {best['abs_err_sum']})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "published_target": target, "our_n_det_faces": round(n_det, 2),
        "grid": results, "best": best,
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
