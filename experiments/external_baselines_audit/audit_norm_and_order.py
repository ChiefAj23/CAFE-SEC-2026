"""External classifiers: joint sweep over normalization and class ordering.

The preprocessing audit established that three external classifiers were fed
un-normalized input. It also concluded that LANMSFF was measured correctly,
because no normalization improved it. That conclusion tested only one of the
two conventions a borrowed model can disagree on.

The second is output class ordering. FER2013 has two orderings in common use:
the canonical order of the original Kaggle release (angry, disgust, fear,
happy, sad, surprise, neutral), and the alphabetical order produced by any
ImageFolder-style directory scan (angry, disgust, fear, happy, neutral, sad,
surprise). They differ only in the last three positions, so a model trained
under one and evaluated under the other loses accuracy on three of seven
classes while still appearing to function -- no collapse, no error, just a
plausible lower number.

A model can disagree on either convention independently, so the two must be
swept jointly rather than one at a time. This script evaluates every external
classifier over the full cross product and reports the best cell, which
identifies the contract each model was actually trained under.

Usage:
    python experiments/external_baselines_audit/audit_norm_and_order.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
TEST_DIR = REPO_ROOT / "data" / "FER2013" / "test"
OUT_DIR = REPO_ROOT / "results" / "validation"

SPLIT_PREFIX = "PrivateTest"

# Directory names on disk are alphabetical; that is the ground-truth source.
DIR_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

ORDERINGS = {
    "alphabetical": DIR_ORDER,
    "kaggle": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PUBLISHED = {"ImpMobNetv3": 23.45, "POSTERv2": 59.25,
             "PCNN": 53.80, "LANMSFF": 38.30}

MODELS = {
    "ImpMobNetv3": {"file": "mobilenetv3_best_opset20_dynB.onnx",
                    "imgsz": 224, "channels": 3, "layout": "nchw"},
    "POSTERv2": {"file": "poster_v2_fer2013_dynB.onnx",
                 "imgsz": 224, "channels": 3, "layout": "nchw"},
    "PCNN": {"file": "pcnn_best_opset20_dynB.onnx",
             "imgsz": 224, "channels": 3, "layout": "nchw"},
    "LANMSFF": {"file": "two_path_massatt_pwfs_opset20.onnx",
                "imgsz": 64, "channels": 1, "layout": "nhwc"},
}

NORMALIZATIONS = ("none", "imagenet", "symmetric")


def load_items() -> list[tuple[Path, str]]:
    items = []
    for name in DIR_ORDER:
        d = TEST_DIR / name
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and (
                    not SPLIT_PREFIX or p.name.startswith(SPLIT_PREFIX)):
                items.append((p, name))
    return items


def prepare(path: Path, meta: dict, norm: str) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    size = meta["imgsz"]
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    img = cv2.resize(img, (size, size), interpolation=interp).astype(np.float32) / 255.0
    if meta["channels"] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, None]
        if norm == "imagenet":
            img = (img - IMAGENET_MEAN.mean()) / IMAGENET_STD.mean()
        elif norm == "symmetric":
            img = (img - 0.5) / 0.5
    else:
        if norm == "imagenet":
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        elif norm == "symmetric":
            img = (img - 0.5) / 0.5
    return (np.transpose(img, (2, 0, 1))[None] if meta["layout"] == "nchw"
            else img[None]).astype(np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_items()
    counts = Counter(c for _, c in items)
    majority = 100.0 * max(counts.values()) / len(items)
    print(f"{len(items)} images, majority-class rate {majority:.2f}%\n")

    rows, report = [], {"n_images": len(items),
                        "majority_class_rate_pct": round(majority, 2),
                        "models": {}}

    for name, meta in MODELS.items():
        path = ARTIFACTS / meta["file"]
        if not path.is_file():
            print(f"{name:12s} SKIPPED (missing {meta['file']})")
            continue
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        out = sess.get_outputs()[0].name

        print(f"{name}  (published {PUBLISHED[name]}%)")
        entry, best = {"published_top1_pct": PUBLISHED[name], "grid": {}}, None

        for norm in NORMALIZATIONS:
            # Predictions depend only on preprocessing; scoring them under
            # each ordering is free, so inference runs once per normalization.
            preds = [int(np.argmax(sess.run([out], {inp: prepare(p, meta, norm)})[0][0]))
                     for p, _ in items]
            truth = [c for _, c in items]
            for oname, order in ORDERINGS.items():
                acc = 100.0 * sum(1 for pi, t in zip(preds, truth)
                                  if order[pi] == t) / len(truth)
                hist = Counter(order[pi] for pi in preds)
                top_cls, top_n = hist.most_common(1)[0]
                cell = {"top1_acc_pct": round(acc, 2),
                        "distinct_classes": len(hist),
                        "dominant_class": top_cls,
                        "dominant_share_pct": round(100.0 * top_n / len(preds), 2)}
                entry["grid"][f"{norm}|{oname}"] = cell
                rows.append({"model": name, "normalization": norm,
                             "class_order": oname,
                             "published_top1_pct": PUBLISHED[name], **cell})
                print(f"   norm={norm:9s} order={oname:12s} "
                      f"top-1 {acc:6.2f}%   dominant {top_cls} "
                      f"{cell['dominant_share_pct']:.1f}%")
                if best is None or acc > best[1]:
                    best = (f"{norm}|{oname}", acc)

        entry["best_config"] = best[0]
        entry["best_top1_pct"] = round(best[1], 2)
        entry["gain_over_published_pp"] = round(best[1] - PUBLISHED[name], 2)
        as_pub = entry["grid"]["none|alphabetical"]["top1_acc_pct"]
        entry["reproduces_published"] = abs(as_pub - PUBLISHED[name]) < 1.5
        print(f"   -> best {best[0]} at {best[1]:.2f}%  "
              f"({entry['gain_over_published_pp']:+.2f} pp vs published; "
              f"as-published cell reproduces: {entry['reproduces_published']})\n")
        report["models"][name] = entry

    pd.DataFrame(rows).to_csv(OUT_DIR / "external_norm_order_audit.csv", index=False)
    (OUT_DIR / "external_norm_order_audit.json").write_text(json.dumps(report, indent=2))
    print(f"wrote 2 files to {OUT_DIR}")


if __name__ == "__main__":
    main()
