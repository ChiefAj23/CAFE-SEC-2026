"""External-classifier preprocessing audit, extended to all four baselines.

`audit_preprocessing.py` established that the ImpMobNetv3 baseline's low top-1
accuracy is caused by feeding raw [0, 1] pixels to a graph that expects
externally normalized input. That raises an immediate question about the rest
of the external comparison: the same shared evaluation path feeds POSTERv2,
PCNN, and LANMSFF, so any of them could be affected the same way.

This script runs the identical preprocessing sweep across all four external
classifiers on the FER2013 held-out split, reporting top-1 accuracy and the
predicted-class distribution per variant. A model whose accuracy jumps under
normalization, and whose prediction histogram goes from collapsed to spread,
was mismeasured; a model that is flat across variants was measured correctly
and is genuinely weak.

LANMSFF differs from the others: 64x64 grayscale input in NHWC layout. Its
variants are constructed accordingly.

Usage:
    python experiments/external_baselines_audit/audit_all_baselines.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"
TEST_DIR = REPO_ROOT / "data" / "FER2013" / "test"
OUT_DIR = REPO_ROOT / "results" / "validation"

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SPLIT_PREFIX = "PrivateTest"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Top-1 accuracy as reported in the paper, for reproduction checking.
PUBLISHED = {
    "ImpMobNetv3": 23.45,
    "POSTERv2": 59.25,
    "PCNN": 53.80,
    "LANMSFF": 38.30,
}

MODELS = {
    "ImpMobNetv3": {"file": "mobilenetv3_best_opset20_dynB.onnx", "imgsz": 224,
                    "channels": 3, "layout": "nchw"},
    "POSTERv2": {"file": "poster_v2_fer2013_dynB.onnx", "imgsz": 224,
                 "channels": 3, "layout": "nchw"},
    "PCNN": {"file": "pcnn_best_opset20_dynB.onnx", "imgsz": 224,
             "channels": 3, "layout": "nchw"},
    "LANMSFF": {"file": "two_path_massatt_pwfs_opset20.onnx", "imgsz": 64,
                "channels": 1, "layout": "nhwc"},
}


def load_images() -> list[tuple[Path, int]]:
    items = []
    for idx, name in enumerate(CLASSES):
        d = TEST_DIR / name
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if SPLIT_PREFIX and not p.name.startswith(SPLIT_PREFIX):
                continue
            items.append((p, idx))
    return items


def decode(path: Path, meta: dict) -> np.ndarray:
    """Decode and resize to the model's input size. Returns HWC float [0,1]."""
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    size = meta["imgsz"]
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    img = cv2.resize(img, (size, size), interpolation=interp)
    if meta["channels"] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, None]
    return img.astype(np.float32) / 255.0


def pack(hwc: np.ndarray, meta: dict) -> np.ndarray:
    if meta["layout"] == "nchw":
        return np.transpose(hwc, (2, 0, 1))[None].astype(np.float32)
    return hwc[None].astype(np.float32)


def build_variants(meta: dict):
    """Preprocessing variants appropriate to the model's channel count."""
    if meta["channels"] == 3:
        return {
            "as_published": lambda x: pack(x, meta),
            "imagenet_norm": lambda x: pack((x - IMAGENET_MEAN) / IMAGENET_STD, meta),
            "symmetric_norm": lambda x: pack((x - 0.5) / 0.5, meta),
        }
    # Single-channel: the ImageNet grayscale convention uses the mean of the
    # RGB constants, which is what a 1-channel port of a torchvision transform
    # produces.
    gray_mean = float(IMAGENET_MEAN.mean())
    gray_std = float(IMAGENET_STD.mean())
    return {
        "as_published": lambda x: pack(x, meta),
        "imagenet_norm": lambda x: pack((x - gray_mean) / gray_std, meta),
        "symmetric_norm": lambda x: pack((x - 0.5) / 0.5, meta),
    }


def has_normalization_prologue(path: Path) -> tuple[bool, list[str]]:
    model = onnx.load(str(path), load_external_data=False)
    ops = [n.op_type for n in model.graph.node]
    return any(op in ("Sub", "Div") for op in ops[:4]), ops[:6]


def evaluate(sess, in_name, out_name, items, meta, fn) -> dict:
    correct, preds = 0, []
    for path, true_idx in items:
        logits = sess.run([out_name], {in_name: fn(decode(path, meta))})[0]
        pred = int(np.argmax(logits[0]))
        preds.append(pred)
        correct += int(pred == true_idx)
    hist = Counter(preds)
    top_class, top_count = hist.most_common(1)[0]
    return {
        "top1_acc_pct": round(100.0 * correct / len(items), 2),
        "distinct_classes_predicted": len(hist),
        "dominant_class": CLASSES[top_class],
        "dominant_class_share_pct": round(100.0 * top_count / len(items), 2),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_images()
    if not items:
        raise SystemExit(f"no images under {TEST_DIR} -- see docs/DATA.md")

    counts = Counter(t for _, t in items)
    majority = 100.0 * max(counts.values()) / len(items)
    print(f"{len(items)} images ({SPLIT_PREFIX} split), "
          f"majority-class rate {majority:.2f}%\n")

    rows, report = [], {"n_images": len(items),
                        "majority_class_rate_pct": round(majority, 2),
                        "models": {}}

    for name, meta in MODELS.items():
        path = ARTIFACTS / meta["file"]
        if not path.is_file():
            print(f"{name:12s} SKIPPED -- {meta['file']} not present")
            continue

        prologue, head = has_normalization_prologue(path)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        entry = {"file": meta["file"], "published_top1_pct": PUBLISHED.get(name),
                 "graph_has_normalization_prologue": prologue,
                 "graph_head_ops": head, "variants": {}}
        print(f"{name}  (input {meta['imgsz']}x{meta['imgsz']}, "
              f"{meta['channels']}ch {meta['layout'].upper()}, "
              f"baked-in norm: {prologue})")

        for vname, fn in build_variants(meta).items():
            res = evaluate(sess, in_name, out_name, items, meta, fn)
            entry["variants"][vname] = res
            rows.append({"model": name, "variant": vname,
                         "published_top1_pct": PUBLISHED.get(name), **res})
            print(f"   {vname:16s} top-1 {res['top1_acc_pct']:6.2f}%   "
                  f"classes {res['distinct_classes_predicted']}/7   "
                  f"dominant {res['dominant_class']} "
                  f"{res['dominant_class_share_pct']:.1f}%")

        best = max(entry["variants"], key=lambda k: entry["variants"][k]["top1_acc_pct"])
        pub = PUBLISHED.get(name)
        delta = entry["variants"][best]["top1_acc_pct"] - entry["variants"]["as_published"]["top1_acc_pct"]
        entry["best_variant"] = best
        entry["gain_over_as_published_pp"] = round(delta, 2)
        entry["reproduces_published"] = (
            pub is not None
            and abs(entry["variants"]["as_published"]["top1_acc_pct"] - pub) < 1.0
        )
        print(f"   -> best {best}, +{delta:.2f} pp over as-published; "
              f"reproduces published {pub}%: {entry['reproduces_published']}\n")
        report["models"][name] = entry

    pd.DataFrame(rows).to_csv(OUT_DIR / "external_baseline_audit.csv", index=False)
    (OUT_DIR / "external_baseline_audit.json").write_text(json.dumps(report, indent=2))
    print(f"wrote 2 files to {OUT_DIR}")


if __name__ == "__main__":
    main()
