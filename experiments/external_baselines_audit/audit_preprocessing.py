"""ImpMobNetv3 accuracy audit.

The ImpMobNetv3 baseline scores 23.45% top-1 on FER2013 when evaluated through
the shared deployment pipeline. FER2013's majority class (happy) accounts for
24.7% of the test split, so the reported figure sits at or below the trivial
majority-class predictor -- a level that indicates a broken evaluation path
rather than a weak model.

This script isolates the cause by holding the model, weights, and test data
fixed and varying only the input preprocessing, then reports top-1 accuracy and
the predicted-class distribution for each variant. A collapsed prediction
histogram (nearly all mass on one class) distinguishes a preprocessing fault
from genuine low accuracy.

The exported graph is checked for a baked-in normalization prologue. If the
graph begins directly at a convolution, the model expects externally normalized
input, and feeding it raw [0, 1] pixels is a preprocessing error.

Usage:
    python experiments/external_baselines_audit/audit_preprocessing.py
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
MODEL = REPO_ROOT / "artifacts" / "mobilenetv3_best_opset20_dynB.onnx"
TEST_DIR = REPO_ROOT / "data" / "FER2013" / "test"
OUT_DIR = REPO_ROOT / "results" / "validation"

# Alphabetical order, as used by the deployment pipeline's ImageFolder scan.
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# The paper evaluates the 3589-image held-out split; FER2013 encodes it in the
# filename prefix. Set to None to evaluate all 7178 test images.
SPLIT_PREFIX = "PrivateTest"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMGSZ = 224


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


def to_rgb_float(path: Path) -> np.ndarray:
    """Decode, RGB-convert, resize, scale to [0, 1]. HWC float32.

    This is the deployment pipeline's path, factored out so that every variant
    below differs only in what happens after this point.
    """
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_AREA if img.shape[0] > IMGSZ else cv2.INTER_LINEAR
    img = cv2.resize(img, (IMGSZ, IMGSZ), interpolation=interp)
    return img.astype(np.float32) / 255.0


def nchw(hwc: np.ndarray) -> np.ndarray:
    return np.transpose(hwc, (2, 0, 1))[None].astype(np.float32)


# Each variant maps an RGB [0,1] HWC array to a model-ready NCHW batch.
VARIANTS = {
    # As implemented in the deployment pipeline: scale to [0,1], nothing else.
    "as_published": lambda x: nchw(x),
    # Standard torchvision preprocessing for ImageNet-pretrained backbones.
    "imagenet_norm": lambda x: nchw((x - IMAGENET_MEAN) / IMAGENET_STD),
    # Symmetric [-1, 1] scaling, the other common convention.
    "symmetric_norm": lambda x: nchw((x - 0.5) / 0.5),
    # Control: channel order swapped, to confirm it is not the cause.
    # FER2013 is grayscale, so this should be a no-op.
    "bgr_imagenet_norm": lambda x: nchw((x[:, :, ::-1] - IMAGENET_MEAN) / IMAGENET_STD),
}


def graph_has_normalization_prologue(path: Path) -> tuple[bool, list[str]]:
    model = onnx.load(str(path), load_external_data=False)
    ops = [n.op_type for n in model.graph.node]
    head = [o for o in ops[:8]]
    has_prologue = any(op in ("Sub", "Div") for op in ops[:4])
    return has_prologue, head


def evaluate(sess, in_name, out_name, items, fn) -> dict:
    correct = 0
    preds = []
    for path, true_idx in items:
        x = fn(to_rgb_float(path))
        logits = sess.run([out_name], {in_name: x})[0]
        pred = int(np.argmax(logits[0]))
        preds.append(pred)
        correct += int(pred == true_idx)
    hist = Counter(preds)
    top_class, top_count = hist.most_common(1)[0]
    return {
        "top1_acc_pct": round(100.0 * correct / len(items), 2),
        "n_images": len(items),
        "distinct_classes_predicted": len(hist),
        "dominant_class": CLASSES[top_class],
        "dominant_class_share_pct": round(100.0 * top_count / len(items), 2),
        "pred_histogram": {CLASSES[k]: v for k, v in sorted(hist.items())},
    }


def best_label_permutation(sess, in_name, out_name, items, fn) -> dict:
    """Check whether a class-index mismatch alone explains the low accuracy.

    Builds the confusion matrix once, then greedily assigns each predicted
    index to the true class it most often coincides with. This upper-bounds the
    accuracy recoverable by relabeling, without refitting anything.
    """
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for path, true_idx in items:
        x = fn(to_rgb_float(path))
        pred = int(np.argmax(sess.run([out_name], {in_name: x})[0][0]))
        conf[pred, true_idx] += 1

    mapping, used = {}, set()
    for pred_idx in np.argsort(-conf.sum(axis=1)):
        order = np.argsort(-conf[pred_idx])
        choice = next((int(c) for c in order if int(c) not in used), None)
        if choice is None:
            continue
        mapping[int(pred_idx)] = choice
        used.add(choice)

    recovered = sum(conf[p, t] for p, t in mapping.items())
    return {
        "upper_bound_acc_pct": round(100.0 * recovered / conf.sum(), 2),
        "permutation": {CLASSES[p]: CLASSES[t] for p, t in sorted(mapping.items())},
        "is_identity": all(p == t for p, t in mapping.items()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_images()
    if not items:
        raise SystemExit(f"no images under {TEST_DIR} -- see docs/DATA.md")

    has_prologue, head_ops = graph_has_normalization_prologue(MODEL)
    class_counts = Counter(t for _, t in items)
    majority = max(class_counts.values()) / len(items)

    print(f"model : {MODEL.name}")
    print(f"graph : first ops {head_ops}")
    print(f"        baked-in normalization prologue: {has_prologue}")
    print(f"images: {len(items)} ({SPLIT_PREFIX or 'all'} split)")
    print(f"majority-class rate: {100 * majority:.2f}%\n")

    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    report = {
        "model": MODEL.name,
        "split": SPLIT_PREFIX,
        "n_images": len(items),
        "majority_class_rate_pct": round(100 * majority, 2),
        "graph_has_normalization_prologue": has_prologue,
        "graph_head_ops": head_ops,
        "variants": {},
    }

    rows = []
    for name, fn in VARIANTS.items():
        res = evaluate(sess, in_name, out_name, items, fn)
        report["variants"][name] = res
        rows.append({"variant": name, **{k: v for k, v in res.items() if k != "pred_histogram"}})
        print(f"{name:20s} top-1 {res['top1_acc_pct']:6.2f}%  "
              f"classes predicted {res['distinct_classes_predicted']}/7  "
              f"dominant {res['dominant_class']} {res['dominant_class_share_pct']:.1f}%")

    best = max(report["variants"], key=lambda k: report["variants"][k]["top1_acc_pct"])
    print(f"\nbest variant: {best}")

    print("\nlabel-permutation upper bound (as_published):")
    perm = best_label_permutation(sess, in_name, out_name, items, VARIANTS["as_published"])
    report["label_permutation_as_published"] = perm
    print(f"  upper bound {perm['upper_bound_acc_pct']}%  identity={perm['is_identity']}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "mobilenetv3_audit.csv", index=False)
    (OUT_DIR / "mobilenetv3_audit.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote 2 files to {OUT_DIR}")


if __name__ == "__main__":
    main()
