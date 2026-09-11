"""Precision/compilation ablation: separating pruning from TensorRT + FP16.

The headline pipeline speedup compares a P0-FP32 configuration executed in
native PyTorch against a P10-FP16 configuration executed as a TensorRT engine.
That comparison changes three variables at once -- pruning ratio, numeric
precision, and runtime -- so it cannot attribute the speedup to any one of
them. Table IV already hints at the answer: P0-FP32 and P10-FP32 differ by
0.02 ms, meaning structured pruning alone buys almost no latency on a premium
GPU.

The missing measurement is the unpruned model compiled to a TensorRT FP16
engine (P0-FP16). With it the speedup decomposes into two independent factors:

    P0-FP32  ->  P0-FP16    compilation + precision, at fixed architecture
    P0-FP16  ->  P10-FP16   pruning, at fixed runtime and precision

Every arm is measured in a single process on a single GPU, with identical
warmup, identical image ordering, and identical timing instrumentation, so the
ratios are internally valid even where absolute latencies differ from the
paper's hardware.

Usage:
    python experiments/precision_ablation/ablate_precision.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "validation"
ENGINE_DIR = Path("/tmp/cafe_ablation_engines")

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLS_TEST_DIR = REPO_ROOT / "data" / "FER2013" / "test"
SPLIT_PREFIX = "PrivateTest"
DET_IMG_DIR = REPO_ROOT / "data" / "Stress_test" / "emotion_raw_images"

CLS_IMGSZ, DET_IMGSZ = 224, 640
WARMUP = 20

# (arm label, pruning tag, weight path relative to repo root, precision)
CLASSIFIER_ARMS = [
    ("P0-FP32", "P0", "weights/P0/classifier.pt", "FP32"),
    ("P0-FP16", "P0", "weights/P0/classifier.pt", "FP16"),
    ("P10-FP32", "P10", "weights/P10/classifier.pt", "FP32"),
    ("P10-FP16", "P10", "weights/P10/classifier.pt", "FP16"),
    ("P15-FP32", "P15", "weights/P15/classifier.pt", "FP32"),
    ("P15-FP16", "P15", "weights/P15/classifier.pt", "FP16"),
]

DETECTOR_ARMS = [
    ("P0-FP32", "P0", "weights/P0/detector.pt", "FP32"),
    ("P0-FP16", "P0", "weights/P0/detector.pt", "FP16"),
    ("P10-FP32", "P10", "weights/P10/detector.pt", "FP32"),
    ("P10-FP16", "P10", "weights/P10/detector.pt", "FP16"),
    ("P15-FP32", "P15", "weights/P15/detector.pt", "FP32"),
    ("P15-FP16", "P15", "weights/P15/detector.pt", "FP16"),
]


def classifier_images(limit: int | None) -> list[tuple[Path, int]]:
    items = []
    for idx, name in enumerate(CLASSES):
        d = CLS_TEST_DIR / name
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if SPLIT_PREFIX and not p.name.startswith(SPLIT_PREFIX):
                continue
            items.append((p, idx))
    if limit:
        # Stratified: take every k-th image so all classes stay represented.
        step = max(1, len(items) // limit)
        items = items[::step][:limit]
    return items


def detector_images() -> list[Path]:
    return sorted(p for p in DET_IMG_DIR.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def export_engine(weight: Path, task: str, imgsz: int, tag: str) -> Path:
    """Compile a TensorRT FP16 engine, reusing it if already built."""
    from ultralytics import YOLO

    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    target = ENGINE_DIR / f"{tag}.engine"
    if target.is_file():
        return target
    exported = YOLO(str(weight), task=task).export(
        format="engine", imgsz=imgsz, batch=1, half=True, int8=False,
        device=0, simplify=True, verbose=False,
    )
    src = Path(str(exported))
    if src.resolve() != target.resolve():
        target.write_bytes(src.read_bytes())
    return target


def time_arm(source: str, task: str, imgsz: int, images, labelled: bool) -> dict:
    """Run one arm end to end, returning mean per-phase latency and accuracy.

    Ultralytics' own speed dictionary is used for both PyTorch and engine paths
    so that the two runtimes are instrumented identically -- timing PyTorch by
    hand while reading engine timings from the framework would bias the very
    comparison this ablation exists to make.
    """
    from ultralytics import YOLO

    model = YOLO(source, task=task)
    warm = images[:WARMUP]
    for item in warm:
        path = item[0] if labelled else item
        model.predict(source=str(path), imgsz=imgsz, device=0, verbose=False)
    torch.cuda.synchronize()

    pre, inf, post, correct = [], [], [], 0
    t_wall = time.perf_counter()
    for item in images:
        path, true_idx = (item if labelled else (item, None))
        r = model.predict(source=str(path), imgsz=imgsz, device=0, verbose=False)[0]
        torch.cuda.synchronize()
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
        if labelled:
            correct += int(int(r.probs.top1) == true_idx)
    wall = time.perf_counter() - t_wall

    out = {
        "n_images": len(images),
        "pre_ms": round(float(np.mean(pre)), 4),
        "infer_ms": round(float(np.mean(inf)), 4),
        "post_ms": round(float(np.mean(post)), 4),
        "total_ms": round(float(np.mean(pre) + np.mean(inf) + np.mean(post)), 4),
        "infer_ms_p50": round(float(np.percentile(inf, 50)), 4),
        "infer_ms_p95": round(float(np.percentile(inf, 95)), 4),
        "wall_s": round(wall, 2),
    }
    if labelled:
        out["top1_acc_pct"] = round(100.0 * correct / len(images), 2)
    return out


def run_stage(name: str, arms, task: str, imgsz: int, images, labelled: bool) -> list[dict]:
    rows = []
    print(f"\n=== {name} ===")
    for label, prune_tag, rel, precision in arms:
        weight = REPO_ROOT / rel
        if not weight.is_file():
            print(f"  {label:10s} SKIPPED (missing {rel})")
            continue
        if precision == "FP32":
            source = str(weight)
        else:
            source = str(export_engine(weight, task, imgsz, f"{name}_{label}"))
        res = time_arm(source, task, imgsz, images, labelled)
        row = {"stage": name, "arm": label, "pruning": prune_tag,
               "precision": precision,
               "runtime": "PyTorch" if precision == "FP32" else "TensorRT",
               "size_mb": round(weight.stat().st_size / 1e6, 3), **res}
        rows.append(row)
        acc = f"  top-1 {res['top1_acc_pct']:.2f}%" if labelled else ""
        print(f"  {label:10s} infer {res['infer_ms']:7.3f} ms   "
              f"total {res['total_ms']:7.3f} ms{acc}")
    return rows


def decompose(rows: list[dict], stage: str) -> dict:
    """Split the headline speedup into compilation+precision and pruning."""
    by = {r["arm"]: r for r in rows if r["stage"] == stage}
    need = ("P0-FP32", "P0-FP16", "P10-FP16")
    if not all(k in by for k in need):
        return {}
    f = lambda a: by[a]["infer_ms"]
    total = f("P0-FP32") / f("P10-FP16")
    compile_only = f("P0-FP32") / f("P0-FP16")
    prune_only = f("P0-FP16") / f("P10-FP16")
    share = np.log(compile_only) / np.log(total) if total > 1 else float("nan")
    return {
        "stage": stage,
        "headline_speedup_P0FP32_to_P10FP16": round(total, 3),
        "compilation_precision_speedup_P0FP32_to_P0FP16": round(compile_only, 3),
        "pruning_speedup_P0FP16_to_P10FP16": round(prune_only, 3),
        "compilation_share_of_log_speedup_pct": round(100 * float(share), 1),
        "pruning_share_of_log_speedup_pct": round(100 * (1 - float(share)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate a stratified subset of the classifier split")
    ap.add_argument("--det-repeat", type=int, default=10,
                    help="passes over the 20-scene detector set, for a stable "
                         "latency estimate from a small image pool")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cls_imgs = classifier_images(args.limit)
    det_imgs = detector_images() * args.det_repeat

    env = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "host": platform.node(),
    }
    try:
        import tensorrt
        env["tensorrt"] = tensorrt.__version__
    except ImportError:
        env["tensorrt"] = None
    try:
        import ultralytics
        env["ultralytics"] = ultralytics.__version__
    except ImportError:
        env["ultralytics"] = None

    print(json.dumps(env, indent=2))
    print(f"classifier images: {len(cls_imgs)}   detector images: {len(det_imgs)}")

    rows = []
    rows += run_stage("classifier", CLASSIFIER_ARMS, "classify", CLS_IMGSZ, cls_imgs, True)
    rows += run_stage("detector", DETECTOR_ARMS, "detect", DET_IMGSZ, det_imgs, False)

    decomp = [d for d in (decompose(rows, "classifier"), decompose(rows, "detector")) if d]
    print("\n=== speedup decomposition ===")
    for d in decomp:
        print(f"{d['stage']}: headline {d['headline_speedup_P0FP32_to_P10FP16']}x = "
              f"compilation+FP16 {d['compilation_precision_speedup_P0FP32_to_P0FP16']}x "
              f"x pruning {d['pruning_speedup_P0FP16_to_P10FP16']}x   "
              f"({d['compilation_share_of_log_speedup_pct']}% of the log-speedup is "
              f"compilation+precision)")

    pd.DataFrame(rows).to_csv(OUT_DIR / "precision_ablation.csv", index=False)
    (OUT_DIR / "precision_ablation.json").write_text(
        json.dumps({"environment": env, "arms": rows, "decomposition": decomp}, indent=2))
    print(f"\nwrote 2 files to {OUT_DIR}")


if __name__ == "__main__":
    main()
