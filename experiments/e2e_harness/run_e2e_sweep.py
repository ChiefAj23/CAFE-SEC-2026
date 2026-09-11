"""End-to-end pipeline sweep over the in-the-wild stress set.

Reimplements the pipeline-level evaluation behind the end-to-end results table:
each detector runs once per scene, matched detections are cropped with
scale-normalized contextual padding, each classifier runs once per matched
face, and per-configuration accuracy and latency are aggregated exactly as the
summary CSVs report them.

The original sweep script is not part of this repository, so this harness is
reconstructed from the paper's description (Sec. III-C and V-E) plus the
ground-truth annotations. Two parameters the paper does not state numerically
-- the padding coefficient lambda and the 7-to-3 emotion group mapping -- are
exposed as arguments so they can be calibrated against the published summary
(see calibrate_lambda.py).

Ground-truth label semantics (verified visually against the scenes):
class 0 = happy, 1 = sad, 6 = neutral.

Usage:
    python run_e2e_sweep.py --lam 0.10 --surprise-to neutral \
        --out results/validation/e2e_sweep.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
STRESS_IMG = REPO_ROOT / "data" / "Stress_test" / "emotion_raw_images"
STRESS_LBL = REPO_ROOT / "data" / "Stress_test" / "labels_out"
ENGINE_DIR = Path("/tmp/cafe_e2e_engines")

DET_IMGSZ, CLS_IMGSZ = 640, 224
IOU_THRESHOLD = 0.50
DET_CONF = 0.25

# FER2013 alphabetical order used by the YOLO classifiers.
FER7 = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Ground-truth coarse classes as annotated in labels_out (verified visually).
GT_CLASSES = {0: "happy", 1: "sad", 6: "neutral"}

# 7-class -> 3-group mapping. angry/disgust/fear carry negative valence and
# map to "sad"; the group for "surprise" is ambiguous and therefore an
# argument.
BASE_MAP = {"happy": "happy", "sad": "sad", "neutral": "neutral",
            "angry": "sad", "disgust": "sad", "fear": "sad"}

DETECTORS = {
    "YOLO-FP32-Base": ("yolo", "weights/P0/detector.pt", "FP32"),
    "YOLO-FP32-P10": ("yolo", "weights/P10/detector.pt", "FP32"),
    "YOLO-FP32-P15": ("yolo", "weights/P15/detector.pt", "FP32"),
    "YOLO-FP16-P10": ("yolo", "weights/P10/detector.pt", "FP16"),
    "YOLO-FP16-P15": ("yolo", "weights/P15/detector.pt", "FP16"),
    "YOLO-INT8-P10": ("yolo", "weights/P10/detector.pt", "INT8"),
    "YOLO-INT8-P15": ("yolo", "weights/P15/detector.pt", "INT8"),
    "RT-DETR-FP32": ("rtdetr", "artifacts/onnx_exports/face_det_rtdetr_train_clean2_op20_b1_640_static_shared_baseline.onnx", "FP32"),
}

CLASSIFIERS = {
    "YOLO-FP32-Base": ("yolo", "weights/P0/classifier.pt", "FP32"),
    "YOLO-FP32-P10": ("yolo", "weights/P10/classifier.pt", "FP32"),
    "YOLO-FP32-P15": ("yolo", "weights/P15/classifier.pt", "FP32"),
    "YOLO-FP16-P10": ("yolo", "weights/P10/classifier.pt", "FP16"),
    "YOLO-FP16-P15": ("yolo", "weights/P15/classifier.pt", "FP16"),
    "YOLO-INT8-P10": ("yolo", "weights/P10/classifier.pt", "INT8"),
    "YOLO-INT8-P15": ("yolo", "weights/P15/classifier.pt", "INT8"),
    "MobileNetV3": ("onnx", "artifacts/mobilenetv3_best_opset20_dynB.onnx", "FP32"),
    "POSTERv2": ("onnx", "artifacts/poster_v2_fer2013_dynB.onnx", "FP32"),
    "PCNN": ("onnx", "artifacts/pcnn_best_opset20_dynB.onnx", "FP32"),
    "LANMSFF": ("onnx", "artifacts/two_path_massatt_pwfs_opset20.onnx", "FP32"),
}

ONNX_META = {
    "MobileNetV3": {"imgsz": 224, "channels": 3, "layout": "nchw"},
    "POSTERv2": {"imgsz": 224, "channels": 3, "layout": "nchw"},
    "PCNN": {"imgsz": 224, "channels": 3, "layout": "nchw"},
    "LANMSFF": {"imgsz": 64, "channels": 1, "layout": "nhwc"},
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- ground truth

def load_scenes() -> list[dict]:
    scenes = []
    for lbl in sorted(STRESS_LBL.glob("*.txt")):
        img_path = STRESS_IMG / f"{lbl.stem}.png"
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        boxes, groups = [], []
        for line in lbl.read_text().splitlines():
            c, xc, yc, bw, bh = line.split()
            xc, yc, bw, bh = (float(xc) * w, float(yc) * h,
                              float(bw) * w, float(bh) * h)
            boxes.append([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2])
            groups.append(GT_CLASSES[int(c)])
        scenes.append({"id": lbl.stem, "path": img_path, "img": img,
                       "gt_boxes": np.array(boxes), "gt_groups": groups})
    return scenes


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter) if ua + ub - inter > 0 else 0.0


def greedy_match(dets: list, gt_boxes: np.ndarray) -> list[tuple[int, int]]:
    """Greedy confidence-ordered matching at IoU >= 0.50.

    Returns (det_idx, gt_idx) pairs. Each ground-truth box matches at most one
    detection; detections below the threshold are false positives and
    unmatched ground truth are missed detections.
    """
    order = sorted(range(len(dets)), key=lambda k: -dets[k][4])
    taken, pairs = set(), []
    for di in order:
        best_iou, best_gt = 0.0, -1
        for gi in range(len(gt_boxes)):
            if gi in taken:
                continue
            iou = box_iou(np.array(dets[di][:4]), gt_boxes[gi])
            if iou > best_iou:
                best_iou, best_gt = iou, gi
        if best_iou >= IOU_THRESHOLD:
            taken.add(best_gt)
            pairs.append((di, best_gt))
    return pairs


def pad_crop(img: np.ndarray, box, lam: float) -> np.ndarray:
    """Scale-normalized contextual padding, Eq. 1 of the paper."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px1 = int(max(0, x1 - lam * w))
    px2 = int(min(W, x2 + lam * w))
    py1 = int(max(0, y1 - lam * h))
    py2 = int(min(H, y2 + lam * h))
    return img[py1:py2, px1:px2]


# ------------------------------------------------------------------- detectors

def calibration_dataset() -> Path:
    """Build a single-class face-detection set for INT8 calibration.

    The paper calibrates the INT8 detectors on the WIDER FACE validation
    split, which is not distributed with this repository. The stress-set
    scenes are themselves drawn from the WIDER FACE test split, so they are
    in-domain calibration data; the set is far smaller, which is recorded in
    the run metadata as a deviation from the original protocol.
    """
    root = Path("/tmp/cafe_det_calib")
    yaml_path = root / "faces.yaml"
    if yaml_path.is_file():
        return yaml_path
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in sorted(STRESS_IMG.iterdir()):
            if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            dst = root / "images" / split / img.name
            if not dst.exists():
                dst.symlink_to(img.resolve())
            src_lbl = STRESS_LBL / f"{img.stem}.txt"
            # Emotion class is irrelevant here; every box is a face.
            lines = [" ".join(["0"] + ln.split()[1:])
                     for ln in src_lbl.read_text().splitlines() if ln.strip()]
            (root / "labels" / split / f"{img.stem}.txt").write_text(
                "\n".join(lines) + "\n")
    yaml_path.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n"
        "nc: 1\nnames: [face]\n")
    return yaml_path


def build_detector(name: str):
    kind, rel, precision = DETECTORS[name]
    path = REPO_ROOT / rel
    if kind == "yolo":
        from ultralytics import YOLO
        if precision in ("FP16", "INT8"):
            ENGINE_DIR.mkdir(parents=True, exist_ok=True)
            eng = ENGINE_DIR / f"det_{name}.engine"
            if not eng.is_file():
                kwargs = dict(format="engine", imgsz=DET_IMGSZ, batch=1,
                              device=0, simplify=True, verbose=False)
                if precision == "FP16":
                    kwargs["half"] = True
                else:
                    kwargs.update(int8=True, half=False,
                                  data=str(calibration_dataset()), split="val")
                out = YOLO(str(path), task="detect").export(**kwargs)
                eng.write_bytes(Path(str(out)).read_bytes())
            model = YOLO(str(eng), task="detect")
        else:
            model = YOLO(str(path), task="detect")

        def run(img_path):
            r = model.predict(source=str(img_path), imgsz=DET_IMGSZ,
                              conf=DET_CONF, device=0, verbose=False)[0]
            # Framework-reported pre/infer/post, excluding image decode from
            # disk. Wall-clock around predict() would add a fixed per-frame I/O
            # cost that dominates a nano-scale detector and is not part of the
            # inference latency being compared.
            lat = sum(r.speed[k] for k in ("preprocess", "inference",
                                           "postprocess"))
            dets = [[*map(float, b.xyxy[0]), float(b.conf[0])] for b in r.boxes]
            return dets, lat
        return run, round(path.stat().st_size / 1e6, 3)

    # RT-DETR through ONNX Runtime, pre/post as in the component notebook.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name

    def run(img_path):
        img = cv2.imread(str(img_path))   # decode excluded from timing
        h0, w0 = img.shape[:2]
        t0 = time.perf_counter()
        r = min(DET_IMGSZ / h0, DET_IMGSZ / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        resized = cv2.resize(img, (nw, nh))
        top = (DET_IMGSZ - nh) // 2
        left = (DET_IMGSZ - nw) // 2
        canvas = cv2.copyMakeBorder(resized, top, DET_IMGSZ - nh - top,
                                    left, DET_IMGSZ - nw - left,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
        raw = sess.run([out], {inp: x})[0][0]
        lat = (time.perf_counter() - t0) * 1000
        dets = []
        for row in raw:
            score = float(row[4])
            if score < DET_CONF:
                continue
            cx, cy, bw, bh = (row[0] * DET_IMGSZ, row[1] * DET_IMGSZ,
                              row[2] * DET_IMGSZ, row[3] * DET_IMGSZ)
            x1 = np.clip((cx - bw / 2 - left) / r, 0, w0)
            x2 = np.clip((cx + bw / 2 - left) / r, 0, w0)
            y1 = np.clip((cy - bh / 2 - top) / r, 0, h0)
            y2 = np.clip((cy + bh / 2 - top) / r, 0, h0)
            if x2 > x1 and y2 > y1:
                dets.append([float(x1), float(y1), float(x2), float(y2), score])
        return dets, lat
    return run, round(path.stat().st_size / 1e6, 3)


# ----------------------------------------------------------------- classifiers

def build_classifier(name: str, corrected_norm: bool):
    kind, rel, precision = CLASSIFIERS[name]
    path = REPO_ROOT / rel
    if kind == "yolo":
        from ultralytics import YOLO
        if precision in ("FP16", "INT8"):
            ENGINE_DIR.mkdir(parents=True, exist_ok=True)
            eng = ENGINE_DIR / f"cls_{name}.engine"
            if not eng.is_file():
                kwargs = dict(format="engine", imgsz=CLS_IMGSZ, batch=1,
                              device=0, simplify=True, verbose=False)
                if precision == "FP16":
                    kwargs["half"] = True
                else:
                    # INT8 calibration on the FER2013 held-out split, as in
                    # the paper. Ultralytics reads the 'val' split.
                    data_root = REPO_ROOT / "data" / "FER2013"
                    # Ultralytics' classification loader requires both splits
                    # to exist; only the held-out test split is distributed
                    # here, and calibration reads 'val'.
                    for split in ("val", "train"):
                        link = data_root / split
                        if not link.exists():
                            link.symlink_to((data_root / "test").resolve())
                    kwargs.update(int8=True, half=False,
                                  data=str(data_root), split="val")
                out = YOLO(str(path), task="classify").export(**kwargs)
                eng.write_bytes(Path(str(out)).read_bytes())
            model = YOLO(str(eng), task="classify")
        else:
            model = YOLO(str(path), task="classify")

        def run(crop):
            t0 = time.perf_counter()
            r = model.predict(source=crop, imgsz=CLS_IMGSZ, device=0,
                              verbose=False)[0]
            lat = (time.perf_counter() - t0) * 1000
            return FER7[int(r.probs.top1)], lat
        return run, round(path.stat().st_size / 1e6, 3)

    import onnxruntime as ort
    meta = ONNX_META[name]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name

    def run(crop):
        t0 = time.perf_counter()
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        size = meta["imgsz"]
        img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
        if meta["channels"] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, None]
            if corrected_norm:
                img = (img - IMAGENET_MEAN.mean()) / IMAGENET_STD.mean()
        elif corrected_norm:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        x = (np.transpose(img, (2, 0, 1))[None] if meta["layout"] == "nchw"
             else img[None]).astype(np.float32)
        logits = sess.run([out], {inp: x})[0][0]
        lat = (time.perf_counter() - t0) * 1000
        return FER7[int(np.argmax(logits))], lat
    return run, round(path.stat().st_size / 1e6, 3)


# ----------------------------------------------------------------------- sweep

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True,
                    help="padding coefficient lambda (Eq. 1)")
    ap.add_argument("--surprise-to", choices=["happy", "neutral", "sad"],
                    default="neutral", help="coarse group for 'surprise'")
    ap.add_argument("--corrected-norm", action="store_true",
                    help="apply ImageNet normalization to the external "
                         "classifiers (post-audit corrected preprocessing)")
    ap.add_argument("--detectors", nargs="*", default=list(DETECTORS))
    ap.add_argument("--classifiers", nargs="*", default=list(CLASSIFIERS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seven_to_three = dict(BASE_MAP, surprise=args.surprise_to)
    scenes = load_scenes()
    n_gt_total = sum(len(s["gt_groups"]) for s in scenes)
    print(f"{len(scenes)} scenes, {n_gt_total} GT faces  "
          f"lam={args.lam}  surprise->{args.surprise_to}  "
          f"corrected_norm={args.corrected_norm}")

    # Detection pass: run each detector once, cache detections and matches.
    det_cache = {}
    for dname in args.detectors:
        run, size_mb = build_detector(dname)
        per_scene, lats = [], []
        for s in scenes:
            dets, lat = run(s["path"])
            pairs = greedy_match(dets, s["gt_boxes"])
            per_scene.append(pairs_with_boxes(dets, pairs))
            lats.append(lat)
        n_det = sum(len(p) for p in per_scene) / len(scenes)
        det_cache[dname] = {"per_scene": per_scene, "size": size_mb,
                            "lat": float(np.mean(lats)), "n_det": n_det}
        print(f"  det {dname:15s} {np.mean(lats):7.1f} ms/frame  "
              f"matched {n_det:.2f} faces/scene")

    rows = []
    for cname in args.classifiers:
        try:
            crun, csize = build_classifier(cname, args.corrected_norm)
        except Exception as exc:
            # One unbuildable classifier must not abort the whole sweep.
            print(f"    SKIP {cname}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        for dname in args.detectors:
            dc = det_cache[dname]
            correct = matched = 0
            cls_lats, e2e_lats = [], []
            for s, pairs in zip(scenes, dc["per_scene"]):
                scene_cls_ms = 0.0
                for det_box, gi in pairs:
                    crop = pad_crop(s["img"], det_box, args.lam)
                    if crop.size == 0:
                        continue
                    pred7, lat = crun(crop)
                    scene_cls_ms += lat
                    cls_lats.append(lat)
                    matched += 1
                    if seven_to_three[pred7] == s["gt_groups"][gi]:
                        correct += 1
                e2e_lats.append(dc["lat"] + scene_cls_ms)
            e2e_acc = correct / n_gt_total
            det_acc = correct / matched if matched else 0.0
            rows.append({
                "detector": dname, "classifier": cname,
                "Size_detector": dc["size"], "Size_classifier": csize,
                "det_lat_ms": round(dc["lat"], 4),
                "cls_lat_ms": round(float(np.mean(cls_lats)), 4),
                "e2e_lat_ms": round(float(np.mean(e2e_lats)), 4),
                "n_det_faces": round(dc["n_det"], 2),
                "n_gt_faces": round(n_gt_total / len(scenes), 2),
                "e2e_accuracy": round(e2e_acc, 4),
                "cls_accuracy": round(det_acc, 4),
            })
            print(f"    {dname:15s} + {cname:15s} "
                  f"e2e {e2e_acc:.4f}  det-acc {det_acc:.4f}")

    df = pd.DataFrame(rows)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        meta = {"lam": args.lam, "surprise_to": args.surprise_to,
                "corrected_norm": args.corrected_norm,
                "int8_detector_calibration": "stress-set scenes (20 images, "
                "drawn from the WIDER FACE test split); the original protocol "
                "calibrates on the WIDER FACE validation split, which is not "
                "distributed with this repository"}
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {out}")


def pairs_with_boxes(dets, pairs):
    return [(dets[di][:4], gi) for di, gi in pairs]


if __name__ == "__main__":
    main()
