"""Scene-level bootstrap of the stress-set ranking.

The 20-scene, 132-face stress set is small: one additional correct
prediction moves end-to-end accuracy by ~0.76 points, so the stability of
the 88-configuration ranking under sampling noise is a fair question. This
experiment answers it directly, and is the basis for Sec. V-F of the paper.

Method: the per-scene sweep (run_e2e_perscene.py, with each external
classifier under its determined normalization contract and the as-published
path for everything else) records
ground-truth, matched, and correct counts per (detector, classifier,
scene). We resample the 20 scenes with replacement B times and recompute
end-to-end and detected accuracy for all 88 configurations per resample.
Sizes, latencies, and mAP are scene-independent and stay fixed.

Two analyses are reported:

  raw       -- accuracies taken directly from the resampled harness
               measurements; one fully consistent basis, but its reference
               ranking is the harness's own, not the paper's table.
  centered  -- each configuration's resampled accuracy is expressed as a
               delta from its full-sample harness value and added to the
               value the paper's table prints for that configuration
               (clipped to [0, 1]). This injects the measured scene-level
               sampling variability around the printed operating point, so
               the stability numbers refer to the table the paper shows.
               The centered analysis is the one quoted in the paper.

Usage:
    python experiments/camera_ready/bootstrap_stress_set.py \
        --per-scene results/camera_ready/perscene_aspub_per_scene.csv \
                    results/camera_ready/perscene_corrected_per_scene.csv \
        --out results/camera_ready/bootstrap_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

NEG = {"Size_detector", "Size_classifier", "det_lat_ms"}
WEIGHTS = {"Size_detector": 0.08, "Size_classifier": 0.42, "det_lat_ms": 0.35,
           "mAP50-95": 0.05, "cls_accuracy": 0.05, "e2e_accuracy": 0.05}
FP16_DETS = {"YOLO-FP16-P10", "YOLO-FP16-P15"}
B_DEFAULT = 10_000


def normalize(col: np.ndarray, negative: bool) -> np.ndarray:
    span = col.max() - col.min()
    if span == 0:
        return np.full_like(col, 0.5)
    unit = (col - col.min()) / span
    return 1.0 - unit if negative else unit


def edus(df: pd.DataFrame, e2e: np.ndarray, det: np.ndarray) -> np.ndarray:
    parts = [
        WEIGHTS["Size_detector"] * normalize(df.Size_detector.values, True),
        WEIGHTS["Size_classifier"] * normalize(df.Size_classifier.values, True),
        WEIGHTS["det_lat_ms"] * normalize(df.det_lat_ms.values, True),
        WEIGHTS["mAP50-95"] * normalize(df["mAP50-95"].values, False),
        WEIGHTS["cls_accuracy"] * normalize(det, False),
        WEIGHTS["e2e_accuracy"] * normalize(e2e, False),
    ]
    return np.sum(parts, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=B_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS / "camera_ready" / "bootstrap_report.json"))
    args = ap.parse_args()

    ps = pd.concat([pd.read_csv(p) for p in args.per_scene], ignore_index=True)
    dup = ps.duplicated(["detector", "classifier", "scene"]).sum()
    assert dup == 0, f"{dup} duplicate per-scene rows across input files"
    configs = ps[["detector", "classifier"]].drop_duplicates()
    n_cfg = len(configs)
    scenes = sorted(ps.scene.unique())
    n_scenes = len(scenes)
    print(f"{n_cfg} configurations, {n_scenes} scenes")

    # counts[c, s, :] = (correct, matched, gt)
    key = ps.set_index(["detector", "classifier", "scene"])
    cfg_index = pd.MultiIndex.from_frame(configs)
    counts = np.zeros((n_cfg, n_scenes, 3), dtype=np.int64)
    for ci, (d, c) in enumerate(cfg_index):
        for si, s in enumerate(scenes):
            row = key.loc[(d, c, s)]
            counts[ci, si] = (row.n_correct, row.n_matched, row.n_gt)

    # Full-sample harness accuracies (centering point of the deltas).
    tot_full = counts.sum(axis=1)
    e2e_h = tot_full[:, 0] / tot_full[:, 2]
    det_h = np.divide(tot_full[:, 0], tot_full[:, 1],
                      where=tot_full[:, 1] > 0, out=np.zeros(n_cfg))

    posterv2_idx = [i for i, (d, c) in enumerate(cfg_index) if c == "POSTERv2"]
    pcnn_idx = [i for i, (d, c) in enumerate(cfg_index) if c == "PCNN"]

    rng = np.random.default_rng(args.seed)
    report = {}
    for platform in ("jetson", "desktop"):
        mat = pd.read_csv(RESULTS / f"e2e_summary_{platform}_with_EDUS.csv")
        mat = mat.set_index(["detector", "classifier"]).loc[cfg_index].reset_index()
        printed = pd.read_csv(
            RESULTS / "camera_ready" / f"table8_camera_ready_{platform}.csv"
        ).set_index(["detector", "classifier"]).loc[cfg_index]
        e2e_p = printed.e2e_accuracy.values
        det_p = printed.cls_accuracy.values

        for mode in ("raw", "centered"):
            if mode == "raw":
                base_e2e, base_det = e2e_h, det_h
            else:
                base_e2e, base_det = e2e_p, det_p
            s_full = edus(mat, base_e2e, base_det)
            order_full = np.argsort(-s_full)
            rank1_full = order_full[0]
            top10_full = set(order_full[:10])

            rng_m = np.random.default_rng(args.seed)  # same draws per mode
            hit_rank1 = hit_fp16 = ext_ok = 0
            overlaps, rank1_acc = [], []
            for _ in range(args.boot):
                pick = rng_m.integers(0, n_scenes, n_scenes)
                tot = counts[:, pick, :].sum(axis=1)
                e2e_b = tot[:, 0] / tot[:, 2]
                det_b = np.divide(tot[:, 0], tot[:, 1], where=tot[:, 1] > 0,
                                  out=np.zeros(n_cfg))
                if mode == "centered":
                    e2e_b = np.clip(e2e_p + (e2e_b - e2e_h), 0.0, 1.0)
                    det_b = np.clip(det_p + (det_b - det_h), 0.0, 1.0)
                s = edus(mat, e2e_b, det_b)
                order = np.argsort(-s)
                hit_rank1 += order[0] == rank1_full
                hit_fp16 += cfg_index[order[0]][0] in FP16_DETS
                overlaps.append(len(top10_full & set(order[:10])))
                rank1_acc.append(e2e_b[rank1_full])
                ranks = np.empty(n_cfg, dtype=int)
                ranks[order] = np.arange(1, n_cfg + 1)
                ext_ok += (ranks[posterv2_idx].min() > 60) and (
                    ranks[pcnn_idx].min() > 60)

            d0, c0 = cfg_index[rank1_full]
            report[f"{platform}_{mode}"] = {
                "reference_rank1": f"{d0} + {c0}",
                "p_rank1_stable": hit_rank1 / args.boot,
                "p_rank1_fp16_detector": hit_fp16 / args.boot,
                "top10_overlap_mean": float(np.mean(overlaps)),
                "top10_overlap_p5": float(np.percentile(overlaps, 5)),
                "rank1_e2e_acc_ci95": [float(np.percentile(rank1_acc, 2.5)),
                                       float(np.percentile(rank1_acc, 97.5))],
                "p_posterv2_and_pcnn_below_rank60": ext_ok / args.boot,
                "n_bootstrap": args.boot,
            }
            print(f"{platform}_{mode}",
                  json.dumps(report[f"{platform}_{mode}"], indent=1))

    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
