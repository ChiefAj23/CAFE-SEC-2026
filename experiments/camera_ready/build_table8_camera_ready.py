"""Camera-ready Table XII: recompute the 88-configuration EDUS ranking.

Two constructions relative to the originally submitted table:

  1. The Eq. 5 memory weights use the adopted in-band split, 0.42 (classifier
     size) / 0.08 (detector size), which lies inside the uncertainty band of
     the Eq. 4 derivation. All other weights are unchanged: detector latency
     0.35, and mAP, detected accuracy, and end-to-end accuracy 0.05 each.
  2. The stress-set accuracy columns of the three external classifiers that
     require ImageNet normalization (ImpMobNetv3, POSTERv2, PCNN) are taken
     under their determined preprocessing contract, as established in
     experiments/external_baselines_audit/. Sizes, latencies, and mAP stay as
     reported per platform; only e2e_accuracy and cls_accuracy (detected
     accuracy) are substituted, since a model's predictions on a fixed image
     set do not depend on the timing platform. LANMSFF keeps its reported
     values, its published grayscale 64x64 path having been confirmed as the
     contract it was trained under; substituting harness-measured LANMSFF
     numbers would import harness differences (lambda calibration, coarse
     label mapping) that are not a preprocessing difference. The 56 in-house
     YOLOv12n rows are untouched for the same reason.

Source for the substituted accuracies: results/validation/
e2e_sweep_corrected.csv, the complete 88-configuration sweep under each
model's determined normalization contract ("corrected" in these filenames
refers to that contract). One file, all 24 rows: 3 models x 8 detectors.

Two caveats are recorded deliberately. The substituted external accuracies
sit on the harness basis of experiments/e2e_harness/, which reproduces the
reported YOLO-family accuracies to a mean absolute error of 0.031. And the
INT8 detector in that sweep was calibrated on the stress scenes rather than
the WIDER FACE validation split, as recorded in the run metadata.

Usage:
    python experiments/camera_ready/build_table8_camera_ready.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
REB = RESULTS / "validation"
OUT = RESULTS / "camera_ready"

CORRECTED = ("MobileNetV3", "POSTERv2", "PCNN")
EXTERNALS = CORRECTED + ("LANMSFF",)
NEGATIVE = {"Size_detector", "Size_classifier", "det_lat_ms"}

WEIGHTS = {
    "Size_detector": 0.08,
    "Size_classifier": 0.42,
    "det_lat_ms": 0.35,
    "mAP50-95": 0.05,
    "cls_accuracy": 0.05,
    "e2e_accuracy": 0.05,
}


def normalize(series: pd.Series, negative: bool) -> pd.Series:
    span = series.max() - series.min()
    unit = (series - series.min()) / span
    return 1.0 - unit if negative else unit


def corrected_accuracies() -> pd.DataFrame:
    """One (detector, classifier) -> (e2e_accuracy, cls_accuracy) map."""
    df = pd.read_csv(REB / "e2e_sweep_corrected.csv")
    acc = df[df.classifier.isin(CORRECTED)][
        ["detector", "classifier", "e2e_accuracy", "cls_accuracy"]
    ]
    dup = acc.duplicated(["detector", "classifier"]).sum()
    assert dup == 0, f"{dup} duplicate (detector, classifier) rows"
    assert len(acc) == 24, f"expected 24 corrected rows, got {len(acc)}"
    return acc.set_index(["detector", "classifier"])


def build(platform: str, acc: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"e2e_summary_{platform}_with_EDUS.csv")
    assert len(df) == 88

    df = df.set_index(["detector", "classifier"])
    n_sub = 0
    for key in acc.index:
        assert key in df.index, f"{key} missing from {platform} matrix"
        df.loc[key, ["e2e_accuracy", "cls_accuracy"]] = acc.loc[
            key, ["e2e_accuracy", "cls_accuracy"]
        ]
        n_sub += 1
    assert n_sub == 24
    df = df.reset_index()

    df["EDUS_cr"] = sum(
        w * normalize(df[c], c in NEGATIVE) for c, w in WEIGHTS.items()
    )
    df["Rank_cr"] = df["EDUS_cr"].rank(ascending=False, method="min").astype(int)
    df["Comb_Size"] = df["Size_detector"] + df["Size_classifier"]
    return df.sort_values("Rank_cr")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    acc = corrected_accuracies()

    frames = {p: build(p, acc) for p in ("jetson", "desktop")}
    desk_rank = frames["desktop"].set_index(["detector", "classifier"])["Rank_cr"]
    jet = frames["jetson"].set_index(["detector", "classifier"])
    jet["Desk_Rank_cr"] = desk_rank
    frames["jetson"] = jet.reset_index()

    report = {}
    for p, df in frames.items():
        df.to_csv(OUT / f"table8_camera_ready_{p}.csv", index=False)
        ext_ranks = {
            c: {
                "best": int(df[df.classifier == c]["Rank_cr"].min()),
                "worst_yolo_det": int(
                    df[(df.classifier == c)
                       & ~df.detector.str.startswith("RT-DETR")]["Rank_cr"].max()
                ),
            }
            for c in EXTERNALS
        }
        top10 = df.nsmallest(10, "Rank_cr")
        report[p] = {
            "rank1": f"{df.iloc[0].detector} + {df.iloc[0].classifier}",
            "rank1_edus": round(float(df.iloc[0].EDUS_cr), 4),
            "external_rank_ranges_over_yolo_dets": ext_ranks,
            "top10": [
                f"{r.detector} + {r.classifier} (EDUS {r.EDUS_cr:.4f})"
                for r in top10.itertuples()
            ],
        }

    jet_top10 = set(
        frames["jetson"].nsmallest(10, "Rank_cr")
        .set_index(["detector", "classifier"]).index
    )
    report["jetson_top10_max_desk_rank"] = int(
        frames["jetson"]
        .set_index(["detector", "classifier"])
        .loc[list(jet_top10), "Desk_Rank_cr"].max()
    )
    desk_top3 = frames["desktop"].nsmallest(3, "Rank_cr").set_index(
        ["detector", "classifier"]).index
    report["desktop_top3_max_jetson_rank"] = int(
        frames["jetson"].set_index(["detector", "classifier"])
        .loc[list(desk_top3), "Rank_cr"].max()
    )

    (OUT / "table8_camera_ready_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
