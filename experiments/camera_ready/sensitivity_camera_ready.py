"""Table VI refresh: sensitivity of the adopted EDUS profile (0.42/0.08).

Re-runs the single-weight and tier-ratio perturbations of the published
Table VI around the camera-ready operating point -- the in-band memory split
adopted in Eq. 5 -- on the camera-ready matrices (corrected external
accuracies). Also reports the exact-derivation split and the prior 0.40/0.10
operational weights as variants.

Usage:
    python experiments/camera_ready/sensitivity_camera_ready.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results" / "camera_ready"

NEG = {"Size_detector", "Size_classifier", "det_lat_ms"}
BASE = {"Size_detector": 0.08, "Size_classifier": 0.42, "det_lat_ms": 0.35,
        "mAP50-95": 0.05, "cls_accuracy": 0.05, "e2e_accuracy": 0.05}


def norm(s, neg):
    u = (s - s.min()) / (s.max() - s.min())
    return 1 - u if neg else u


def score(df, w):
    t = sum(w.values())
    return sum(v / t * norm(df[k], k in NEG) for k, v in w.items())


def main() -> None:
    rows = []
    for p in ("jetson", "desktop"):
        df = pd.read_csv(OUT / f"table8_camera_ready_{p}.csv")
        base = score(df, BASE)
        b10, b1 = set(base.nlargest(10).index), base.idxmax()

        def add(name, w):
            s = score(df, w)
            rows.append({
                "platform": p, "variant": name,
                "top10_overlap": len(b10 & set(s.nlargest(10).index)),
                "rank1_match": s.idxmax() == b1,
                "spearman_rho": round(float(spearmanr(base, s).statistic), 4),
            })

        for k in BASE:
            for f in (0.5, 1.5):
                w = dict(BASE); w[k] *= f
                add(f"{k} x{f}", w)
        for hw in (0.90, 0.85, 0.80, 0.75, 0.70):
            bc = 1 - hw
            add(f"tier {hw:.2f}/{bc:.2f}", {
                "Size_detector": 0.08 / 0.85 * hw,
                "Size_classifier": 0.42 / 0.85 * hw,
                "det_lat_ms": 0.35 / 0.85 * hw,
                "mAP50-95": bc / 3, "cls_accuracy": bc / 3,
                "e2e_accuracy": bc / 3,
            })
        add("exact 0.4342/0.0658",
            {**BASE, "Size_classifier": 0.434211, "Size_detector": 0.065789})
        add("prior 0.40/0.10",
            {**BASE, "Size_classifier": 0.40, "Size_detector": 0.10})

    out = OUT / "edus_sensitivity_camera_ready.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    for r in rows:
        if not r["rank1_match"]:
            print("RANK-1 CHANGED:", r)


if __name__ == "__main__":
    main()
