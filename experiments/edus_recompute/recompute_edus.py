"""EDUS weight-sensitivity recompute.

Reruns the Edge Deployment Unified Score over the existing 88-configuration
sweep under alternative weight vectors, to test whether the paper's ranking
conclusions depend on the specific operational weights reported in Section V-E.

Three questions are answered:

  1. Can the published EDUS column be reproduced exactly from the raw metric
     columns?  (Establishes that every variant below differs from the paper
     only in its weights, not in its implementation.)
  2. Do the rankings survive bringing the classifier/detector size weights
     inside the derivation band stated in Section V-E?
  3. Do the rankings survive dropping the per-face memory-scaling assumption
     altogether -- including reassigning the per-face factor from memory to
     latency, where it actually belongs?

Usage:
    python experiments/edus_recompute/recompute_edus.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "validation"

PLATFORMS = ("desktop", "jetson")

# Metric columns in the sweep CSVs, and their polarity.
# Negative = smaller is better and the term is inverted before weighting.
NEGATIVE = {"Size_detector", "Size_classifier", "det_lat_ms", "cls_lat_ms"}

# Average faces per image on the in-the-wild stress set (paper, Section V-E).
N_BAR = 6.6

# Total budgets held fixed across every variant so that only the *split*
# changes, never the overall memory / latency / correctness balance:
#   memory 0.50, detector latency 0.35, behavioral correctness 0.15
W_MEM, W_LAT, W_BEHAVIOR = 0.50, 0.35, 0.15
BEHAVIOR = {"mAP50-95": 0.05, "cls_accuracy": 0.05, "e2e_accuracy": 0.05}


def _derived_size_weights(n_bar: float = N_BAR) -> tuple[float, float]:
    """Equation 4: split the memory budget by per-face execution count."""
    w_cls = n_bar / (n_bar + 1.0) * W_MEM
    w_det = 1.0 / (n_bar + 1.0) * W_MEM
    return w_cls, w_det


_W_CLS_DERIVED, _W_DET_DERIVED = _derived_size_weights()

VARIANTS: dict[str, dict[str, float]] = {
    # As published in Equation 5.
    "calibrated": {
        "Size_detector": 0.10,
        "Size_classifier": 0.40,
        "det_lat_ms": 0.35,
        **BEHAVIOR,
    },
    # The published 0.40 / 0.10 fall outside the derivation ranges stated in
    # the same section (0.417-0.444 and 0.056-0.083). This variant moves them
    # back inside that band.
    "in_band": {
        "Size_detector": 0.08,
        "Size_classifier": 0.42,
        "det_lat_ms": 0.35,
        **BEHAVIOR,
    },
    # Equation 4 evaluated exactly at n_bar = 6.6, with no rounding.
    "exact_derivation": {
        "Size_detector": round(_W_DET_DERIVED, 6),
        "Size_classifier": round(_W_CLS_DERIVED, 6),
        "det_lat_ms": 0.35,
        **BEHAVIOR,
    },
    # Model weights are loaded once under sequential batch-1 inference, so
    # classifier memory does NOT scale with the number of faces. This drops the
    # per-face memory argument entirely and splits the memory budget evenly.
    "equal_memory_split": {
        "Size_detector": 0.25,
        "Size_classifier": 0.25,
        "det_lat_ms": 0.35,
        **BEHAVIOR,
    },
    # The correction taken to its conclusion: memory does not scale per face,
    # but *latency* does. Memory is split evenly and the per-face factor is
    # moved onto the latency budget, which is split between the detector (once
    # per frame) and the classifier (once per face).
    "latency_scaled": {
        "Size_detector": 0.25,
        "Size_classifier": 0.25,
        "det_lat_ms": round(1.0 / (N_BAR + 1.0) * W_LAT, 6),
        "cls_lat_ms": round(N_BAR / (N_BAR + 1.0) * W_LAT, 6),
        **BEHAVIOR,
    },
}


def normalize(series: pd.Series, negative: bool) -> pd.Series:
    """Min-max normalize a metric column onto [0, 1], higher = better.

    This reproduces the published EDUS column exactly. Note that per-term
    min-max of a z-score is identical to per-term min-max of the raw metric,
    so this is the same transform the paper describes as a z-score
    normalization, up to a per-term affine rescaling.
    """
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    unit = (series - series.min()) / span
    return 1.0 - unit if negative else unit


def score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"weights must sum to 1.0, got {total:.6f}")
    return sum(
        w * normalize(df[col], col in NEGATIVE) for col, w in weights.items()
    )


def zscore_variant(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Literal weighted z-score aggregation, as Equation 5 is written.

    The published implementation uses per-term min-max normalization instead.
    Unlike the min-max form this is not bounded to [0, 1], so only the induced
    ranking is compared.
    """
    out = 0.0
    for col, w in weights.items():
        z = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        out = out + w * (-z if col in NEGATIVE else z)
    return out


def config_label(row: pd.Series) -> str:
    return f"{row.detector} + {row.classifier}"


def analyze(df: pd.DataFrame, platform: str) -> tuple[dict, pd.DataFrame]:
    baseline = score(df, VARIANTS["calibrated"])
    published = df["EDUS"]
    repro_err = float(np.abs(baseline - published).max())

    base_order = baseline.rank(ascending=False, method="min")
    base_top10 = set(baseline.nlargest(10).index)
    base_rank1 = int(baseline.idxmax())

    rows, summary = [], []
    for name, weights in VARIANTS.items():
        s = score(df, weights)
        order = s.rank(ascending=False, method="min")
        top10 = set(s.nlargest(10).index)
        rank1 = int(s.idxmax())
        rho = float(spearmanr(base_order, order).statistic)
        summary.append(
            {
                "platform": platform,
                "variant": name,
                "w_det_size": weights["Size_detector"],
                "w_cls_size": weights["Size_classifier"],
                "w_det_lat": weights["det_lat_ms"],
                "w_cls_lat": weights.get("cls_lat_ms", 0.0),
                "rank1": config_label(df.loc[rank1]),
                "rank1_matches_calibrated": rank1 == base_rank1,
                "top10_overlap": len(top10 & base_top10),
                "spearman_rho": round(rho, 5),
                "posterv2_best_rank": int(
                    order[df.classifier == "POSTERv2"].min()
                ),
                "lanmsff_best_rank": int(order[df.classifier == "LANMSFF"].min()),
                "fp16_det_in_top10": int(
                    df.loc[list(top10), "detector"].str.contains("FP16").sum()
                ),
            }
        )
        for idx in s.nlargest(10).index:
            rows.append(
                {
                    "platform": platform,
                    "variant": name,
                    "rank": int(order[idx]),
                    "config": config_label(df.loc[idx]),
                    "score": round(float(s[idx]), 4),
                }
            )

    # Literal z-score form, ranking comparison only.
    z = zscore_variant(df, VARIANTS["calibrated"])
    z_order = z.rank(ascending=False, method="min")
    zinfo = {
        "rank1": config_label(df.loc[int(z.idxmax())]),
        "rank1_matches_calibrated": int(z.idxmax()) == base_rank1,
        "top10_overlap": len(set(z.nlargest(10).index) & base_top10),
        "spearman_rho": round(float(spearmanr(base_order, z_order).statistic), 5),
    }

    meta = {
        "platform": platform,
        "n_configs": int(len(df)),
        "reproduction_max_abs_error": repro_err,
        "reproduces_published_edus": repro_err < 1e-4,
        "calibrated_rank1": config_label(df.loc[base_rank1]),
        "zscore_form": zinfo,
        "variants": summary,
    }
    return meta, pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report, top_tables = {}, []

    for platform in PLATFORMS:
        df = pd.read_csv(RESULTS_DIR / f"e2e_summary_{platform}_with_EDUS.csv")
        meta, tops = analyze(df, platform)
        report[platform] = meta
        top_tables.append(tops)

        print(f"\n=== {platform} ({meta['n_configs']} configs) ===")
        print(
            f"reproduces published EDUS: {meta['reproduces_published_edus']} "
            f"(max abs err {meta['reproduction_max_abs_error']:.2e})"
        )
        print(pd.DataFrame(meta["variants"]).drop(columns=["platform"]).to_string(index=False))
        print(f"literal z-score form: {meta['zscore_form']}")

    # Cross-platform agreement of the top tier, per variant.
    desk = pd.read_csv(RESULTS_DIR / "e2e_summary_desktop_with_EDUS.csv")
    jets = pd.read_csv(RESULTS_DIR / "e2e_summary_jetson_with_EDUS.csv")
    cross = []
    for name, weights in VARIANTS.items():
        d_top = {config_label(desk.loc[i]) for i in score(desk, weights).nlargest(10).index}
        j_top = {config_label(jets.loc[i]) for i in score(jets, weights).nlargest(10).index}
        cross.append(
            {
                "variant": name,
                "desktop_jetson_top10_overlap": len(d_top & j_top),
            }
        )
    report["cross_platform"] = cross
    print("\n=== cross-platform top-10 agreement ===")
    print(pd.DataFrame(cross).to_string(index=False))

    pd.concat(top_tables).to_csv(OUT_DIR / "edus_variant_top10.csv", index=False)
    rows = [v for p in PLATFORMS for v in report[p]["variants"]]
    pd.DataFrame(rows).to_csv(OUT_DIR / "edus_variant_summary.csv", index=False)
    (OUT_DIR / "edus_recompute_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote 3 files to {OUT_DIR}")


if __name__ == "__main__":
    main()
