"""How much accuracy would a large classifier need to become deployable?

The external classifiers were evaluated through a shared export and runtime
path, and an audit of that path found a preprocessing fault affecting three of
the four (see experiments/external_baselines_audit/). Any corrected accuracy figure
therefore invites a follow-up question: if the external baselines are more
accurate than reported, does the deployment ranking still demote them?

Arguing over the specific corrected number is the weak form of that answer.
The strong form is a bound. This script replaces a classifier's behavioral
terms with a *perfect* score -- 100% detected accuracy and 100% end-to-end
accuracy, an upper bound no correction can exceed -- and recomputes the
ranking over all 88 configurations. If a configuration remains outside the
deployable tier even when handed perfect accuracy, then its ranking is
determined entirely by its memory and latency envelope, and no accuracy
correction, ours or anyone's, can change the conclusion.

The same sweep is run under the published weights and under the equal
memory-split weights, so the bound does not depend on the disputed per-face
memory argument either.

Usage:
    python experiments/edus_recompute/accuracy_headroom.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from recompute_edus import VARIANTS, config_label, score

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "validation"

PLATFORMS = ("desktop", "jetson")
BEHAVIORAL = ["cls_accuracy", "e2e_accuracy"]

# Weightings under which the bound is evaluated. `equal_memory_split` drops the
# per-face memory-scaling argument entirely.
WEIGHT_SETS = ("calibrated", "equal_memory_split")

# Deployment tiers, for reporting where a configuration lands.
TOP_TIER = 10


def headroom_for(df: pd.DataFrame, classifier: str, weights: dict) -> dict:
    """Rank the classifier's best configuration when given perfect accuracy."""
    perfect = df.copy()
    mask = perfect.classifier == classifier
    perfect.loc[mask, BEHAVIORAL] = 1.0

    s_before = score(df, weights)
    s_after = score(perfect, weights)
    r_before = s_before.rank(ascending=False, method="min")
    r_after = s_after.rank(ascending=False, method="min")

    best_before = int(r_before[mask].min())
    best_after = int(r_after[mask].min())
    return {
        "classifier": classifier,
        "size_mb": float(df.loc[mask, "Size_classifier"].iloc[0]),
        "observed_e2e_acc_max": round(float(df.loc[mask, "e2e_accuracy"].max()), 4),
        "best_rank_observed": best_before,
        "best_rank_with_perfect_accuracy": best_after,
        "rank_gain": best_before - best_after,
        "reaches_top_tier_with_perfect_accuracy": best_after <= TOP_TIER,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, report = [], {}

    for platform in PLATFORMS:
        df = pd.read_csv(RESULTS_DIR / f"e2e_summary_{platform}_with_EDUS.csv")
        classifiers = sorted(df.classifier.unique())
        report[platform] = {}

        for wname in WEIGHT_SETS:
            weights = VARIANTS[wname]
            entries = [headroom_for(df, c, weights) for c in classifiers]
            report[platform][wname] = entries
            for e in entries:
                rows.append({"platform": platform, "weights": wname, **e})

            print(f"\n=== {platform} / {wname} ===")
            print(f"{'classifier':16s} {'size MB':>9s} {'obs acc':>8s} "
                  f"{'rank':>5s} {'rank@100%':>10s} {'top-10?':>8s}")
            for e in sorted(entries, key=lambda x: -x["size_mb"]):
                print(f"{e['classifier']:16s} {e['size_mb']:9.3f} "
                      f"{e['observed_e2e_acc_max']:8.3f} "
                      f"{e['best_rank_observed']:5d} "
                      f"{e['best_rank_with_perfect_accuracy']:10d} "
                      f"{str(e['reaches_top_tier_with_perfect_accuracy']):>8s}")

        # The headline case: the largest external classifier, handed a perfect
        # score, under weights that make no per-face memory assumption.
        big = max(classifiers, key=lambda c: df.loc[df.classifier == c, "Size_classifier"].iloc[0])
        e = headroom_for(df, big, VARIANTS["equal_memory_split"])
        print(f"\n  bound: {big} ({e['size_mb']:.1f} MB) at 100% accuracy still ranks "
              f"{e['best_rank_with_perfect_accuracy']}/88 on {platform}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "accuracy_headroom.csv", index=False)
    (OUT_DIR / "accuracy_headroom.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote 2 files to {OUT_DIR}")


if __name__ == "__main__":
    main()
