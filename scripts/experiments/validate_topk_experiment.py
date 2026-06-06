from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RELEASE_ROOT = Path(os.environ.get("CATSOLV_RELEASE_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("CATSOLV_TOPK_OUTPUT_ROOT", RELEASE_ROOT / "results" / "topk_matched_feature_count"))
OUT = ROOT / "outputs"
PHYS_EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PHYS_EXPERIMENT_ROOT))

from run_physchem_cross_20seed import (  # noqa: E402
    ORIGINAL_V1_OOD_SOLVENTS,
    SEEDS,
    build_model_frames,
    build_test_manifest,
    feature_cols,
    idx_from_row_ids,
)


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    failures: list[str] = []
    frames, _ = build_model_frames()
    v1 = frames["v1"]
    v2 = frames["v2"]
    full = frames["v2_physchem_class_cross"]

    selected = pd.read_csv(OUT / "selected_features_by_seed.csv")
    overall = pd.read_csv(OUT / "overall_seed_metrics.csv")
    preds = pd.read_csv(OUT / "ood_predictions_all_seeds.csv", usecols=["Common_RowID", "model", "seed", "SMILES_Solvent"])
    rankings = pd.read_csv(OUT / "validation_shap_rankings_all_seeds.csv")

    check(len(v1) == len(v2) == len(full) == 103262, "Unexpected row count in model frames.", failures)
    check(v1["Common_RowID"].is_unique, "v1 Common_RowID is not unique.", failures)
    check(v2["Common_RowID"].is_unique, "v2 Common_RowID is not unique.", failures)
    check(full["Common_RowID"].is_unique, "full Common_RowID is not unique.", failures)
    check(
        v1["Common_RowID"].tolist() == v2["Common_RowID"].tolist() == full["Common_RowID"].tolist(),
        "Model frames are not aligned in identical Common_RowID order.",
        failures,
    )
    check(
        np.allclose(v1["LogS"].to_numpy(), v2["LogS"].to_numpy())
        and np.allclose(v1["LogS"].to_numpy(), full["LogS"].to_numpy()),
        "LogS targets differ across frames.",
        failures,
    )
    check(
        (v1[["SMILES_Solute", "SMILES_Solvent"]].astype(str).to_numpy()
         == v2[["SMILES_Solute", "SMILES_Solvent"]].astype(str).to_numpy()).all(),
        "v1/v2 SMILES columns are not row-aligned.",
        failures,
    )

    _, remain_row_ids, test_row_ids = build_test_manifest(v1)
    test_set = set(test_row_ids)
    remain_set = set(remain_row_ids)
    check(len(test_set) == 5424, "Unexpected fixed OOD test size.", failures)
    check(test_set.isdisjoint(remain_set), "OOD test rows overlap with training pool.", failures)
    check(
        set(v1.loc[v1["Common_RowID"].isin(test_set), "SMILES_Solvent"]) == set(ORIGINAL_V1_OOD_SOLVENTS),
        "OOD test solvents do not match the original-five protocol.",
        failures,
    )

    full_features = set(feature_cols(full))
    v1_features = set(feature_cols(v1))
    v2_features = set(feature_cols(v2))
    selected_features = set(selected["feature"])
    ranking_features = set(rankings["feature"])
    check(selected_features.issubset(full_features), "Some selected top-k features are missing from full98 frame.", failures)
    check(ranking_features == full_features, "Validation SHAP ranking does not cover exactly the full98 feature set.", failures)
    check(len(full_features) == 98, "Full class-cross feature count is not 98.", failures)
    check(len(v1_features) == 33, "v1 feature count is not 33.", failures)
    check(len(v2_features) == 47, "v2 feature count is not 47.", failures)

    expected_models = {
        "v1",
        "v2",
        "class_cross_top33_valshap",
        "class_cross_top47_valshap",
        "class_cross_top60_valshap",
        "class_cross_full98",
    }
    expected_seeds = set(SEEDS)
    check(set(overall["model"]) == expected_models, "overall metrics model set mismatch.", failures)
    check(set(overall["seed"]) == expected_seeds, "overall metrics seed set mismatch.", failures)
    check(len(overall) == 120, "overall metrics should have 20 seeds x 6 models = 120 rows.", failures)

    expected_selected_rows = len(SEEDS) * (33 + 47 + 60)
    check(len(selected) == expected_selected_rows, "selected_features_by_seed row count mismatch.", failures)
    selected_counts = selected.groupby(["seed", "top_k"]).size().unstack()
    check((selected_counts[33] == 33).all(), "Not every seed has 33 top-33 features.", failures)
    check((selected_counts[47] == 47).all(), "Not every seed has 47 top-47 features.", failures)
    check((selected_counts[60] == 60).all(), "Not every seed has 60 top-60 features.", failures)

    for seed in SEEDS:
        train_ids, val_ids = train_test_split(remain_row_ids, test_size=0.1, random_state=seed)
        train_set = set(train_ids)
        val_set = set(val_ids)
        check(train_set.isdisjoint(test_set), f"Seed {seed}: train rows overlap OOD test rows.", failures)
        check(val_set.isdisjoint(test_set), f"Seed {seed}: validation rows overlap OOD test rows.", failures)
        check(train_set.isdisjoint(val_set), f"Seed {seed}: train rows overlap validation rows.", failures)
        check(len(idx_from_row_ids(full, sorted(train_ids))) == 88054, f"Seed {seed}: unexpected train size.", failures)
        check(len(idx_from_row_ids(full, sorted(val_ids))) == 9784, f"Seed {seed}: unexpected validation size.", failures)

    pred_group = preds.groupby(["seed", "model"])["Common_RowID"].nunique()
    check((pred_group == 5424).all(), "Every seed/model should predict exactly 5424 unique OOD rows.", failures)
    pred_solvents = set(preds["SMILES_Solvent"])
    check(pred_solvents == set(ORIGINAL_V1_OOD_SOLVENTS), "Prediction file contains unexpected OOD solvents.", failures)

    numeric_na = {
        "v1_na": int(v1[feature_cols(v1)].isna().sum().sum()),
        "v2_na": int(v2[feature_cols(v2)].isna().sum().sum()),
        "full98_na": int(full[feature_cols(full)].isna().sum().sum()),
    }

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "row_counts": {
            "v1": int(len(v1)),
            "v2": int(len(v2)),
            "full98": int(len(full)),
            "ood_test_rows": int(len(test_set)),
            "train_rows_per_seed": 88054,
            "validation_rows_per_seed": 9784,
        },
        "feature_counts": {
            "v1": int(len(v1_features)),
            "v2": int(len(v2_features)),
            "full98": int(len(full_features)),
            "selected_feature_rows": int(len(selected)),
            "overall_metric_rows": int(len(overall)),
        },
        "numeric_missing_values": numeric_na,
        "ood_solvents": ORIGINAL_V1_OOD_SOLVENTS,
    }
    (OUT / "sanity_check_topk_experiment.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
