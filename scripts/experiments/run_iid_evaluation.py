"""
IID (random-split) evaluation to complement the OOD benchmark.

Evaluates all key models on IID random splits, 20 seeds, using the EXACT
same training pool (non-OOD rows), hyperparameters, and seed values as the
OOD experiment. The only difference: the test set is a random holdout from
the training pool, not the 5 held-out OOD solvents.

Uses pre-computed feature CSVs (no RDKit needed).
"""
from __future__ import annotations

import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RELEASE_ROOT = Path(os.environ.get("CATSOLV_RELEASE_ROOT", Path(__file__).resolve().parents[2]))
THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("CATSOLV_IID_OUTPUT_ROOT", RELEASE_ROOT / "results" / "iid_evaluation")) / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config from the OOD protocol ──────────────────────────────────────────
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
OOD_SOLVENTS = ["CS(C)=O", "C1CCCCC1", "OCCO", "CN1CCCC1=O", "CCCOC(C)=O"]
TRAIN_POOL_ROWS = 97838
OOD_TEST_ROWS = 5424

XGB_PARAMS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=30,
)

# Input CSVs (same as OOD experiment)
RAW_DATA_ROOT = Path(os.environ.get("CATSOLV_RAW_DATA_ROOT", RELEASE_ROOT / "data" / "raw"))
V1_INPUT = RAW_DATA_ROOT / "BigSolDB_features_v1_common.csv"
V2_INPUT = RAW_DATA_ROOT / "BigSolDB_features_v2_common.csv"
V3_INPUT = RAW_DATA_ROOT / "BigSolDB_features_v3_common.csv"

# Optional full OOD prediction table for side-by-side IID/OOD printing.
PHYSCHEM_MERGED = Path(
    os.environ.get(
        "CATSOLV_OOD_PREDICTIONS",
        RELEASE_ROOT / "data" / "raw" / "ood_predictions_all_seeds.csv",
    )
)

# ── Feature set definitions (from protocol) ───────────────────────────────
FEATURE_SETS = {
    "2D-Min": [
        "Solute_MolWt","Solute_HeavyAtomCount","Solute_NumRotatableBonds",
        "Solute_RingCount","Solute_NumAromaticRings","Solute_BertzCT",
        "Solute_BalabanJ","Solute_MolLogP","Solute_TPSA","Solute_NumHDonors",
        "Solute_NumHAcceptors","Solute_LabuteASA",
        "Solvent_MolWt","Solvent_HeavyAtomCount","Solvent_NumRotatableBonds",
        "Solvent_RingCount","Solvent_NumAromaticRings","Solvent_BertzCT",
        "Solvent_BalabanJ","Solvent_MolLogP","Solvent_TPSA","Solvent_NumHDonors",
        "Solvent_NumHAcceptors","Solvent_LabuteASA",
        "Temperature_K",
    ],
    "2D-Elec": [
        "Solute_MolWt","Solute_HeavyAtomCount","Solute_NumRotatableBonds",
        "Solute_RingCount","Solute_NumAromaticRings","Solute_BertzCT",
        "Solute_BalabanJ","Solute_MolLogP","Solute_TPSA","Solute_NumHDonors",
        "Solute_NumHAcceptors","Solute_LabuteASA",
        "Solute_MaxPartialCharge","Solute_MinPartialCharge",
        "Solute_MaxAbsPartialCharge","Solute_MinAbsPartialCharge",
        "Solvent_MolWt","Solvent_HeavyAtomCount","Solvent_NumRotatableBonds",
        "Solvent_RingCount","Solvent_NumAromaticRings","Solvent_BertzCT",
        "Solvent_BalabanJ","Solvent_MolLogP","Solvent_TPSA","Solvent_NumHDonors",
        "Solvent_NumHAcceptors","Solvent_LabuteASA",
        "Solvent_MaxPartialCharge","Solvent_MinPartialCharge",
        "Solvent_MaxAbsPartialCharge","Solvent_MinAbsPartialCharge",
        "Temperature_K",
    ],
    "Sigma-noElec": [
        "Solute_MolWt","Solute_HeavyAtomCount","Solute_NumRotatableBonds",
        "Solute_RingCount","Solute_NumAromaticRings","Solute_BertzCT",
        "Solute_BalabanJ","Solute_MolLogP","Solute_TPSA","Solute_NumHDonors",
        "Solute_NumHAcceptors","Solute_LabuteASA",
        "Solvent_MolWt","Solvent_HeavyAtomCount","Solvent_NumRotatableBonds",
        "Solvent_RingCount","Solvent_NumAromaticRings","Solvent_BertzCT",
        "Solvent_BalabanJ","Solvent_MolLogP","Solvent_TPSA","Solvent_NumHDonors",
        "Solvent_NumHAcceptors","Solvent_LabuteASA",
        "Temperature_K",
        "Solute_sig2","Solute_sig3","Solute_M_hb_acc","Solute_M_hb_don",
        "Solute_M_nonpolar","Solute_pos_area_frac","Solute_neg_area_frac",
        "Solvent_sig2","Solvent_sig3","Solvent_M_hb_acc","Solvent_M_hb_don",
        "Solvent_M_nonpolar","Solvent_pos_area_frac","Solvent_neg_area_frac",
    ],
    "Sigma-Solv": [
        "Solute_MolWt","Solute_HeavyAtomCount","Solute_NumRotatableBonds",
        "Solute_RingCount","Solute_NumAromaticRings","Solute_BertzCT",
        "Solute_BalabanJ","Solute_MolLogP","Solute_TPSA","Solute_NumHDonors",
        "Solute_NumHAcceptors","Solute_LabuteASA",
        "Solute_MaxPartialCharge","Solute_MinPartialCharge",
        "Solute_MaxAbsPartialCharge","Solute_MinAbsPartialCharge",
        "Solvent_MolWt","Solvent_HeavyAtomCount","Solvent_NumRotatableBonds",
        "Solvent_RingCount","Solvent_NumAromaticRings","Solvent_BertzCT",
        "Solvent_BalabanJ","Solvent_MolLogP","Solvent_TPSA","Solvent_NumHDonors",
        "Solvent_NumHAcceptors","Solvent_LabuteASA",
        "Solvent_MaxPartialCharge","Solvent_MinPartialCharge",
        "Solvent_MaxAbsPartialCharge","Solvent_MinAbsPartialCharge",
        "Temperature_K",
        "Solute_sig2","Solute_sig3","Solute_M_hb_acc","Solute_M_hb_don",
        "Solute_M_nonpolar","Solute_pos_area_frac","Solute_neg_area_frac",
        "Solvent_sig2","Solvent_sig3","Solvent_M_hb_acc","Solvent_M_hb_don",
        "Solvent_M_nonpolar","Solvent_pos_area_frac","Solvent_neg_area_frac",
    ],
}

MODEL_ORDER = ["2D-Min", "2D-Elec", "Sigma-noElec", "Sigma-Solv"]


def load_data_frames():
    """Load feature CSVs and split into train_pool (in-dist) and OOD."""
    v1 = pd.read_csv(V1_INPUT)
    v2 = pd.read_csv(V2_INPUT)
    v3 = pd.read_csv(V3_INPUT)

    ood_mask = v1["SMILES_Solvent"].isin(OOD_SOLVENTS)
    train_pool_mask = ~ood_mask

    # Verify row counts
    assert train_pool_mask.sum() == TRAIN_POOL_ROWS, \
        f"Expected {TRAIN_POOL_ROWS} train pool rows, got {train_pool_mask.sum()}"
    assert ood_mask.sum() == OOD_TEST_ROWS, \
        f"Expected {OOD_TEST_ROWS} OOD rows, got {ood_mask.sum()}"

    # Map model name -> DataFrame (training pool only, reset index)
    frames = {}
    # 2D-Min and 2D-Elec from v1 (2D-Min = v1 minus Gasteiger, 2D-Elec = v1)
    frames["2D-Min"] = v1[train_pool_mask].copy().reset_index(drop=True)
    frames["2D-Elec"] = v1[train_pool_mask].copy().reset_index(drop=True)
    # Sigma-noElec from v3 (v3 = v2 - Gasteiger)
    frames["Sigma-noElec"] = v3[train_pool_mask].copy().reset_index(drop=True)
    # Sigma-Solv from v2
    frames["Sigma-Solv"] = v2[train_pool_mask].copy().reset_index(drop=True)

    # Verify feature columns exist
    for model_name, cols in FEATURE_SETS.items():
        missing = [c for c in cols if c not in frames[model_name].columns]
        if missing:
            print(f"WARNING: {model_name} missing features: {missing}")

    return frames


def evaluate_model(df, features, seed, model_name):
    """Train on 80% IID split, evaluate on 20% IID holdout, with 10% val for early stopping."""
    X = df[features].to_numpy()
    y = df["LogS"].to_numpy()

    # 80/20 IID train/test split, then 10% of train for validation
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=seed
    )
    train_sub_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, random_state=seed
    )

    model = xgb.XGBRegressor(
        **XGB_PARAMS, random_state=seed, n_jobs=4,
        eval_metric="rmse",
    )
    model.fit(
        X[train_sub_idx], y[train_sub_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        verbose=False,
    )

    y_pred = model.predict(X[test_idx])
    return {
        "seed": seed,
        "model": model_name,
        "feature_count": len(features),
        "train_n": len(train_sub_idx),
        "val_n": len(val_idx),
        "test_n": len(test_idx),
        "R2": float(r2_score(y[test_idx], y_pred)),
        "MAE": float(mean_absolute_error(y[test_idx], y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y[test_idx], y_pred))),
    }


def aggregate(results_df):
    """Compute mean/std across seeds."""
    return results_df.groupby("model").agg(
        feature_count=("feature_count", "first"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        R2_median=("R2", "median"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
    ).reset_index()


def main():
    print("Loading data frames...")
    frames = load_data_frames()
    seeds = SEEDS[:int(os.environ.get("SEED_LIMIT"))] if os.environ.get("SEED_LIMIT") else SEEDS
    print(f"Evaluating {len(MODEL_ORDER)} models x {len(seeds)} seeds...")

    rows = []
    for model_name in MODEL_ORDER:
        df = frames[model_name]
        features = [c for c in FEATURE_SETS[model_name] if c in df.columns]
        print(f"  {model_name}: {len(features)} features, {len(df)} rows")
        for seed in seeds:
            row = evaluate_model(df, features, seed, model_name)
            rows.append(row)
        print(f"    done ({len(seeds)} seeds)")

    results = pd.DataFrame(rows).sort_values(["seed", "model"]).reset_index(drop=True)
    agg = aggregate(results)

    # Save
    results.to_csv(OUT_DIR / "iid_per_seed_metrics.csv", index=False)
    agg.to_csv(OUT_DIR / "iid_aggregate_metrics.csv", index=False)
    (OUT_DIR / "iid_protocol.json").write_text(
        json.dumps({
            "description": "IID random-split evaluation (interpolation)",
            "ood_solvents": OOD_SOLVENTS,
            "seeds": seeds,
            "models": MODEL_ORDER,
            "train_pool_rows": TRAIN_POOL_ROWS,
            "ood_test_rows": OOD_TEST_ROWS,
            "xgb_params": XGB_PARAMS,
            "split_method": "random 80/20 train/test, 10% of train for validation",
        }, indent=2, ensure_ascii=False) + "\n"
    )

    # Pretty print
    print("\n" + "=" * 90)
    print("IID EVALUATION RESULTS (20-seed mean ± std)")
    print("=" * 90)
    header = f"{'Model':<18} {'Feat':>5} {'IID R2':>14} {'IID MAE':>11} {'IID RMSE':>11}"
    print(header)
    print("-" * 65)
    for _, r in agg.iterrows():
        print(f"{r['model']:<18} {int(r['feature_count']):>5} "
              f"{r['R2_mean']:>7.4f}±{r['R2_std']:.4f} "
              f"{r['MAE_mean']:>7.4f} {r['RMSE_mean']:>7.4f}")

    # Also load OOD metrics for gap computation
    ood_path = Path(
        os.environ.get(
            "CATSOLV_OOD_AGGREGATE",
            RELEASE_ROOT / "source_data" / "experiments" / "physchem_cross_20seed" / "aggregate_seed_metrics.csv",
        )
    )
    if ood_path.exists():
        ood_agg = pd.read_csv(ood_path)
        name_map = {
            "base_no_gasteiger": "2D-Min", "base_gasteiger": "2D-Elec",
            "sigma_no_gasteiger": "Sigma-noElec", "sigma_gasteiger": "Sigma-Solv",
        }
        ood_agg["model"] = ood_agg["model"].map(name_map)

        print("\n" + "=" * 90)
        print("IID → OOD GENERALIZATION GAP")
        print("=" * 90)
        header = f"{'Model':<18} {'IID R2':>14} {'OOD R2':>14} {'Δ R2':>10} {'Gap %':>8}"
        print(header)
        print("-" * 72)
        for _, iid_row in agg.iterrows():
            ood_row = ood_agg[ood_agg["model"] == iid_row["model"]]
            if len(ood_row) == 1:
                iid_r2 = iid_row["R2_mean"]
                ood_r2 = ood_row.iloc[0]["R2_mean"]
                delta = iid_r2 - ood_r2
                gap_pct = delta / iid_r2 * 100
                print(f"{iid_row['model']:<18} {iid_r2:>7.4f}±{iid_row['R2_std']:.4f} "
                      f"{ood_r2:>7.4f}±{ood_row.iloc[0]['R2_std']:.4f} "
                      f"{delta:>+8.4f} {gap_pct:>6.1f}%")

    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
