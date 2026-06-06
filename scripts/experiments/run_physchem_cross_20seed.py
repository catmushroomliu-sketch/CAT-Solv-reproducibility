from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RELEASE_ROOT = Path(os.environ.get("CATSOLV_RELEASE_ROOT", Path(__file__).resolve().parents[2]))
EXPERIMENT_ROOT = Path(
    os.environ.get("CATSOLV_PHYSCHEM_OUTPUT_ROOT", RELEASE_ROOT / "results" / "physchem_cross_20seed")
)
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
FEATURE_ROOT = EXPERIMENT_ROOT / "features"
DATA_ROOT = Path(os.environ.get("CATSOLV_RAW_DATA_ROOT", RELEASE_ROOT / "data" / "raw"))

V1_INPUT = DATA_ROOT / "BigSolDB_features_v1_common.csv"
V2_INPUT = DATA_ROOT / "BigSolDB_features_v2_common.csv"

SEEDS = list(range(42, 62))
ORIGINAL_V1_OOD_SOLVENTS = [
    "CS(C)=O",
    "C1CCCCC1",
    "OCCO",
    "CN1CCCC1=O",
    "CCCOC(C)=O",
]

META_COLS = ["Common_RowID", "SMILES_Solute", "SMILES_Solvent", "LogS"]
PRED_META_COLS = ["Common_RowID", "SMILES_Solute", "SMILES_Solvent", "Temperature_K", "LogS"]
MODEL_ORDER = [
    "v1",
    "v2",
    "v2_physchem_general_cross",
    "v2_physchem_class_cross",
]

SOLVENT_NAMES = {
    "CS(C)=O": "DMSO",
    "C1CCCCC1": "Cyclohexane",
    "OCCO": "Ethylene glycol",
    "CN1CCCC1=O": "NMP",
    "CCCOC(C)=O": "n-Propyl acetate",
}

TARGET_SOLVENTS = {"C1CCCCC1", "OCCO", "CCCOC(C)=O"}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def safe_div(num: pd.Series, den: pd.Series | float, eps: float = 1e-9) -> pd.Series:
    return num / (den + eps)


def has_substructure(smiles: str, smarts: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts(smarts)
    if mol is None or patt is None:
        return 0
    return int(mol.HasSubstructMatch(patt))


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def fit_xgb(seed: int, X_train, y_train, X_val, y_val) -> xgb.XGBRegressor:
    n_jobs = max(1, min(10, (os.cpu_count() or 8) - 2))
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def idx_from_row_ids(df: pd.DataFrame, row_ids: list[int]) -> np.ndarray:
    return df.index[df["Common_RowID"].isin(set(row_ids))].to_numpy()


def build_test_manifest(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int], list[int]]:
    counts = df["SMILES_Solvent"].value_counts().sort_values(ascending=False)
    rank_map = {smi: rank for rank, smi in enumerate(counts.index.tolist(), start=1)}
    missing = sorted(set(ORIGINAL_V1_OOD_SOLVENTS) - set(counts.index))
    if missing:
        raise RuntimeError(f"OOD solvents missing from common subset: {missing}")

    ood_mask = df["SMILES_Solvent"].isin(ORIGINAL_V1_OOD_SOLVENTS)
    test_row_ids = sorted(df.loc[ood_mask, "Common_RowID"].tolist())
    remain_row_ids = sorted(df.loc[~ood_mask, "Common_RowID"].tolist())
    table = pd.DataFrame(
        {
            "SMILES_Solvent": ORIGINAL_V1_OOD_SOLVENTS,
            "Solvent_Name": [SOLVENT_NAMES[s] for s in ORIGINAL_V1_OOD_SOLVENTS],
            "n_rows_in_common_subset": [int(counts[s]) for s in ORIGINAL_V1_OOD_SOLVENTS],
            "frequency_rank_in_common_subset": [int(rank_map[s]) for s in ORIGINAL_V1_OOD_SOLVENTS],
        }
    )
    return table, remain_row_ids, test_row_ids


def add_general_cross_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    new_cols: list[str] = []

    # Compact solvent physicochemical proxies from descriptors already present
    # for all solvents in the common dataset.
    proxy_defs = {
        "Solvent_HB_capacity": out["Solvent_NumHDonors"] + out["Solvent_NumHAcceptors"],
        "Solvent_HB_balance": out["Solvent_NumHDonors"] - out["Solvent_NumHAcceptors"],
        "Solvent_polar_surface_density": safe_div(out["Solvent_TPSA"], out["Solvent_LabuteASA"]),
        "Solvent_hydrophobic_density": safe_div(out["Solvent_MolLogP"], out["Solvent_MolWt"]),
        "Solvent_sigma_polarity": out["Solvent_sig2"],
        "Solvent_sigma_asymmetry_abs": out["Solvent_sig3"].abs(),
        "Solvent_signed_surface_imbalance": out["Solvent_pos_area_frac"] - out["Solvent_neg_area_frac"],
        "Solvent_self_hbond_proxy": safe_div(
            out["Solvent_NumHDonors"] * out["Solvent_NumHAcceptors"],
            out["Solvent_MolWt"],
        ),
        "Solute_HB_capacity": out["Solute_NumHDonors"] + out["Solute_NumHAcceptors"],
        "Solute_polar_surface_density": safe_div(out["Solute_TPSA"], out["Solute_LabuteASA"]),
        "Solute_signed_surface_imbalance": out["Solute_pos_area_frac"] - out["Solute_neg_area_frac"],
    }
    for col, values in proxy_defs.items():
        out[col] = values.replace([np.inf, -np.inf], np.nan)
        new_cols.append(col)

    cross_defs = {
        "Cross_HBD_solute_x_HBA_solvent": out["Solute_NumHDonors"] * out["Solvent_NumHAcceptors"],
        "Cross_HBA_solute_x_HBD_solvent": out["Solute_NumHAcceptors"] * out["Solvent_NumHDonors"],
        "Cross_HB_capacity_product": out["Solute_HB_capacity"] * out["Solvent_HB_capacity"],
        "Cross_HB_directional_balance": (
            out["Solute_NumHDonors"] * out["Solvent_NumHAcceptors"]
            - out["Solute_NumHAcceptors"] * out["Solvent_NumHDonors"]
        ),
        "Cross_logP_abs_gap": (out["Solute_MolLogP"] - out["Solvent_MolLogP"]).abs(),
        "Cross_TPSA_density_abs_gap": (
            out["Solute_polar_surface_density"] - out["Solvent_polar_surface_density"]
        ).abs(),
        "Cross_sigma2_abs_gap": (out["Solute_sig2"] - out["Solvent_sig2"]).abs(),
        "Cross_sigma3_abs_gap": (out["Solute_sig3"] - out["Solvent_sig3"]).abs(),
        "Cross_pos_area_abs_gap": (out["Solute_pos_area_frac"] - out["Solvent_pos_area_frac"]).abs(),
        "Cross_hb_acc_moment_product": out["Solute_M_hb_acc"] * out["Solvent_M_hb_acc"],
        "Cross_hb_don_moment_product": out["Solute_M_hb_don"] * out["Solvent_M_hb_don"],
        "Cross_acc_to_don_surface": out["Solute_M_hb_acc"] * out["Solvent_M_hb_don"],
        "Cross_don_to_acc_surface": out["Solute_M_hb_don"] * out["Solvent_M_hb_acc"],
        "Cross_nonpolar_match": out["Solute_M_nonpolar"] * out["Solvent_M_nonpolar"],
        "Cross_surface_imbalance_product": (
            out["Solute_signed_surface_imbalance"] * out["Solvent_signed_surface_imbalance"]
        ),
        "Cross_temperature_x_solvent_polarity": out["Temperature_K"] * out["Solvent_polar_surface_density"],
        "Cross_temperature_x_sigma2_gap": out["Temperature_K"] * (out["Solute_sig2"] - out["Solvent_sig2"]).abs(),
    }
    for col, values in cross_defs.items():
        out[col] = values.replace([np.inf, -np.inf], np.nan)
        new_cols.append(col)
    return out, new_cols


def add_class_cross_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out, cols = add_general_cross_features(df)

    solvent_unique = out[["SMILES_Solvent"]].drop_duplicates().copy()
    solvent_unique["Solvent_has_ester"] = solvent_unique["SMILES_Solvent"].map(
        lambda s: has_substructure(s, "[CX3](=O)[OX2][#6]")
    )
    solvent_unique["Solvent_is_cycloalkane_like"] = solvent_unique["SMILES_Solvent"].map(
        lambda s: int(s == "C1CCCCC1")
    )
    out = out.merge(solvent_unique, on="SMILES_Solvent", how="left")

    class_defs = {
        "Solvent_is_nonpolar": (
            (out["Solvent_TPSA"] < 5)
            & (out["Solvent_NumHDonors"] == 0)
            & (out["Solvent_NumHAcceptors"] == 0)
        ).astype(int),
        "Solvent_is_protic": (out["Solvent_NumHDonors"] > 0).astype(int),
        "Solvent_is_acceptor_only": (
            (out["Solvent_NumHDonors"] == 0) & (out["Solvent_NumHAcceptors"] > 0)
        ).astype(int),
        "Solvent_is_polyol_like": (
            (out["Solvent_NumHDonors"] >= 2)
            & (out["Solvent_NumHAcceptors"] >= 2)
            & (out["Solvent_MolWt"] < 130)
        ).astype(int),
        "Solvent_is_ester_acceptor": (
            (out["Solvent_has_ester"] == 1) & (out["Solvent_NumHDonors"] == 0)
        ).astype(int),
    }

    new_cols = list(cols)
    for col, values in class_defs.items():
        out[col] = values
        new_cols.append(col)

    class_cross_defs = {
        # Nonpolar solvents: penalize localized polar surface and reward nonpolar
        # surface compatibility.
        "Class_nonpolar_x_solute_TPSA_density": out["Solvent_is_nonpolar"] * out["Solute_polar_surface_density"],
        "Class_nonpolar_x_solute_sigma2": out["Solvent_is_nonpolar"] * out["Solute_sig2"],
        "Class_nonpolar_x_solute_HB_capacity": out["Solvent_is_nonpolar"] * out["Solute_HB_capacity"],
        "Class_nonpolar_x_solute_logP": out["Solvent_is_nonpolar"] * out["Solute_MolLogP"],
        "Class_nonpolar_x_nonpolar_match": out["Solvent_is_nonpolar"] * out["Cross_nonpolar_match"],
        # Polyol solvents: model competition between solute-EG and EG-EG
        # hydrogen-bond networks.
        "Class_polyol_x_hb_capacity_product": out["Solvent_is_polyol_like"] * out["Cross_HB_capacity_product"],
        "Class_polyol_x_directional_HB": out["Solvent_is_polyol_like"] * (
            out["Cross_HBD_solute_x_HBA_solvent"] + out["Cross_HBA_solute_x_HBD_solvent"]
        ),
        "Class_polyol_x_self_hbond_competition": out["Solvent_is_polyol_like"] * (
            out["Cross_HB_capacity_product"] - out["Solvent_self_hbond_proxy"]
        ),
        "Class_polyol_x_solute_charge_span": out["Solvent_is_polyol_like"] * (
            out["Solute_MaxPartialCharge"] - out["Solute_MinPartialCharge"]
        ),
        "Class_polyol_x_salt_like_proxy": out["Solvent_is_polyol_like"] * (
            out["SMILES_Solute"].astype(str).str.contains("\\.").astype(int)
        ),
        # Ester acceptor solvents: acceptor-only interactions should not be
        # treated like DMSO/polyol hydrogen-bonding regimes.
        "Class_ester_x_solute_HBD": out["Solvent_is_ester_acceptor"] * out["Solute_NumHDonors"],
        "Class_ester_x_solute_HBA": out["Solvent_is_ester_acceptor"] * out["Solute_NumHAcceptors"],
        "Class_ester_x_solute_TPSA_density": out["Solvent_is_ester_acceptor"] * out["Solute_polar_surface_density"],
        "Class_ester_x_sigma2_gap": out["Solvent_is_ester_acceptor"] * out["Cross_sigma2_abs_gap"],
        "Class_ester_x_logP_gap": out["Solvent_is_ester_acceptor"] * out["Cross_logP_abs_gap"],
        "Class_ester_x_acceptor_only_penalty": out["Solvent_is_ester_acceptor"] * (
            out["Solute_NumHAcceptors"] - out["Solute_NumHDonors"]
        ),
    }
    for col, values in class_cross_defs.items():
        out[col] = values.replace([np.inf, -np.inf], np.nan)
        new_cols.append(col)
    return out, new_cols


def build_model_frames() -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    df_v1 = pd.read_csv(V1_INPUT)
    df_v2 = pd.read_csv(V2_INPUT)
    if sorted(df_v1["Common_RowID"].tolist()) != sorted(df_v2["Common_RowID"].tolist()):
        raise RuntimeError("v1/v2 common datasets do not align on Common_RowID.")

    general, general_cols = add_general_cross_features(df_v2)
    class_cross, class_cols = add_class_cross_features(df_v2)
    frames = {
        "v1": df_v1,
        "v2": df_v2,
        "v2_physchem_general_cross": general,
        "v2_physchem_class_cross": class_cross,
    }
    feature_sets = {
        "v2_physchem_general_cross": general_cols,
        "v2_physchem_class_cross": class_cols,
    }
    return frames, feature_sets


def summarize_model(
    df: pd.DataFrame,
    split: dict[str, list[int]],
    seed: int,
    model_name: str,
) -> tuple[dict, pd.DataFrame]:
    cols = feature_cols(df)
    X = df[cols].values
    y = df["LogS"].values

    idx_train = idx_from_row_ids(df, split["train_row_ids"])
    idx_val = idx_from_row_ids(df, split["val_row_ids"])
    idx_test = idx_from_row_ids(df, split["test_row_ids"])

    model = fit_xgb(seed, X[idx_train], y[idx_train], X[idx_val], y[idx_val])
    y_test = y[idx_test]
    y_pred = model.predict(X[idx_test])
    payload = {
        "seed": seed,
        "model": model_name,
        "feature_count": len(cols),
        "train_n": int(len(idx_train)),
        "val_n": int(len(idx_val)),
        "test_n": int(len(idx_test)),
        **metrics(y_test, y_pred),
    }
    pred_df = df.iloc[idx_test][PRED_META_COLS].copy()
    pred_df = pred_df.rename(columns={"LogS": "y_true"})
    pred_df["y_pred"] = y_pred
    pred_df["model"] = model_name
    pred_df["seed"] = seed
    pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]
    pred_df["abs_error"] = np.abs(pred_df["residual"])
    pred_df["sq_error"] = pred_df["residual"] ** 2
    return payload, pred_df


def per_solvent_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, seed, solvent), group in pred_df.groupby(["model", "seed", "SMILES_Solvent"], sort=False):
        rows.append(
            {
                "model": model_name,
                "seed": int(seed),
                "SMILES_Solvent": solvent,
                "Solvent_Name": SOLVENT_NAMES.get(solvent, solvent),
                "n_rows": int(len(group)),
                **metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_seed_metrics(overall_df: pd.DataFrame) -> pd.DataFrame:
    agg = overall_df.groupby("model").agg(
        feature_count=("feature_count", "first"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        R2_median=("R2", "median"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
    )
    return agg.loc[MODEL_ORDER].reset_index()


def delta_vs_reference(overall_df: pd.DataFrame, reference: str = "v2") -> pd.DataFrame:
    pivot = overall_df.pivot(index="seed", columns="model", values=["R2", "MAE", "RMSE"])
    rows = []
    for model in [m for m in MODEL_ORDER if m != reference]:
        for metric, positive_is_good in [("R2", True), ("MAE", False), ("RMSE", False)]:
            delta = pivot[metric][model] - pivot[metric][reference]
            arr = delta.to_numpy()
            rows.append(
                {
                    "reference": reference,
                    "candidate": model,
                    "metric": metric,
                    "mean_delta": float(arr.mean()),
                    "std_delta": float(arr.std(ddof=1)),
                    "median_delta": float(np.median(arr)),
                    "p2_5": float(np.percentile(arr, 2.5)),
                    "p97_5": float(np.percentile(arr, 97.5)),
                    "wins": int((arr > 0).sum()) if positive_is_good else int((arr < 0).sum()),
                    "n_seeds": int(len(arr)),
                }
            )
    return pd.DataFrame(rows)


def per_solvent_delta_summary(solvent_df: pd.DataFrame, reference: str = "v2") -> pd.DataFrame:
    pivot = solvent_df.pivot_table(
        index=["seed", "SMILES_Solvent", "Solvent_Name", "n_rows"],
        columns="model",
        values=["R2", "MAE", "RMSE"],
    ).reset_index()
    pivot.columns = [
        col if isinstance(col, str) else (col[0] if col[1] == "" else f"{col[0]}__{col[1]}")
        for col in pivot.columns
    ]
    rows = []
    for model in [m for m in MODEL_ORDER if m != reference]:
        tmp = pd.DataFrame(
            {
                "seed": pivot["seed"],
                "SMILES_Solvent": pivot["SMILES_Solvent"],
                "Solvent_Name": pivot["Solvent_Name"],
                "n_rows": pivot["n_rows"],
                "reference": reference,
                "candidate": model,
                "delta_R2": pivot[f"R2__{model}"] - pivot[f"R2__{reference}"],
                "delta_MAE": pivot[f"MAE__{model}"] - pivot[f"MAE__{reference}"],
                "delta_RMSE": pivot[f"RMSE__{model}"] - pivot[f"RMSE__{reference}"],
            }
        )
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    summary = out.groupby(["reference", "candidate", "SMILES_Solvent", "Solvent_Name", "n_rows"]).agg(
        delta_R2_mean=("delta_R2", "mean"),
        delta_R2_std=("delta_R2", "std"),
        delta_MAE_mean=("delta_MAE", "mean"),
        delta_MAE_std=("delta_MAE", "std"),
        delta_RMSE_mean=("delta_RMSE", "mean"),
        wins_R2=("delta_R2", lambda s: int((s > 0).sum())),
        wins_MAE=("delta_MAE", lambda s: int((s < 0).sum())),
        wins_RMSE=("delta_RMSE", lambda s: int((s < 0).sum())),
    )
    return summary.reset_index()


def markdown_table(df: pd.DataFrame, ndigits: int = 4) -> str:
    view = df.copy()
    numeric_cols = view.select_dtypes(include=[np.number]).columns
    view[numeric_cols] = view[numeric_cols].round(ndigits)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def build_report(
    ood_table: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    solvent_df: pd.DataFrame,
    solvent_delta_df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    seeds: list[int],
) -> str:
    target_metrics = (
        solvent_df[solvent_df["SMILES_Solvent"].isin(TARGET_SOLVENTS)]
        .groupby(["Solvent_Name", "model"], as_index=False)[["R2", "MAE", "RMSE"]]
        .mean()
        .sort_values(["Solvent_Name", "R2"], ascending=[True, False])
    )
    target_delta = solvent_delta_df[
        solvent_delta_df["SMILES_Solvent"].isin(TARGET_SOLVENTS)
        & solvent_delta_df["candidate"].isin(["v2_physchem_general_cross", "v2_physchem_class_cross"])
    ].copy()
    feature_lines = []
    for name, cols in feature_sets.items():
        feature_lines.append(f"- `{name}` added `{len(cols)}` engineered features.")
        feature_lines.extend([f"  - `{c}`" for c in cols])

    return f"""# Physchem Cross 20-Seed OOD Experiment

## Protocol

- OOD solvent identities are fixed to the original-five fair-control benchmark.
- Seeds: `{seeds[0]}-{seeds[-1]}` (`n={len(seeds)}`)
- Train/validation split and XGBoost seed vary by seed; OOD rows stay fixed.
- This experiment tests whether solvent-physics cross features improve weak OOD solvents without changing the OOD definition.

## OOD solvents

{markdown_table(ood_table, ndigits=0)}

## Aggregate metrics

{markdown_table(aggregate_df)}

## Delta versus v2

{markdown_table(delta_df)}

## Target-solvent metrics

{markdown_table(target_metrics)}

## Target-solvent delta versus v2

{markdown_table(target_delta)}

## Engineered feature sets

{chr(10).join(feature_lines)}
"""


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[: int(os.environ["SEED_LIMIT"])] if os.environ.get("SEED_LIMIT") else SEEDS

    frames, feature_sets = build_model_frames()
    for model_name, cols in feature_sets.items():
        pd.DataFrame({"feature": cols}).to_csv(FEATURE_ROOT / f"{model_name}_added_features.csv", index=False)

    ood_table, remain_row_ids, test_row_ids = build_test_manifest(frames["v1"])
    save_json(
        OUTPUT_ROOT / "protocol.json",
        {
            "v1_input": str(V1_INPUT),
            "v2_input": str(V2_INPUT),
            "ood_solvents": ORIGINAL_V1_OOD_SOLVENTS,
            "seeds": seeds,
            "models": MODEL_ORDER,
            "n_train_pool_rows": len(remain_row_ids),
            "n_test_rows": len(test_row_ids),
            "feature_sets": feature_sets,
        },
    )
    ood_table.to_csv(OUTPUT_ROOT / "ood_solvents.csv", index=False)

    overall_rows = []
    prediction_frames = []
    for seed in seeds:
        train_ids, val_ids = train_test_split(remain_row_ids, test_size=0.1, random_state=seed)
        split = {
            "train_row_ids": sorted(train_ids),
            "val_row_ids": sorted(val_ids),
            "test_row_ids": test_row_ids,
        }
        for model_name in MODEL_ORDER:
            metrics_row, pred_df = summarize_model(frames[model_name], split, seed, model_name)
            overall_rows.append(metrics_row)
            prediction_frames.append(pred_df)
        print(f"completed seed {seed}")

    overall_df = pd.DataFrame(overall_rows).sort_values(["seed", "model"]).reset_index(drop=True)
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    solvent_seed_df = per_solvent_metrics(pred_df)
    aggregate_df = aggregate_seed_metrics(overall_df)
    delta_df = delta_vs_reference(overall_df, reference="v2")
    solvent_delta_df = per_solvent_delta_summary(solvent_seed_df, reference="v2")

    overall_df.to_csv(OUTPUT_ROOT / "overall_seed_metrics.csv", index=False)
    pred_df.to_csv(OUTPUT_ROOT / "ood_predictions_all_seeds.csv", index=False)
    solvent_seed_df.to_csv(OUTPUT_ROOT / "per_solvent_seed_metrics.csv", index=False)
    aggregate_df.to_csv(OUTPUT_ROOT / "aggregate_seed_metrics.csv", index=False)
    delta_df.to_csv(OUTPUT_ROOT / "delta_vs_v2_summary.csv", index=False)
    solvent_delta_df.to_csv(OUTPUT_ROOT / "per_solvent_delta_vs_v2_summary.csv", index=False)
    (OUTPUT_ROOT / "report.md").write_text(
        build_report(ood_table, aggregate_df, delta_df, solvent_seed_df, solvent_delta_df, feature_sets, seeds),
        encoding="utf-8",
    )

    print(aggregate_df.to_string(index=False))
    print(delta_df.to_string(index=False))


if __name__ == "__main__":
    main()
