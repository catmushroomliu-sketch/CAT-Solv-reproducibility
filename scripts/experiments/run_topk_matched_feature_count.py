from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RELEASE_ROOT = Path(os.environ.get("CATSOLV_RELEASE_ROOT", Path(__file__).resolve().parents[2]))
EXPERIMENT_ROOT = Path(
    os.environ.get("CATSOLV_TOPK_OUTPUT_ROOT", RELEASE_ROOT / "results" / "topk_matched_feature_count")
)
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
PHYS_EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PHYS_EXPERIMENT_ROOT))

from run_physchem_cross_20seed import (  # noqa: E402
    PRED_META_COLS,
    SEEDS,
    SOLVENT_NAMES,
    build_model_frames,
    build_test_manifest,
    feature_cols,
    fit_xgb,
    idx_from_row_ids,
)


TOP_K_VALUES = [33, 47, 60]
REFERENCE_MODELS = ["v1", "v2"]
FULL_MODEL = "v2_physchem_class_cross"
MODEL_ORDER = [
    "v1",
    "v2",
    "class_cross_top33_valshap",
    "class_cross_top47_valshap",
    "class_cross_top60_valshap",
    "class_cross_full98",
]


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame, ndigits: int = 4) -> str:
    view = df.copy()
    numeric_cols = view.select_dtypes(include=[np.number]).columns
    view[numeric_cols] = view[numeric_cols].round(ndigits)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def select_top_features_by_validation_shap(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, xgb.XGBRegressor]:
    cols = feature_cols(df)
    X = df[cols]
    y = df["LogS"].to_numpy()

    selector_model = fit_xgb(
        seed,
        X.iloc[train_idx].to_numpy(),
        y[train_idx],
        X.iloc[val_idx].to_numpy(),
        y[val_idx],
    )
    val_dm = xgb.DMatrix(X.iloc[val_idx], feature_names=cols)
    contrib = selector_model.get_booster().predict(val_dm, pred_contribs=True)
    shap_values = contrib[:, :-1]
    ranking = pd.DataFrame(
        {
            "feature": cols,
            "mean_abs_shap_val": np.abs(shap_values).mean(axis=0),
            "mean_signed_shap_val": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap_val", ascending=False, ignore_index=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking, selector_model


def evaluate_model(
    df: pd.DataFrame,
    cols: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    model_name: str,
    prefit_model: xgb.XGBRegressor | None = None,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    X = df[cols]
    y = df["LogS"].to_numpy()
    model = prefit_model
    if model is None:
        model = fit_xgb(
            seed,
            X.iloc[train_idx].to_numpy(),
            y[train_idx],
            X.iloc[val_idx].to_numpy(),
            y[val_idx],
        )

    y_true = y[test_idx]
    y_pred = model.predict(X.iloc[test_idx].to_numpy())
    metrics = metric_dict(y_true, y_pred)
    row = {
        "seed": seed,
        "model": model_name,
        "feature_count": len(cols),
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
        **metrics,
    }
    pred_df = df.iloc[test_idx][PRED_META_COLS].copy()
    pred_df = pred_df.rename(columns={"LogS": "y_true"})
    pred_df["y_pred"] = y_pred
    pred_df["model"] = model_name
    pred_df["seed"] = seed
    pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]
    pred_df["abs_error"] = np.abs(pred_df["residual"])
    pred_df["sq_error"] = pred_df["residual"] ** 2
    return row, pred_df


def aggregate_metrics(overall_df: pd.DataFrame) -> pd.DataFrame:
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


def delta_table(overall_df: pd.DataFrame, reference: str) -> pd.DataFrame:
    pivot = overall_df.pivot(index="seed", columns="model", values=["R2", "MAE", "RMSE"])
    rows = []
    for model in [m for m in MODEL_ORDER if m != reference]:
        for metric, positive_is_good in [("R2", True), ("MAE", False), ("RMSE", False)]:
            arr = (pivot[metric][model] - pivot[metric][reference]).to_numpy()
            rows.append(
                {
                    "reference": reference,
                    "candidate": model,
                    "metric": metric,
                    "mean_delta": float(arr.mean()),
                    "std_delta": float(arr.std(ddof=1)),
                    "median_delta": float(np.median(arr)),
                    "wins": int((arr > 0).sum()) if positive_is_good else int((arr < 0).sum()),
                    "n_seeds": int(len(arr)),
                }
            )
    return pd.DataFrame(rows)


def per_solvent_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed, solvent), group in pred_df.groupby(["model", "seed", "SMILES_Solvent"], sort=False):
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "SMILES_Solvent": solvent,
                "Solvent_Name": SOLVENT_NAMES.get(solvent, solvent),
                "n_rows": int(len(group)),
                **metric_dict(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    aggregate_df: pd.DataFrame,
    delta_v1: pd.DataFrame,
    delta_v2: pd.DataFrame,
    selected_summary: pd.DataFrame,
    seeds: list[int],
) -> str:
    r2_delta_v1 = delta_v1[delta_v1["metric"] == "R2"].copy()
    r2_delta_v2 = delta_v2[delta_v2["metric"] == "R2"].copy()
    return f"""# Top-k Matched Feature Count Control

## Purpose

This experiment tests whether the 98-feature physchem-cross model improves OOD performance merely because it has more descriptors.

Feature selection is intentionally performed without OOD leakage:

- For each seed, the original-five OOD solvents are held out exactly as in the fair 20-seed protocol.
- A full 98-feature `v2_physchem_class_cross` model is trained on the training split.
- SHAP values are computed only on the validation split from the training pool.
- The top-33, top-47, and top-60 features are selected from validation SHAP rankings.
- Each selected feature subset is retrained from scratch and evaluated on the fixed OOD set.

Seeds: `{seeds[0]}-{seeds[-1]}` (`n={len(seeds)}`)

## Aggregate OOD metrics

{markdown_table(aggregate_df)}

## R2 delta versus v1

{markdown_table(r2_delta_v1)}

## R2 delta versus v2

{markdown_table(r2_delta_v2)}

## Features repeatedly selected across seeds

{markdown_table(selected_summary.head(40))}
"""


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[: int(os.environ["SEED_LIMIT"])] if os.environ.get("SEED_LIMIT") else SEEDS

    frames, _ = build_model_frames()
    ood_table, remain_row_ids, test_row_ids = build_test_manifest(frames["v1"])
    save_json(
        OUTPUT_ROOT / "protocol.json",
        {
            "seeds": seeds,
            "ood_solvents": ood_table.to_dict(orient="records"),
            "feature_selection": "validation_shap_only_no_ood_leakage",
            "top_k_values": TOP_K_VALUES,
            "models": MODEL_ORDER,
            "source_experiment": str(PHYS_EXPERIMENT_ROOT),
        },
    )
    ood_table.to_csv(OUTPUT_ROOT / "ood_solvents.csv", index=False)

    overall_rows = []
    pred_frames = []
    ranking_frames = []
    selected_rows = []

    for seed in seeds:
        train_ids, val_ids = train_test_split(remain_row_ids, test_size=0.1, random_state=seed)

        idx = {}
        for model_name, df in frames.items():
            idx[model_name] = {
                "train": idx_from_row_ids(df, sorted(train_ids)),
                "val": idx_from_row_ids(df, sorted(val_ids)),
                "test": idx_from_row_ids(df, test_row_ids),
            }

        full_df = frames[FULL_MODEL]
        ranking, full_model = select_top_features_by_validation_shap(
            full_df,
            idx[FULL_MODEL]["train"],
            idx[FULL_MODEL]["val"],
            seed,
        )
        ranking["seed"] = seed
        ranking_frames.append(ranking)

        for model_name in REFERENCE_MODELS:
            df = frames[model_name]
            cols = feature_cols(df)
            row, pred_df = evaluate_model(
                df,
                cols,
                idx[model_name]["train"],
                idx[model_name]["val"],
                idx[model_name]["test"],
                seed,
                model_name,
            )
            overall_rows.append(row)
            pred_frames.append(pred_df)

        full_cols = feature_cols(full_df)
        row, pred_df = evaluate_model(
            full_df,
            full_cols,
            idx[FULL_MODEL]["train"],
            idx[FULL_MODEL]["val"],
            idx[FULL_MODEL]["test"],
            seed,
            "class_cross_full98",
            prefit_model=full_model,
        )
        overall_rows.append(row)
        pred_frames.append(pred_df)

        for top_k in TOP_K_VALUES:
            model_name = f"class_cross_top{top_k}_valshap"
            cols = ranking.head(top_k)["feature"].tolist()
            selected_rows.extend(
                {"seed": seed, "top_k": top_k, "rank": rank, "feature": feature}
                for rank, feature in enumerate(cols, start=1)
            )
            row, pred_df = evaluate_model(
                full_df,
                cols,
                idx[FULL_MODEL]["train"],
                idx[FULL_MODEL]["val"],
                idx[FULL_MODEL]["test"],
                seed,
                model_name,
            )
            overall_rows.append(row)
            pred_frames.append(pred_df)

        print(f"completed seed {seed}", flush=True)

    overall_df = pd.DataFrame(overall_rows).sort_values(["seed", "model"]).reset_index(drop=True)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    rankings_df = pd.concat(ranking_frames, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    solvent_df = per_solvent_metrics(pred_df)
    aggregate_df = aggregate_metrics(overall_df)
    delta_v1 = delta_table(overall_df, reference="v1")
    delta_v2 = delta_table(overall_df, reference="v2")
    selected_summary = (
        selected_df.groupby("feature", as_index=False)
        .agg(
            selected_count=("feature", "size"),
            mean_rank=("rank", "mean"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        )
        .sort_values(["selected_count", "mean_rank"], ascending=[False, True])
        .reset_index(drop=True)
    )

    overall_df.to_csv(OUTPUT_ROOT / "overall_seed_metrics.csv", index=False)
    aggregate_df.to_csv(OUTPUT_ROOT / "aggregate_seed_metrics.csv", index=False)
    delta_v1.to_csv(OUTPUT_ROOT / "delta_vs_v1_summary.csv", index=False)
    delta_v2.to_csv(OUTPUT_ROOT / "delta_vs_v2_summary.csv", index=False)
    pred_df.to_csv(OUTPUT_ROOT / "ood_predictions_all_seeds.csv", index=False)
    solvent_df.to_csv(OUTPUT_ROOT / "per_solvent_seed_metrics.csv", index=False)
    rankings_df.to_csv(OUTPUT_ROOT / "validation_shap_rankings_all_seeds.csv", index=False)
    selected_df.to_csv(OUTPUT_ROOT / "selected_features_by_seed.csv", index=False)
    selected_summary.to_csv(OUTPUT_ROOT / "selected_feature_frequency.csv", index=False)
    (OUTPUT_ROOT / "report.md").write_text(
        build_report(aggregate_df, delta_v1, delta_v2, selected_summary, seeds),
        encoding="utf-8",
    )

    print(aggregate_df.to_string(index=False))
    print(delta_v1[delta_v1["metric"] == "R2"].to_string(index=False))
    print(delta_v2[delta_v2["metric"] == "R2"].to_string(index=False))


if __name__ == "__main__":
    main()
