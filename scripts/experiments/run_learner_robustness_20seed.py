from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RELEASE_ROOT = Path(os.environ.get("CATSOLV_RELEASE_ROOT", Path(__file__).resolve().parents[2]))
EXPERIMENT_ROOT = Path(
    os.environ.get("CATSOLV_LEARNER_OUTPUT_ROOT", RELEASE_ROOT / "results" / "learner_robustness_20seed")
)
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
PHYS_EXPERIMENT_ROOT = Path(__file__).resolve().parent
TOP47_FEATURES = Path(
    os.environ.get(
        "CATSOLV_TOP47_FEATURES",
        RELEASE_ROOT / "source_data" / "experiments" / "topk_matched_feature_count" / "selected_features_by_seed.csv",
    )
)

sys.path.insert(0, str(PHYS_EXPERIMENT_ROOT))
from run_physchem_cross_20seed import (  # noqa: E402
    PRED_META_COLS,
    SEEDS,
    SOLVENT_NAMES,
    build_model_frames,
    build_test_manifest,
    feature_cols,
    idx_from_row_ids,
)


GASTEIGER_SUFFIXES = (
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
)

FEATURE_LAYERS = ["2D-Min", "2D-Elec", "Sigma-Solv", "CAT-Solv-47"]
LEARNERS = ["XGBoost", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
LAYER_TO_FRAME = {
    "2D-Min": "v1_2d_min",
    "2D-Elec": "v1",
    "Sigma-Solv": "v2",
    "CAT-Solv-47": "v2_physchem_class_cross",
}


def n_jobs() -> int:
    return max(1, min(10, (os.cpu_count() or 8) - 2))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def load_top47_by_seed() -> dict[int, list[str]]:
    selected = pd.read_csv(TOP47_FEATURES)
    selected = selected[selected["top_k"].eq(47)].copy()
    out: dict[int, list[str]] = {}
    for seed, group in selected.groupby("seed"):
        cols = group.sort_values("rank")["feature"].tolist()
        if len(cols) != 47:
            raise RuntimeError(f"Seed {seed} has {len(cols)} top-47 features, expected 47")
        out[int(seed)] = cols
    return out


def gasteiger_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if c.endswith(GASTEIGER_SUFFIXES)]


def add_2d_min_frame(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = dict(frames)
    drop_cols = gasteiger_cols(out["v1"].columns.tolist())
    if len(drop_cols) != 8:
        raise RuntimeError(f"Expected 8 Gasteiger columns in v1, found {len(drop_cols)}: {drop_cols}")
    out["v1_2d_min"] = out["v1"].drop(columns=drop_cols)
    return out


def layer_feature_columns(df: pd.DataFrame, layer: str, seed: int, top47_by_seed: dict[int, list[str]]) -> list[str]:
    if layer in {"2D-Min", "2D-Elec", "Sigma-Solv"}:
        return feature_cols(df)
    if layer == "CAT-Solv-47":
        cols = top47_by_seed[seed]
        missing = sorted(set(cols) - set(df.columns))
        if missing:
            raise RuntimeError(f"Missing CAT-Solv-47 features for seed {seed}: {missing[:10]}")
        return cols
    raise KeyError(layer)


def fit_xgboost(seed: int, X_train, y_train, X_val, y_val):
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=n_jobs(),
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fit_random_forest(seed: int, X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(
        n_estimators=200,
        max_features=0.75,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=n_jobs(),
    )
    model.fit(X_train, y_train)
    return model


def fit_extra_trees(seed: int, X_train, y_train, X_val, y_val):
    model = ExtraTreesRegressor(
        n_estimators=200,
        max_features=0.75,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=n_jobs(),
    )
    model.fit(X_train, y_train)
    return model


def fit_hist_gradient_boosting(seed: int, X_train, y_train, X_val, y_val):
    model = HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.05,
        max_leaf_nodes=63,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


FITTERS: dict[str, Callable] = {
    "XGBoost": fit_xgboost,
    "RandomForest": fit_random_forest,
    "ExtraTrees": fit_extra_trees,
    "HistGradientBoosting": fit_hist_gradient_boosting,
}


def evaluate_one(
    df: pd.DataFrame,
    cols: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    learner: str,
    layer: str,
) -> tuple[dict, pd.DataFrame]:
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["LogS"].to_numpy(dtype=np.float32)

    fit_start = time.perf_counter()
    model = FITTERS[learner](seed, X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    fit_seconds = time.perf_counter() - fit_start

    pred_start = time.perf_counter()
    y_true = y[test_idx]
    y_pred = model.predict(X[test_idx])
    predict_seconds = time.perf_counter() - pred_start

    row = {
        "seed": seed,
        "learner": learner,
        "feature_layer": layer,
        "feature_count": len(cols),
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        **metrics(y_true, y_pred),
    }
    pred_df = df.iloc[test_idx][PRED_META_COLS].copy()
    pred_df = pred_df.rename(columns={"LogS": "y_true"})
    pred_df["y_pred"] = y_pred
    pred_df["learner"] = learner
    pred_df["feature_layer"] = layer
    pred_df["seed"] = seed
    pred_df["residual"] = pred_df["y_pred"] - pred_df["y_true"]
    pred_df["abs_error"] = np.abs(pred_df["residual"])
    pred_df["sq_error"] = pred_df["residual"] ** 2
    return row, pred_df


def aggregate(overall_df: pd.DataFrame) -> pd.DataFrame:
    agg = overall_df.groupby(["learner", "feature_layer"], as_index=False).agg(
        feature_count=("feature_count", "first"),
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        R2_median=("R2", "median"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        fit_seconds_mean=("fit_seconds", "mean"),
        fit_seconds_total=("fit_seconds", "sum"),
        predict_seconds_mean=("predict_seconds", "mean"),
    )
    learner_order = {v: i for i, v in enumerate(LEARNERS)}
    layer_order = {v: i for i, v in enumerate(FEATURE_LAYERS)}
    agg["_learner_order"] = agg["learner"].map(learner_order)
    agg["_layer_order"] = agg["feature_layer"].map(layer_order)
    return agg.sort_values(["_learner_order", "_layer_order"]).drop(columns=["_learner_order", "_layer_order"]).reset_index(drop=True)


def per_solvent_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (learner, layer, seed, solvent), group in pred_df.groupby(["learner", "feature_layer", "seed", "SMILES_Solvent"], sort=False):
        rows.append(
            {
                "learner": learner,
                "feature_layer": layer,
                "seed": int(seed),
                "SMILES_Solvent": solvent,
                "Solvent_Name": SOLVENT_NAMES.get(solvent, solvent),
                "n_rows": int(len(group)),
                **metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def descriptor_delta(overall_df: pd.DataFrame) -> pd.DataFrame:
    pivot = overall_df.pivot_table(index=["learner", "seed"], columns="feature_layer", values=["R2", "MAE", "RMSE"])
    rows = []
    comparisons = [
        ("2D-Elec", "2D-Min"),
        ("Sigma-Solv", "2D-Elec"),
        ("Sigma-Solv", "2D-Min"),
        ("CAT-Solv-47", "Sigma-Solv"),
        ("CAT-Solv-47", "2D-Elec"),
        ("CAT-Solv-47", "2D-Min"),
    ]
    for learner in LEARNERS:
        for candidate, reference in comparisons:
            for metric, positive_is_good in [("R2", True), ("MAE", False), ("RMSE", False)]:
                arr = (pivot.loc[learner, metric][candidate] - pivot.loc[learner, metric][reference]).to_numpy()
                rows.append(
                    {
                        "learner": learner,
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "mean_delta": float(arr.mean()),
                        "std_delta": float(arr.std(ddof=1)),
                        "median_delta": float(np.median(arr)),
                        "wins": int((arr > 0).sum()) if positive_is_good else int((arr < 0).sum()),
                        "n_seeds": int(len(arr)),
                    }
                )
    return pd.DataFrame(rows)


def build_report(ood_table: pd.DataFrame, aggregate_df: pd.DataFrame, delta_df: pd.DataFrame, seeds: list[int]) -> str:
    r2_delta = delta_df[delta_df["metric"].eq("R2")].copy()
    return f"""# Learner Robustness 20-Seed Control

## Purpose

This experiment checks whether the fixed solvent-OOD gain is descriptor-driven rather than specific to XGBoost.

## Protocol

- OOD solvents: original-five fixed solvent-OOD benchmark.
- Seeds: `{seeds[0]}-{seeds[-1]}` (`n={len(seeds)}`).
- Feature layers: `2D-Min`, `2D-Elec`, `Sigma-Solv`, `CAT-Solv-47`.
- Learners: `{', '.join(LEARNERS)}`.
- `2D-Min` is the minimal 2D descriptor baseline obtained by removing the eight Gasteiger partial-charge proxy columns from the v1 feature table.
- `CAT-Solv-47` uses the seed-specific validation-SHAP top-47 feature lists from the matched feature-count experiment. No OOD rows are used for feature selection.

## OOD solvents

{markdown_table(ood_table, ndigits=0)}

## Aggregate OOD metrics

{markdown_table(aggregate_df)}

## R2 descriptor deltas

{markdown_table(r2_delta)}

## Interpretation for manuscript

Use this as a Supporting Information robustness control. The primary model remains XGBoost; this experiment asks whether the descriptor progression remains directionally consistent across tree learners.
"""


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[: int(os.environ["SEED_LIMIT"])] if os.environ.get("SEED_LIMIT") else SEEDS
    learners = os.environ.get("LEARNERS")
    learner_list = [x.strip() for x in learners.split(",") if x.strip()] if learners else LEARNERS
    bad = sorted(set(learner_list) - set(LEARNERS))
    if bad:
        raise RuntimeError(f"Unknown learners: {bad}")

    frames, _ = build_model_frames()
    frames = add_2d_min_frame(frames)
    top47_by_seed = load_top47_by_seed()
    missing_seeds = sorted(set(seeds) - set(top47_by_seed))
    if missing_seeds:
        raise RuntimeError(f"Missing top47 feature lists for seeds: {missing_seeds}")

    ood_table, remain_row_ids, test_row_ids = build_test_manifest(frames["v1"])
    ood_table.to_csv(OUTPUT_ROOT / "ood_solvents.csv", index=False)
    save_json(
        OUTPUT_ROOT / "protocol.json",
        {
            "seeds": seeds,
            "learners": learner_list,
            "feature_layers": FEATURE_LAYERS,
            "ood_solvents": ood_table.to_dict(orient="records"),
            "top47_feature_source": str(TOP47_FEATURES),
            "source_physchem_experiment": str(PHYS_EXPERIMENT_ROOT),
            "2d_min_definition": "v1 feature table with the eight Gasteiger partial-charge proxy columns removed",
            "note": "CAT-Solv-47 uses seed-specific validation-SHAP top47 lists from the matched feature-count experiment.",
        },
    )

    feature_rows = []
    overall_rows = []
    prediction_frames = []

    for seed in seeds:
        train_ids, val_ids = train_test_split(remain_row_ids, test_size=0.1, random_state=seed)
        for layer in FEATURE_LAYERS:
            frame_name = LAYER_TO_FRAME[layer]
            df = frames[frame_name]
            cols = layer_feature_columns(df, layer, seed, top47_by_seed)
            feature_rows.extend(
                {"seed": seed, "feature_layer": layer, "rank": i, "feature": col}
                for i, col in enumerate(cols, start=1)
            )
            train_idx = idx_from_row_ids(df, sorted(train_ids))
            val_idx = idx_from_row_ids(df, sorted(val_ids))
            test_idx = idx_from_row_ids(df, test_row_ids)
            for learner in learner_list:
                row, pred_df = evaluate_one(df, cols, train_idx, val_idx, test_idx, seed, learner, layer)
                overall_rows.append(row)
                prediction_frames.append(pred_df)
                print(
                    f"completed seed={seed} learner={learner} layer={layer} "
                    f"R2={row['R2']:.4f} fit={row['fit_seconds']:.1f}s",
                    flush=True,
                )

    overall_df = pd.DataFrame(overall_rows).sort_values(["seed", "learner", "feature_layer"]).reset_index(drop=True)
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    solvent_df = per_solvent_metrics(pred_df)
    aggregate_df = aggregate(overall_df)
    delta_df = descriptor_delta(overall_df)
    feature_df = pd.DataFrame(feature_rows)

    overall_df.to_csv(OUTPUT_ROOT / "overall_seed_metrics.csv", index=False)
    aggregate_df.to_csv(OUTPUT_ROOT / "aggregate_metrics.csv", index=False)
    delta_df.to_csv(OUTPUT_ROOT / "descriptor_delta_summary.csv", index=False)
    solvent_df.to_csv(OUTPUT_ROOT / "per_solvent_seed_metrics.csv", index=False)
    pred_df.to_csv(OUTPUT_ROOT / "ood_predictions_all_seeds.csv", index=False)
    feature_df.to_csv(OUTPUT_ROOT / "feature_lists_by_seed_layer.csv", index=False)
    (OUTPUT_ROOT / "report.md").write_text(build_report(ood_table, aggregate_df, delta_df, seeds), encoding="utf-8")

    print(aggregate_df.to_string(index=False))
    print(delta_df[delta_df["metric"].eq("R2")].to_string(index=False))


if __name__ == "__main__":
    main()
