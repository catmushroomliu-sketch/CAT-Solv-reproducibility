from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
PLOT_DATA = OUTPUT_ROOT / "plot_data"

LEARNER_ORDER = ["XGBoost", "RandomForest", "ExtraTrees", "HistGradientBoosting"]
LAYER_ORDER = ["2D-Min", "2D-Elec", "Sigma-Solv", "CAT-Solv-47"]


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "learner" in out.columns:
        out["learner"] = pd.Categorical(out["learner"], LEARNER_ORDER, ordered=True)
    if "feature_layer" in out.columns:
        out["feature_layer"] = pd.Categorical(out["feature_layer"], LAYER_ORDER, ordered=True)
    return out.sort_values([c for c in ["learner", "feature_layer", "seed"] if c in out.columns]).reset_index(drop=True)


def load_outputs() -> dict[str, pd.DataFrame]:
    required = {
        "overall": "overall_seed_metrics.csv",
        "aggregate": "aggregate_metrics.csv",
        "delta": "descriptor_delta_summary.csv",
        "per_solvent": "per_solvent_seed_metrics.csv",
        "ood": "ood_solvents.csv",
    }
    missing = [name for name in required.values() if not (OUTPUT_ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected experiment outputs: {missing}")
    return {key: pd.read_csv(OUTPUT_ROOT / filename) for key, filename in required.items()}


def validate(overall: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    expected_rows = len(LEARNER_ORDER) * len(LAYER_ORDER) * 20
    if len(overall) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} overall rows, found {len(overall)}")
    counts = aggregate.set_index("feature_layer")["feature_count"].to_dict()
    expected_counts = {"2D-Min": 25, "2D-Elec": 33, "Sigma-Solv": 47, "CAT-Solv-47": 47}
    for layer, expected in expected_counts.items():
        actual = int(counts[layer])
        if actual != expected:
            raise RuntimeError(f"{layer} feature count should be {expected}, found {actual}")
    xgb_min = aggregate[(aggregate["learner"].eq("XGBoost")) & (aggregate["feature_layer"].eq("2D-Min"))]["R2_mean"].iloc[0]
    if not 0.52 <= xgb_min <= 0.56:
        raise RuntimeError(f"XGBoost 2D-Min R2_mean={xgb_min:.4f}; expected near 0.541")


def write_plot_data(data: dict[str, pd.DataFrame]) -> None:
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    overall = ordered(data["overall"])
    aggregate = ordered(data["aggregate"])
    delta = data["delta"].copy()
    per_solvent = ordered(data["per_solvent"])

    aggregate.to_csv(PLOT_DATA / "aggregate_metrics_ordered.csv", index=False)
    overall.to_csv(PLOT_DATA / "seed_distribution_source.csv", index=False)

    r2_mean = aggregate.pivot(index="learner", columns="feature_layer", values="R2_mean").reindex(LEARNER_ORDER)[LAYER_ORDER]
    r2_std = aggregate.pivot(index="learner", columns="feature_layer", values="R2_std").reindex(LEARNER_ORDER)[LAYER_ORDER]
    mae_mean = aggregate.pivot(index="learner", columns="feature_layer", values="MAE_mean").reindex(LEARNER_ORDER)[LAYER_ORDER]
    rmse_mean = aggregate.pivot(index="learner", columns="feature_layer", values="RMSE_mean").reindex(LEARNER_ORDER)[LAYER_ORDER]
    r2_mean.to_csv(PLOT_DATA / "r2_mean_wide_by_learner.csv")
    r2_std.to_csv(PLOT_DATA / "r2_std_wide_by_learner.csv")
    mae_mean.to_csv(PLOT_DATA / "mae_mean_wide_by_learner.csv")
    rmse_mean.to_csv(PLOT_DATA / "rmse_mean_wide_by_learner.csv")

    progression = aggregate[
        [
            "learner",
            "feature_layer",
            "feature_count",
            "R2_mean",
            "R2_std",
            "MAE_mean",
            "MAE_std",
            "RMSE_mean",
            "RMSE_std",
            "fit_seconds_mean",
            "predict_seconds_mean",
        ]
    ].copy()
    progression.to_csv(PLOT_DATA / "descriptor_progression_source.csv", index=False)

    r2_delta = delta[delta["metric"].eq("R2")].copy()
    r2_delta.to_csv(PLOT_DATA / "r2_descriptor_delta_source.csv", index=False)
    for candidate, reference, filename in [
        ("2D-Elec", "2D-Min", "electrostatic_gain_bar_source.csv"),
        ("Sigma-Solv", "2D-Elec", "sigma_gain_bar_source.csv"),
        ("CAT-Solv-47", "Sigma-Solv", "compact_gain_bar_source.csv"),
        ("CAT-Solv-47", "2D-Min", "full_progression_bar_source.csv"),
    ]:
        subset = r2_delta[(r2_delta["candidate"].eq(candidate)) & (r2_delta["reference"].eq(reference))].copy()
        subset["comparison"] = f"{candidate} - {reference}"
        subset.to_csv(PLOT_DATA / filename, index=False)

    solvent_agg = (
        per_solvent.groupby(["learner", "feature_layer", "Solvent_Name"], as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            MAE_mean=("MAE", "mean"),
            RMSE_mean=("RMSE", "mean"),
            n_rows=("n_rows", "first"),
        )
    )
    solvent_agg = ordered(solvent_agg)
    solvent_agg.to_csv(PLOT_DATA / "per_solvent_aggregate_source.csv", index=False)

    xgb_solvent = solvent_agg[solvent_agg["learner"].astype(str).eq("XGBoost")].copy()
    xgb_solvent.to_csv(PLOT_DATA / "xgboost_per_solvent_progression_source.csv", index=False)


def main() -> None:
    data = load_outputs()
    validate(data["overall"], data["aggregate"])
    write_plot_data(data)
    print(f"Wrote plot data to {PLOT_DATA}")


if __name__ == "__main__":
    main()
