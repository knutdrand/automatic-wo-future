"""N-BEATS v2: Population normalization + rainfall lag features.

Improvements over baseline:
1. Population normalization: Model incidence rates instead of raw counts
2. Rainfall lag features: 1-2 month lagged rainfall for mosquito breeding cycle
3. Calibrated dispersion_scale=0.03 for ~80% coverage
"""

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from darts import TimeSeries
from darts.models import NBEATSModel
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner

log = structlog.get_logger()


class DiseaseModelConfig(BaseConfig):
    """Configuration for disease prediction model."""

    input_chunk_length: int = 12
    output_chunk_length: int = 3
    num_stacks: int = 10
    num_blocks: int = 1
    num_layers: int = 2
    layer_widths: int = 64
    n_epochs: int = 30
    batch_size: int = 32
    n_samples: int = 100
    min_dispersion: float = 1.0
    dispersion_scale: float = 0.03  # Calibrated for ~80% coverage
    population_scale: float = 100000.0  # Normalize to cases per 100k


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns to numeric types where possible."""
    numeric_cols = ["rainfall", "mean_temperature", "disease_cases", "population"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_time_period(time_str: str) -> pd.Timestamp:
    """Parse time period string, handling both single dates and week ranges."""
    if "/" in str(time_str):
        start_date = time_str.split("/")[0]
        return pd.to_datetime(start_date)
    return pd.to_datetime(time_str)


def _add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Fourier features for annual seasonality."""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        month = df.index.month
    else:
        month = pd.to_datetime(df.index).month

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["month_sin2"] = np.sin(4 * np.pi * month / 12)
    df["month_cos2"] = np.cos(4 * np.pi * month / 12)

    return df


def _add_rainfall_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged rainfall features for mosquito breeding cycle.
    
    Dengue transmission typically has 1-2 month lag after rainfall
    due to mosquito breeding and development time.
    """
    df = df.copy()
    if "rainfall" in df.columns:
        # 1-month lag (primary mosquito breeding response)
        df["rainfall_lag_1"] = df["rainfall"].shift(1)
        # 2-month lag (extended breeding cycle)
        df["rainfall_lag_2"] = df["rainfall"].shift(2)
        # Fill NaN values with column mean
        df["rainfall_lag_1"] = df["rainfall_lag_1"].bfill().fillna(df["rainfall"].mean())
        df["rainfall_lag_2"] = df["rainfall_lag_2"].bfill().fillna(df["rainfall"].mean())
    return df


def _prepare_time_series(
    df: pd.DataFrame,
    location: str,
    config: DiseaseModelConfig,
    target_col: str = "disease_cases",
    covariate_cols: list[str] | None = None,
    add_seasonality: bool = True,
    normalize_population: bool = True,
) -> tuple[TimeSeries | None, TimeSeries | None, float | None]:
    """Prepare darts TimeSeries for a single location.
    
    Returns: (target_series, covariate_series, population) 
    """
    loc_df = df[df["location"] == location].copy()
    if loc_df.empty:
        return None, None, None

    loc_df = _convert_numeric_columns(loc_df)
    loc_df = loc_df.sort_values("time_period")
    loc_df["time_period"] = loc_df["time_period"].apply(_parse_time_period)
    loc_df = loc_df.set_index("time_period")

    # Get population for denormalization later
    population = None
    if "population" in loc_df.columns and normalize_population:
        population = loc_df["population"].median()
        if population and population > 0:
            # Convert to incidence rate (cases per 100k)
            loc_df[target_col] = loc_df[target_col] / population * config.population_scale
            log.info("normalized_to_incidence", location=location, population=population)
        else:
            population = None

    # Add rainfall lag features
    loc_df = _add_rainfall_lag_features(loc_df)

    if add_seasonality:
        loc_df = _add_seasonal_features(loc_df)

    if target_col in loc_df.columns:
        loc_df[target_col] = loc_df[target_col].fillna(0)

    target_series = None
    if target_col in loc_df.columns and loc_df[target_col].notna().any():
        target_series = TimeSeries.from_dataframe(
            loc_df[[target_col]],
            value_cols=target_col,
            freq=pd.infer_freq(loc_df.index) or "MS",
        )

    covariate_series = None
    all_cov_cols = list(covariate_cols or [])
    # Add rainfall lag features to covariates
    all_cov_cols.extend(["rainfall_lag_1", "rainfall_lag_2"])
    if add_seasonality:
        all_cov_cols.extend(["month_sin", "month_cos", "month_sin2", "month_cos2"])

    if all_cov_cols:
        available_covs = [c for c in all_cov_cols if c in loc_df.columns]
        if available_covs:
            cov_df = loc_df[available_covs].copy()
            for col in available_covs:
                cov_df[col] = cov_df[col].fillna(cov_df[col].mean())
            covariate_series = TimeSeries.from_dataframe(
                cov_df,
                value_cols=available_covs,
                freq=pd.infer_freq(loc_df.index) or "MS",
            )

    return target_series, covariate_series, population


async def on_train(
    config: DiseaseModelConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    """Train NBEATSModel per location with population normalization and rainfall lags."""
    df = data.to_pandas()
    locations = df["location"].unique().tolist()

    covariate_cols = ["rainfall", "mean_temperature"]
    models = {}
    training_stats = {}

    for location in locations:
        target_series, covariate_series, population = _prepare_time_series(
            df, location, config, covariate_cols=covariate_cols
        )

        if target_series is None or len(target_series) < config.input_chunk_length + config.output_chunk_length + 2:
            log.warning("insufficient_data", location=location)
            training_stats[location] = {"status": "skipped", "reason": "insufficient_data"}
            continue

        try:
            model = NBEATSModel(
                input_chunk_length=config.input_chunk_length,
                output_chunk_length=config.output_chunk_length,
                num_stacks=config.num_stacks,
                num_blocks=config.num_blocks,
                num_layers=config.num_layers,
                layer_widths=config.layer_widths,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                random_state=42,
                pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False},
            )

            model.fit(target_series, past_covariates=covariate_series)
            log.info("model_fit_complete", location=location)

            # Compute dispersion from residuals
            try:
                fitted_values = model.historical_forecasts(
                    target_series,
                    past_covariates=covariate_series,
                    start=config.input_chunk_length,
                    forecast_horizon=1,
                    stride=1,
                    retrain=False,
                    verbose=False,
                )
                fitted_vals = fitted_values.values().flatten()
                actual_vals = target_series.values().flatten()[config.input_chunk_length : config.input_chunk_length + len(fitted_vals)]

                residuals = actual_vals - fitted_vals
                mean_fitted = np.mean(fitted_vals[fitted_vals > 0]) if np.any(fitted_vals > 0) else 1.0
                var_residuals = np.var(residuals)

                if var_residuals > mean_fitted and mean_fitted > 0:
                    dispersion = (mean_fitted ** 2) / (var_residuals - mean_fitted)
                    dispersion = max(config.min_dispersion, min(dispersion, 100.0))
                else:
                    dispersion = config.min_dispersion

                log.info("dispersion_estimated", location=location, dispersion=dispersion)
            except Exception as e:
                log.warning("dispersion_estimation_failed", location=location, error=str(e))
                dispersion = config.min_dispersion

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pt")
                model.save(model_path)

                with open(model_path, "rb") as f:
                    model_bytes = f.read()
                ckpt_path = model_path + ".ckpt"
                with open(ckpt_path, "rb") as f:
                    ckpt_bytes = f.read()

            target_bytes = pickle.dumps(target_series)
            cov_bytes = pickle.dumps(covariate_series) if covariate_series else None

            models[location] = {
                "model_bytes": model_bytes,
                "ckpt_bytes": ckpt_bytes,
                "target_bytes": target_bytes,
                "cov_bytes": cov_bytes,
                "dispersion": dispersion,
                "population": population,  # Store for denormalization
            }
            training_stats[location] = {
                "status": "trained",
                "n_samples": len(target_series),
                "dispersion": dispersion,
                "population": population,
            }
            log.info("model_trained", location=location, n_samples=len(target_series), population=population)

        except Exception as e:
            log.error("training_failed", location=location, error=str(e))
            training_stats[location] = {"status": "failed", "error": str(e)}

    return {
        "models": models,
        "covariate_cols": covariate_cols,
        "training_stats": training_stats,
        "config": config.model_dump(),
    }


def _extend_covariates_with_seasonality(
    last_covariates: TimeSeries,
    n_periods: int,
) -> TimeSeries:
    """Extend covariate series into the future with seasonal features."""
    last_time = last_covariates.end_time()
    freq = last_covariates.freq

    future_times = pd.date_range(start=last_time + freq, periods=n_periods, freq=freq)

    months = future_times.month
    future_data = pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
        "month_sin2": np.sin(4 * np.pi * months / 12),
        "month_cos2": np.cos(4 * np.pi * months / 12),
    }, index=future_times)

    cov_cols = last_covariates.components.tolist()
    for col in cov_cols:
        if col not in future_data.columns:
            historical_mean = float(last_covariates[col].values().mean())
            future_data[col] = historical_mean

    future_data = future_data[cov_cols]

    future_series = TimeSeries.from_dataframe(
        future_data,
        value_cols=cov_cols,
        freq=freq,
    )

    return last_covariates.append(future_series)


async def on_predict(
    config: DiseaseModelConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Generate predictions using trained NBEATS models."""
    models = model["models"]
    covariate_cols = model["covariate_cols"]

    future_df = future.to_pandas()
    historic_df = historic.to_pandas()
    locations = future_df["location"].unique().tolist()

    results = []

    for location in locations:
        loc_future = future_df[future_df["location"] == location].copy()
        n_periods = len(loc_future)

        if location not in models:
            log.warning("no_model_for_location", location=location)
            mean_val = historic_df["disease_cases"].fillna(0).mean()
            samples = [[max(0, mean_val)] * config.n_samples for _ in range(n_periods)]
        else:
            model_bytes = models[location]["model_bytes"]
            ckpt_bytes = models[location]["ckpt_bytes"]
            population = models[location].get("population")

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pt")
                ckpt_path = model_path + ".ckpt"

                with open(model_path, "wb") as f:
                    f.write(model_bytes)
                with open(ckpt_path, "wb") as f:
                    f.write(ckpt_bytes)

                loc_model = NBEATSModel.load(model_path)

            last_target = pickle.loads(models[location]["target_bytes"])
            cov_bytes = models[location]["cov_bytes"]
            last_covariates = pickle.loads(cov_bytes) if cov_bytes else None
            dispersion = models[location].get("dispersion", config.min_dispersion)

            target_series, covariate_series, _ = _prepare_time_series(
                historic_df, location, config, covariate_cols=covariate_cols
            )

            if target_series is not None and target_series.end_time() > last_target.end_time():
                try:
                    new_portion = target_series.drop_before(last_target.end_time())
                    if len(new_portion) > 0:
                        combined_target = last_target.append(new_portion)
                    else:
                        combined_target = last_target
                except Exception:
                    combined_target = last_target
            else:
                combined_target = last_target

            if covariate_series is not None and last_covariates is not None:
                if covariate_series.end_time() > last_covariates.end_time():
                    try:
                        new_cov_portion = covariate_series.drop_before(last_covariates.end_time())
                        if len(new_cov_portion) > 0:
                            combined_covariates = last_covariates.append(new_cov_portion)
                        else:
                            combined_covariates = last_covariates
                    except Exception:
                        combined_covariates = last_covariates
                else:
                    combined_covariates = last_covariates
            else:
                combined_covariates = last_covariates

            if combined_covariates is not None:
                combined_covariates = _extend_covariates_with_seasonality(
                    combined_covariates, n_periods
                )

            try:
                predictions = loc_model.predict(
                    n=n_periods,
                    series=combined_target,
                    past_covariates=combined_covariates,
                )

                pred_values = predictions.values().flatten()

                samples = []
                for pred_val in pred_values:
                    # Predictions are in incidence rate space
                    mean_pred = max(0.1, float(pred_val))
                    
                    # Convert back to raw counts if population normalization was used
                    if population and population > 0:
                        mean_pred_counts = mean_pred * population / config.population_scale
                    else:
                        mean_pred_counts = mean_pred
                    
                    # Generate samples using negative binomial
                    r = max(0.5, dispersion * config.dispersion_scale)
                    p = r / (r + mean_pred_counts)
                    sample_vals = np.random.negative_binomial(r, p, config.n_samples).tolist()
                    samples.append(sample_vals)

            except Exception as e:
                log.error("prediction_failed", location=location, error=str(e))
                mean_val = historic_df["disease_cases"].fillna(0).mean()
                samples = [[max(0, mean_val)] * config.n_samples for _ in range(n_periods)]

        for idx, (_, row) in enumerate(loc_future.iterrows()):
            row_samples = samples[idx] if idx < len(samples) else samples[-1]
            result_row = {
                "time_period": row["time_period"],
                "location": location,
            }
            for i, sample_val in enumerate(row_samples):
                result_row[f"sample_{i}"] = float(sample_val)
            results.append(result_row)

    result_df = pd.DataFrame(results)
    log.info("predictions_made", n_predictions=len(result_df))

    return DataFrame.from_pandas(result_df)


info = MLServiceInfo(
    display_name="NBEATS v2: Population Norm + Rainfall Lag",
    version="2.0.0",
    summary="N-BEATS with population normalization and rainfall lag features",
    description="Improvements: (1) Models incidence rates instead of raw counts, (2) Includes 1-2 month lagged rainfall for mosquito breeding cycle, (3) Calibrated dispersion_scale=0.03",
    author="CHAP Team",
    author_assessed_status=AssessedStatus.yellow,
    contact_email="chap@example.com",
)

HIERARCHY = ArtifactHierarchy(
    name="disease_model",
    level_labels={0: "ml_training", 1: "ml_prediction"},
)

runner: FunctionalModelRunner[DiseaseModelConfig] = FunctionalModelRunner(
    on_train=on_train, on_predict=on_predict
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/chapkit.db")
if DATABASE_URL.startswith("sqlite") and ":///" in DATABASE_URL:
    db_path = Path(DATABASE_URL.split("///")[1])
    db_path.parent.mkdir(parents=True, exist_ok=True)

app = (
    MLServiceBuilder(
        info=info,
        config_schema=DiseaseModelConfig,
        hierarchy=HIERARCHY,
        runner=runner,
        database_url=DATABASE_URL,
    )
    .build()
)


if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main_nbeats_v2:app")
