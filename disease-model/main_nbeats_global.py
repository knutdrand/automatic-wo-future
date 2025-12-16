"""Spatio-temporal disease prediction model using darts with GLOBAL model.

This variant trains a SINGLE N-BEATS model on ALL locations together,
with location encoded as a categorical covariate (one-hot encoding).

This allows the model to learn cross-location patterns and share parameters
across all time series.

Comparison:
- main_nbeats.py: Independent model per location (no parameter sharing)
- main_nbeats_global.py: Global model with location as covariate (this file)
"""

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from darts import TimeSeries, concatenate
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

    input_chunk_length: int = 12  # Look-back window for NBEATS
    output_chunk_length: int = 3  # Direct 3-step prediction
    num_stacks: int = 10  # Number of stacks in NBEATS
    num_blocks: int = 1  # Number of blocks per stack
    num_layers: int = 2  # Number of fully-connected layers per block
    layer_widths: int = 64  # Width of each layer
    n_epochs: int = 30  # Training epochs
    batch_size: int = 32  # Training batch size
    n_samples: int = 100  # Monte Carlo samples for uncertainty
    min_dispersion: float = 1.0  # Minimum overdispersion factor
    dispersion_scale: float = 0.1  # Scale factor for dispersion


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


def _prepare_all_time_series(
    df: pd.DataFrame,
    locations: list[str],
    target_col: str = "disease_cases",
    covariate_cols: list[str] | None = None,
) -> tuple[list[TimeSeries], list[TimeSeries], dict[str, int]]:
    """Prepare darts TimeSeries for ALL locations with one-hot location encoding.

    Returns:
        target_series_list: List of target TimeSeries (one per location)
        covariate_series_list: List of covariate TimeSeries with location encoding
        location_to_idx: Mapping from location name to one-hot index
    """
    # Create location to index mapping for one-hot encoding
    location_to_idx = {loc: idx for idx, loc in enumerate(sorted(locations))}
    n_locations = len(locations)

    target_series_list = []
    covariate_series_list = []

    for location in locations:
        loc_df = df[df["location"] == location].copy()
        if loc_df.empty:
            continue

        loc_df = _convert_numeric_columns(loc_df)
        loc_df = loc_df.sort_values("time_period")
        loc_df["time_period"] = loc_df["time_period"].apply(_parse_time_period)
        loc_df = loc_df.set_index("time_period")

        # Add seasonal features
        loc_df = _add_seasonal_features(loc_df)

        # Handle missing values in target
        if target_col in loc_df.columns:
            loc_df[target_col] = loc_df[target_col].fillna(0)

        # Create target series
        if target_col not in loc_df.columns or not loc_df[target_col].notna().any():
            continue

        target_series = TimeSeries.from_dataframe(
            loc_df[[target_col]],
            value_cols=target_col,
            freq=pd.infer_freq(loc_df.index) or "MS",
        )

        # Build covariate dataframe with location one-hot encoding
        cov_df = loc_df.copy()

        # Add one-hot encoded location columns
        for loc_name, loc_idx in location_to_idx.items():
            cov_df[f"loc_{loc_idx}"] = 1.0 if loc_name == location else 0.0

        # Select covariate columns
        all_cov_cols = list(covariate_cols or [])
        all_cov_cols.extend(["month_sin", "month_cos", "month_sin2", "month_cos2"])
        all_cov_cols.extend([f"loc_{i}" for i in range(n_locations)])

        available_covs = [c for c in all_cov_cols if c in cov_df.columns]
        for col in available_covs:
            if col not in [f"loc_{i}" for i in range(n_locations)]:
                cov_df[col] = cov_df[col].fillna(cov_df[col].mean())

        covariate_series = TimeSeries.from_dataframe(
            cov_df[available_covs],
            value_cols=available_covs,
            freq=pd.infer_freq(loc_df.index) or "MS",
        )

        target_series_list.append(target_series)
        covariate_series_list.append(covariate_series)

    return target_series_list, covariate_series_list, location_to_idx


def _prepare_single_location_series(
    df: pd.DataFrame,
    location: str,
    location_to_idx: dict[str, int],
    target_col: str = "disease_cases",
    covariate_cols: list[str] | None = None,
) -> tuple[TimeSeries | None, TimeSeries | None]:
    """Prepare TimeSeries for a single location with location encoding."""
    loc_df = df[df["location"] == location].copy()
    if loc_df.empty:
        return None, None

    loc_df = _convert_numeric_columns(loc_df)
    loc_df = loc_df.sort_values("time_period")
    loc_df["time_period"] = loc_df["time_period"].apply(_parse_time_period)
    loc_df = loc_df.set_index("time_period")
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

    # Build covariates with location encoding
    n_locations = len(location_to_idx)
    cov_df = loc_df.copy()

    for loc_name, loc_idx in location_to_idx.items():
        cov_df[f"loc_{loc_idx}"] = 1.0 if loc_name == location else 0.0

    all_cov_cols = list(covariate_cols or [])
    all_cov_cols.extend(["month_sin", "month_cos", "month_sin2", "month_cos2"])
    all_cov_cols.extend([f"loc_{i}" for i in range(n_locations)])

    available_covs = [c for c in all_cov_cols if c in cov_df.columns]
    for col in available_covs:
        if col not in [f"loc_{i}" for i in range(n_locations)]:
            cov_df[col] = cov_df[col].fillna(cov_df[col].mean())

    covariate_series = TimeSeries.from_dataframe(
        cov_df[available_covs],
        value_cols=available_covs,
        freq=pd.infer_freq(loc_df.index) or "MS",
    )

    return target_series, covariate_series


async def on_train(
    config: DiseaseModelConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    """Train a SINGLE global NBEATSModel on all locations with location encoding."""
    df = data.to_pandas()
    locations = sorted(df["location"].unique().tolist())

    covariate_cols = ["rainfall", "mean_temperature"]

    log.info("preparing_global_training_data", n_locations=len(locations))

    # Prepare all time series with location encoding
    target_list, covariate_list, location_to_idx = _prepare_all_time_series(
        df, locations, covariate_cols=covariate_cols
    )

    if not target_list:
        log.error("no_valid_training_data")
        return {"error": "No valid training data"}

    # Filter series that are too short
    min_length = config.input_chunk_length + config.output_chunk_length + 2
    valid_targets = []
    valid_covariates = []
    for t, c in zip(target_list, covariate_list):
        if len(t) >= min_length:
            valid_targets.append(t)
            valid_covariates.append(c)

    log.info("training_global_model",
             n_series=len(valid_targets),
             total_samples=sum(len(t) for t in valid_targets))

    try:
        # Create and train GLOBAL model on all locations
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

        # Train on multiple series (darts handles this natively)
        model.fit(valid_targets, past_covariates=valid_covariates)
        log.info("global_model_fit_complete")

        # Estimate dispersion per location using the global model
        location_dispersions = {}
        for idx, (target, covariate, location) in enumerate(zip(valid_targets, valid_covariates, locations)):
            try:
                fitted_values = model.historical_forecasts(
                    target,
                    past_covariates=covariate,
                    start=config.input_chunk_length,
                    forecast_horizon=1,
                    stride=1,
                    retrain=False,
                    verbose=False,
                )
                fitted_vals = fitted_values.values().flatten()
                actual_vals = target.values().flatten()[config.input_chunk_length:config.input_chunk_length + len(fitted_vals)]

                residuals = actual_vals - fitted_vals
                mean_fitted = np.mean(fitted_vals[fitted_vals > 0]) if np.any(fitted_vals > 0) else 1.0
                var_residuals = np.var(residuals)

                if var_residuals > mean_fitted and mean_fitted > 0:
                    dispersion = (mean_fitted ** 2) / (var_residuals - mean_fitted)
                    dispersion = max(config.min_dispersion, min(dispersion, 100.0))
                else:
                    dispersion = config.min_dispersion

                location_dispersions[location] = dispersion
            except Exception as e:
                log.warning("dispersion_estimation_failed", location=location, error=str(e))
                location_dispersions[location] = config.min_dispersion

        # Save global model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pt")
            model.save(model_path)

            with open(model_path, "rb") as f:
                model_bytes = f.read()
            with open(model_path + ".ckpt", "rb") as f:
                ckpt_bytes = f.read()

        # Save training series for each location
        location_data = {}
        for target, covariate, location in zip(valid_targets, valid_covariates, locations):
            location_data[location] = {
                "target_bytes": pickle.dumps(target),
                "cov_bytes": pickle.dumps(covariate),
            }

        return {
            "model_bytes": model_bytes,
            "ckpt_bytes": ckpt_bytes,
            "location_to_idx": location_to_idx,
            "location_data": location_data,
            "location_dispersions": location_dispersions,
            "covariate_cols": covariate_cols,
            "config": config.model_dump(),
            "model_type": "global",
        }

    except Exception as e:
        log.error("global_training_failed", error=str(e))
        return {"error": str(e)}


def _extend_covariates_with_seasonality(
    last_covariates: TimeSeries,
    n_periods: int,
) -> TimeSeries:
    """Extend covariate series into the future with seasonal features only."""
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
    """Generate predictions using the global NBEATS model."""
    if "error" in model:
        raise ValueError(f"Model training failed: {model['error']}")

    location_to_idx = model["location_to_idx"]
    location_data = model["location_data"]
    location_dispersions = model["location_dispersions"]
    covariate_cols = model["covariate_cols"]

    # Load global model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pt")
        with open(model_path, "wb") as f:
            f.write(model["model_bytes"])
        with open(model_path + ".ckpt", "wb") as f:
            f.write(model["ckpt_bytes"])
        global_model = NBEATSModel.load(model_path)

    future_df = future.to_pandas()
    historic_df = historic.to_pandas()
    locations = future_df["location"].unique().tolist()

    results = []

    for location in locations:
        loc_future = future_df[future_df["location"] == location].copy()
        n_periods = len(loc_future)

        if location not in location_data:
            log.warning("no_data_for_location", location=location)
            mean_val = historic_df["disease_cases"].fillna(0).mean()
            samples = [[max(0, mean_val)] * config.n_samples for _ in range(n_periods)]
        else:
            # Load stored series for this location
            last_target = pickle.loads(location_data[location]["target_bytes"])
            last_covariates = pickle.loads(location_data[location]["cov_bytes"])
            dispersion = location_dispersions.get(location, config.min_dispersion)

            # Update with new historic data
            target_series, covariate_series = _prepare_single_location_series(
                historic_df, location, location_to_idx, covariate_cols=covariate_cols
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

            if covariate_series is not None and covariate_series.end_time() > last_covariates.end_time():
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

            # Extend covariates for prediction
            combined_covariates = _extend_covariates_with_seasonality(
                combined_covariates, n_periods
            )

            try:
                predictions = global_model.predict(
                    n=n_periods,
                    series=combined_target,
                    past_covariates=combined_covariates,
                )

                pred_values = predictions.values().flatten()

                samples = []
                for pred_val in pred_values:
                    mean_pred = max(0.1, float(pred_val))
                    r = max(0.5, dispersion * config.dispersion_scale)
                    p = r / (r + mean_pred)
                    sample_vals = np.random.negative_binomial(r, p, config.n_samples).tolist()
                    samples.append(sample_vals)

            except Exception as e:
                log.error("prediction_failed", location=location, error=str(e))
                mean_val = historic_df["disease_cases"].fillna(0).mean()
                samples = [[max(0, mean_val)] * config.n_samples for _ in range(n_periods)]

        # Build result rows
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


# Service metadata
info = MLServiceInfo(
    display_name="NBEATS Disease Model (GLOBAL - Location Encoded)",
    version="1.0.0",
    summary="Global spatio-temporal disease prediction using NBEATS with location as categorical covariate",
    description="Uses a SINGLE NBEATSModel trained on ALL locations with one-hot encoded location covariates. This allows cross-location learning and parameter sharing.",
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

    run_app("main_nbeats_global:app")
