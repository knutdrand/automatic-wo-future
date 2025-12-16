"""Spatio-temporal disease prediction model using darts.

This model uses TCNModel (Temporal Convolutional Network) with historical weather as past_covariates.
TCNModel is a "block" model that predicts output_chunk_length steps at once,
meaning when n <= output_chunk_length, it doesn't need future covariate values.

Features used:
- Historical disease cases (target)
- Historical weather (rainfall, mean_temperature) as past_covariates
- Fourier seasonal features for annual/semi-annual patterns
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
from darts.models import TCNModel
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner

log = structlog.get_logger()


class DiseaseModelConfig(BaseConfig):
    """Configuration for disease prediction model."""

    input_chunk_length: int = 12  # Look-back window for TCN
    output_chunk_length: int = 3  # Direct 3-step prediction
    kernel_size: int = 3  # TCN kernel size
    num_filters: int = 32  # Number of filters per layer
    dilation_base: int = 2  # Dilation base for TCN
    n_epochs: int = 50  # Training epochs
    batch_size: int = 32  # Training batch size
    n_samples: int = 100  # Monte Carlo samples for uncertainty
    min_dispersion: float = 1.0  # Minimum overdispersion factor
    dispersion_scale: float = 0.1  # Scale factor for dispersion (lower = wider intervals)


def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns to numeric types where possible."""
    numeric_cols = ["rainfall", "mean_temperature", "disease_cases", "population"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parse_time_period(time_str: str) -> pd.Timestamp:
    """Parse time period string, handling both single dates and week ranges.

    Handles formats like:
    - "2019-01" (monthly)
    - "2019-01-21/2019-01-27" (weekly range - uses start date)
    - "2019-01-21" (daily)
    """
    if "/" in str(time_str):
        # Week range format: use start date
        start_date = time_str.split("/")[0]
        return pd.to_datetime(start_date)
    return pd.to_datetime(time_str)


def _add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Fourier features for annual seasonality.

    Uses sin/cos encoding of month-of-year to capture cyclical patterns.
    """
    df = df.copy()

    # Get month from index (should be DatetimeIndex)
    if isinstance(df.index, pd.DatetimeIndex):
        month = df.index.month
    else:
        # Fallback: try to extract from index
        month = pd.to_datetime(df.index).month

    # Annual cycle: period = 12 months
    # First harmonic captures main annual pattern
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Second harmonic for semi-annual patterns (e.g., bimodal disease peaks)
    df["month_sin2"] = np.sin(4 * np.pi * month / 12)
    df["month_cos2"] = np.cos(4 * np.pi * month / 12)

    return df


def _prepare_time_series(
    df: pd.DataFrame,
    location: str,
    target_col: str = "disease_cases",
    covariate_cols: list[str] | None = None,
    add_seasonality: bool = True,
) -> tuple[TimeSeries | None, TimeSeries | None]:
    """Prepare darts TimeSeries for a single location."""
    loc_df = df[df["location"] == location].copy()
    if loc_df.empty:
        return None, None

    # Convert string columns to numeric (workaround for GAP-001 in chapkit)
    loc_df = _convert_numeric_columns(loc_df)

    loc_df = loc_df.sort_values("time_period")
    loc_df["time_period"] = loc_df["time_period"].apply(_parse_time_period)
    loc_df = loc_df.set_index("time_period")

    # Add seasonal features
    if add_seasonality:
        loc_df = _add_seasonal_features(loc_df)

    # Handle missing values in target
    if target_col in loc_df.columns:
        loc_df[target_col] = loc_df[target_col].fillna(0)

    # Create target series if column exists and has data
    target_series = None
    if target_col in loc_df.columns and loc_df[target_col].notna().any():
        target_series = TimeSeries.from_dataframe(
            loc_df[[target_col]],
            value_cols=target_col,
            freq=pd.infer_freq(loc_df.index) or "MS",
        )

    # Create covariate series (include seasonal features)
    covariate_series = None
    all_cov_cols = list(covariate_cols or [])
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

    return target_series, covariate_series


async def on_train(
    config: DiseaseModelConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    """Train TCNModel per location with historical weather covariates."""
    df = data.to_pandas()
    locations = df["location"].unique().tolist()

    covariate_cols = ["rainfall", "mean_temperature"]
    models = {}
    training_stats = {}

    for location in locations:
        target_series, covariate_series = _prepare_time_series(
            df, location, covariate_cols=covariate_cols
        )

        if target_series is None or len(target_series) < config.input_chunk_length + config.output_chunk_length + 2:
            log.warning("insufficient_data", location=location)
            training_stats[location] = {"status": "skipped", "reason": "insufficient_data"}
            continue

        try:
            # TCNModel - supports past_covariates
            # When n <= output_chunk_length, no future covariate values are needed
            model = TCNModel(
                input_chunk_length=config.input_chunk_length,
                output_chunk_length=config.output_chunk_length,
                kernel_size=config.kernel_size,
                num_filters=config.num_filters,
                dilation_base=config.dilation_base,
                dropout=0.1,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs,
                random_state=42,
                pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False},
            )

            # Train with past_covariates (rainfall, temperature, seasonal features)
            model.fit(target_series, past_covariates=covariate_series)
            log.info("model_fit_complete", location=location)

            # Compute in-sample residuals to estimate overdispersion
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

                # Estimate dispersion: Var(Y) = mu + mu^2/r => r = mu^2 / (Var(Y) - mu)
                residuals = actual_vals - fitted_vals
                mean_fitted = np.mean(fitted_vals[fitted_vals > 0]) if np.any(fitted_vals > 0) else 1.0
                var_residuals = np.var(residuals)

                if var_residuals > mean_fitted and mean_fitted > 0:
                    dispersion = (mean_fitted ** 2) / (var_residuals - mean_fitted)
                    dispersion = max(config.min_dispersion, min(dispersion, 100.0))  # Clamp
                else:
                    dispersion = config.min_dispersion

                log.info("dispersion_estimated", location=location, dispersion=dispersion, mean_fitted=mean_fitted, var_residuals=var_residuals)
            except Exception as e:
                log.warning("dispersion_estimation_failed", location=location, error=str(e))
                dispersion = config.min_dispersion

            # Save model using darts' native serialization
            # Darts creates TWO files: .pt (model state) and .pt.ckpt (PyTorch Lightning checkpoint)
            # Both are required for proper loading
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pt")
                model.save(model_path)

                # Read both files
                with open(model_path, "rb") as f:
                    model_bytes = f.read()
                ckpt_path = model_path + ".ckpt"
                with open(ckpt_path, "rb") as f:
                    ckpt_bytes = f.read()

            # Serialize TimeSeries using pickle (they work fine)
            target_bytes = pickle.dumps(target_series)
            cov_bytes = pickle.dumps(covariate_series) if covariate_series else None

            models[location] = {
                "model_bytes": model_bytes,  # Darts .pt file
                "ckpt_bytes": ckpt_bytes,  # PyTorch Lightning .ckpt file
                "target_bytes": target_bytes,  # Pickled TimeSeries
                "cov_bytes": cov_bytes,  # Pickled TimeSeries or None
                "dispersion": dispersion,
            }
            training_stats[location] = {
                "status": "trained",
                "n_samples": len(target_series),
                "dispersion": dispersion,
            }
            log.info("model_trained", location=location, n_samples=len(target_series), dispersion=dispersion)

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
    """Extend covariate series into the future with seasonal features only.

    Since we don't have future weather, we extend with just the deterministic
    seasonal features (sin/cos of month).
    """
    # Get the last timestamp and frequency
    last_time = last_covariates.end_time()
    freq = last_covariates.freq

    # Generate future timestamps
    future_times = pd.date_range(start=last_time + freq, periods=n_periods, freq=freq)

    # Create seasonal features for future periods
    months = future_times.month
    future_data = pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
        "month_sin2": np.sin(4 * np.pi * months / 12),
        "month_cos2": np.cos(4 * np.pi * months / 12),
    }, index=future_times)

    # Add placeholder values for weather columns (use historical means or zeros)
    # BlockRNNModel uses past_covariates, so it only looks back, not forward
    # But we need to extend the series to cover the prediction horizon
    cov_cols = last_covariates.components.tolist()
    for col in cov_cols:
        if col not in future_data.columns:
            # Use mean of historical values as placeholder
            historical_mean = float(last_covariates[col].values().mean())
            future_data[col] = historical_mean

    # Reorder columns to match original
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
    """Generate predictions using trained TCN models."""
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
            # Fallback: use global mean from historic data
            log.warning("no_model_for_location", location=location)
            mean_val = historic_df["disease_cases"].fillna(0).mean()
            samples = [[max(0, mean_val)] * config.n_samples for _ in range(n_periods)]
        else:
            # Load model from bytes using darts' native deserialization
            # Must restore BOTH .pt and .pt.ckpt files to same temp directory
            model_bytes = models[location]["model_bytes"]
            ckpt_bytes = models[location]["ckpt_bytes"]

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pt")
                ckpt_path = model_path + ".ckpt"

                with open(model_path, "wb") as f:
                    f.write(model_bytes)
                with open(ckpt_path, "wb") as f:
                    f.write(ckpt_bytes)

                loc_model = TCNModel.load(model_path)

            # Load TimeSeries from pickled bytes
            last_target = pickle.loads(models[location]["target_bytes"])
            cov_bytes = models[location]["cov_bytes"]
            last_covariates = pickle.loads(cov_bytes) if cov_bytes else None
            dispersion = models[location].get("dispersion", config.min_dispersion)

            # Update target series with historic data if available
            target_series, covariate_series = _prepare_time_series(
                historic_df, location, covariate_cols=covariate_cols
            )

            # Combine training target with any new historic data
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

            # Combine covariates similarly
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

            # Extend covariates into the future with seasonal features
            # This is needed because BlockRNNModel needs covariates to cover the prediction range
            if combined_covariates is not None:
                combined_covariates = _extend_covariates_with_seasonality(
                    combined_covariates, n_periods
                )

            try:
                # Generate predictions with past_covariates
                log.info("predicting", location=location, n_periods=n_periods,
                        target_end=str(combined_target.end_time()),
                        cov_end=str(combined_covariates.end_time()) if combined_covariates else None)

                predictions = loc_model.predict(
                    n=n_periods,
                    series=combined_target,
                    past_covariates=combined_covariates,
                )

                # Generate samples using prediction uncertainty
                pred_values = predictions.values().flatten()
                log.info("prediction_values", location=location, pred_values=pred_values[:3].tolist())

                # Create probabilistic samples using negative binomial for overdispersion
                samples = []
                for pred_val in pred_values:
                    mean_pred = max(0.1, float(pred_val))
                    # Apply dispersion_scale: lower r = more variance = wider intervals
                    r = max(0.5, dispersion * config.dispersion_scale)
                    p = r / (r + mean_pred)
                    sample_vals = np.random.negative_binomial(r, p, config.n_samples).tolist()
                    samples.append(sample_vals)

                log.info("samples_generated", location=location,
                        sample_mean=np.mean(samples[0]) if samples else 0,
                        sample_std=np.std(samples[0]) if samples else 0)

            except Exception as e:
                log.error("prediction_failed", location=location, error=str(e), error_type=type(e).__name__)
                mean_val = historic_df["disease_cases"].fillna(0).mean()
                samples = [[max(0, mean_val)] * config.n_samples for _ in range(n_periods)]

        # Build result rows
        for idx, (_, row) in enumerate(loc_future.iterrows()):
            row_samples = samples[idx] if idx < len(samples) else samples[-1]
            result_row = {
                "time_period": row["time_period"],
                "location": location,
            }
            # Add individual sample columns (sample_0, sample_1, etc.)
            for i, sample_val in enumerate(row_samples):
                result_row[f"sample_{i}"] = float(sample_val)
            results.append(result_row)

    result_df = pd.DataFrame(results)
    log.info("predictions_made", n_predictions=len(result_df))

    return DataFrame.from_pandas(result_df)


# Service metadata
info = MLServiceInfo(
    display_name="TCN Disease Model (Historical Weather)",
    version="1.0.0",
    summary="Spatio-temporal disease prediction using TCN with historical weather",
    description="Uses TCNModel (Temporal Convolutional Network) with past_covariates for historical climate data (rainfall, temperature) and Fourier seasonal features. Future predictions use seasonal features only.",
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

    run_app("main_tcn:app")
