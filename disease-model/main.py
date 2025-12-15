"""Spatio-temporal disease prediction model using darts."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.ensemble import RandomForestRegressor
from geojson_pydantic import FeatureCollection

from chapkit import BaseConfig
from chapkit.api import AssessedStatus, MLServiceBuilder, MLServiceInfo
from chapkit.artifact import ArtifactHierarchy
from chapkit.data import DataFrame
from chapkit.ml import FunctionalModelRunner

log = structlog.get_logger()


class DiseaseModelConfig(BaseConfig):
    """Configuration for disease prediction model."""

    lags: int = 12
    lags_past_covariates: int = 12
    output_chunk_length: int = 1
    n_samples: int = 100
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
    """Train darts model per location."""
    df = data.to_pandas()
    locations = df["location"].unique().tolist()

    covariate_cols = ["rainfall", "mean_temperature"]
    models = {}
    training_stats = {}

    for location in locations:
        target_series, covariate_series = _prepare_time_series(
            df, location, covariate_cols=covariate_cols
        )

        if target_series is None or len(target_series) < config.lags + 2:
            log.warning("insufficient_data", location=location)
            training_stats[location] = {"status": "skipped", "reason": "insufficient_data"}
            continue

        try:
            model = RegressionModel(
                lags=config.lags,
                lags_past_covariates=config.lags_past_covariates if covariate_series else None,
                output_chunk_length=config.output_chunk_length,
                model=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            )

            model.fit(target_series, past_covariates=covariate_series)

            # Compute in-sample residuals to estimate overdispersion
            try:
                fitted_values = model.historical_forecasts(
                    target_series,
                    past_covariates=covariate_series,
                    start=config.lags,
                    forecast_horizon=1,
                    stride=1,
                    retrain=False,
                    verbose=False,
                )
                fitted_vals = fitted_values.values().flatten()
                actual_vals = target_series.values().flatten()[config.lags : config.lags + len(fitted_vals)]

                # Estimate dispersion: Var(Y) = mu + mu^2/r => r = mu^2 / (Var(Y) - mu)
                residuals = actual_vals - fitted_vals
                mean_fitted = np.mean(fitted_vals[fitted_vals > 0])
                var_residuals = np.var(residuals)

                if var_residuals > mean_fitted and mean_fitted > 0:
                    dispersion = (mean_fitted ** 2) / (var_residuals - mean_fitted)
                    dispersion = max(config.min_dispersion, min(dispersion, 100.0))  # Clamp
                else:
                    dispersion = config.min_dispersion

                log.info("dispersion_estimated", location=location, dispersion=dispersion)
            except Exception as e:
                log.warning("dispersion_estimation_failed", location=location, error=str(e))
                dispersion = config.min_dispersion

            models[location] = {
                "model": model,
                "last_target": target_series,
                "last_covariates": covariate_series,
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


async def on_predict(
    config: DiseaseModelConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Generate predictions using trained models."""
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
            loc_model = models[location]["model"]
            last_target = models[location]["last_target"]
            last_covariates = models[location]["last_covariates"]

            # Update target series with historic data if available
            target_series, covariate_series = _prepare_time_series(
                historic_df, location, covariate_cols=covariate_cols
            )

            if target_series is not None:
                # Concatenate with last training data
                try:
                    combined_target = last_target.append(target_series)
                except Exception:
                    combined_target = target_series
            else:
                combined_target = last_target

            # Prepare future covariates
            future_covariates = None
            if last_covariates is not None:
                _, future_cov_series = _prepare_time_series(
                    loc_future, location, covariate_cols=covariate_cols
                )
                if future_cov_series is not None:
                    if covariate_series is not None:
                        try:
                            future_covariates = covariate_series.append(future_cov_series)
                        except Exception:
                            future_covariates = future_cov_series
                    else:
                        future_covariates = future_cov_series

            try:
                # Generate predictions
                predictions = loc_model.predict(
                    n=n_periods,
                    series=combined_target,
                    past_covariates=future_covariates,
                )

                # Generate samples using prediction uncertainty
                pred_values = predictions.values().flatten()
                dispersion = models[location].get("dispersion", config.min_dispersion)

                # Create probabilistic samples using negative binomial for overdispersion
                # Negative binomial: Var = mu + mu^2/r where r is dispersion
                samples = []
                for pred_val in pred_values:
                    mean_pred = max(0.1, float(pred_val))
                    # Convert to numpy negative binomial parameters: n (successes), p (probability)
                    # For NB: mean = n*(1-p)/p, var = n*(1-p)/p^2
                    # With our parameterization: var = mu + mu^2/r
                    # So: n = r, p = r/(r+mu)
                    # Apply dispersion_scale: lower r = more variance = wider intervals
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
            # Add individual sample columns (sample_0, sample_1, etc.)
            for i, sample_val in enumerate(row_samples):
                result_row[f"sample_{i}"] = float(sample_val)
            results.append(result_row)

    result_df = pd.DataFrame(results)
    log.info("predictions_made", n_predictions=len(result_df))

    return DataFrame.from_pandas(result_df)


# Service metadata
info = MLServiceInfo(
    display_name="Darts Disease Model",
    version="1.4.0",
    summary="Spatio-temporal disease prediction using darts time series library",
    description="Uses RandomForestRegressor with climate covariates (rainfall, temperature) and Fourier seasonal features for disease case forecasting.",
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

    run_app("main:app")
