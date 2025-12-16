# Disease Prediction Model (No Future Weather)

A CHAP-compatible spatio-temporal disease prediction model that forecasts disease cases using only historical data. Unlike models that require future weather predictions, this model uses autoregressive features and seasonal patterns that can be computed without external forecasts.

## Key Features

- **No Future Weather Required**: Predictions rely solely on historical disease case patterns
- **Fourier Seasonal Encoding**: Captures annual and semi-annual disease cycles
- **Negative Binomial Uncertainty**: Proper count data uncertainty quantification with overdispersion
- **Per-Location Models**: Trains separate RandomForest models for each geographic location
- **Direct Multi-Step Forecasting**: Predicts multiple time steps without autoregressive error accumulation

## Model Architecture

```
Input: time_period, location, disease_cases (historical)
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Feature Engineering          │
    │  - 12 autoregressive lags     │
    │  - 4 Fourier seasonal features│
    │    (sin/cos annual + semi)    │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  RandomForest Regressor       │
    │  - 100 estimators             │
    │  - max_depth = 10             │
    │  - Per-location training      │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Uncertainty Quantification   │
    │  - Learned dispersion from    │
    │    in-sample residuals        │
    │  - Negative Binomial sampling │
    │  - 100 Monte Carlo samples    │
    └───────────────────────────────┘
                    │
                    ▼
    Output: sample_0, sample_1, ..., sample_99
```

## Why No Weather Data?

Traditional disease prediction models often use weather data (rainfall, temperature) as covariates. However, this creates challenges:

1. **Future weather requires forecasts**: Weather forecasts degrade quickly beyond 7-10 days
2. **darts covariate limitations**: Even models with `past_covariates` require covariate values extending into the forecast horizon during prediction
3. **Spurious correlations**: Models may overfit to weather patterns that don't generalize

### Technical Note: darts Covariate Behavior

We investigated using historical weather during training with darts models:
- `RegressionModel` with `lags_past_covariates` requires covariates during prediction
- `BlockRNNModel`, `TCNModel`, `NBEATSModel` with `past_covariates` also need covariate values at prediction time
- Even with `output_chunk_length >= prediction_horizon`, these models require covariates extending into the forecast window

**Conclusion**: For truly weather-independent prediction with darts, we use only target series features.

This model captures weather effects implicitly through:
- **Fourier seasonal features**: Encode the annual cycle when weather-driven outbreaks typically occur
- **Autoregressive lags**: Recent case counts reflect current environmental conditions

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lags` | 12 | Number of historical periods for autoregression |
| `output_chunk_length` | 3 | Direct prediction horizon (months) |
| `n_samples` | 100 | Monte Carlo samples for uncertainty |
| `min_dispersion` | 1.0 | Minimum overdispersion factor |
| `dispersion_scale` | 0.1 | Scale for uncertainty width (lower = wider intervals) |

## Installation

```bash
cd disease-model
uv sync
```

## Usage

### Start the Model Server

```bash
uv run uvicorn main:app --port 8080
```

### Run Evaluation with chap-core

```bash
chap evaluate2 http://localhost:8080 \
    /path/to/dataset.csv \
    output.nc \
    --run-config.is-chapkit-model
```

### Export Metrics

```bash
chap export-metrics output.nc --output-file metrics.csv
```

### Generate Reports

```bash
# Interactive HTML plot
chap plot-backtest output.nc backtest.html

# Multi-page PDF report
chap generate-pdf-report output.nc report.pdf
```

## Performance

Evaluated on Southeast Asian dengue datasets:

| Country   | RMSE | MAE | CRPS | Coverage 10-90 | Coverage 25-75 |
|-----------|------|-----|------|----------------|----------------|
| Thailand  | 121  | 42  | 15.5 | 88%            | 63%            |
| Vietnam   | 270  | 143 | 47.9 | 99%            | 90%            |
| Laos      | 248  | 96  | 55.9 | 82%            | 53%            |

Target coverage for 10-90 prediction interval is 80%.

## Technical Details

### Seasonal Features

The model uses Fourier encoding to capture cyclical disease patterns:

```python
# Annual cycle (12-month period)
month_sin  = sin(2π × month / 12)
month_cos  = cos(2π × month / 12)

# Semi-annual cycle (6-month period)
month_sin2 = sin(4π × month / 12)
month_cos2 = cos(4π × month / 12)
```

This captures both:
- **Unimodal patterns**: Single annual peak (e.g., monsoon-driven outbreaks)
- **Bimodal patterns**: Two peaks per year (e.g., post-monsoon + dry season)

### Uncertainty Quantification

Disease counts exhibit overdispersion (variance > mean), so we use the Negative Binomial distribution:

```
Var(Y) = μ + μ²/r
```

Where:
- `μ` = predicted mean from RandomForest
- `r` = dispersion parameter (learned from residuals)

The dispersion is estimated per-location from in-sample prediction errors, then scaled by `dispersion_scale` to calibrate prediction intervals.

### Direct Multi-Step Prediction

Setting `output_chunk_length=3` enables direct 3-step forecasting:
- The model learns to predict 3 future values simultaneously
- Avoids error accumulation from iterative single-step predictions
- Critical for removing dependency on future covariate values

## Input Data Format

The model expects CSV data with columns:
- `time_period`: Date string (e.g., "2019-01", "2019-01-21/2019-01-27")
- `location`: Geographic unit identifier
- `disease_cases`: Target variable (non-negative integers)

**Note**: Weather columns (`rainfall`, `mean_temperature`) may be present in the data but are **not used** by this model. Weather effects are captured implicitly through seasonal features.

## Project Structure

```
disease-model/
├── main.py              # Model implementation
├── pyproject.toml       # Dependencies
├── Dockerfile           # Container build
├── compose.yml          # Docker Compose config
├── results/             # Evaluation outputs
│   ├── *_eval.nc        # NetCDF evaluation data
│   ├── *_metrics.csv    # Performance metrics
│   ├── *_backtest.html  # Interactive plots
│   └── *_report.pdf     # PDF reports
└── tests/               # Unit tests
```

## API Endpoints

- `GET /health` - Health check
- `GET /api/v1/info` - Model metadata
- `POST /api/v1/configs` - Create configuration
- `POST /api/v1/ml/$train` - Train model
- `POST /api/v1/ml/$predict` - Generate predictions

## License

MIT

## References

- [CHAP - Climate Health Analytics Platform](https://github.com/dhis2-chap)
- [chapkit - ML Service Framework](https://dhis2-chap.github.io/chapkit)
- [darts - Time Series Library](https://github.com/unit8co/darts)
