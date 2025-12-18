# Disease Prediction Model (No Future Weather)

A CHAP-compatible spatio-temporal disease prediction model using N-BEATS architecture. This model forecasts disease cases using only historical data, without requiring future weather predictions.

## Key Features

- **No Future Weather Required**: Predictions rely solely on historical disease case patterns
- **N-BEATS Architecture**: Neural Basis Expansion Analysis for interpretable time series forecasting
- **Fourier Seasonal Encoding**: Captures annual and semi-annual disease cycles
- **Calibrated Uncertainty**: Negative binomial distribution tuned for ~80% prediction interval coverage
- **Per-Location Models**: Trains separate models for each geographic location

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
    │  N-BEATS Model (per location) │
    │  - 10 stacks, 1 block each    │
    │  - 2 layers, width 64         │
    │  - 30 epochs training         │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Uncertainty Quantification   │
    │  - Negative Binomial sampling │
    │  - Dispersion from residuals  │
    │  - dispersion_scale = 0.03    │
    │  - 100 Monte Carlo samples    │
    └───────────────────────────────┘
                    │
                    ▼
    Output: sample_0, sample_1, ..., sample_99
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_chunk_length` | 12 | Historical periods for autoregression |
| `output_chunk_length` | 3 | Direct prediction horizon (months) |
| `n_samples` | 100 | Monte Carlo samples for uncertainty |
| `dispersion_scale` | 0.03 | Calibrated for ~80% coverage |
| `min_dispersion` | 1.0 | Minimum overdispersion factor |

## Installation

```bash
cd disease-model
uv sync
```

## Usage

### Start the Model Server

```bash
uv run uvicorn main_nbeats:app --port 8080
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

Evaluated on Southeast Asian dengue datasets (12-split backtesting):

| Country   | RMSE  | 80% Coverage | Notes |
|-----------|-------|--------------|-------|
| Thailand  | 95.8  | 83%          | Well-calibrated |
| Vietnam   | 147.0 | 70%          | Under-coverage |
| Laos      | 193.5 | 58%          | Under-coverage |

Target coverage for 10th-90th percentile prediction interval is 80%.

## Why N-BEATS?

N-BEATS was selected after extensive model comparison:

| Model | RMSE | CRPS | 80% Coverage |
|-------|------|------|--------------|
| **N-BEATS** | **32.8** | **12.6** | 67% → 83% (calibrated) |
| LSTM | 34.1 | 15.5 | 88% |
| TCN | 44.3 | 15.4 | 87% |
| Transformer | 60.8 | 23.9 | 69% |

N-BEATS achieves the best point accuracy (RMSE, CRPS). With `dispersion_scale=0.03` tuning, it also achieves good uncertainty calibration.

## Technical Details

### Why No Weather Data?

Traditional disease prediction models use weather data as covariates. However:

1. **Future weather requires forecasts**: Weather forecasts degrade quickly beyond 7-10 days
2. **darts limitations**: Even `past_covariates` models require covariate values extending into the forecast horizon
3. **Implicit encoding**: Seasonal patterns capture weather effects through Fourier features

### Seasonal Features

```python
# Annual cycle (12-month period)
month_sin  = sin(2π × month / 12)
month_cos  = cos(2π × month / 12)

# Semi-annual cycle (6-month period)
month_sin2 = sin(4π × month / 12)
month_cos2 = cos(4π × month / 12)
```

### Uncertainty Calibration

Disease counts exhibit overdispersion (variance > mean). The Negative Binomial distribution models this:

```
Var(Y) = μ + μ²/r
```

Where `r = dispersion × dispersion_scale`. Lower `dispersion_scale` → wider intervals → higher coverage.

The `dispersion_scale=0.03` was tuned to achieve ~80% coverage on Thailand data.

## Input Data Format

CSV with columns:
- `time_period`: Date string (e.g., "2019-01", "2019-01-21/2019-01-27")
- `location`: Geographic unit identifier
- `disease_cases`: Target variable (non-negative integers)

Weather columns (`rainfall`, `mean_temperature`) may be present but are not used.

## Project Structure

```
disease-model/
├── main_nbeats.py           # Production model (recommended)
├── main_nbeats_v2.py        # Experimental: population normalization
├── main_lstm.py             # Alternative: LSTM model
├── main_tcn.py              # Alternative: TCN model
├── main_transformer.py      # Alternative: Transformer model
├── pyproject.toml           # Dependencies
├── Dockerfile               # Container build
├── results/                 # Evaluation outputs
└── docs/
    └── EXPERIMENTAL_MODELS.md  # Documentation for alternatives
```

## Further Documentation

- [MODEL_COMPARISON_REPORT.md](MODEL_COMPARISON_REPORT.md) - Detailed model comparison
- [NBEATS_INVESTIGATION_REPORT.md](NBEATS_INVESTIGATION_REPORT.md) - N-BEATS tuning experiments
- [docs/EXPERIMENTAL_MODELS.md](docs/EXPERIMENTAL_MODELS.md) - Alternative model implementations

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
- [N-BEATS Paper](https://arxiv.org/abs/1905.10437)
