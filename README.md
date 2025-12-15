# automatic-model

A CHAP-compatible disease prediction model using RandomForest and time series forecasting.

## Overview

This project implements a spatio-temporal disease prediction model built with:
- **[chapkit](https://github.com/dhis2-chap/chapkit)** - ML service framework for CHAP
- **[darts](https://github.com/unit8co/darts)** - Time series forecasting library
- **scikit-learn RandomForest** - Core regression model

## Features

- RandomForest-based time series regression with 12-period lag features
- Fourier seasonal encoding (annual and semi-annual harmonics)
- Negative binomial uncertainty quantification for count data
- Supports both monthly and weekly disease case data
- Climate covariates: rainfall and temperature

## Installation

```bash
cd disease-model
uv sync
```

## Usage

### Start the model server

```bash
./scripts/start_server.sh
# or manually:
cd disease-model
uv run uvicorn main:app --port 8080
```

### Run evaluation with chap-core

```bash
chap evaluate2 http://localhost:8080 \
    /path/to/dataset.csv \
    output.nc \
    --run-config.is-chapkit-model
```

### Export metrics

```bash
chap export-metrics output.nc --output-file metrics.csv
```

## Model Architecture

```
Input: time_period, location, rainfall, mean_temperature, disease_cases
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Feature Engineering          │
    │  - 12 lags of target          │
    │  - 12 lags of covariates      │
    │  - 4 Fourier seasonal features│
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  RandomForest Regressor       │
    │  - 100 estimators             │
    │  - max_depth=10               │
    └───────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Uncertainty Quantification   │
    │  - Negative Binomial sampling │
    │  - Dispersion from residuals  │
    │  - 100 Monte Carlo samples    │
    └───────────────────────────────┘
                    │
                    ▼
    Output: sample_0, sample_1, ..., sample_99
```

## Performance

Evaluated across 10 countries with 80% prediction interval coverage target:

| Country   | RMSE | Coverage 10-90 |
|-----------|------|----------------|
| Cambodia  | 103  | 93%            |
| Argentina | 177  | 85%            |
| Colombia  | 146  | 82%            |
| Thailand  | 181  | 84%            |
| Vietnam   | 325  | 98%            |
| Mexico    | 506  | 84%            |
| Malaysia  | 919  | 86%            |
| Brazil    | 4790 | 96%            |

## License

MIT
