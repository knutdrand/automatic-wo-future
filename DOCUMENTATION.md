# Building CHAP-Compatible Models: A Complete Guide

This guide documents how to use **chapkit**, **chap-python-sdk**, and **chap-core** together to build, test, and evaluate spatio-temporal disease prediction models.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Model Development Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. CREATE           2. TEST              3. EVALUATE            │
│  ┌──────────┐       ┌──────────────┐     ┌────────────────┐     │
│  │ chapkit  │  ───▶ │ chap-python- │ ───▶│   chap-core    │     │
│  │  init    │       │     sdk      │     │   evaluate2    │     │
│  └──────────┘       └──────────────┘     └────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Create a New Model with chapkit

### Initialize Project

```bash
cd your-project-directory
uvx chapkit init my-model --template ml
cd my-model
```

This creates:
- `main.py` - FastAPI service with model runner
- `pyproject.toml` - Dependencies
- `Dockerfile` - Container build file

### Implement Your Model

Edit `main.py` to implement:

1. **Configuration** (pydantic model):
```python
class MyModelConfig(BaseConfig):
    lags: int = 12
    n_samples: int = 100
```

2. **Training function**:
```python
async def on_train(
    config: MyModelConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    df = data.to_pandas()
    # Train your model here
    return {"model": trained_model}
```

3. **Prediction function**:
```python
async def on_predict(
    config: MyModelConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    # Generate predictions with samples
    # IMPORTANT: Return format with sample_0, sample_1, etc. columns
    return DataFrame.from_pandas(result_df)
```

### Critical: Output Format

Predictions MUST have this format:

```
time_period | location | sample_0 | sample_1 | sample_2 | ...
2019-01-21  | Bokeo    | 1.2      | 3.4      | 5.6      | ...
```

NOT this (will fail):
```
time_period | location | samples
2019-01-21  | Bokeo    | [1.2, 3.4, 5.6, ...]
```

### Handling Data Issues

#### Issue: String columns instead of numeric
```python
def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["rainfall", "mean_temperature", "disease_cases"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
```

#### Issue: Weekly date ranges
```python
def _parse_time_period(time_str: str) -> pd.Timestamp:
    if "/" in str(time_str):
        start_date = time_str.split("/")[0]
        return pd.to_datetime(start_date)
    return pd.to_datetime(time_str)
```

## Step 2: Test with chap-python-sdk

### Add to Dependencies

```toml
[dependency-groups]
dev = [
    "chap-python-sdk",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]

[tool.uv.sources]
chap-python-sdk = { path = "../chap-python-sdk", editable = true }
```

### Write Tests

```python
import pytest
from chap_python_sdk.testing import get_example_data, validate_model_io
from main import runner, MyModelConfig

class TestMyModel:
    @pytest.mark.asyncio
    async def test_model_validation(self) -> None:
        example_data = get_example_data(country="laos", frequency="monthly")
        config = MyModelConfig(lags=6, n_samples=10)

        result = await validate_model_io(runner, example_data, config)

        assert result.success, f"Validation failed: {result.errors}"
```

### Run Tests

```bash
uv run pytest tests/ -v
```

## Step 3: Evaluate with chap-core

### Start Your Model Service

```bash
uv run uvicorn main:app --port 8080
```

### Run Evaluation

```bash
chap evaluate2 http://localhost:8080 \
    --dataset-csv path/to/data.csv \
    --output-file results/evaluation.nc \
    --run-config.is-chapkit-model
```

### Export Metrics

```bash
chap export-metrics results/evaluation.nc \
    --output-file results/metrics.csv
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| RMSE | Root Mean Square Error | Lower is better |
| MAE | Mean Absolute Error | Lower is better |
| CRPS | Continuous Ranked Probability Score | Lower is better |
| Coverage 10th-90th | % predictions in 80% interval | ~80% |
| Coverage 25th-75th | % predictions in 50% interval | ~50% |

## Dataset Format

CSV files must have:

```csv
time_period,disease_cases,rainfall,mean_temperature,population,location,parent
2019-01,10,100.5,25.3,50000,Bokeo,-
```

Required columns:
- `time_period` - Date or date range
- `disease_cases` - Target variable
- `location` - Spatial identifier
- `population` - Population count (for some models)

Optional covariates:
- `rainfall` - Precipitation
- `mean_temperature` - Temperature

## Common Issues and Solutions

### "Could not convert string to numeric"
CSV data loaded as strings. Add numeric conversion in your model.

### "invalid tzoffset" error
Weekly date ranges like "2019-01-21/2019-01-27". Parse start date only.

### "All fields must be same length"
Wrong prediction format. Use `sample_0`, `sample_1`, etc. columns.

### Port already in use
```bash
lsof -ti:8080 | xargs kill -9
```

## Example Project Structure

```
my-model/
├── main.py                 # Model service
├── pyproject.toml          # Dependencies
├── Dockerfile              # Container build
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Test fixtures
│   └── test_model.py       # Model tests
├── data/
│   └── example.csv         # Test data
└── results/
    ├── evaluation.nc       # Evaluation results
    └── metrics.csv         # Exported metrics
```

## Case Study: Building v1.4.0 RandomForest Model

This section documents the complete development process for our disease prediction model.

### Iteration 1: Baseline with Linear Regression

Started with darts LinearRegressionModel:

```python
from darts.models import LinearRegressionModel

model = LinearRegressionModel(
    lags=12,
    lags_past_covariates=12,
    output_chunk_length=1,
)
model.fit(target_series, past_covariates=covariate_series)
```

**Problem**: Coverage was only ~60% (target: 80%). Uncertainty was overconfident.

### Iteration 2: Negative Binomial Uncertainty

Switched from Poisson to Negative Binomial sampling for overdispersed count data:

```python
# Estimate dispersion from training residuals
# Var(Y) = mu + mu^2/r  =>  r = mu^2 / (Var(Y) - mu)
mean_fitted = np.mean(fitted_vals[fitted_vals > 0])
var_residuals = np.var(residuals)

if var_residuals > mean_fitted and mean_fitted > 0:
    dispersion = (mean_fitted ** 2) / (var_residuals - mean_fitted)
    dispersion = max(1.0, min(dispersion, 100.0))  # Clamp

# Sample using negative binomial
r = dispersion * dispersion_scale  # Scale to widen intervals
p = r / (r + mean_pred)
samples = np.random.negative_binomial(r, p, n_samples)
```

**Result**: Coverage improved from 60% to 68%.

### Iteration 3: Fourier Seasonal Features

Added cyclical encoding of month-of-year:

```python
def _add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    month = df.index.month

    # First harmonic: annual cycle
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Second harmonic: semi-annual patterns
    df["month_sin2"] = np.sin(4 * np.pi * month / 12)
    df["month_cos2"] = np.cos(4 * np.pi * month / 12)

    return df
```

### Iteration 4: RandomForest (Major Breakthrough)

Switched from LinearRegressionModel to RegressionModel with RandomForest:

```python
from darts.models import RegressionModel
from sklearn.ensemble import RandomForestRegressor

model = RegressionModel(
    lags=12,
    lags_past_covariates=12,
    output_chunk_length=1,
    model=RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ),
)
```

**Results**:
- RMSE: 30% improvement
- Coverage 10-90: 50% → 97% (exceeds 80% target!)
- Better point predictions lead to better uncertainty calibration

### Final Model Architecture

```
Input: time_period, location, rainfall, mean_temperature
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
    │  - dispersion from residuals  │
    │  - 100 Monte Carlo samples    │
    └───────────────────────────────┘
                    │
                    ▼
    Output: sample_0, sample_1, ..., sample_99
```

---

## Using chap-core evaluate2

The `evaluate2` command runs time-series cross-validation (backtesting).

### Basic Usage

```bash
cd /path/to/chap-core

uv run chap evaluate2 http://localhost:8080 \
    /path/to/dataset.csv \
    target/output.nc \
    --run-config.is-chapkit-model
```

### Backtest Parameters

```bash
uv run chap evaluate2 http://localhost:8080 \
    /path/to/dataset.csv \
    target/output.nc \
    --backtest-params.n-periods 3 \      # prediction horizon (months)
    --backtest-params.n-splits 9 \        # number of train/test splits
    --backtest-params.stride 1 \          # step between splits
    --run-config.is-chapkit-model
```

**Weekly vs Monthly Datasets**:
- Monthly: `--backtest-params.n-periods 3` (3-month forecast)
- Weekly: `--backtest-params.n-periods 12` (12-week forecast)

### Export Metrics

```bash
uv run chap export-metrics target/output.nc \
    --output-file target/metrics.csv
```

### Metrics CSV Format

```csv
filename,model_name,model_version,rmse_aggregate,mae_aggregate,crps,ratio_within_10th_90th,ratio_within_25th_75th,test_sample_count
laos_eval.nc,http://localhost:8080,unknown,225.0,94.2,57.8,0.783,0.376,18900.0
```

---

## Evaluation Results Across 10 Countries

Using evaluate2 with n-splits=9, prediction_length=3:

| Country     | RMSE    | Coverage 10-90 | Coverage 25-75 | Status |
|-------------|---------|----------------|----------------|--------|
| Cambodia    | 103     | 93%            | 70%            | Pass   |
| Argentina   | 177     | 85%            | 59%            | Pass   |
| Colombia    | 146     | 82%            | 59%            | Pass   |
| Thailand    | 181     | 84%            | 63%            | Pass   |
| Vietnam     | 325     | 98%            | 77%            | Pass   |
| Mexico      | 506     | 84%            | 60%            | Pass   |
| Malaysia    | 919     | 86%            | 58%            | Pass   |
| Brazil      | 4790    | 96%            | 76%            | Pass   |
| Laos        | 225     | 78%            | 38%            | Review |
| Indonesia   | 538     | 67%            | 47%            | Review |

**Key Insight**: Model generalizes well across 8/10 countries with proper uncertainty calibration.

---

## Development Workflow Scripts

### Start Server Script (`scripts/start_server.sh`)

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$0")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/disease-model"

pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 2
rm -f data/chapkit.db

uv run uvicorn main:app --port 8080 &
sleep 5
curl -s http://localhost:8080/health
```

### Run Evaluation Script (`scripts/run_eval.sh`)

```bash
#!/bin/bash
set -e

CHAP_CORE_DIR="${CHAP_CORE_DIR:-/path/to/chap-core}"
DATASET="${1:-/path/to/dataset.csv}"

cd "$CHAP_CORE_DIR"
uv run chap evaluate2 http://localhost:8080 \
    "$DATASET" \
    target/eval.nc \
    --backtest-params.n-periods 3 \
    --backtest-params.n-splits 9 \
    --run-config.is-chapkit-model

uv run chap export-metrics target/eval.nc --output-file target/metrics.csv
cat target/metrics.csv
```

---

## Known Gaps and Workarounds

### GAP-001: DataFrame Numeric Conversion
**Issue**: `DataFrame.from_csv()` reads all values as strings
**Workaround**: Add `_convert_numeric_columns()` function

### GAP-002: Time Period Parsing (Monthly and Weekly)
**Issue**: Various time period formats fail to parse consistently
**Workaround**: Extract start date with `_parse_time_period()`

#### Investigation: Consistent Time Period Representation for Monthly and Weekly Data

**Goal**: Define a consistent approach for representing monthly and weekly time periods in CSV format that works across pandas, darts, and xarray.

**Current Problems**:
- Weekly interval format `2019-01-21/2019-01-27` fails: `ValueError: invalid tzoffset`
- Monthly format `2019-01` works in pandas but needs explicit handling
- Inconsistent formats between monthly and weekly datasets cause confusion

---

#### Recommended Unified Format

**Principle**: Use the **first day of the period** as a full ISO 8601 date (`YYYY-MM-DD`).

| Frequency | CSV Format | Example | pandas freq | darts freq | xarray freq |
|-----------|------------|---------|-------------|------------|-------------|
| Monthly | `YYYY-MM-01` | `2019-01-01` | `MS` | `MS` | `MS` |
| Weekly | `YYYY-MM-DD` (Monday) | `2019-01-07` | `W-MON` | `W-MON` | `W-MON` |
| Daily | `YYYY-MM-DD` | `2019-01-15` | `D` | `D` | `D` |

**Why this works:**
- All formats parse natively with `pd.to_datetime()` - no special handling needed
- Consistent pattern: always a full date representing period start
- Frequency is explicit in metadata/code, not encoded in the date string
- Works identically across all three libraries

---

#### Format Comparison Table

| Format | Monthly Example | Weekly Example | pandas | darts | xarray |
|--------|-----------------|----------------|--------|-------|--------|
| **Period start (recommended)** | `2019-01-01` | `2019-01-07` | ✅ Native | ✅ Native | ✅ Native |
| ISO truncated | `2019-01` | N/A | ✅ Native | ⚠️ Needs conversion | ⚠️ Needs conversion |
| ISO week | N/A | `2019-W02` | ⚠️ Recent pandas | ❌ Convert first | ❌ Convert first |
| ISO interval | `2019-01-01/2019-01-31` | `2019-01-07/2019-01-13` | ❌ Needs aniso8601 | ❌ Convert first | ❌ Convert first |
| Epiweek | N/A | `201902` | ❌ Needs epiweeks | ❌ Convert first | ❌ Convert first |

---

#### Monthly Data

**Recommended CSV format:**
```csv
time_period,disease_cases,location
2019-01-01,150,Bokeo
2019-02-01,142,Bokeo
2019-03-01,168,Bokeo
```

**Alternative format (also widely supported):**
```csv
time_period,disease_cases,location
2019-01,150,Bokeo
2019-02,142,Bokeo
```

The `YYYY-MM` format is valid [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) and pandas parses it natively. However, `YYYY-MM-01` is more explicit and consistent with weekly format.

**pandas handling:**
```python
# Both formats work
pd.to_datetime('2019-01')     # Timestamp('2019-01-01 00:00:00')
pd.to_datetime('2019-01-01')  # Timestamp('2019-01-01 00:00:00')

# Create PeriodIndex for month spans
pd.Period('2019-01', freq='M')  # Period('2019-01', 'M')
period.start_time  # Timestamp('2019-01-01 00:00:00')
period.end_time    # Timestamp('2019-01-31 23:59:59.999999999')
```

**darts handling:**
```python
from darts import TimeSeries

# Monthly data with MS (month start) frequency
ts = TimeSeries.from_dataframe(
    df,
    time_col='time_period',
    value_cols='disease_cases',
    freq='MS',  # Month Start
    fill_missing_dates=True
)
```

**xarray handling:**
```python
# Resample to monthly (MS = month start, M = month end)
ds.resample(time='MS').mean()  # Monthly averages, labeled at month start
ds.resample(time='M').sum()    # Monthly sums, labeled at month end
```

---

#### Weekly Data

**Recommended CSV format:**
```csv
time_period,disease_cases,location
2019-01-07,10,Bokeo
2019-01-14,15,Bokeo
2019-01-21,12,Bokeo
```

Use the **Monday** of each week as the date (or Sunday, depending on your week definition).

**pandas handling:**
```python
# Parse directly
pd.to_datetime('2019-01-07')  # Timestamp('2019-01-07 00:00:00')

# Create PeriodIndex for week spans
pd.Period('2019-01-07', freq='W-MON')  # Week ending Monday
period.start_time  # Start of week
period.end_time    # End of week

# Week anchoring options:
# W-MON: Week ends on Monday (so period is Tue-Mon)
# W-SUN: Week ends on Sunday (so period is Mon-Sun) - default
```

**darts handling:**
```python
from darts import TimeSeries

ts = TimeSeries.from_dataframe(
    df,
    time_col='time_period',
    value_cols='disease_cases',
    freq='W-MON',  # Weekly, anchored to Monday
    fill_missing_dates=True
)
```

**xarray handling:**
```python
# Weekly resampling
ds.resample(time='W-MON').mean()  # Weekly, anchored to Monday

# Access week components
ds['time.week']       # Week of year (1-53)
ds['time.dayofweek']  # Day of week (0=Monday)
```

---

#### pandas PeriodIndex Deep Dive

`PeriodIndex` represents time **spans** rather than points, making it semantically correct for aggregated data.

**Monthly periods:**
```python
# Create from various formats
pd.Period('2019-01', freq='M')      # Period('2019-01', 'M')
pd.Period('2019-01-01', freq='M')   # Period('2019-01', 'M')
pd.Period(year=2019, month=1, freq='M')  # Period('2019-01', 'M')

# Properties
period = pd.Period('2019-01', freq='M')
period.start_time    # Timestamp('2019-01-01 00:00:00')
period.end_time      # Timestamp('2019-01-31 23:59:59.999999999')
period.to_timestamp('S')  # Start: Timestamp('2019-01-01')
period.to_timestamp('E')  # End: Timestamp('2019-01-31')
```

**Weekly periods:**
```python
# From ISO week string (pandas >= 1.1)
pd.Period('2019-W04', freq='W')  # Period('2019-01-21/2019-01-27', 'W-SUN')

# From explicit week/year
pd.Period(week=4, year=2019, freq='W')

# From date with frequency
pd.Period('2019-01-21', freq='W-MON')

# Properties
period = pd.Period('2019-W04', freq='W')
period.start_time  # Timestamp('2019-01-21 00:00:00')
period.end_time    # Timestamp('2019-01-27 23:59:59.999999999')
```

**Limitations:**
- [Known issue #20818](https://github.com/pandas-dev/pandas/issues/20818): `Timestamp.to_period()` can be off-by-one for weekly frequencies
- ISO week string parsing (`2019-W04`) only works in pandas >= 1.1

---

#### xarray Handling

xarray primarily uses `datetime64[ns]` (DatetimeIndex) or `CFTimeIndex` for time coordinates.

**PeriodIndex support is limited:**
- Stored internally but displayed as integers in repr ([issue #645](https://github.com/pydata/xarray/issues/645))
- Datetime attributes like `time.month` may not work correctly with PeriodIndex ([issue #1565](https://github.com/pydata/xarray/issues/1565))

**Recommended approach:**
```python
import xarray as xr

# Monthly aggregation
ds.resample(time='MS').mean()   # Month start
ds.resample(time='M').mean()    # Month end

# Weekly aggregation
ds.resample(time='W').mean()      # Week end (Sunday)
ds.resample(time='W-MON').sum()   # Week end (Monday)

# Virtual coordinates for grouping
ds.groupby('time.month').mean()   # By month of year
ds.groupby('time.week').mean()    # By week of year
```

**For non-standard calendars:** Use `CFTimeIndex` with `cftime` library.

---

#### darts Handling

darts **does not support PeriodIndex**. Only `DatetimeIndex` and `RangeIndex` are accepted.

**Requirements:**
- Strictly monotonically increasing time index
- Well-defined frequency without gaps
- If gaps exist, use `fill_missing_dates=True`

**Monthly data:**
```python
from darts import TimeSeries

ts = TimeSeries.from_dataframe(
    df,
    time_col='time_period',
    value_cols='disease_cases',
    freq='MS',  # Month start (or 'M' for month end)
    fill_missing_dates=True
)
```

**Weekly data:**
```python
ts = TimeSeries.from_dataframe(
    df,
    time_col='time_period',
    value_cols='disease_cases',
    freq='W-MON',  # Weekly, anchored to Monday
    fill_missing_dates=True
)

# Resample between frequencies
ts_monthly = ts.resample(freq='MS')
```

**Handling gaps (workarounds):**
1. Use `fill_missing_dates=True` with explicit `freq` parameter
2. Use `extract_subseries()` to split at significant gaps
3. Pre-fill with `df.reindex()` and `ffill()` before creating TimeSeries

**Limitation:** Functions like `plot_acf()` and `check_seasonality()` don't work well with NaN-filled gaps.

---

#### Universal Parsing Function

```python
import re
import pandas as pd

def parse_time_period(df: pd.DataFrame, time_col: str = 'time_period') -> pd.DataFrame:
    """
    Parse various time period formats to DatetimeIndex.

    Handles:
    - "2019-01-01" (full date - monthly or weekly)
    - "2019-01" (ISO month)
    - "2019-01-21/2019-01-27" (ISO interval - uses start date)
    - "2019-W04" (ISO week - converts to Monday)

    Returns DataFrame with parsed datetime column.
    """
    df = df.copy()
    sample = str(df[time_col].iloc[0])

    # ISO 8601 interval (start/end) -> extract start date
    if '/' in sample:
        df[time_col] = df[time_col].str.split('/').str[0]
        df[time_col] = pd.to_datetime(df[time_col])

    # ISO week (YYYY-Www) -> convert to Monday of that week
    elif re.match(r'^\d{4}-W\d{2}$', sample):
        df[time_col] = pd.to_datetime(
            df[time_col] + '-1', format='%G-W%V-%u'
        )

    # ISO month (YYYY-MM) -> first day of month
    elif re.match(r'^\d{4}-\d{2}$', sample):
        df[time_col] = pd.to_datetime(df[time_col] + '-01')

    # Full date (YYYY-MM-DD) -> parse directly
    else:
        df[time_col] = pd.to_datetime(df[time_col])

    return df


def infer_frequency(df: pd.DataFrame, time_col: str = 'time_period') -> str:
    """
    Infer frequency from time series data.

    Returns pandas frequency string: 'MS' (monthly), 'W-MON' (weekly), 'D' (daily).
    """
    dates = pd.to_datetime(df[time_col])
    diff = dates.diff().median()

    if diff >= pd.Timedelta(days=28):
        return 'MS'  # Monthly
    elif diff >= pd.Timedelta(days=6):
        # Determine week anchor from first date's day of week
        dow = dates.iloc[0].dayofweek
        anchors = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN']
        return anchors[dow]
    else:
        return 'D'  # Daily
```

---

#### Summary: Consistent Format Recommendation

| Aspect | Monthly | Weekly |
|--------|---------|--------|
| **CSV format** | `2019-01-01` | `2019-01-07` (Monday) |
| **Alternative** | `2019-01` | - |
| **pandas freq** | `MS` (month start) | `W-MON` (week ending Monday) |
| **darts freq** | `MS` | `W-MON` |
| **xarray freq** | `MS` | `W-MON` |
| **PeriodIndex freq** | `M` | `W-MON` |

**Key principles:**
1. **Use full dates** (`YYYY-MM-DD`) for maximum compatibility
2. **Use period start** (first day of month/week) as the date value
3. **Specify frequency explicitly** in code, not encoded in the date format
4. **Be consistent** - same pattern for all temporal granularities

---

#### Sources

- [pandas Time Series Documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [pandas PeriodIndex](https://pandas.pydata.org/docs/reference/api/pandas.PeriodIndex.html)
- [pandas Period.start_time](https://pandas.pydata.org/docs/reference/api/pandas.Period.start_time.html)
- [pandas Date Offsets](https://pandas.pydata.org/docs/reference/offset_frequency.html)
- [xarray Time Series Data](https://docs.xarray.dev/en/stable/user-guide/time-series.html)
- [xarray resample](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.resample.html)
- [darts TimeSeries](https://unit8co.github.io/darts/generated_api/darts.timeseries.html)
- [darts Issue #284 - Natural gaps](https://github.com/unit8co/darts/issues/284)
- [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) - International date/time standard
- [ISO 8601 Week Date](https://en.wikipedia.org/wiki/ISO_week_date)
- [aniso8601](https://aniso8601.readthedocs.io/) - For parsing ISO 8601 intervals
- [epiweeks](https://pypi.org/project/epiweeks/) - For epidemiological week handling

### GAP-003: Prediction Output Format (FIXED)
**Issue**: List-based samples column fails validation
**Status**: FIXED in chap-python-sdk - validation now correctly handles both formats
**Details**: Models can now return predictions in either format:
- Nested format: `samples` column with lists `[[1.0, 2.0], [3.0, 4.0]]`
- Wide format: `sample_0`, `sample_1`, etc. columns

### GAP-004: chap-core/chapkit API Mismatch
**Issue**: Published chap-core expects `model_artifact_id` field
**Workaround**: Use local chap-core from source with `--is-chapkit-model`

---

## Next Steps for Model Improvement

1. **Better uncertainty quantification** - Use proper probabilistic models
2. **Add more covariates** - Population density, land use, etc.
3. **Seasonal modeling** - Explicit seasonal components
4. **Spatial modeling** - Account for spatial autocorrelation
5. **Ensemble methods** - Combine multiple models
6. **Per-region tuning** - Adjust dispersion_scale for problem datasets
7. **Try other darts models** - XGBoost, LightGBM, TFT
