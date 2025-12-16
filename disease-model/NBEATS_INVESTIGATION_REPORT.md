# N-BEATS Model Investigation Report

## Overview

This investigation compares different N-BEATS model configurations to determine the optimal approach for disease prediction:

1. **Location handling**: Independent time series vs. Global model with location as categorical
2. **Hyperparameters**: Model size and input window length

All experiments use 12 backtesting splits on the Thailand dengue dataset.

---

## Experimental Setup

### Models Compared

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| **Baseline** | Independent model per location | 12-month lookback, 64-wide, 2 layers |
| **Global** | Single model, location one-hot encoded | Same as baseline + location covariates |
| **Large** | Larger architecture | 128-wide, 4 layers, 15 stacks, 50 epochs |
| **Long** | Longer lookback window | 24-month lookback (vs 12) |

### Evaluation

- Dataset: Thailand dengue cases (monthly, 77 provinces)
- Backtesting: 12 splits, 3-period forecasts
- Metrics: RMSE, MAE, CRPS, Coverage (10-90%, 25-75%)

---

## Results

| Model | RMSE | MAE | CRPS | Coverage 10-90% | Coverage 25-75% |
|-------|------|-----|------|-----------------|-----------------|
| **Baseline** | **92.81** | **33.63** | **26.61** | 71.5% | 48.6% |
| **Global** | 93.77 | 34.70 | 27.03 | **96.3%** | **74.9%** |
| **Large** | 110.59 | 40.80 | 31.25 | 75.1% | 50.6% |
| **Long** | 115.40 | 36.51 | 28.80 | 67.1% | 43.9% |

---

## Key Findings

### 1. Location Handling: Independent Models Win

**Finding**: The baseline approach (independent model per location) outperforms the global model with one-hot location encoding on point accuracy metrics.

| Approach | RMSE | CRPS |
|----------|------|------|
| Independent (baseline) | 92.81 | 26.61 |
| Global (location encoded) | 93.77 | 27.03 |

**Why?**
- Disease dynamics are location-specific (different climate, population density, urbanization)
- Global model may dilute location-specific patterns when learning shared parameters
- One-hot encoding adds many sparse features (77 provinces = 77 binary columns)

**Trade-off**: Global model has much better calibration (96.3% vs 71.5% coverage), but this comes at the cost of over-wide prediction intervals.

### 2. Larger Models Hurt Performance

**Finding**: Increasing model capacity (more layers, wider networks) degraded results.

| Model | RMSE | Training Time |
|-------|------|---------------|
| Baseline (64-wide, 2 layers) | 92.81 | ~2 min/location |
| Large (128-wide, 4 layers) | 110.59 | ~4 min/location |

**Why?**
- Overfitting on relatively short time series (~20 years monthly data)
- More parameters need more data to generalize
- Disease prediction benefits from simpler, more regularized models

### 3. Longer Lookback Window Hurts Performance

**Finding**: Extending the lookback from 12 to 24 months worsened results.

| Lookback | RMSE | Coverage |
|----------|------|----------|
| 12 months | 92.81 | 71.5% |
| 24 months | 115.40 | 67.1% |

**Why?**
- Disease patterns are more influenced by recent history than distant past
- Longer windows may capture spurious correlations
- 12-month window captures one full seasonal cycle (sufficient for annual patterns)

---

## Recommendations

### For Production Use

**Use the baseline N-BEATS with independent time series approach:**
- Best point accuracy (RMSE 92.81, CRPS 26.61)
- Simpler implementation (no location encoding needed)
- Scales linearly with number of locations

### If Better Calibration is Needed

Consider the global model if uncertainty quantification is critical:
- 96.3% coverage (vs 71.5% for baseline)
- But accept ~1% degradation in point accuracy

### Hyperparameter Guidance

Stick with baseline configuration:
- `input_chunk_length`: 12 (one year)
- `layer_widths`: 64
- `num_layers`: 2
- `num_stacks`: 10
- `n_epochs`: 30

---

## Files Created

### Model Implementations
- `main_nbeats.py` - Baseline (independent models)
- `main_nbeats_global.py` - Global model with location encoding
- `main_nbeats_large.py` - Larger architecture
- `main_nbeats_long.py` - Longer lookback window

### Evaluation Results
- `results/nbeats_baseline_12splits.nc`
- `results/nbeats_global_12splits.nc`
- `results/nbeats_large_12splits.nc`
- `results/nbeats_long_12splits.nc`

### Metrics
- `results/nbeats_baseline_12splits_metrics.csv`
- `results/nbeats_global_12splits_metrics.csv`
- `results/nbeats_large_12splits_metrics.csv`
- `results/nbeats_long_12splits_metrics.csv`

---

## Conclusion

The independent time series approach with baseline hyperparameters remains the best configuration for N-BEATS on this disease prediction task. Location encoding as a categorical variable does not improve performance and larger/longer models overfit. The baseline configuration balances point accuracy with reasonable calibration.
