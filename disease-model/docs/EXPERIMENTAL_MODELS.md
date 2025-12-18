# Experimental Models

This document describes alternative model implementations that were explored during development. These models are kept for reference and potential future use, but the recommended production model is `main_nbeats.py`.

## Quick Reference

| File | Model | Status | Notes |
|------|-------|--------|-------|
| `main_nbeats.py` | N-BEATS (calibrated) | **Production** | Best overall performance |
| `main_nbeats_v2.py` | N-BEATS + population norm | Experimental | Better RMSE, worse coverage |
| `main_lstm.py` | LSTM (BlockRNNModel) | Alternative | Good calibration, moderate accuracy |
| `main_tcn.py` | TCN | Alternative | Good calibration, moderate accuracy |
| `main_transformer.py` | Transformer | Experimental | Under-performs on this data |
| `main_nbeats_global.py` | N-BEATS (global) | Experimental | Location encoding variant |
| `main_nbeats_large.py` | N-BEATS (large) | Experimental | More parameters |
| `main_nbeats_long.py` | N-BEATS (long input) | Experimental | 24-month input window |
| `main_nbeats_lagged.py` | N-BEATS + weather lags | Experimental | Lagged rainfall features |
| `main_nbeats_improved.py` | N-BEATS (tuned) | Experimental | Calibration experiment |

---

## N-BEATS Variants

### main_nbeats_v2.py - Population Normalization

**Features:**
- Normalizes disease cases to per-100k population incidence rates
- Adds 1-2 month rainfall lag features
- Uses calibrated dispersion_scale=0.03

**Results (12-split backtesting):**

| Country | RMSE | 80% Coverage |
|---------|------|--------------|
| Thailand | **91.8** | 75.8% |
| Vietnam | 181.3 | 65.5% |
| Laos | 208.9 | 52.4% |

**Conclusion:** Improved Thailand RMSE but worse coverage and worse performance on Vietnam/Laos. The population normalization doesn't generalize well across countries.

**Usage:**
```bash
uv run uvicorn main_nbeats_v2:app --port 8080
```

---

### main_nbeats_global.py - Global Location Encoding

**Features:**
- One-hot encodes locations instead of training separate models
- Enables cross-location learning
- Single model for all locations

**Conclusion:** Over-calibrated (96% coverage vs 80% target). The shared parameters produce overly conservative predictions.

---

### main_nbeats_large.py - Larger Architecture

**Features:**
- Increased layer widths and stack count
- More model parameters

**Conclusion:** Similar performance to baseline with longer training time. No significant benefit.

---

### main_nbeats_long.py - Extended Input Window

**Features:**
- 24-month input window (vs 12-month default)
- Captures longer-term patterns

**Conclusion:** No improvement. 12 months is sufficient for seasonal disease patterns.

---

### main_nbeats_lagged.py - Weather Lag Features

**Features:**
- Adds lagged rainfall as covariates
- Tests mosquito breeding cycle hypothesis (1-2 month lag)

**Conclusion:** Marginal improvement in some cases, but adds complexity. The seasonal Fourier features already capture most weather-related patterns implicitly.

---

## Alternative Architectures

### main_lstm.py - LSTM (BlockRNNModel)

**Architecture:**
- darts BlockRNNModel with LSTM cells
- 32 hidden units, 2 RNN layers
- 50 training epochs

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 60.2 |
| CRPS | 22.9 |
| 80% Coverage | 76.4% |

**Notes:**
- Requires special serialization (model.pt + model.pt.ckpt files)
- Good calibration but lower point accuracy than N-BEATS

---

### main_tcn.py - Temporal Convolutional Network

**Architecture:**
- darts TCNModel
- 32 filters, kernel size 3
- Dilation base 2

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 44.3 |
| CRPS | 15.4 |
| 80% Coverage | 87.1% |

**Notes:**
- Best inner-interval coverage (68.5% for 25th-75th)
- Good balance of accuracy and calibration

---

### main_transformer.py - Transformer

**Architecture:**
- darts TransformerModel
- d_model=32, 4 attention heads
- 2 encoder/decoder layers

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 60.8 |
| CRPS | 23.9 |
| 80% Coverage | 69.2% |

**Notes:**
- Under-performs compared to simpler models
- May require more data or hyperparameter tuning

---

## Calibration Experiments

### Dispersion Scale Tuning

The `dispersion_scale` parameter controls prediction interval width in the Negative Binomial uncertainty model:

| dispersion_scale | 80% Coverage | RMSE |
|------------------|--------------|------|
| 0.1 (default) | 67-72% | ~96 |
| 0.05 | ~75% | ~97 |
| **0.03** | **~83%** | ~98 |
| 0.01 | ~92% | ~99 |

**Finding:** `dispersion_scale=0.03` achieves the target 80% coverage with minimal impact on point accuracy.

---

## Technical Notes

### Darts Model Serialization

All darts Torch models (BlockRNNModel, TCNModel, NBEATSModel, TransformerModel) create two files when saved:

```python
# Saving
model.save("model.pt")
# Creates: model.pt, model.pt.ckpt

# Loading (both files must be present)
model = NBEATSModel.load("model.pt")
```

When using chapkit's pickle-based artifacts, both files must be stored:

```python
# In on_train:
with open(model_path, "rb") as f:
    model_bytes = f.read()
with open(model_path + ".ckpt", "rb") as f:
    ckpt_bytes = f.read()

# In on_predict:
with open(model_path, "wb") as f:
    f.write(model_bytes)
with open(model_path + ".ckpt", "wb") as f:
    f.write(ckpt_bytes)
model = NBEATSModel.load(model_path)
```

### Per-Location vs Global Models

All production models use **per-location training**:
- Each geographic unit gets its own model instance
- No parameter sharing between locations
- Allows location-specific disease dynamics
- Scales linearly with number of locations

Global models (one model for all locations) were tested but showed over-calibration issues.

---

## Future Directions

1. **Conformal Prediction**: Post-hoc calibration method that could improve coverage across countries
2. **Ensemble Methods**: Combining N-BEATS with TCN or LSTM
3. **Country-Specific Tuning**: Different dispersion_scale per country
4. **Temperature Features**: Adding temperature threshold features for dengue transmission

---

## Related Documentation

- [MODEL_COMPARISON_REPORT.md](../MODEL_COMPARISON_REPORT.md) - Full model comparison with metrics
- [NBEATS_INVESTIGATION_REPORT.md](../NBEATS_INVESTIGATION_REPORT.md) - N-BEATS hyperparameter tuning
