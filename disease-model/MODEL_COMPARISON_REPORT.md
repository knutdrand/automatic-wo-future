# Model Comparison Report: Past-Covariates Only Models

## Overview

This report compares four darts models that natively support `past_covariates` (historical data only) without requiring future covariate values during prediction. All models were evaluated on the Thailand dengue dataset using backtesting with 7 splits and 3-period forecasts.

## Architecture: Location Handling

All models in this repository use a **separate model per location** approach:

- **No categorical encoding**: Locations are not one-hot encoded or embedded
- **Independent training**: Each location gets its own isolated model instance
- **No cross-location learning**: Zero parameter sharing between locations
- **Location-specific uncertainty**: Dispersion parameter learned separately per location

This approach scales linearly with the number of locations but allows each location's unique disease dynamics to be captured independently.

## Key Finding: Darts Model Serialization

**Critical Discovery**: Darts Torch models (BlockRNNModel, TCNModel, NBEATSModel, TransformerModel) require special serialization handling when used with chapkit's pickle-based artifact system.

When calling `model.save(path)`, darts creates **two files**:
- `model.pt` - Model state dictionary
- `model.pt.ckpt` - PyTorch Lightning checkpoint

Both files must be saved as bytes and restored to the same directory during prediction. See `main_lstm.py:229-253` for the serialization pattern.

---

## Results Summary

### Thailand Model Comparison

| Model | RMSE | MAE | CRPS | Coverage 10-90% | Coverage 25-75% | File |
|-------|------|-----|------|-----------------|-----------------|------|
| **N-BEATS** | **32.78** | **17.12** | **12.59** | 67.0% | 47.2% | `main_nbeats.py` |
| **Original LSTM** | 34.08 | 18.86 | 15.49 | **88.1%** | 63.3% | `main.py` |
| **TCN** | 44.29 | 20.20 | 15.42 | 87.1% | **68.5%** | `main_tcn.py` |
| **LSTM v4** | 60.20 | 27.40 | 22.87 | 76.4% | 60.8% | `main_lstm.py` |
| **Transformer** | 60.76 | 28.20 | 23.93 | 69.2% | 50.3% | `main_transformer.py` |

**Best metrics highlighted in bold**

### Cross-Country Results (Original Model)

The original model (`main.py`) was evaluated across three countries:

| Country | RMSE | MAE | CRPS | Coverage 10-90% | Coverage 25-75% | Samples |
|---------|------|-----|------|-----------------|-----------------|---------|
| **Thailand** | 34.08 | 18.86 | 15.49 | 88.1% | 63.3% | 60,900 |
| **Vietnam** | 87.64 | 53.46 | 47.90 | 99.2% | 89.7% | 39,900 |
| **Laos** | 225.90 | 77.44 | 55.93 | 82.3% | 53.1% | 14,700 |

**Observations**:
- Thailand has the best overall metrics (lowest RMSE, MAE, CRPS)
- Vietnam has the widest prediction intervals (highest coverage) but higher point errors
- Laos has the highest RMSE/CRPS - likely due to smaller sample size (14,700 vs 60,900)

---

## Detailed Model Reports

### 0. Original Model (Baseline)

**File**: `main.py`

**Configuration**:
- `input_chunk_length`: 12
- `output_chunk_length`: 3
- `hidden_dim`: 32
- `n_rnn_layers`: 2
- `n_epochs`: 50

**Metrics** (Thailand):
- RMSE: 34.08
- MAE: 18.86
- CRPS: 15.49
- Coverage 10-90%: 88.1% (well-calibrated)
- Coverage 25-75%: 63.3%

**Assessment**: Second-best point accuracy and excellent uncertainty calibration. Serves as a strong baseline.

---

### 1. LSTM v4 (BlockRNNModel)

**File**: `main_lstm.py`

**Configuration**:
- `input_chunk_length`: 12
- `output_chunk_length`: 3
- `hidden_dim`: 32
- `n_rnn_layers`: 2
- `n_epochs`: 50

**Metrics**:
- RMSE: 60.20
- MAE: 27.40
- CRPS: 22.87
- Coverage 10-90%: 76.4% (slight under-coverage)
- Coverage 25-75%: 60.8%

**Assessment**: Moderate point accuracy with reasonable calibration. Uses proper model serialization.

**Note**: LSTM v1-v3 had broken uncertainty estimation (0% coverage) due to serialization issues. v4 fixed this.

---

### 2. TCN (Temporal Convolutional Network)

**File**: `main_tcn.py`

**Configuration**:
- `input_chunk_length`: 12
- `output_chunk_length`: 3
- `kernel_size`: 3
- `num_filters`: 32
- `dilation_base`: 2
- `n_epochs`: 50

**Metrics**:
- RMSE: 44.29
- MAE: 20.20
- CRPS: 15.42
- Coverage 10-90%: 87.1% (well-calibrated)
- Coverage 25-75%: 68.5% (best inner interval)

**Assessment**: Good balance of point accuracy and calibration. Third-best RMSE with excellent coverage metrics.

---

### 3. N-BEATS (Neural Basis Expansion Analysis)

**File**: `main_nbeats.py`

**Configuration**:
- `input_chunk_length`: 12
- `output_chunk_length`: 3
- `num_stacks`: 10
- `num_blocks`: 1
- `num_layers`: 2
- `layer_widths`: 64
- `n_epochs`: 30

**Metrics**:
- RMSE: 32.78 (best!)
- MAE: 17.12 (best!)
- CRPS: 12.59 (best!)
- Coverage 10-90%: 67.0% (under-coverage)
- Coverage 25-75%: 47.2% (under-coverage)

**Assessment**: Best point accuracy (RMSE, MAE) and probabilistic accuracy (CRPS). However, uncertainty intervals are too narrow (under-coverage). Consider increasing `dispersion_scale` parameter for better calibration.

---

### 4. Transformer

**File**: `main_transformer.py`

**Configuration**:
- `input_chunk_length`: 12
- `output_chunk_length`: 3
- `d_model`: 32
- `nhead`: 4
- `num_encoder_layers`: 2
- `num_decoder_layers`: 2
- `dim_feedforward`: 64
- `n_epochs`: 30

**Metrics**:
- RMSE: 60.76
- MAE: 28.20
- CRPS: 23.93
- Coverage 10-90%: 69.2% (under-coverage)
- Coverage 25-75%: 50.3%

**Assessment**: Moderate point accuracy with under-calibrated uncertainty intervals. Needs tuning for this dataset.

---

## Recommendations

### For Best Point Accuracy
**N-BEATS** is the top performer with:
- Best RMSE (32.78), MAE (17.12), and CRPS (12.59)
- However, needs `dispersion_scale` tuning for better coverage

### For Best Calibration
**Original Model** (`main.py`) or **TCN** recommended:
- Original: 88.1% coverage (target 80%), good point accuracy (RMSE 34.08)
- TCN: 87.1% coverage, best inner interval (68.5%)

### For Production Use
Consider **Original Model** as it offers:
- Second-best RMSE (34.08)
- Best uncertainty calibration (88.1% coverage)
- Simpler architecture, proven stability

### Future Improvements
1. **N-BEATS**: Increase `dispersion_scale` to widen prediction intervals (currently 67% vs target 80%)
2. **Transformer/LSTM v4**: Tune hyperparameters - currently underperforming compared to simpler models
3. **Cross-country**: Investigate Laos high error (RMSE 225.90) - may need country-specific tuning or more training data

---

## Technical Notes

### Past Covariates Only

All models use `past_covariates` (historical weather data) during training but do NOT require future weather data during prediction. This is achieved by:

1. Setting `output_chunk_length >= n_periods` for direct multi-step forecasting
2. Using darts' "block" models that predict multiple steps at once
3. Extending covariates with seasonal features (Fourier sin/cos) for the prediction horizon

### Serialization Pattern

```python
# Save (in on_train)
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "model.pt")
    model.save(model_path)

    with open(model_path, "rb") as f:
        model_bytes = f.read()
    with open(model_path + ".ckpt", "rb") as f:
        ckpt_bytes = f.read()

# Load (in on_predict)
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = os.path.join(tmpdir, "model.pt")
    with open(model_path, "wb") as f:
        f.write(model_bytes)
    with open(model_path + ".ckpt", "wb") as f:
        f.write(ckpt_bytes)

    loc_model = BlockRNNModel.load(model_path)
```

---

## Files Created

### Model Implementations
- `main.py` - Original LSTM (BlockRNNModel) baseline
- `main_lstm.py` - LSTM v4 (BlockRNNModel) with fixed serialization
- `main_tcn.py` - TCN implementation
- `main_nbeats.py` - N-BEATS implementation
- `main_transformer.py` - Transformer implementation

### Evaluation Results
- `results/thailand_metrics.csv` - Original model on Thailand
- `results/vietnam_metrics.csv` - Original model on Vietnam
- `results/laos_metrics.csv` - Original model on Laos
- `results/thailand_lstm_v4_metrics.csv`
- `results/thailand_tcn_v1_metrics.csv`
- `results/thailand_nbeats_v1_metrics.csv`
- `results/thailand_transformer_v1_metrics.csv`

---

## Glossary

- **RMSE**: Root Mean Square Error - measures point prediction accuracy (lower is better)
- **MAE**: Mean Absolute Error - measures average prediction error magnitude (lower is better)
- **CRPS**: Continuous Ranked Probability Score - measures probabilistic forecast quality (lower is better)
- **Coverage 10-90%**: Percentage of actual values within the 10th-90th percentile interval (target: 80%)
- **Coverage 25-75%**: Percentage of actual values within the 25th-75th percentile interval (target: 50%)
