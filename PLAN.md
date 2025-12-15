# Automatic Model Development Plan

This document outlines the plan for developing a CHAP-compatible predictive model through chapkit, using the three interconnected packages: chap-core, chapkit, and chap-python-sdk.

## Goals

1. **Find missing functionality** in helper repos - identify gaps and either implement or document TODOs
2. **Develop documentation** for using the three packages together
3. **Develop a good model** through iterative improvement using evaluation feedback

## Architecture Overview

```
+------------------+     +------------------+     +------------------+
|   chap-core      |     |    chapkit       |     | chap-python-sdk  |
|------------------|     |------------------|     |------------------|
| - CLI (evaluate2,|     | - MLServiceBuilder|    | - Test datasets  |
|   export-metrics)|     | - BaseModelRunner|     | - validate_model_io|
| - Model templates|     | - FunctionalModel|     | - Assertions     |
| - Backtesting    |<--->|   Runner         |     | - Format utils   |
| - Metrics        |REST | - ShellModelRunner|     |                  |
+------------------+ API +------------------+     +------------------+
```

**Integration Flow:**
1. Create chapkit model service (FastAPI app)
2. Run the service locally
3. Use chap-core CLI with `--is-chapkit-model` flag pointing to service URL
4. Evaluate with `chap evaluate2` and `chap export-metrics`

---

## Phase 1: Project Setup and Basic Model

### 1.1 Initialize Project Structure

```bash
cd /Users/knutdr/Sources/automatic-model
uvx chapkit init disease-model --template ml
```

This creates:
- `main.py` - Service entry point with model runner
- `pyproject.toml` - Dependencies
- `compose.yml` - Docker compose for local dev

### 1.2 Create Minimal Model

Create a simple mean-baseline model that:
- Returns the mean of historical disease cases as prediction
- Generates probabilistic samples using a Poisson distribution

**Files to create:**
- `main.py` - Model service with train/predict functions
- `tests/test_model.py` - Tests using chap-python-sdk

### 1.3 Validate with chap-python-sdk

```python
from chap_python_sdk.testing import get_example_data, validate_model_io

example_data = get_example_data(country="laos", frequency="monthly")
result = await validate_model_io(runner, example_data, config)
assert result.success
```

### 1.4 Run Service and Test with chap-core

```bash
# Start chapkit service
fastapi dev main.py --port 8000

# In another terminal, run evaluation
chap evaluate2 http://localhost:8000 \
    --dataset-csv path/to/test_data.csv \
    --output-file evaluation.nc \
    --is-chapkit-model
```

---

## Phase 2: Evaluation Pipeline

### 2.1 Set Up Evaluation Dataset

Use example dataset from chap-python-sdk or chap-core:
- Export Laos monthly data to CSV format
- Create matching GeoJSON file for polygons

### 2.2 Run Backtest Evaluation

```bash
# Run evaluate2 to generate NetCDF evaluation file
chap evaluate2 http://localhost:8000 \
    --dataset-csv data/laos_monthly.csv \
    --output-file results/baseline_eval.nc \
    --backtest-params.n-periods 3 \
    --backtest-params.n-splits 7 \
    --is-chapkit-model

# Export metrics to CSV
chap export-metrics results/baseline_eval.nc --output-file results/metrics.csv
```

### 2.3 Analyze Metrics

Key metrics to track:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- CRPS (Continuous Ranked Probability Score)
- Coverage (prediction interval calibration)

---

## Phase 3: Model Improvement Iterations

### Iteration 1: Add Climate Features

Enhance model to use:
- `rainfall` - precipitation data
- `mean_temperature` - temperature data

Model approach: Linear regression with climate features

### Iteration 2: Add Seasonality

- Detect seasonal patterns in disease cases
- Add month-of-year features
- Consider sine/cosine encoding for cyclical features

### Iteration 3: Add Spatial Features (if geo available)

- Use population data if available
- Consider spatial autocorrelation

### Iteration 4: Advanced Models

Consider more sophisticated approaches:
- Random Forest / XGBoost for non-linear relationships
- Time series models (SARIMA, Prophet)
- Neural networks if sufficient data

---

## Phase 4: Documentation Development

### 4.1 Getting Started Guide

Document:
1. How to create a new chapkit model
2. How to test with chap-python-sdk
3. How to evaluate with chap-core

### 4.2 API Reference

Document the key interfaces:
- `BaseModelRunner.on_train()` signature and return type
- `BaseModelRunner.on_predict()` signature and return type
- Prediction format requirements

### 4.3 Best Practices

- Model development workflow
- Testing strategies
- Common pitfalls and solutions

---

## Phase 5: Gap Analysis and TODOs

Track missing functionality discovered during development:

### Potential Gaps to Investigate

1. **chap-python-sdk**:
   - [ ] More test datasets (different countries, frequencies)
   - [ ] CLI for running validation
   - [ ] Integration with pytest fixtures

2. **chapkit**:
   - [ ] Model versioning support
   - [ ] Hyperparameter configuration
   - [ ] Built-in feature engineering utilities

3. **chap-core**:
   - [ ] Better error messages for chapkit integration
   - [ ] Documentation for chapkit model flag
   - [ ] Example datasets export utilities

---

## File Structure

```
automatic-model/
  PLAN.md                    # This plan document
  FINDINGS.md                # Gap analysis and findings
  DOCUMENTATION.md           # Usage documentation draft
  main.py                    # Chapkit service entry point
  pyproject.toml             # Dependencies
  compose.yml                # Docker compose
  scripts/
    train.py                 # Training logic (if using ShellModelRunner)
    predict.py               # Prediction logic
  tests/
    test_model.py            # Model tests with chap-python-sdk
    conftest.py              # Test fixtures
  data/
    laos_monthly.csv         # Test dataset
    laos_monthly.geojson     # Polygons
  results/
    baseline_eval.nc         # Evaluation results
    metrics.csv              # Exported metrics
```

---

## Success Criteria

1. **Model runs end-to-end**: Train and predict through chap-core evaluation
2. **Metrics tracked**: At least MSE and CRPS computed and improving
3. **Documentation complete**: Clear guide for others to follow
4. **Gaps documented**: All missing functionality identified with TODOs
5. **Tests pass**: Model validates with chap-python-sdk

---

## Next Steps

1. Start Phase 1.1 - Initialize project with chapkit
2. Create minimal baseline model
3. Run first validation with chap-python-sdk
4. Attempt evaluation with chap-core
5. Document findings and iterate
