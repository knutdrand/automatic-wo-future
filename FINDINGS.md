# Gap Analysis and Findings

This document tracks missing functionality, bugs, and improvement opportunities discovered while developing a CHAP-compatible model using chapkit, chap-python-sdk, and chap-core.

---

## GAP-001: chapkit DataFrame.from_csv() does not convert numeric types

**Package:** chapkit
**Severity:** High
**Status:** Identified

### Description
The `DataFrame.from_csv()` method in `chapkit/data/dataframe.py` reads all CSV values as strings. When the DataFrame is converted to pandas via `to_pandas()`, numeric columns remain as string objects, causing operations like `.mean()` to concatenate strings instead of computing numeric averages.

### Reproduction
```python
from chapkit.data import DataFrame

# Load CSV with numeric columns
df = DataFrame.from_csv("data.csv")
pdf = df.to_pandas()

# This fails - rainfall column contains strings like "430.119"
print(pdf["rainfall"].mean())  # Raises TypeError or concatenates strings
```

### Impact
- Models cannot directly use numeric features from CSV-loaded data
- Breaks integration between chap-python-sdk example data and model training

### Suggested Fix
Update `from_csv()` to infer types after reading:

```python
@classmethod
def from_csv(cls, path, ...):
    # ... existing code ...

    # Auto-convert numeric values
    converted_data = []
    for row in data:
        converted_row = []
        for val in row:
            try:
                # Try int first, then float
                converted_row.append(int(val))
            except ValueError:
                try:
                    converted_row.append(float(val))
                except ValueError:
                    converted_row.append(val)
        converted_data.append(converted_row)

    return cls(columns=columns, data=converted_data)
```

### Workaround
Convert columns explicitly in model code:
```python
df = data.to_pandas()
for col in ["rainfall", "mean_temperature", "disease_cases"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```

---

---

## GAP-002: Weekly date range format handling

**Package:** Model-specific (but affects all packages)
**Severity:** Medium
**Status:** Fixed locally

### Description
Weekly datasets from CHAP use date range formats like "2019-01-21/2019-01-27" which pandas `to_datetime()` cannot parse directly. This causes training to fail with:

```
ValueError: Parsed string "2019-01-21/2019-01-27" gives an invalid tzoffset
```

### Suggested Fix
Parse date ranges by extracting the start date:

```python
def _parse_time_period(time_str: str) -> pd.Timestamp:
    if "/" in str(time_str):
        start_date = time_str.split("/")[0]
        return pd.to_datetime(start_date)
    return pd.to_datetime(time_str)
```

### Impact
- Models must implement custom parsing for weekly data
- Should be standardized at SDK/chapkit level

---

## GAP-003: Prediction output format not documented

**Package:** chap-core / chapkit
**Severity:** High
**Status:** Fixed locally

### Description
The prediction output format expected by chap-core is not clearly documented. Models must output samples as individual columns (`sample_0`, `sample_1`, etc.) instead of a list in a single `samples` column.

### Expected Format
```
time_period | location | sample_0 | sample_1 | sample_2 | ...
2019-01-21  | Bokeo    | 1.2      | 3.4      | 5.6      | ...
```

### Incorrect Format (what models might naively return)
```
time_period | location | samples
2019-01-21  | Bokeo    | [1.2, 3.4, 5.6, ...]
```

### Error Message
```
AssertionError: All fields in a npdataclass need to be of the same length: [3, 0]
```

### Suggested Fix
1. Add clear documentation for prediction output format
2. Consider adding a helper function in chapkit/SDK to convert list format to columnar format

---

## GAP-004: API mismatch between published chap-core and chapkit

**Package:** chap-core (published version)
**Severity:** High
**Status:** Identified

### Description
The published version of chap-core (installed via `uvx --from chap-core`) expects a `model_artifact_id` field in the train response, but chapkit doesn't return this field. This causes evaluation to fail:

```
KeyError: 'model_artifact_id'
```

### Reproduction
```bash
uvx --from chap-core chap evaluate http://localhost:8080 \
    --dataset-csv data.csv \
    --is-chapkit-model
```

### Error Location
`chap_core/models/chapkit_rest_api_wrapper.py:339`:
```python
return {"job_id": result["job_id"], "model_artifact_id": result["model_artifact_id"]}
```

### Workaround
Use the local chap-core from source instead of the published version:
```bash
cd /Users/knutdr/Sources/chap-core
uv run chap evaluate http://localhost:8080 --dataset-csv data.csv --is-chapkit-model
```

### Suggested Fix
Either:
1. Update chapkit to return `model_artifact_id` in train response
2. Update chap-core to handle missing `model_artifact_id` gracefully
3. Release a new version of chap-core that's compatible with current chapkit

---

## TODOs for Helper Packages

### chapkit
- [ ] **GAP-001**: Add type inference to `DataFrame.from_csv()`
- [ ] Add optional `dtype` parameter to `from_csv()` for explicit type specification
- [ ] Consider adding a `DataFrame.convert_dtypes()` method

### chap-python-sdk
- [ ] Document that loaded example data may have string types if using `from_csv()`
- [ ] Consider pre-converting numeric columns in example data loader
- [ ] Add more test datasets (weekly, different countries)

### chap-core
- [ ] Improve error messages when chapkit model integration fails
- [ ] Add documentation for `--is-chapkit-model` flag
- [ ] Add utility to export example datasets to CSV format

---

## Log of Testing Iterations

### Iteration 1 - Initial Model Test
- **Date:** 2024-12-10
- **Issue:** Test failed with "Could not convert string to numeric"
- **Root Cause:** GAP-001 - CSV data loaded as strings
- **Solution:** Add explicit type conversion in model code

### Iteration 2 - Weekly Date Format
- **Date:** 2024-12-10
- **Issue:** Training failed with "invalid tzoffset" error
- **Root Cause:** GAP-002 - Weekly date ranges not parseable by pandas
- **Solution:** Custom parsing to extract start date from ranges

### Iteration 3 - Prediction Output Format
- **Date:** 2024-12-10
- **Issue:** "All fields must be same length: [3, 0]"
- **Root Cause:** GAP-003 - Wrong output format (list vs columns)
- **Solution:** Output `sample_0`, `sample_1`, etc. columns

### Iteration 4 - Successful Evaluation
- **Date:** 2024-12-10
- **Status:** SUCCESS
- **Metrics:**
  | Metric | Value |
  |--------|-------|
  | RMSE | 4.99 |
  | MAE | 3.48 |
  | CRPS | 2.15 |
  | Coverage (10th-90th) | 59.6% |
  | Coverage (25th-75th) | 41.2% |
  | Test samples | 35,700 |

- **Analysis:**
  - Model is functional end-to-end
  - Coverage metrics suggest overconfident uncertainty (should be ~80% and ~50%)
  - MAE of ~3.5 cases is reasonable for weekly disease counts
  - Improvement opportunities: better uncertainty quantification
