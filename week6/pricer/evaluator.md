# `evaluator.py` — Model Evaluation & Visualisation

Tests price predictors against ground-truth data and generates diagnostic plots.

## Class: `Tester`

### Constructor
`Tester(predictor, data, title, size=200, workers=5)` — configures the evaluator with a predictor function, test data, and display title.

### Key Methods

| Method | Description |
|--------|-------------|
| `make_title(predictor)` | Generates a human-readable title from the predictor function name |
| `post_process(value)` | Cleans model output: strips `$` and commas, extracts the first number via regex |
| `color_for(error, truth)` | Classifies error severity: green (<$40 or <20%), orange (<$80 or <40%), red otherwise |
| `run_datapoint(i)` | Runs one prediction, post-processes, computes error, assigns colour |
| `run()` | Iterates over `size` datapoints using `ThreadPoolExecutor(workers)` to parallelise predictions, prints coloured error feedback inline, then calls `report()` |
| `report()` | Computes average error, MSE, and R², then renders the error trend chart and scatter plot |

### Charts

#### `chart(title)` — Predicted vs Actual Scatter
- Plotly scatter plot with colour-coded points (green/orange/red)
- Reference line `y = x` (dashed sky-blue)
- Hover tooltips show title, guess, and actual price
- Axes share the same range `[0, max_value]`

#### `error_trend_chart()` — Cumulative Average Error
- Running mean and standard deviation of absolute errors
- 95% confidence interval band (grey shading)
- Title shows final `Avg Error ± 95% CI`
- Hover shows `n`, `Avg Error`, and `±95% CI`

## Standalone Function

### `evaluate(function, data, size=200, workers=5)`
Convenience wrapper that creates a `Tester` and runs it.

## Metrics Reported
- **Average Error** ($)
- **Mean Squared Error (MSE)**
- **R² Score** (as percentage)
