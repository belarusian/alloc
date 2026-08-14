# TICKET-012: alloc/core.py — Results Serialization and JSON Persistence

**Module:** `alloc/core.py` (serialization helpers)
**Test:** `tests/test_core.py`
**Priority:** Medium — results must be serializable for downstream analysis and model comparison.

## What to Implement

Serialization helpers and result-persistence logic that replace the ad-hoc JSON dumping scattered across the seed's `main()` and `run_portfolio_simulation()`. The seed has two distinct serialization paths: backtest results (lines ~856–865) and prediction results (lines ~934–960), each with its own numpy-to-list conversion logic. The alloc version consolidates this into reusable helpers.

### Function 1: `serialize_results(results: dict) -> dict`

Converts numpy arrays to Python lists for JSON serialization. Recursively processes dict values.

**Input:** results dict from `SimulationRunner.run()` containing numpy arrays, floats, strings
**Output:** dict with same structure but numpy arrays replaced with lists, ndarrays converted via `tolist()`

### Function 2: `save_results(results: dict, path: str, mode: str = "backtest") -> None`

Writes serialized results to JSON file. Creates parent directories if needed.

- `mode` = "backtest" → save to `{path}/backtest_results.json`
- `mode` = "predict" → save to `{path}/prediction_results.json`

### Function 3: `load_results(path: str) -> dict`

Loads JSON results file and returns dict.

### Design Improvements Over Seed

1. **Single serialization path** — seed duplicates logic in main() and run_portfolio_simulation()
2. **Type-safe conversion** — explicit handling of numpy types
3. **No print statements** — use logging for errors

### Dependencies

- `TICKET-009` — SimulationRunner must exist first

### Tests

**File:** `tests/test_core.py`

| Test | Verifies |
|---|---|
| `test_serialize_results_numpy_arrays` | numpy arrays converted to lists |
| `test_serialize_results_nested_dict` | nested dicts handled recursively |
| `test_save_results_creates_file` | file written to disk |
| `test_save_results_backtest_mode` | backtest path used |
| `test_save_results_predict_mode` | predict path used |
| `test_load_results_roundtrip` | save then load returns equivalent data |
