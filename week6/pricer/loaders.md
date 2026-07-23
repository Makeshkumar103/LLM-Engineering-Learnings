# `loaders.py` — Parallel Dataset Loader

Efficiently loads and parses Amazon Review datasets using **multi-processing**.

## Class: `ItemLoader`

### Constructor
`ItemLoader(category)` — stores the product category (e.g. `"Electronics"`).

### Methods

- **`from_datapoint(datapoint)`** — Delegates to `parser.parse(datapoint, category)`. Returns an `Item` or `None`.

- **`from_chunk(chunk)`** — Processes a chunk of datapoints (default `CHUNK_SIZE = 1000`) through `from_datapoint`, filtering out `None` results.

- **`chunk_generator()`** — Yields successive slices of the loaded HuggingFace dataset as chunked subsets.

- **`load_in_parallel(workers)`** — Uses `ProcessPoolExecutor` to map `from_chunk` across all chunks in parallel. Each worker process handles entire chunks, significantly speeding up the CPU-bound parsing work.

- **`load(workers=WORKERS)`** — Main entry point:
  1. Loads the raw dataset from HuggingFace Hub (`McAuley-Lab/Amazon-Reviews-2023` split `"full"`)
  2. Calls `load_in_parallel` with `max(cpu_count - 1, 1)` workers
  3. Prints timing stats

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CHUNK_SIZE` | 1000 | Number of datapoints per chunk |
| `WORKERS` | `max(cpu_count - 1, 1)` | Parallel worker count |
