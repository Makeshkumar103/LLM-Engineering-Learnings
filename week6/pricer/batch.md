# `batch.py` — Groq Batch API Submission

Handles **batch inference** via the [Groq API](https://console.groq.com) to summarise thousands of product descriptions efficiently.

## Configuration

| Constant | Value |
|----------|-------|
| `MODEL` | `"openai/gpt-oss-20b"` |
| `BATCHES_FOLDER` | `"batches"` |
| `OUTPUT_FOLDER` | `"output"` |
| `state` | `Path("batches.pkl")` — pickle file to persist batch state |

Uses the same `SYSTEM_PROMPT` as `preprocessor.py`.

## Class: `Batch`

Each `Batch` instance represents **1,000 items** (one `.jsonl` file).

### Instance Methods

| Method | Description |
|--------|-------------|
| `make_jsonl(item)` | Formats a single item into a Groq-compatible JSONL request line with `custom_id`, `model`, `messages`, and `reasoning_effort: "low"` |
| `make_file()` | Writes all items in `[start:end]` to a JSONL file |
| `send_file()` | Uploads the JSONL file to Groq via `groq.files.create()` |
| `submit_batch()` | Submits a batch job referencing the uploaded file (`groq.batches.create()`) with a 24-hour completion window |
| `is_ready()` | Polls batch status; returns `True` when completed and captures the `output_file_id` |
| `fetch_output()` | Downloads the batch output file from Groq |
| `apply_output()` | Parses the output JSONL and assigns the LLM-generated `summary` back to each `Item` by `custom_id` |

### Class Methods

| Method | Description |
|--------|-------------|
| `create(items, lite)` | Partitions items into `BATCH_SIZE`-sized `Batch` objects, stored in `cls.batches` |
| `run()` | For each batch: create file → send → submit (with `tqdm` progress) |
| `fetch()` | Polls all unfinished batches, downloads and applies outputs when ready |
| `save()` | Serialises batch state (minus items reference) to `batches.pkl` |
| `load(items)` | Deserialises batch state and re-attaches the items reference |

## Workflow
```
create() → run() → fetch() → save() / load()
```
Supports both `"lite"` and `"full"` modes, saved to separate subdirectories.
