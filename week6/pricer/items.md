# `items.py` — Product Item Data Model

Defines the **`Item`** Pydantic `BaseModel` — the core data structure representing a single product with a price.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `title` | `str` | Product title |
| `category` | `str` | Product category (e.g. "Electronics") |
| `price` | `float` | Actual price |
| `full` | `Optional[str]` | Raw concatenated product text (title + description + features + details) |
| `weight` | `Optional[float]` | Item weight in pounds (parsed from details) |
| `summary` | `Optional[str]` | LLM-generated concise summary |
| `prompt` | `Optional[str]` | Training prompt built from the summary + price |
| `id` | `Optional[int]` | Index / custom ID for batch processing |

## Methods

- **`make_prompt(text)`** — Constructs a prompt string: the question *"What does this cost to the nearest dollar?"* followed by the product text, then `"Price is $X.00"` with the rounded price.
- **`test_prompt()`** — Returns the prompt *without* the actual price (used at inference time to let the model predict).
- **`push_to_hub(dataset_name, train, val, test)`** — `@staticmethod` that serialises three `Item` lists into a HuggingFace `DatasetDict` (train / validation / test) and pushes to the Hub.
- **`from_hub(dataset_name)`** — `@classmethod` that loads a `DatasetDict` from the Hub and reconstructs three `list[Item]`.

## Role

Serves as the **universal data container** throughout the pipeline: parsing → preprocessing → training → evaluation.
