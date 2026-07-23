# `parser.py` — Amazon Review Dataset Parsing

Transforms raw datapoints from the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset into clean `Item` objects.

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MIN_CHARS` | 600 | Minimum cleaned text length to keep an item |
| `MIN_PRICE` | 0.5 | Minimum valid price |
| `MAX_PRICE` | 999.49 | Maximum valid price |
| `MAX_TEXT_EACH` | 3000 | Character limit for each text field |
| `MAX_TEXT_TOTAL` | 4000 | Total character limit for the combined product text |
| `REMOVALS` | list of field names (e.g. "Part Number", "Best Sellers Rank") | Fields to strip from the `details` dict |

## Functions

### `simplify(text_list) -> str`
Converts a text field (often a list) into a single-line string with excess whitespace removed, truncated to `MAX_TEXT_EACH` characters.

### `scrub(title, description, features, details) -> str`
Combines all product text fields into a single cleaned string:
1. Removes unwanted `details` keys (part numbers, ranks, etc.)
2. Concatenates title, description, features, and JSON-serialised details
3. Strips alphanumeric tokens that look like product codes (7+ chars, mix of letters & digits)
4. Truncates to `MAX_TEXT_TOTAL` characters

### `get_weight(details) -> float`
Reads the `"Item Weight"` field from details and normalises it to **pounds** (supports pounds, ounces, grams, milligrams, kilograms, hundredths-of-pounds).

### `parse(datapoint, category) -> Item | None`
Main entry point:
1. Extracts and validates the price (must be between `MIN_PRICE` and `MAX_PRICE`)
2. Scrapes title, description, features, details
3. Calls `scrub()` to build the cleaned `full` text
4. Returns an `Item` if text length ≥ `MIN_CHARS`; otherwise `None`
