# `preprocessor.py` — LLM-Based Product Summarisation

Uses a language model (via **LiteLLM**) to generate concise, structured summaries of raw product text.

## Constants

| Constant | Value |
|----------|-------|
| `DEFAULT_MODEL_NAME` | `"groq/openai/gpt-oss-20b"` |
| `DEFAULT_REASONING_EFFORT` | `"low"` |

## System Prompt
Instructs the model to output a structured description with four fields:
- **Title** — rewritten short precise title
- **Category** — e.g. Electronics
- **Brand** — brand name
- **Description** — 1-sentence description
- **Details** — 1 sentence on features

Part numbers must be excluded.

## Class: `Preprocessor`

### Constructor
`Preprocessor(model_name, reasoning_effort)` — initialises token/cost counters and stores model config.

### Methods

- **`messages_for(text) -> list[dict]`** — Builds the chat message list (system prompt + user text).

- **`preprocess(text) -> str`** — Sends the text to the LLM and returns the generated summary. Tracks:
  - `total_input_tokens`
  - `total_output_tokens`
  - `total_cost` (from `response._hidden_params["response_cost"]`)

## Role
Transforms verbose, noisy product descriptions into **clean, structured summaries** that are fed into the DNN for price prediction. This step dramatically reduces input dimensionality and improves model generalisation.
