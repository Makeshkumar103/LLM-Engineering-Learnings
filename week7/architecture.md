# Week 7 Architecture

## Overview

Week 7 fine-tunes Llama 3.2 (3B parameters) using QLoRA on the product price prediction task from Week 6. The core training happens in Google Colab (GPU-accelerated), while local modules handle data preparation and evaluation.

---

## Core Modules

### `pricer/items.py` — Data model

**`class Item`** (Pydantic `BaseModel`):
- Fields: `title`, `category`, `price`, `full`, `weight`, `summary`, `prompt`, `completion`
- `make_prompts(tokenizer, max_tokens=110, do_round=True)`:
  - Truncates summary to `max_tokens` tokens
  - Sets `prompt = f"What does this cost to the nearest dollar?\n\n{summary}\n\nPrice is $"`
  - Sets `completion = f"{round(price)}.00"` (rounded to integer dollar)
- `from_hub()` / `push_to_hub()` / `push_prompts_to_hub()` — HuggingFace dataset I/O

### `pricer/evaluator.py` — Multi-threaded evaluation

**`class Tester`:**
- `predictor` function, test data, `ThreadPoolExecutor` (5 workers)
- `post_process(value)` — regex price extraction from free-text LLM output
- `run()` → `report()` → scatter plot (predicted vs actual, color-coded by error) + cumulative error trend (95% CI)

### `util.py` — Single-threaded evaluation variant

Same logic as `evaluator.py` but single-threaded with `IPython.display.clear_output()` for inline notebook progress.

---

## Data Flow

### 1. Data Preparation (Day 2 — local)

```
Item.from_hub("ed-donner/items_full")
  → 800K train + 10K val + 10K test items
  → Token distribution analysis (histogram)
  → CUTOFF = 110 tokens (truncates only 5.7% of items)
  → Item.make_prompts(tokenizer, CUTOFF, do_round=True) for train/val
  → Item.make_prompts(tokenizer, CUTOFF, do_round=False) for test
  → push_prompts_to_hub("items_prompts_full")
```

### 2. Fine-Tuning (Days 3-4 — Colab)

```
Load Meta-Llama-3.2-3B in 4-bit (BitsAndBytesConfig)
  → QLoRA adapters (rank=16, alpha=16, target_modules=["q_proj", "v_proj"])
  → SFTTrainer from TRL:
      max_seq_length=512
      dataset_text_field="prompt" + "completion"
      per_device_train_batch_size=... (GPU dependent)
      gradient_accumulation_steps=...
      learning_rate=2e-4
  → Save + push adapter to HuggingFace Hub
```

### 3. Evaluation (Day 5 — Colab / local)

```
Load base model + LoRA adapter
  → For each test item:
      generate(prompt) → extract price via regex
      compute |guess - truth|
  → Aggregate: average error, MSE, R²
  → Visualize: scatter plot + cumulative error trend
```

---

## Architecture Patterns

### 1. QLoRA Fine-Tuning

**4-bit NormalFloat quantization** — Loads the model in 4-bit precision (NF4 type), reducing memory from ~16GB to ~4GB for a 3B model.

**LoRA adapters** — Rank-16 matrices injected at attention projections (`q_proj`, `v_proj`). Only ~0.1% of parameters are trainable.

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
```

### 2. Prompt Format for SFT

```
Training format:
  Prompt:  "What does this cost to the nearest dollar?\n\n{summary}\n\nPrice is $"
  Completion: "{rounded_price}.00"

Full text for SFT: prompt + completion (concatenated)
Loss computed only on completion tokens (via tokenizer's labels masking)
```

### 3. Prompt Dataset Structure

Pushed to HuggingFace Hub as a `DatasetDict`:

```
items_prompts_full/
├── train/     (800K rows: prompt, completion)
├── validation/(10K rows: prompt, completion)
└── test/      (10K rows: prompt, completion)
```

### 4. Multi-Threaded Post-Processing

```python
def post_process(value):
    numbers = re.findall(r'[\d.,]+', str(value))
    if numbers:
        return float(numbers[0].replace(',', ''))
    return 0.0
```

Extracts first number from free-text LLM output. Handles commas, dollar signs, trailing text.

---

## Results

| Model | Avg Error |
|---|---|
| Base Llama 3.2 (4-bit, no fine-tuning) | $110.72 |
| **Fine-tuned Lite** (smaller dataset) | **$65.40** |
| **Fine-tuned Full** (800K items) | **$39.85** |
| GPT 5.1 (zero-shot, Week 6 best) | $44.74 |

Fine-tuned Llama 3.2 (3B) outperforms GPT 5.1 ($39.85 vs $44.74) on the domain-specific pricing task.

---

## Key Concepts

1. **QLoRA** — 4-bit quantization + Low-Rank Adaptation for consumer GPU fine-tuning
2. **BitsAndBytes** — NF4 quantization, double quantization, bfloat16 compute
3. **SFTTrainer** (TRL) — Supervised fine-tuning with prompt/completion format
4. **Token analysis** — Histogram-based cutoff selection to minimize data loss
5. **Prompt engineering for SFT** — Structured prompt prefix + completion suffix
6. **Post-processing LLM output** — Regex extraction of structured values
7. **Open-source beats frontier** — Fine-tuned 3B model outperforms GPT 5.1 on domain task

---

## Directory Structure

```
week7/
├── pricer/
│   ├── __init__.py
│   ├── items.py              # Item data model, prompt construction, HF I/O
│   └── evaluator.py          # Multi-threaded evaluation + visualization
├── util.py                   # Single-threaded evaluation variant
├── day1.ipynb                # QLoRA theory (markdown only)
├── day2.ipynb                # Data preparation + push to Hub
├── day3 and 4.ipynb          # Training execution (markdown → Colab link)
├── day5.ipynb                # Evaluation (markdown → Colab link)
├── results.ipynb             # Final comparison chart
├── Outputs/                  # Evaluation PNGs
│   ├── lite-false-graph0.png
│   ├── lite-true-graph0.png
│   └── lite-ture-graph1.png
└── community-contributions/
```