# Week 3 Architecture

## Overview

Week 3 introduces the HuggingFace Transformers ecosystem and OpenAI API internals. Days 1-4 are external Colab notebooks covering pipelines, tokenizers, and models. Day 5 has two tracks: a local token visualization tool and an external Colab for meeting minutes creation.

---

## Core Modules

### `visualizer.py` — Token-level inference visualizer

A standalone module that connects to OpenAI's API, generates text token-by-token with `logprobs`, and builds/visualizes a directed graph of token predictions and alternatives.

**`class TokenPredictor`:**
- Calls `client.chat.completions.create()` with `temperature=0`, `logprobs=True`, `top_logprobs=3`, `stream=True`
- Extracts top token + logprob + 2 alternative tokens per step
- Converts logprobs to probabilities via `math.exp()`
- Returns `List[{token, probability, alternatives}]`

**`create_token_graph()` → `networkx.DiGraph`:**
- `START` node (green) → main token nodes `t0..tN` (lightblue) → `END` (red)
- Alternative nodes (lightgray) branch from their parent
- Each node stores token text and probability

**`visualize_predictions()` → `matplotlib` figure:**
- Custom vertical layout with `spacing_y=5`
- Alternatives offset horizontally (`spacing_x=5`)

---

## Data Flow

### Local: Token Visualizer (Day 5)

```
Text prompt
  → TokenPredictor.predict_tokens()
    → OpenAI Chat Completions (streaming + logprobs)
    → List[Dict] of per-token predictions
  → create_token_graph() → networkx.DiGraph
  → visualize_predictions() → matplotlib graph
  → plt.show()
```

### External Colab: Meeting Minutes Creator (Day 5)

```
Audio file (meeting recording)
  → Google Drive mount
  → Whisper transcription (speech-to-text)
  → LLM summarization → structured meeting minutes
```

---

## Architecture Patterns

### 1. Token-Level Probing

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    logprobs=True,
    top_logprobs=3,
    stream=True,
    temperature=0
)
for chunk in response:
    top = chunk.choices[0].logprobs.content[0].top_logprobs
    # top[0] = most likely token + logprob
    # top[1:3] = alternative tokens
```

Reveals the model's probability distribution over the vocabulary at each generation step.

### 2. Graph Visualization of LLM Outputs

Token generation modeled as a directed graph using `networkx`:
- Main tokens form the sequential path
- Alternative tokens branch off with their probabilities
- Rendered with `matplotlib` for publication-quality figures

### 3. External Colab Pattern

Days 1-5 reference external Colab notebooks hosted by the instructor:

| Day | Topic | External Resource |
|---|---|---|
| 1 | Google Colab Setup | Colab fundamentals notebook |
| 2 | HuggingFace Pipelines | `pipeline("text-generation")` demo |
| 3 | Tokenizers | BPE, WordPiece, SentencePiece comparison |
| 4 | Models | `AutoModelForCausalLM`, device mapping |
| 5 | Meeting Minutes | Whisper + LLM summarization |

### 4. Diagnostic Pipeline

`Diagnose Llama Access.ipynb` follows a sequential validation flow:

```
1. Check HF_TOKEN prefix ("hf_")
2. GET huggingface.co/api/whoami-v2 → validate token
3. HEAD meta-llama/Meta-Llama-3.1-8B-Instruct → check 403
4. HF API whoami → check "read" scope
5. AutoTokenizer.from_pretrained() → test model loading
  → On failure: use Qwen 2.5/3 as fallback
```

---

## Key Concepts

1. **HuggingFace Pipeline API** — `pipeline("text-generation")` high-level abstraction
2. **Tokenizer types** — BPE (GPT-2), WordPiece (BERT), SentencePiece (T5)
3. **AutoModel classes** — `AutoModelForCausalLM`, device mapping
4. **Token-level inference** — `logprobs`, `top_logprobs`, probability distributions
5. **Model confidence** — Visible through alternative token probabilities
6. **Audio transcription** — Whisper speech-to-text
7. **HF authentication** — Token-based access to gated models

---

## Directory Structure

```
week3/
├── visualizer.py                    # Token prediction graph visualizer
├── Diagnose Llama Access.ipynb      # HuggingFace auth troubleshooting
├── QUESTHER_V3_WEEK3.md             # Reference multi-provider Q&A architecture
├── day1.ipynb                       # Google Colab setup (external Colab link)
├── day2.ipynb                       # HuggingFace pipelines (external Colab link)
├── day3.ipynb                       # Tokenizers (external Colab link)
├── day4.ipynb                       # Models (external Colab link)
├── day5.ipynb                       # Token visualizer + Meeting Minutes Creator
├── Oday1-5.ipynb                    # Ollama variants
└── community-contributions/
```