# Week 1 Architecture

## Overview

Week 1 builds a progressive 5-day curriculum toward a **multi-LLM web summarization and content generation pipeline**. Each day introduces a new layer of capability, culminating in a **Brochure Generator** that chains multiple LLM calls and web scraping into a single business solution.

---

## Core Modules

### `scraper.py` — Utility layer

A reusable web scraping module consumed by all notebooks and `solution.py`.

| Function | Description |
|---|---|
| `fetch_website_contents(url)` | HTTP GET via `requests` → parse with BeautifulSoup → strip `<script>`, `<style>`, `<img>`, `<input>` → return title + truncated text (2000 chars) |
| `fetch_website_links(url)` | Extract all `<a href>` links from a page |

**Design decision**: Pure functions with no state; accepts URL, returns parsed data. This keeps scraping testable and swappable (e.g., community contributions replace it with Selenium/Playwright for JS-heavy pages).

### `solution.py` — CLI application

Standalone script using `scraper.py` + OpenAI-compatible client pointing at Ollama (`http://localhost:11434/v1`). Prompts for a URL and prints a snarky LLM summary.

---

## Data Flow

### Day 1: Simple Website Summarizer

```
User URL → scraper.fetch_website_contents(url)
              ↕
         Raw HTML → BeautifulSoup → cleaned text
              ↕
  [system prompt (snarky)] + [user prompt (text)] → messages[]
              ↕
  Groq client.chat.completions.create(model="openai/gpt-oss-20b")
              ↕
         Markdown summary displayed in notebook
```

### Day 5: Brochure Generator (culminating pipeline)

```
Company URL → scraper.fetch_website_contents(landing page)
                                     ↕
                        scraper.fetch_website_links(landing page)
                                     ↕
              LLM Call #1: "Which links are relevant for a brochure?"
              One-shot prompt + response_format={"type": "json_object"}
                                     ↕
              Returns JSON: {"links": [{"type": "...", "url": "..."}]}
                                     ↕
              scraper.fetch_website_contents() for each relevant link
                                     ↕
              Aggregate: landing page + all relevant page content
                                     ↕
              LLM Call #2: "Generate a professional brochure in markdown"
                                     ↕
                         Final brochure output
```

### End-of-Week Exercise: Technical Q&A Tutor

```
Technical question → Two parallel LLM calls:
  ├── GPT-4o-mini (streaming response)
  └── Llama 3.2 via Ollama (streaming response)
```

---

## Architecture Patterns

### 1. Provider Abstraction Layer

The same Chat Completions API format works across all providers:

| Provider | Client | Endpoint | Key Model |
|---|---|---|---|
| Groq | `groq.Groq()` | Cloud API | `openai/gpt-oss-20b` |
| Ollama (OpenAI compat) | `openai.OpenAI(base_url=ollama)` | `localhost:11434/v1` | `llama3.2:3b` |
| Ollama (native) | `ollama.chat()` | Local daemon | `llama3.2:3b` |
| OpenAI (direct) | `openai.OpenAI()` | Cloud API | `gpt-4o-mini` |

Messages are always `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]` — provider switching requires only changing the client import and model name.

### 2. Prompt Engineering Patterns

| Pattern | Used In | Description |
|---|---|---|
| **System persona** | Day 1, Day 2, Exercise | Sets assistant role (snarky commentator, helpful tutor) |
| **One-shot prompting** | Day 5 | Provides an example of the desired JSON output in the prompt |
| **Structured output** | Day 5 | `response_format={"type": "json_object"}` for parseable link classification |
| **Stateless design** | Day 4 | Each LLM call is independent; conversation history must be passed manually |

### 3. Environment Configuration

All secrets and endpoints loaded from `.env` via `python-dotenv`:

```
GROQ_API_KEY=...
OPENAI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Never hardcoded. Day 1 validates the key prefix (`sk-proj-`) at startup.

---

## Key Lessons (as reflected in architecture)

1. **LLM calls are stateless** — The model has no memory between calls. "Memory" is achieved by passing the full conversation history as messages.

2. **Tokenization is lossy** — `tiktoken` (OpenAI) and `transformers.AutoTokenizer` (HuggingFace) are demonstrated to show how text is split into tokens and decoded back.

3. **Chaining enables complex workflows** — Day 5 shows how two LLM calls (link selection + brochure generation) combined with web scraping produce a result no single call could.

4. **Local vs. cloud tradeoffs** — Ollama runs offline with smaller models (3B params); Groq/OpenAI provide larger models via API. The abstraction layer makes switching trivial.

---

## Community Contributions

The `community-contributions/` directory (~700+ entries) extends the core architecture with:

- **Alternative scrapers**: Selenium, Playwright for JS-rendered pages
- **Domain-specific apps**: Clinical trial analysis, restaurant finder, bedtime story generator, YouTube summarizer
- **Backend integrations**: FastAPI endpoints, API client templates
- **Multi-model comparisons**: Phi3, DeepSeek, and other models alongside Llama and GPT

---

## Directory Structure

```
week1/
├── scraper.py                 # Reusable scraping utilities
├── solution.py                # CLI summarizer using Ollama
├── day1.ipynb                 # Groq-based website summarizer
├── day1o.ipynb                # Ollama variant of Day 1
├── day2.ipynb                 # Ollama Chat Completions API
├── day2o.ipynb                # Variant of Day 2
├── day4.ipynb                 # Tokenization & statelessness
├── day4o.ipynb                # Variant of Day 4
├── day5.ipynb                 # Brochure Generator (multi-step pipeline)
├── day5o.ipynb                # Ollama variant of Day 5
├── week1 EXERCISE.ipynb       # Technical Q&A Tutor exercise
├── solutions/                 # Completed exercise solutions
│   ├── day1_with_ollama.ipynb
│   ├── day2 SOLUTION.ipynb
│   └── week1 SOLUTION.ipynb
└── community-contributions/   # Student projects and extensions
```