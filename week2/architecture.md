# Week 2 Architecture

## Overview

Week 2 progresses from multi-provider LLM APIs through Gradio UI frameworks, conversational chatbots, tool/function calling, and culminates in a multi-modal Airline Assistant. Two parallel tracks exist: **Groq-based** (day*.ipynb) and **Ollama-based** (oday*.ipynb).

---

## Core Modules

### `scraper.py` — Web scraping utility

Reused from Week 1. `fetch_website_contents(url)` returns cleaned page text; `fetch_website_links(url)` returns all hyperlinks. Used in Day 2's Brochure Generator.

### `revealer.py` — SVG animation utility

Injects CSS keyframe animations into SVG elements with staggered delays. Used in `extra.ipynb` to animate LLM-generated SVGs piece by piece.

### `prices.db` — SQLite database

Stores airline ticket prices by destination. Used by Day 4 and Day 5 tool/function calling to look up real price data.

---

## Data Flow by Day

### Day 1: Frontier Model APIs

```
Unified OpenAI Python Client
  ├── base_url = api.openai.com           → GPT models
  ├── base_url = api.anthropic.com        → Claude
  ├── base_url = generativelanguageapi... → Gemini
  ├── base_url = api.groq.com             → Groq (openai/gpt-oss-20b)
  ├── base_url = api.x.ai                 → Grok
  ├── base_url = openrouter.ai            → OpenRouter (multi-model)
  ├── base_url = localhost:11434/v1       → Ollama
  └── Native clients: google.genai, anthropic.Anthropic

Abstraction layers: LangChain (langchain_openai), LiteLLM (litellm.completion)
Key concept: reasoning_effort parameter for reasoning models
```

### Day 2: Gradio UI Framework

```
gradio.Interface(fn=function, inputs=..., outputs=...)
  ├── fn = stream_generator (yield) → streaming output
  ├── inputs = gr.Textbox, gr.Dropdown → multi-input
  ├── outputs = gr.Markdown, gr.Textbox
  ├── auth = (username, password)
  ├── share = True → public HTTP tunnel
  └── examples = [...] → pre-filled examples
```

### Day 3: Conversational AI (Chatbots)

```
gr.ChatInterface(fn=chat, type="messages")
  └── def chat(message, history):
        history → convert to OpenAI format
        messages = [system_prompt] + history + [user_message]
        response = LLM call (stream or non-stream)
        yield response (or return)
        
Dynamic system prompt modification based on user intent (e.g., belt detection)
```

### Day 4: Tools / Function Calling

```
User: "How much is a ticket to London?"
  → LLM returns finish_reason="tool_calls"
  → Extract function name + args: get_ticket_price("London")
  → Execute Python function (SQLite lookup)
  → Append "tool" role response → re-query LLM
  → LLM returns natural language answer
  → Loop while finish_reason == "tool_calls" (for chained calls)

Tool Schema:
{
  "name": "get_ticket_price",
  "description": "...",
  "parameters": {
    "type": "object",
    "properties": { "destination_city": { "type": "string" } },
    "required": ["destination_city"]
  }
}
```

### Day 5: Multi-Modal Airline Assistant

```
gr.Blocks custom UI
  ├── gr.Row → [gr.Chatbot | gr.Image]
  ├── gr.Row → [gr.Audio]
  ├── gr.Row → [gr.Textbox]
  └── Event chain: submit → put_message_in_chatbot → then → chat()

chat() returns (history, audio_bytes, image)
  ├── LLM tool calling (SQLite price lookup)
  ├── image = openai.images.generate(model, prompt)
  └── audio = openai.audio.speech.create(model, voice, input)
```

### extra.ipynb: SVG Generation

```
Multiple models via OpenRouter → SVG code output
  → IPython.display.SVG renders inline
  → revealer.py animates with CSS keyframes
Models tested: GPT-OSS-120B, GPT-5-Nano, DeepSeek V3.2, Kimi K2, Grok 4.1, Claude Opus 4.5, GPT 5.2, Gemini 3 Pro
```

---

## Architecture Patterns

### 1. Universal OpenAI Client Adapter

```python
OpenAI(api_key=key, base_url=provider_endpoint)
```
All providers (OpenAI, Anthropic, Gemini, Groq, Grok, Ollama, OpenRouter) expose the same `/v1/chat/completions` interface. Only `base_url` and `api_key` change.

### 2. Provider Abstraction Layers

| Layer | Library | Use Case |
|---|---|---|
| Direct | `openai` / `groq` / `anthropic` | Individual provider access |
| Router | `litellm.completion()` | Unified API across providers |
| Framework | `langchain_openai.ChatOpenAI` | LangChain ecosystem |
| Meta-Router | OpenRouter API | Access 200+ models via one endpoint |

### 3. Tool/Function Calling Loop

```
1. LLM call with tools=[...]
2. If finish_reason == "tool_calls":
     for each tool_call:
       execute function with parsed arguments
       append "tool" role message
     re-call LLM with updated messages
3. If finish_reason == "stop":
     return content
```

### 4. Gradio UI Progression

| Day | Widget | Pattern |
|---|---|---|
| 2 | `gr.Interface` | Single function → single UI |
| 3 | `gr.ChatInterface` | `chat(message, history)` callback |
| 5 | `gr.Blocks` | Custom layout with `gr.Row`, event chaining with `.submit().then()` |

### 5. Streaming Responses

```python
def stream_fn(prompt):
    stream = client.chat.completions.create(..., stream=True)
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result
```

---

## Key Concepts

1. **Multi-provider LLM access** via single API pattern
2. **Gradio UI framework** — Interface, ChatInterface, Blocks
3. **Streaming** with Python generators
4. **Tool/Function calling** — schema definition, execution, looping
5. **Multi-modal output** — text + image generation + text-to-speech
6. **SQLite integration** for persistent data in tool calls
7. **SVG code generation** by LLMs

---

## Directory Structure

```
week2/
├── scraper.py              # Web scraping utility
├── revealer.py             # SVG animation utility
├── hamlet.txt              # Shakespeare text sample
├── prices.db               # SQLite airline prices
├── day1.ipynb              # Frontier Model APIs
├── day2.ipynb              # Gradio UI Framework
├── day3.ipynb              # Conversational AI
├── day4.ipynb              # Tools/Function Calling
├── day5.ipynb              # Multi-Modal Airline Assistant
├── extra.ipynb             # SVG Generation Challenge
├── week2 EXERCISE.ipynb    # End-of-week exercise
├── oday1-5.ipynb           # Ollama variants
├── oextra.ipynb            # Ollama SVG variant
└── community-contributions/
```