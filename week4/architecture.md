# Week 4 Architecture

## Overview

Week 4 uses frontier and open-source LLMs to translate Python algorithms into highly optimized C++ and Rust, benchmarking performance speedups. The core theme is **LLM-as-Compiler** with system-aware code generation across 7 providers.

---

## Core Modules

### `system_info.py` — System introspection

Reports OS, CPU features (SIMD flags), compiler toolchain, package managers, and Rust toolchain. The output is fed into LLM prompts so generated code is tailored to the specific machine.

| Block | Data Collected |
|---|---|
| OS | name, arch, kernel, WSL/Rosetta detection |
| CPU | brand, logical/physical cores, SIMD (AVX512, AVX2, NEON, SVE) |
| Toolchain | gcc/clang/msvc versions, cmake/ninja/make |
| Rust | rustc/cargo/rustup versions, installed targets |

### `styles.py` — Gradio CSS

Provides custom styling for Gradio interfaces with color-coded themes (blue for Python, yellow for C++, purple accent).

### Generated Files

- `main.cpp` — LLM-generated C++ Pi computation (Gregory-Leibniz series, 200M iterations)
- `main.rs` — Placeholder for LLM-generated Rust code

---

## Data Flow

```
.env (API keys)
  → load_dotenv()
    → Multiple OpenAI() clients (one per provider)

system_info.py
  → retrieve_system_info()
  → rust_toolchain_info()

User Python code + system_info + compile flags
  → LLM prompt construction
    → system_prompt (role as code porter)
    → user_prompt (Python source + system info)
  → client.chat.completions.create(model=..., reasoning_effort=...)
  → Strip markdown fences (```cpp, ```rust, ```)
  → Write to main.cpp / main.rs
  → subprocess.run(compile_cmd)
  → subprocess.run(run_cmd)
  → Compare timing: python_time / cpp_time → X speedup
```

---

## Architecture Patterns

### 1. Multi-Provider via OpenAI-Compatible Endpoint

```python
providers = {
    "OpenAI":     OpenAI(api_key=..., base_url="https://api.openai.com/v1"),
    "Anthropic":  OpenAI(api_key=..., base_url="https://api.anthropic.com/v1/"),
    "Gemini":     OpenAI(api_key=..., base_url="https://generativelanguage.googleapis.com/v1beta/openai/"),
    "Grok":       OpenAI(api_key=..., base_url="https://api.x.ai/v1"),
    "Groq":       OpenAI(api_key=..., base_url="https://api.groq.com/openai/v1"),
    "OpenRouter": OpenAI(api_key=..., base_url="https://openrouter.ai/api/v1"),
    "Ollama":     OpenAI(api_key=..., base_url="http://localhost:11434/v1"),
}
```

### 2. System-Aware Prompting

```python
system_info = retrieve_system_info()  # dict with OS, CPU, toolchain
prompt = f"""Python code:
{code}

System info:
{system_info}

Compile with: clang++ -std=c++17 -Ofast -mcpu=native -flto=thin

Generate only valid C++ code."""
```

The LLM tailors SIMD vectorization, loop unrolling, and data types to the detected CPU.

### 3. Compile → Execute → Benchmark Pipeline

| Step | Command Pattern |
|---|---|
| C++ compile | `clang++ -std=c++17 -Ofast -mcpu=native -flto=thin main.cpp -o main` |
| C++ run | `./main` (capture stdout) |
| Rust compile | `rustc -C opt-level=3 -C target-cpu=native -C lto=fat main.rs` |
| Rust run | `./main` |
| Python exec | `exec(python_code)` wrapped in `time.time()` |

### 4. Gradio UI Progression

| Day | Layout | Features |
|---|---|---|
| 4 | `gr.Blocks` 2-column | Python input → C++ output, model dropdown, Convert button |
| 5 | `gr.Blocks` with `gr.Row`/`gr.Column` | Python editor + generated editor, Run Python/Run C++ buttons, custom CSS theming, language toggle (C++/Rust) |

---

## Benchmark Results

### Pi Calculation (200M iterations)

| Model | Speedup | Provider |
|---|---|---|
| Python baseline | 1X (40.8s) | — |
| Claude Sonnet 4.5 | 184X | Anthropic |
| GPT-5 | 233X | OpenAI |
| Grok 4 | 1060X | xAI |
| **Gemini 2.5 Pro** | **1440X** | Google |

### Max Subarray Sum (n=10000, 20 runs)

| Model | Speedup | Provider |
|---|---|---|
| Python baseline | 1X (33.8s) | — |
| GPT-OSS-20B | 238X | Groq |
| Grok 4 | ~106,000X | xAI |
| **GPT-OSS-120B** | **~111,037X** | Groq |

---

## Key Concepts

1. **LLM-as-Compiler** — Translating Python to optimized C++/Rust
2. **System-aware prompting** — Feed machine metadata for tailored code
3. **Multi-provider routing** — Single API for 7 providers
4. **Performance benchmarking** — Controlled experiment: same algorithm, same machine, different models
5. **Optimization flags** — `-Ofast`, `-march=native`, `-flto=thin` (C++); `-C opt-level=3`, `-C target-cpu=native` (Rust)
6. **Model capability variance** — Frontier models fail on harder algorithms; smaller models sometimes succeed
7. **Gradio app development** — Custom layouts, theming, event wiring

---

## Directory Structure

```
week4/
├── system_info.py           # System introspection utility
├── styles.py                # Gradio CSS styling
├── main.cpp                 # LLM-generated C++ (Pi computation)
├── main.rs                  # LLM-generated Rust placeholder
├── day3.ipynb               # Frontier model code generation
├── day4.ipynb               # Open-source models + Gradio UI
├── day5.ipynb               # Advanced UI + Rust + harder algorithm
├── Oday3.ipynb              # Ollama variant
├── Oday4.ipynb              # Ollama variant
├── Oday5 .ipynb             # Ollama variant (note: space in name)
└── community-contributions/
```