# Week 5 Architecture

## Overview

Week 5 builds a complete Retrieval-Augmented Generation (RAG) system for an insurance company knowledge base. Two parallel implementations exist: **basic** (LangChain-based) and **pro** (native Python with advanced techniques). Both are backed by a Gradio chat UI and a separate evaluation dashboard.

---

## Core Packages

### `implementation/` — Basic RAG

| File | Purpose |
|---|---|
| `ingest.py` | LangChain DirectoryLoader → RecursiveCharacterTextSplitter (500 chars, 200 overlap) → HuggingFace `all-MiniLM-L6-v2` (384 dims) → ChromaDB |
| `answer.py` | Chroma retriever (k=10) + `ChatOpenAI` (Groq: `openai/gpt-oss-20b`) → RAG answer with history |

### `pro_implementation/` — Advanced RAG

| Module | Technique |
|---|---|
| `ingest.py` | LLM-based semantic chunking (Pydantic `Chunk(headline, summary, original_text)`), OpenAI `text-embedding-3-large`, ChromaDB `PersistentClient`, multiprocessing Pool (3 workers), `tenacity` retry |
| `answer.py` | **Query rewriting** (LLM rewrites question), **dual retrieval** (original + rewritten, k=20 each), **merge + deduplicate**, **LLM-based reranking** (Pydantic `RankOrder`), top 10 returned |

### `evaluation/` — Test & Evaluation

| File | Purpose |
|---|---|
| `test.py` | Pydantic `TestQuestion` model (question, keywords, reference_answer, category) |
| `eval.py` | Retrieval metrics (MRR, nDCG, keyword coverage) + Answer metrics (LLM-as-judge: accuracy, completeness, relevance scored 1-5) |
| `tests.jsonl` | 150 test questions across 7 categories |

---

## Data Flow

### Ingestion

```
KNOWLEDGE BASE (markdown files)
  company/ (4 files)  employees/ (32)  products/ (8)  contracts/ (30+)

BASIC:                              PRO:
  DirectoryLoader                     Manual file walking
  RecursiveCharSplitter(500,200)      LLM semantic chunking (headline+summary+text)
  HF all-MiniLM-L6-v2 (384d)          OpenAI text-embedding-3-large
  LangChain Chroma                    ChromaDB PersistentClient
  → vector_db/ (413 chunks)           → preprocessed_db/ (617 chunks)
```

### Retrieval + Generation

```
User question
  → combined_question(question, history) — concatenates prior turns

BASIC:                              PRO:
  retriever.invoke(k=10)              1. rewrite_query(question)
                                        2. retrieve original (k=20)
                                        3. retrieve rewritten (k=20)
                                        4. merge_chunks (deduplicate)
                                        5. rerank (LLM relevance ranking)
                                        6. return top 10

  → context = "\n\n".join(docs)
  → system_prompt.format(context)
  → LLM.generate(messages)
  → answer + source documents
```

### Evaluation

```
tests.jsonl (150 questions, 7 categories)
  → evaluate_retrieval()
     MRR: mean(1/rank per keyword)
     nDCG: discounted cumulative gain
     Keyword coverage: % keywords found

  → evaluate_answer()
     LLM-as-judge (Groq) scores:
       accuracy (1-5), completeness (1-5), relevance (1-5)
```

---

## Architecture Patterns

### 1. Basic vs. Pro RAG Comparison

| Aspect | Basic | Pro |
|---|---|---|
| Chunking | Fixed size (500 chars) | LLM semantic boundaries |
| Embeddings | HF MiniLM (384d, free) | OpenAI text-embedding-3-large (paid) |
| Retrieval | Single query, k=10 | Dual query (original + rewritten), k=20 each |
| Reranking | None | LLM-based relevance reranking |
| Framework | LangChain | Native Python |
| Parallelism | None | multiprocessing.Pool (3 workers) |
| Retry | None | tenacity exponential backoff |
| Chunks | 413 | 617 |

### 2. LLM-as-Judge Evaluation

```python
judge_prompt = f"""Compare the answer to the reference answer.
Score accuracy (1-5), completeness (1-5), relevance (1-5).
Output as JSON."""
```

Retrieval metrics: MRR, nDCG (position-aware), keyword coverage (recall-oriented).

### 3. Gradio UI Layout

```
app.py:
  ┌─────────────────┐  ┌──────────────────────────────┐
  │ Chatbot          │  │ Retrieved Context (Markdown)  │
  │ [Conversation]   │  │ Source: employees/emily.txt   │
  │                  │  │ [document content]            │
  │ [Text Input]     │  │ Source: contracts/...         │
  └─────────────────┘  └──────────────────────────────┘

evaluator.py:
  ┌──────────────────────┐  ┌──────────────────────┐
  │ Retrieval Eval       │  │ Answer Eval           │
  │ MRR, nDCG, Coverage  │  │ Accuracy, Completeness│
  │ (color-coded bars)   │  │ Relevance (1-5 bars)  │
  └──────────────────────┘  └──────────────────────┘
```

### 4. Groq JSON Mode Workaround

Groq requires the word `"json"` in the messages to enable `response_format={"type": "json_object"}`. This is documented in `groq-gpt.txt`.

---

## Knowledge Base Structure

```
knowledge-base/
├── company/       about.md, careers.md, culture.md, overview.md
├── employees/     32 employee HR records (emily.md, james.md, ...)
├── products/      8 product descriptions
└── contracts/     30+ legal contracts
```

Test question categories: `direct_fact` (70), `temporal` (20), `spanning` (20), `comparative` (10), `numerical` (10), `relationship` (10), `holistic` (10).

---

## Key Concepts

1. **RAG pipeline** — Retrieve → Augment → Generate
2. **Chunking strategies** — Fixed-size vs. LLM semantic boundaries
3. **Embedding models** — Free (MiniLM) vs. paid (OpenAI) tradeoffs
4. **Vector databases** — ChromaDB with LangChain wrapper vs. native client
5. **Query rewriting** — LLM reformulates questions for better retrieval
6. **Dual retrieval** — Original + rewritten query, merged and reranked
7. **Evaluation metrics** — MRR, nDCG, keyword coverage, LLM-as-judge
8. **Parallel ingestion** — Multiprocessing for concurrent LLM chunking

---

## Directory Structure

```
week5/
├── app.py                         # Gradio chat UI (basic)
├── oapp.py                        # Gradio chat UI (Ollama variant)
├── evaluator.py                   # Gradio evaluation dashboard
├── implementation/
│   ├── ingest.py                  # Basic ingestion (LangChain)
│   └── answer.py                  # Basic RAG answer
├── pro_implementation/
│   ├── ingest.py                  # Pro ingestion (LLM chunking)
│   └── answer.py                  # Pro RAG (rewrite + dual + rerank)
├── oimplementation/               # Ollama variants
│   ├── app.py / ingest.py / answer.py
├── evaluation/
│   ├── test.py                    # Pydantic test models
│   ├── eval.py                    # Evaluation engine
│   └── tests.jsonl                # 150 test questions
├── knowledge-base/                # Source documents
├── vector_db/                     # Basic ChromaDB store
├── preprocessed_db/               # Pro ChromaDB store
└── community-contributions/
```