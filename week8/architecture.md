# Week 8 Architecture

## Overview

Week 8 is the capstone: a **multi-agent autonomous deal-finding system** called "The Price is Right." It combines a weighted ensemble of three pricing models (frontier LLM + fine-tuned Llama + deep neural network) with RSS scraping, RAG, function-calling agents, push notifications, and a Gradio dashboard.

---

## Core Modules

### `agents/` — Multi-Agent System

| Agent | File | Role |
|---|---|---|
| `Agent` (base) | `agent.py` | Abstract base with ANSI colored logging |
| `PlanningAgent` | `planning_agent.py` | Orchestrator: scan → price → notify |
| `AutonomousPlanningAgent` | `autonomous_planning_agent.py` | GPT-5.1 with function-calling tools (autonomous workflow) |
| `ScannerAgent` | `scanner_agent.py` | RSS feed scraping + GPT-5-mini deal selection |
| `EnsembleAgent` | `ensemble_agent.py` | Weighted ensemble (Frontier 80% + Specialist 10% + Neural 10%) |
| `FrontierAgent` | `frontier_agent.py` | GPT-5.1 with ChromaDB RAG context |
| `SpecialistAgent` | `specialist_agent.py` | Fine-tuned Llama 3.2 on Modal (serverless GPU) |
| `NeuralNetworkAgent` | `neural_network_agent.py` | PyTorch 10-layer residual network |
| `Preprocessor` | `preprocessor.py` | LLM text rewriting (Ollama/Llama 3.2) |
| `MessagingAgent` | `messaging_agent.py` | Pushover push notifications + Claude message crafting |
| `Deal / Opportunity / ScrapedDeal` | `deals.py` | Pydantic data models for deals and opportunities |
| `Tester` | `evaluator.py` | Multi-threaded evaluation + visualization |
| `Item` | `items.py` | Pydantic product data model |

### Services

| File | Technology | Purpose |
|---|---|---|
| `pricer_service2.py` | Modal (class-based) | Deployed fine-tuned Llama 3.2 3B with 4-bit quantization |
| `pricer_service.py` | Modal (function-based) | Simpler deployment variant |
| `pricer_ephemeral.py` | Modal (ephemeral) | Single-run testing |
| `llama.py` | Modal | Basic Llama 3.2 inference |
| `hello.py` | Modal | Geolocation demo |

### UI & Orchestration

| File | Purpose |
|---|---|
| `deal_agent_framework.py` | Top-level orchestrator connecting agents, managing memory |
| `price_is_right.py` | Gradio dashboard with live logs, 3D vector plot, deals table |
| `log_utils.py` | ANSI color → HTML conversion for Gradio log rendering |
| `memory.json` | Persistent deal memory (prevents duplicate alerts) |
| `results.ipynb` | Final model comparison bar chart |

---

## Data Flow

### Agent Pipeline

```
DealAgentFramework.run()
  → PlanningAgent.plan(memory)
    → ScannerAgent.scan(memory)
        1. Scrape 3 RSS feeds (dealnews.com: Electronics, Computers, Smart-Home)
        2. Filter URLs already in memory
        3. GPT-5-mini (Structured Outputs) → select 5 best deals
        → returns DealSelection (5 Deals)

    → EnsembleAgent.price(description)  (for each of 5 deals)
        1. Preprocessor → LLM rewrites description into structured format
        2. FrontierAgent:
            → SentenceTransformer(all-MiniLM-L6-v2) → embed description
            → ChromaDB query → 5 similar products (RAG context)
            → GPT-5.1 with context → price estimate
        3. SpecialistAgent:
            → modal.Cls.from_name("pricer-service", "Pricer").price.remote()
            → Fine-tuned Llama 3.2 → price estimate
        4. NeuralNetworkAgent:
            → HashingVectorizer(5000) → 10-layer Residual Network → price estimate
        5. Weighted average: frontier*0.8 + specialist*0.1 + neural*0.1

    → Sort by discount (estimate - deal_price)
    → Select best deal where discount > $50 (DEAL_THRESHOLD)

    → MessagingAgent.notify(opportunity)
        1. Claude Sonnet 4.5 crafts compelling 2-3 sentence message
        2. Pushover API sends push notification to phone
        3. Opportunity appended to memory.json
```

### Autonomous Variant

```
AutonomousPlanningAgent.plan(memory)
  → GPT-5.1 with 3 function-calling tools:
      1. scan_the_internet_for_bargains()
      2. estimate_true_value(description)
      3. notify_user_of_deal(...)
  → LLM autonomously decides:
      When to scan → which deals to price → which to notify about
  → Loops until LLM decides to stop (no more tool calls)
```

### Gradio UI

```
price_is_right.py
  ┌──────────────────────────────────────────────────────────────┐
  │  🏆 The Price is Right — Autonomous Deal Hunter              │
  ├──────────────────────────────────────────────────────────────┤
  │  ┌─────────────────────────┐  ┌──────────────────────────┐   │
  │  │ Deals Table             │  │ 3D Vector Store Plot     │   │
  │  │ (live-updating, sortable)│  │ (t-SNE, color by category)│   │
  │  └─────────────────────────┘  └──────────────────────────┘   │
  ├──────────────────────────────────────────────────────────────┤
  │  ┌──────────────────────────────────────────────────────────┐│
  │  │ Live Agent Log (colored by agent, streaming from queue)  ││
  │  └──────────────────────────────────────────────────────────┘│
  ├──────────────────────────────────────────────────────────────┤
  │  [Run Now] [Autonomous] [Notify]  Auto-refresh every 5min    │
  └──────────────────────────────────────────────────────────────┘
```

---

## Architecture Patterns

### 1. Weighted Ensemble

```python
weights = {"frontier": 0.8, "specialist": 0.1, "neural": 0.1}
estimate = frontier * 0.8 + specialist * 0.1 + neural * 0.1
```

- **Frontier (80%)** — GPT-5.1 with RAG, strongest single model
- **Specialist (10%)** — Fine-tuned Llama 3.2, domain-tuned on pricing
- **Neural (10%)** — Classical ML, consistent and deterministic

### 2. RAG via ChromaDB

```python
# FrontierAgent
chroma_collection = chromadb.PersistentClient(path="products_vectorstore")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def find_similars(description):
    embedding = embedder.encode(description).tolist()
    results = chroma_collection.query(query_embeddings=[embedding], n_results=5)
    return [(meta["description"], meta["price"]) for meta in results["metadatas"][0]]
```

Context includes prices of similar products, enabling GPT-5.1 to anchor estimates.

### 3. Serverless GPU Inference (Modal)

```python
# pricer_service2.py
app = modal.App("pricer-service")
volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

@app.cls(gpu="T4", image=image, volumes={"/hf-cache": volume}, 
         container_idle_timeout=300, secrets=[...])
class Pricer:
    @modal.enter()
    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
        self.model = PeftModel.from_pretrained(self.model, ...)
        self.tokenizer = AutoTokenizer.from_pretrained(...)

    @modal.method()
    def price(self, description: str) -> float:
        # Generate price from description
```

Modal handles cold starts, GPU provisioning, auto-scaling, and container lifecycle.

### 4. Function-Calling Autonomous Agent

```python
# AutonomousPlanningAgent
tools = [
    {"type": "function", "function": {"name": "scan_the_internet_for_bargains", ...}},
    {"type": "function", "function": {"name": "estimate_true_value", ...}},
    {"type": "function", "function": {"name": "notify_user_of_deal", ...}},
]

def plan(self, memory):
    messages = [system_prompt, user_prompt]
    while True:
        response = client.chat.completions.create(model="gpt-5.1", messages=messages, tools=tools)
        if response.choices[0].finish_reason == "stop":
            break
        # Execute tool calls, append results, continue loop
```

The LLM autonomously decides the workflow — scanning, pricing, and notifying based on its own judgment.

### 5. Structured Outputs (Pydantic)

```python
class DealSelection(BaseModel):
    deals: list[Deal]
    
class Deal(BaseModel):
    product_description: str
    price: float
    url: str

# OpenAI call with response_format=DealSelection
```

Guarantees parseable JSON for downstream agent consumption.

### 6. Push Notifications

```
opportunity (discount > $50)
  → Claude Sonnet 4.5 crafts message: "🔥 Huge deal! Hisense TV at $299..."
  → Pushover API: requests.post("https://api.pushover.net/1/messages.json", data={...})
  → Phone receives push notification with sound
```

---

## Results

| Approach | Avg Error |
|---|---|
| Ensemble (Frontier 80% + Specialist 10% + Neural 10%) | **$29.90** |
| GPT 5.1 RAG | $30.19 |
| Fine-tuned Full (Llama 3.2) | $39.85 |
| GPT 5.1 (no RAG) | $44.74 |
| Deep Neural Network | $46.49 |
| Base Llama 3.2 4-bit | $110.72 |

The ensemble achieves the best overall error, benefiting from complementary strengths of frontier, fine-tuned, and classical models.

---

## Key Concepts

1. **Multi-agent architecture** — Specialized agents with single responsibilities coordinated by a planner
2. **Ensemble inference** — Weighted combination of frontier LLM, fine-tuned model, and classical ML
3. **RAG for pricing** — Similar products as context for LLM price estimates
4. **Function-calling agents** — LLM autonomously decides tool invocation order
5. **Serverless GPU deployment** — Modal for cost-effective, scalable model serving
6. **Push notifications** — Pushover API for real-time deal alerts
7. **Structured outputs** — Pydantic models with OpenAI `response_format`
8. **Persistence** — `memory.json` for deduplication across runs
9. **Real-time Gradio UI** — Streaming logs, live-updating tables, 3D vector visualization

---

## Directory Structure

```
week8/
├── agents/
│   ├── agent.py                      # Base Agent class
│   ├── autonomous_planning_agent.py  # GPT-5.1 function-calling orchestrator
│   ├── deals.py                      # Deal, Opportunity, ScrapedDeal models
│   ├── deep_neural_network.py        # 10-layer Residual Network
│   ├── ensemble_agent.py             # Weighted ensemble (0.8/0.1/0.1)
│   ├── evaluator.py                  # Multi-threaded evaluation
│   ├── frontier_agent.py             # GPT-5.1 + ChromaDB RAG
│   ├── items.py                      # Pydantic Item model
│   ├── messaging_agent.py            # Pushover + Claude notifications
│   ├── neural_network_agent.py       # PyTorch DNN wrapper
│   ├── planning_agent.py             # Scan → Price → Notify orchestrator
│   ├── preprocessor.py               # LLM text rewriting
│   ├── scanner_agent.py              # RSS feed scraping + GPT-5-mini
│   └── specialist_agent.py           # Modal fine-tuned Llama client
├── deal_agent_framework.py           # Top-level orchestrator
├── price_is_right.py                 # Gradio dashboard
├── pricer_service.py                 # Modal function-based service
├── pricer_service2.py                # Modal class-based service
├── pricer_ephemeral.py               # Modal single-run testing
├── log_utils.py                      # ANSI → HTML log formatting
├── memory.json                       # Persistent deal memory
├── hello.py                          # Modal geolocation demo
├── llama.py                          # Modal Llama 3.2 inference
├── day1-5.ipynb                      # Lecture notebooks
├── results.ipynb                     # Final comparison chart
└── community-contributions/
```