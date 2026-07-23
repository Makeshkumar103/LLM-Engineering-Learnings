# LLM Engineering — Architecture

> 8-week progressive curriculum building toward a multi-agent Autonomous AI system ("The Price is Right").

---

## Course Progression Overview

```mermaid
flowchart LR
    subgraph Foundations["Weeks 1–2: Foundations"]
        direction LR
        W1["Week 1<br/>First LLM Project<br/>(Summarizer)"]
        W2["Week 2<br/>Frontier Model APIs<br/>(Multi-Provider)"]
    end

    subgraph DeepDive["Weeks 3–4: Deeper Understanding"]
        direction LR
        W3["Week 3<br/>Open Source Models<br/>(Colab + Token Viz)"]
        W4["Week 4<br/>Code Generation<br/>(C++/Rust via LLM)"]
    end

    subgraph RAG["Week 5: RAG System"]
        W5["Retrieval Augmented Generation<br/>Insurellm Expert Q&A"]
    end

    subgraph Capstone["Weeks 6–7: Price Prediction"]
        direction LR
        W6["Week 6<br/>Data + Modeling<br/>(ML + DNN)"]
        W7["Week 7<br/>Fine-Tuning<br/>(QLoRA Llama 3.2)"]
    end

    subgraph Agents["Week 8: Autonomous Agents"]
        W8["Multi-Agent System<br/>'The Price is Right'"]
    end

    Foundations --> DeepDive --> RAG --> Capstone --> Agents
```

---

## Week 5 — RAG System Architecture

```mermaid
flowchart LR
    KB["Knowledge Base<br/>(Markdown Docs)"]
    Split["Text Splitting<br/>(RecursiveCharacter)"]
    Embed["Embedding Model<br/>(MiniLM-L6-v2)"]
    Chroma["ChromaDB<br/>(Vector Store)"]
    Query["User Query"]
    Retriever["Semantic Search<br/>(Similarity Top-K)"]
    Context["Retrieved Context"]
    Prompt["Augmented Prompt<br/>(Context + Question)"]
    LLM["LLM<br/>(GPT-4o / Claude)"]
    Answer["Generated Answer"]
    Eval["Evaluation<br/>MRR, nDCG, LLM Judge"]

    KB --> Split --> Embed --> Chroma
    Query --> Retriever
    Chroma --> Retriever
    Retriever --> Context
    Context --> Prompt
    Query --> Prompt
    Prompt --> LLM --> Answer
    Answer --> Eval
    Query --> Eval
    Context --> Eval

    style KB fill:#e1f5fe
    style Chroma fill:#fff3e0
    style LLM fill:#e8f5e9
    style Eval fill:#fce4ec
```

---

## Weeks 6–7 — Price Prediction Pipeline

```mermaid
flowchart LR
    Dataset["Amazon Reviews<br/>(HuggingFace Dataset)"]
    Parser["Data Parser<br/>(Parse + Filter)"]
    Items["Item Model<br/>(Pydantic Schema)"]
    Preprocessor["LLM Preprocessor<br/>(Text Rewriting)"]
    Loader["Parallel Loader<br/>(ProcessPoolExecutor)"]

    subgraph Models["Prediction Models"]
        ML["Traditional ML<br/>(scikit-learn + XGBoost)"]
        DNN["Deep Neural Network<br/>(PyTorch Residual Blocks)"]
        FT["Fine-Tuned LLM<br/>(QLoRA Llama 3.2 3B)"]
    end

    Eval["Evaluator<br/>(Scatter Plots + Error Trends)"]

    Dataset --> Parser --> Items --> Preprocessor
    Items --> Loader
    Preprocessor --> Loader
    Loader --> ML
    Loader --> DNN
    Preprocessor --> FT
    ML --> Eval
    DNN --> Eval
    FT --> Eval

    style Dataset fill:#e1f5fe
    style FT fill:#e8f5e9
    style Eval fill:#fce4ec
```

---

## Week 8 — Multi-Agent System Architecture

```mermaid
flowchart TB
    User["User / Gradio Dashboard"]

    subgraph Orchestration["Orchestration Layer"]
        DF["DealAgentFramework<br/>(Main Entry Point)"]
        PA["PlanningAgent<br/>(Orchestrator)"]
        AA["AutonomousPlanningAgent<br/>(GPT-5.1 Tool-Use)"]
    end

    subgraph Data["Data Layer"]
        RSS["RSS Feeds<br/>(dealnews.com)"]
        Mem["Memory<br/>(memory.json)"]
        VDB["Vector DB<br/>(ChromaDB from Week 5)"]
    end

    subgraph Agents["Agent Layer"]
        SA["ScannerAgent<br/>(RSS → Deals)"]
        EA["EnsembleAgent<br/>(Weighted Ensemble)"]
        MA["MessagingAgent<br/>(Pushover / SMS)"]
    end

    subgraph Models["Model Layer"]
        SP["SpecialistAgent<br/>(Fine-Tuned Llama 3.2<br/>on Modal)"]
        FR["FrontierAgent<br/>(GPT + RAG)"]
        NN["NeuralNetworkAgent<br/>(PyTorch DNN<br/>from Week 6)"]
        PP["Preprocessor<br/>(LLM Text Rewriting)"]
    end

    subgraph Notifications["Notification Layer"]
        PO["Pushover<br/>(Mobile Push)"]
    end

    User --> DF
    DF --> PA
    DF --> AA
    PA --> SA
    AA --> SA
    SA --> RSS
    PA --> EA
    AA --> EA
    EA --> PP
    PP --> SP
    PP --> FR
    PP --> NN
    SP --> VDB
    FR --> VDB
    PA --> MA
    AA --> MA
    MA --> PO
    EA --> Mem

    style Orchestration fill:#e3f2fd
    style Agents fill:#f3e5f5
    style Models fill:#e8f5e9
    style Notifications fill:#fff3e0
    style Data fill:#fce4ec
```

---

## Agent Ensemble — Pricing Flow

```mermaid
flowchart LR
    Deal["Scraped Deal"]
    PR["Preprocessor<br/>(Rewrite text)"]
    EN["EnsembleAgent"]

    subgraph Models["Weighted Ensemble Models"]
        FR["FrontierAgent<br/>Weight: 80%"]
        SP["SpecialistAgent<br/>Weight: 10%"]
        NN["NeuralNetworkAgent<br/>Weight: 10%"]
    end

    Eval["Price Evaluation<br/>Discount > $50?"]
    Notif["Push Notification"]

    Deal --> PR --> EN
    EN --> FR
    EN --> SP
    EN --> NN
    FR --> Eval
    SP --> Eval
    NN --> Eval
    Eval -->|Yes| Notif

    style Models fill:#e8f5e9
    style Eval fill:#fff3e0
    style Notif fill:#ffebee
```

---

## Infrastructure & Deployment

```mermaid
flowchart LR
    Dev["Local Development<br/>(uv + Python 3.12)"]
    Modal["Modal Serverless<br/>(GPU Inference)"]
    HF["HuggingFace Hub<br/>(Model Registry)"]
    Env["Environment<br/>(.env → API Keys)"]
    Gradio["Gradio UI<br/>(Web Interface)"]

    Dev --> Env
    Dev --> Gradio
    Dev --> Modal
    Modal --> HF
    HF -->|Fine-Tuned Model| Modal

    style Dev fill:#e3f2fd
    style Modal fill:#e8f5e9
    style Gradio fill:#f3e5f5
```

---

## Key Files & Entry Points

| Path | Purpose |
|---|---|
| `week1/day1.ipynb` | First lab — start here |
| `week5/app.py` | RAG Expert Assistant (Gradio) |
| `week5/evaluator.py` | RAG Evaluation Dashboard |
| `week8/deal_agent_framework.py` | Autonomous agent system entry point |
| `week8/price_is_right.py` | Full Gradio UI for agent system |
| `week8/pricer_service.py` | Modal-deployed fine-tuned model |

---

## Cross-Week Dependencies

```mermaid
flowchart LR
    W5["Week 5<br/>vector_db/"] -->|ChromaDB used by| W8FR["Week 8<br/>FrontierAgent"]
    W6["Week 6<br/>DNN weights"] -->|Loaded by| W8NN["Week 8<br/>NeuralNetworkAgent"]
    W7["Week 7<br/>Fine-Tuned Llama"] -->|Deployed on Modal| W8SP["Week 8<br/>SpecialistAgent"]
    W1["Week 1<br/>scraper.py"] -->|Reused in| W2["Week 2"]

    style W5 fill:#e1f5fe
    style W6 fill:#fff3e0
    style W7 fill:#e8f5e9
    style W8FR fill:#fce4ec
    style W8NN fill:#fce4ec
    style W8SP fill:#fce4ec
```
