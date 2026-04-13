# RAG-Render
# 🛡️ Insurance Policy Advisor — Production RAG Pipeline

> **Ask questions about insurance policies in plain English. Get precise, cited answers instantly.**

A **Retrieval-Augmented Generation (RAG)** pipeline deployed as a production API with a web frontend. Upload insurance policy PDFs, ask any question, and receive accurate answers with source citations powered by OpenAI, ChromaDB, and FastAPI.

🔗 **Live App:** [rag-render-vsxx.onrender.com](https://rag-render-vsxx.onrender.com)
📡 **API Docs:** [rag-render-vsxx.onrender.com/docs](https://rag-render-vsxx.onrender.com/docs)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Production_API-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render&logoColor=white)

---

## 📌 Problem

Insurance policy documents are dense, jargon-heavy, and often span 20–50 pages. Finding specific details like hospitalization coverage limits, waiting periods, or claim procedures means manually reading through pages of legal text across multiple documents.

**This project solves that.** Six HDFC Life insurance PDFs (488 text chunks) are ingested, embedded, and searchable. Ask a question in plain English and get a precise answer with the exact policy name and page number cited.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   WEB FRONTEND                          │
│         User types question → Gets cited answer         │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │      FASTAPI BACKEND        │
          │                             │
          │  /ask    → Answer questions  │
          │  /health → Check status     │
          │  /ingest → Reload PDFs      │
          │  /docs   → Swagger UI       │
          └──────────────┬──────────────┘
                         │
     ┌───────────────────▼───────────────────┐
     │          RAG PIPELINE                  │
     │                                        │
     │  📄 PDF Extraction (pdfplumber)        │
     │      ↓                                 │
     │  📊 Table Detection → NL Conversion    │
     │      ↓                                 │
     │  🧹 OCR Error Cleaning                 │
     │      ↓                                 │
     │  ✂️  Chunking (150 words / 30 overlap)  │
     │      ↓                                 │
     │  🔢 Embedding (text-embedding-3-small) │
     │      ↓                                 │
     │  💾 ChromaDB (Persistent Storage)      │
     │      ↓                                 │
     │  🔍 Semantic Search (Top 10)           │
     │      ↓                                 │
     │  🏆 LLM Re-ranking (GPT-4o-mini → 3)  │
     │      ↓                                 │
     │  💬 Answer Generation (Cited)          │
     └───────────────────────────────────────┘
```

---

## ✨ Features

- **Smart PDF Parsing** — Extracts regular text and tables separately. Tables are converted to natural language for better embedding quality
- **OCR Error Resilience** — Cleans duplicate character artifacts and instructs the LLM to interpret remaining misspellings
- **Semantic Chunking** — 150-word chunks with 30-word overlap preserve context while keeping embeddings focused
- **Two-Stage Retrieval** — Bi-encoder retrieves top 10 candidates, then GPT-4o-mini re-ranks to select the 3 most relevant
- **Cited Answers** — Every response includes source citations with policy name and page number
- **Query Caching** — Previous results are cached to avoid redundant searches and reduce API costs
- **Production API** — FastAPI with Swagger docs, health checks, and proper error handling
- **Web Frontend** — Clean, user-friendly interface with example questions — no coding required

---

## 📁 Project Structure

```
RAG-Render/
├── api.py              # FastAPI endpoints + HTML +Web frontend
├── pipeline.py         # Orchestrates the full RAG pipeline
├── ingestion.py        # PDF extraction, table handling, chunking
├── vectorstore.py      # ChromaDB setup, embedding, retrieval, caching
├── reranker.py         # LLM-based re-ranking with GPT-4o-mini
├── generator.py        # Answer generation with engineered prompt
├── config.py           # All settings (models, chunk size, prompts)
├── requirements.txt    # Pinned dependencies
├── Dockerfile          # Container deployment
├── HDFC_/              # Insurance policy PDFs
│   ├── HDFC-Life-Easy-Health-101N110V03-Policy-Bond-Single-Pay.pdf
│   ├── HDFC-Life-Easy-Health-Customer-Information-Sheet.pdf
│   ├── HDFC-Life-Cancer-Care-101N106V04-Policy-Document.pdf
│   ├── HDFC-Life-Saral-Jeevan-UIN-101N160V05-Policy-Document.pdf
│   └── ...
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology | Why This Choice |
|---|---|---|
| **Backend** | FastAPI | Production-grade API with auto-generated Swagger docs |
| **Frontend** | Vanilla HTML/CSS/JS | Lightweight, no build tools, served from the same API |
| **PDF Parsing** | pdfplumber | Best table detection among Python PDF libraries |
| **Embeddings** | OpenAI text-embedding-3-small | 5x cheaper than ada-002, better retrieval quality |
| **Vector Store** | ChromaDB | Simple API, persistent storage, built-in HNSW search |
| **Re-ranking** | GPT-4o-mini | Handles OCR errors that cross-encoders cannot |
| **Generation** | GPT-4o-mini | Best quality-per-dollar for answer generation |
| **Deployment** | Render | Free tier, auto-deploy from GitHub |

---


## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web frontend — user-friendly Q&A interface |
| `GET` | `/health` | Pipeline status and document count |
| `POST` | `/ask` | Ask a question, get a cited answer |
| `POST` | `/ingest` | Re-process all PDFs (force recreate) |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Example API Call

```bash
curl -X POST https://rag-render-vsxx.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How much is covered for hospitalization?", "use_cache": true}'
```

### Example Response

```json
{
  "answer": "**Answer:** The policy provides Daily Hospital Cash Benefit as a percentage of Sum Insured. For Non-ICU admission, 1% is payable per day. For ICU admission, 2% is payable per day [Source 1].\n\n**Key Details:**\n- Non-ICU: Max 20 days/year, 60 days during policy term [Source 1]\n- ICU: Max 10 days/year, 30 days during policy term [Source 2]",
  "sources": [
    "HDFC-Life-Easy-Health-101N110V03 | Page7",
    "HDFC-Life-Easy-Health-Customer-Information-Sheet | Page2",
    "HDFC-Life-Easy-Health-101N110V03 | Page8"
  ],
  "cached": false
}
```

---

## 💡 Example Questions

| Question | What It Retrieves |
|---|---|
| "How much is covered for hospitalization?" | 1% Sum Insured (Non-ICU), 2% (ICU) with day limits |
| "What is the waiting period?" | 60 days for DHCB/Surgical, 90 days for Critical Illness |
| "What surgeries are covered?" | 138 surgeries across 4 categories with % payouts |
| "Can I surrender my policy?" | Surrender formula: 70% × Premium × (1 - M/P) |
| "What are the plan options?" | 7 options (A–G) combining DHCB, SB, and CIB |

---

## 🔧 Pipeline Deep Dive

### Why LLM Re-ranking Over Cross-Encoders?

Initially implemented cross-encoder re-ranking using `ms-marco-MiniLM-L-6-v2`. All scores were negative despite correct bi-encoder retrieval. Root cause:

1. **OCR Errors** — PDFs had systematic artifacts ("wil"→"will", "admited"→"admitted"). Cross-encoders trained on clean text couldn't match queries against corrupted text.
2. **Table Formatting** — Documents with list-formatted tables scored poorly because cross-encoders expect natural prose.

GPT-4o-mini handles both naturally — it understands misspelled text and interprets table structures regardless of format.

### Prompt Engineering

The answer generation prompt follows a structured format:

```
Role       → Expert insurance policy advisor
Context    → OCR errors present, tables in list/NL format, top 3 docs only
Task       → Answer from provided documents only, no hallucination
Guidelines → Be concise, cite sources, fix spelling, extract specific details
Output     → Direct answer + bullet points + source citations
```

### Why 150-Word Chunks?

| Chunk Size | Issue |
|---|---|
| 50–100 words | Too small — loses context, fragments sentences |
| 150 words | Focused embeddings, retains meaning, fits model limits |
| 300+ words | Blurry embeddings mixing multiple topics |

30-word overlap ensures sentences at boundaries aren't lost.

---

## 💰 Cost

| Operation | Cost |
|---|---|
| Embedding 488 chunks (one-time) | ~$0.005 |
| Per query (rerank + answer) | ~$0.001 |
| 1000 queries | ~$1.00 |
| **Complete project** | **$1–3** |

---

## 🔀 Local Alternative with Ollama (Zero Cost)

The pipeline can run entirely locally using [Ollama](https://ollama.com) — no API keys, no internet, no cost.

| | OpenAI | Ollama |
|---|---|---|
| Cost | ~$0.001/query | Free |
| Quality | Excellent | Good |
| Privacy | Data sent to OpenAI | Stays on your machine |
| Hardware | Any computer | 8GB+ RAM needed |

---

## 🚧 Future Improvements

- [ ] Hybrid search (semantic + BM25 keyword matching)
- [ ] FAISS/Pinecone for scalable vector storage
- [ ] Conversation memory for multi-turn Q&A
- [ ] RAGAS evaluation framework for automated quality testing
- [ ] Streaming responses for better UX
- [ ] Ollama toggle in the UI
- [ ] React frontend for richer interactions

---

## 🧪 Related

Development notebook with step-by-step pipeline building: [kvora09/RAGg](https://github.com/kvora09/RAGg)


<p align="center">
⭐ Star this repo if you found it helpful!
</p>
