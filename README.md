# ⚔️ RAGnarok — Evidence-First Editor

### *Write with Receipts*

**A retrieval-augmented writing assistant that fact-checks every sentence against a trusted document corpus, in real time, as you type — and refuses to guess when the evidence isn't there.**

[![Status](https://img.shields.io/badge/status-active--development-orange?style=for-the-badge)](#-project-status)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](#-license)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#%EF%B8%8F-technology-stack)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#%EF%B8%8F-technology-stack)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](#%EF%B8%8F-technology-stack)
[![Claude](https://img.shields.io/badge/LLM-Claude-D97757?style=for-the-badge)](#%EF%B8%8F-technology-stack)

---

## 🌌 Why RAGnarok?

Most writing tools check grammar and style — none of them check whether what you just wrote is actually **true**. Writers, students, and researchers routinely state facts without verifying them against a source, and by the time an error gets caught (if it ever does), it's already published.

General AI writing tools don't fix this either — a chatbot answering from memory can hallucinate just as easily as a human can misremember a fact.

**RAGnarok is built differently.** Instead of a Q&A chatbot bolted onto your documents, it sits inside the writing process itself: every sentence you finish is checked against a corpus of trusted sources, classified as **Supported**, **Contradicted**, or **Unverified**, and backed by the exact evidence span that justifies the verdict.

> 🧭 **The core bet:** grounding beats fluency. A model that writes confidently but ungrounded is worse than one that says "I don't have evidence for this." Refusing to guess *is* the feature.

---

## 🎯 Objectives

| #   | Goal                                                                        |
| --- | ---------------------------------------------------------------------------- |
| 1️⃣ | Ingest a corpus of trusted documents and index it for fast, accurate lookup |
| 2️⃣ | Check user-written claims against that corpus with hybrid retrieval          |
| 3️⃣ | Classify each claim as supported / contradicted / unverified, never guessing |
| 4️⃣ | Surface the exact evidence span behind every verdict, not a black-box label  |
| 5️⃣ | Ship a real-time editor UI with a live, document-level grounding score       |

---

## 🧠 System Architecture

```
                ┌───────────────────┐
                │   Source Corpus     │
                │   (.txt / .md)       │
                └─────────┬──────────┘
                          │ chunk (sentence-level)
                ┌─────────▼──────────┐
                │    Embed + Index      │
                └─────────┬──────────┘
         ┌────────────────┴────────────────┐
         │                                  │
┌────────▼─────────┐              ┌─────────▼─────────┐
│   Dense Search      │              │   Sparse Search      │
│  (Chroma / cosine)  │              │  (BM25 / keyword)   │
└────────┬─────────┘              └─────────┬─────────┘
         │                                  │
         └────────────────┬─────────────────┘
                          │ merge + normalize + weight
                ┌─────────▼──────────┐
                │  Hybrid Retrieval     │
                │     (top-k chunks)     │
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │   LLM Judge (Claude)  │
                │  claim vs. evidence    │
                └─────────┬──────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   🟢 SUPPORTED     🔴 CONTRADICTED     ⚪ UNVERIFIED
```

The user-facing loop: type → pause → sentence sent to the backend → hybrid retrieval → LLM judgment → Evidence Panel updates live, with a running grounding score for the whole document.

---

## 🔬 End-to-End Pipeline

```
📥 Ingest Docs  →  ✂️ Sentence Chunking  →  🧮 Embed (bge-small)  →  🗂️ Index (Chroma + BM25)
     │
     ▼
⌨️ User Types  →  ⏱️ Debounce  →  🔎 Hybrid Retrieval  →  ⚖️ LLM Classification
     │
     ▼
🟢🔴⚪ Evidence Panel  →  📊 Grounding Score  →  🧪 Eval Harness
```

---

## ⚙️ Technology Stack

| Layer                   | Technology                               |
| ------------------------ | ------------------------------------------ |
| 🧮 Language               | Python (backend) · JavaScript (frontend)  |
| ⚡ API                    | FastAPI · Uvicorn                          |
| 🗂️ Vector Store           | ChromaDB (dense / cosine search)          |
| 🔤 Sparse Retrieval       | BM25 (`rank-bm25`)                         |
| 🧠 Embeddings             | `sentence-transformers` — BAAI/bge-small-en-v1.5 |
| ⚖️ Claim Classification   | Anthropic Claude API                       |
| 🧾 Schemas / Validation   | Pydantic · pydantic-settings               |
| 💻 Frontend               | React · Vite                               |
| 🌐 HTTP Client            | Axios                                      |
| 🧪 Evaluation             | Hand-labeled eval set + Python scoring script |

**Deep learning used:** two pretrained Transformer models — an embedding model for dense retrieval and an LLM for claim classification. No model is trained or fine-tuned from scratch in this project; the engineering is in the retrieval pipeline and classification logic around them.

---

## 📂 Corpus Strategy

The starter corpus is a small set of neutral, independently verifiable facts (the solar system), chosen deliberately so retrieval quality and classification accuracy can be evaluated against clean ground truth.

**Designed to be swapped:** drop any `.txt` / `.md` files into `backend/corpus/sample_docs/` and re-run `/ingest` — the pipeline doesn't assume anything domain-specific about the source documents.

---

## 📊 Evaluation

`Accuracy` · `Per-class breakdown` · `Confusion (expected → predicted)` · `Confidence scores`

The `eval/` directory ships with 18 hand-labeled sentence → expected-label pairs spanning all three classes (supported / contradicted / unverified), and a script that runs them against the live backend and reports results.

> ⚡ **The metric that matters most:** how the system behaves on the *unverified* class — a RAG system that never says "I don't know" isn't trustworthy, no matter how good its retrieval is.

---

## 🔍 Explainability

Every verdict ships with the retrieved evidence chunks, the specific evidence span the model relied on, its source document, and a one-line explanation. No bare labels — every classification is traceable back to the exact text that produced it.

---

## 🌍 Real-World Applications

| 📝 Content Fact-Checking        | 🎓 Academic Writing Support           | 📚 Research Note-Taking        |
| -------------------------------- | -------------------------------------- | -------------------------------- |
| ⚖️ Policy & Compliance Drafting  | 🗞️ Journalism Verification Workflows  | 🏢 Internal Knowledge-Base QA   |

---

## 📈 Roadmap

- [x] Sentence-level chunking + ingestion pipeline
- [x] Hybrid retrieval (dense + BM25)
- [x] LLM claim classification with strict JSON output
- [x] "Unverified" fallback when no evidence is retrieved
- [x] React editor with live Evidence Panel
- [x] Hand-labeled eval harness
- [ ] In-place highlighting inside the editor (contentEditable decorations)
- [ ] Cross-encoder re-ranking stage before classification
- [ ] WebSocket-based live updates (replace debounced HTTP)
- [ ] Multi-document contradiction detection within the corpus itself
- [ ] Deployed public demo

---

## 📁 Project Structure

```
ragnarok/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app, endpoints
│   │   ├── config.py        # env vars / settings
│   │   ├── models.py        # pydantic request/response schemas
│   │   ├── ingest.py        # chunking + indexing
│   │   ├── retrieval.py     # hybrid (dense + BM25) search
│   │   ├── classify.py      # LLM claim-vs-evidence classifier
│   │   └── store.py         # shared Chroma + BM25 state
│   ├── corpus/sample_docs/  # starter corpus
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── components/
│           ├── Editor.jsx
│           └── ClaimBadge.jsx
├── eval/
│   ├── eval_set.json
│   └── run_eval.py
└── README.md
```

---

## 💡 Key Challenges

| Challenge                    | Why It's Hard                                                                  |
| ------------------------------ | --------------------------------------------------------------------------------- |
| 🎯 Precision at claim-level    | Sentence-level claims are shorter and more ambiguous than typical RAG queries    |
| ⚖️ Avoiding false confidence   | The system must prefer "unverified" over a wrong guess when evidence is weak     |
| 🔤 Exact-term sensitivity      | Numbers and named facts need keyword matching, not just semantic similarity      |
| ⏱️ Real-time latency           | Retrieval + LLM judgment has to feel responsive inside a writing flow            |
| 🗂️ Corpus generality           | The pipeline must work on arbitrary document sets, not just the sample corpus    |

---

## 📌 Project Status

> 🚧 **Under Active Development**

The core pipeline — ingestion, hybrid retrieval, LLM classification, and the editor UI — is functional end to end against the starter corpus. Current focus is expanding the eval set and moving from the side-panel UI toward in-place highlighting.

---

## 🤝 Contributing

Contributions are welcome — whether that's corpus curation, retrieval tuning, frontend polish, or documentation.

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. 💻 Make your changes
4. ✅ Open a pull request

Interested in retrieval-augmented generation, grounding, or fact-verification systems? Open an issue and let's talk.

---

## 📜 License

Released under the **MIT License**.

---

## ⭐ Acknowledgements

Built on top of the open-source retrieval and NLP ecosystem — ChromaDB, sentence-transformers, rank-bm25, and Anthropic's Claude API.

---

*If RAGnarok is useful to you, consider ⭐ starring the repo.*
