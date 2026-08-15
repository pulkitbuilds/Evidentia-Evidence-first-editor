# ⚡ Evidentia — Evidence-First Editor

<p align="center">
  <strong>Write. Verify. Understand your evidence.</strong><br/>
  Real-time factual-claim checking against a trusted document corpus.
</p>

<p align="center">
  <a href="#-what-is Evidentia">What is it?</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#setup">Quick Start</a> •
  <a href="#using-it">Usage</a> •
  <a href="#notes-on-design-decisions">Design Decisions</a>
</p>

---

## 🎯 What is Evidentia?

Evidentia is an **Evidence-First Editor** that checks factual claims against a trusted document corpus in real time.

| Verdict | Meaning |
|---|---|
| 🟢 **Supported** | Evidence in the corpus supports the sentence |
| 🔴 **Contradicted** | Evidence in the corpus conflicts with the sentence |
| ⚪ **Unverified** | No sufficiently relevant evidence was found |

<details>
<summary><strong>✨ Click to see the core workflow</strong></summary>

```text
Write sentence
      ↓
Hybrid retrieval
      ↓
Find relevant evidence
      ↓
LLM judge
      ↓
┌────────────┬──────────────┬────────────┐
│ 🟢 Support │ 🔴 Conflict  │ ⚪ No proof │
└────────────┴──────────────┴────────────┘
      ↓
Show exact source inline
```

</details>



A writing tool that checks factual claims against a trusted document corpus in
real time, highlighting each sentence as **supported**, **contradicted**, or
**unverified**, with the exact source shown inline.

## 🧭 Interactive Feature Tour

<details>
<summary>📝 <strong>Draft</strong></summary>

The live editor automatically checks completed sentences. The preview shows a colored underline, a ✓ / ✕ / — mark, and a verification stamp containing the running grounding score.

</details>

<details>
<summary>📚 <strong>Library</strong></summary>

Upload or paste `.txt` / `.md` documents, browse them as index cards, read full text, or remove sources. Documents are persisted in SQLite.

</details>

<details>
<summary>🔎 <strong>Research</strong></summary>

Give Evidentia a topic. It expands the topic into search queries, searches the web, fetches and cleans pages, ingests them into the active case, and drafts a grounded starting paragraph.

</details>

<details>
<summary>📜 <strong>Case Log</strong></summary>

Inspect the permanent, timestamped history of checked claims, verdicts, explanations, and matched sources.

</details>

<details>
<summary>🗂️ <strong>Cases</strong></summary>

Create, switch, rename, or delete isolated case files. Each case owns its corpus, search index, and history.

</details>

<details>
<summary>🧠 <strong>How It Works</strong></summary>

A plain-English walkthrough of the complete verification pipeline.

</details>

## Architecture

```
Case File ("Q3 Report", "Thesis Ch. 2", ...)
        |  each case = its own corpus + search index + history
        v
Source Corpus (.txt/.md)  <-----------------------------.
        |  saved permanently in SQLite (survives restarts)  \
        v                                                     \  Research Assistant:
        |  chunk (sentence-level, 3 sentences/chunk)            topic -> search queries
        v                                                        -> web search -> fetch
   Embed + Index -----------------------------                   -> clean -> ingest as a
        |                                     |                  source, same as above
   Dense Search (Chroma / cosine)     Sparse Search (BM25 / keyword)
        |____________________  ______________|
                             \/
              merge + normalize + weight
                             |
                    Hybrid Retrieval (top-k)
                             |
                    LLM Judge (NVIDIA NIM)
                             |
        supported  /  contradicted  /  unverified
                             |
              also logged permanently to SQLite (Case Log)
```

- **Backend**: Python + FastAPI, ChromaDB (dense), rank-bm25 (sparse),
  `sentence-transformers` (`BAAI/bge-small-en-v1.5`) for embeddings, NVIDIA NIM
  API (OpenAI-compatible) for the supported/contradicted/unverified judgment,
  **SQLite (via SQLAlchemy)** as the durable source of truth — every case,
  every uploaded document, and every claim ever checked is persisted, so a
  server restart doesn't lose anything — and a **research assistant**
  (`duckduckgo-search` for web search, `requests` + `beautifulsoup4` for
  fetching/cleaning pages, no API key required for search) that can go find
  and ingest sources on its own. Each case gets its own isolated Chroma
  collection + BM25 index, rebuilt from SQLite automatically on startup; the
  embedding model and Chroma connection are shared across cases since they're
  expensive to set up.
- **Frontend**: React + Vite + React Router, plain CSS (an "evidence ledger /
  case file" design system), Axios for API calls. A six-page app, plus a
  case switcher in the header visible everywhere:
  - **Draft** — the live editor. Completed sentences are debounced and sent
    to the backend automatically; the preview marks each one with a colored
    underline and a ✓ / ✕ / — mark, and a rotated "verification stamp"
    tracks your overall grounding score.
  - **Library** — manage the active case's corpus: upload/paste documents,
    browse them as index cards, expand to read the full text, or remove one
    (which also cleans it out of the search index).
  - **Research** — give it a topic; it autonomously searches the web,
    fetches and cleans pages, ingests them into the case's corpus, and drafts
    a grounded starting paragraph you can send straight to Draft.
  - **Case Log** — a permanent, timestamped history of every claim ever
    checked in the active case and what it was judged against.
  - **Cases** — create, switch between, rename, or delete case files. Each
    case is fully isolated: its own corpus, its own search index, its own log.
  - **How It Works** — a plain-English walkthrough of the whole pipeline.
- **Eval**: a small Python script that scores the running pipeline against a
  hand-labeled JSON set of (sentence, expected verdict) pairs.

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite + React Router + Axios + plain CSS |
| Backend | Python + FastAPI |
| Dense retrieval | ChromaDB / cosine similarity |
| Sparse retrieval | rank-bm25 |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| LLM Judge | NVIDIA NIM (OpenAI-compatible API) |
| Persistence | SQLite via SQLAlchemy |
| Research | DuckDuckGo Search + Requests + BeautifulSoup |
| Evaluation | Python evaluation script |

---

## 🚀 Quick Start

> **Prerequisite:** Python 3, Node.js/npm, and an NVIDIA API key.

<details open>
<summary><strong>1️⃣ Backend</strong></summary>

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set NVIDIA_API_KEY

uvicorn app.main:app --reload --port 8000
```

</details>

<details>
<summary><strong>2️⃣ Frontend</strong></summary>

```bash
cd frontend
npm install

cp .env.example .env
# Adjust VITE_API_BASE_URL if needed

npm run dev
```

Default local URL:

```text
http://localhost:5173
```

</details>

<details>
<summary><strong>3️⃣ Evaluation</strong></summary>

```bash
cd eval
pip install requests

python eval.py labeled_set.example.json   --base-url http://localhost:8000   --verbose
```

If you have more than one case:

```bash
python eval.py labeled_set.example.json   --base-url http://localhost:8000   --case-id <id>
```

</details>

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set NVIDIA_API_KEY (get one at build.nvidia.com)
uvicorn app.main:app --reload --port 8000
```

The first run will download the `BAAI/bge-small-en-v1.5` embedding model
(~130MB) from Hugging Face, so it needs network access once.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env   # adjust VITE_API_BASE_URL if needed
npm run dev
```

Open the printed local URL (default `http://localhost:5173`).

### 🖱️ Using It

### 1. 🗂️ Pick or create a case
Use the case switcher in the header or open **Cases**.

### 2. 📚 Add trusted evidence
Upload/paste `.txt` or `.md` documents into **Library**.

### 3. ✍️ Write
Start typing in **Draft**. Completed sentences are checked automatically.

### 4. 🔍 Inspect evidence
Click any checked sentence to view the exact passage and source.

### 5. 🌐 Research
Use **Research** when your case needs additional sources.

### 6. 🔄 Verify again
Research-generated text is inserted into Draft as normal text and goes through the same sentence-by-sentence verification.

> 💡 **Tip:** Treat the Research draft as a starting point, not a finished answer.

## Using it

**First, pick or create a case.** The header always shows a case switcher
(top right) — a dropdown of your existing cases plus a "Manage" link to the
**Cases** page, where you can create new ones, rename them, or delete them.
On a fresh install, Evidentia creates one starter case for you automatically
so you're never staring at an empty app. Everything below — corpus, drafting,
history — belongs to whichever case is currently selected.

**Draft** (the home page)
1. Start typing in the draft box. Whenever you finish a sentence (end
   punctuation + a short pause), it's sent to the backend automatically.
2. The **live preview** marks each checked sentence with a colored underline
   and a small ink mark:
   - 🟢 ✓ = supported by the corpus
   - 🔴 ✕ = contradicted by the corpus
   - ⚪ — = unverified (no relevant evidence, or not a checkable factual claim)
3. Click any checked sentence to see the exact passage and source it was
   checked against.
4. The right panel shows a rotated **verification stamp**: your running
   grounding score, the share of checked claims that are supported.

**Library**
- Upload or paste trusted source documents (`.txt`/`.md` files or pasted
  text) into the active case. Each becomes an index card showing a preview,
  chunk count, and date added.
- Click "Read full text" to expand a document, or "Remove" to delete it
  permanently (this also removes it from that case's search index).
- Everything here is stored in SQLite on the backend — it's all still here
  after you restart the server.

**Research**
- Type a topic or question, hit Start Research. In the background it: expands
  your topic into a handful of distinct search queries, runs a web search for
  each, fetches and cleans up to `RESEARCH_MAX_SOURCES` pages, ingests every
  one straight into the active case's corpus (they show up in Library too),
  and drafts a short paragraph using only what it found.
- The page polls for progress and shows a live log while it runs.
- Once done, review the sources it added and the draft — click "Insert into
  Draft" to send the paragraph to the Draft page, where it runs through the
  exact same sentence-by-sentence verification as anything you type yourself.
  Treat the draft as a rough starting point to edit, not a finished answer.
- Web search is via DuckDuckGo and needs no API key. It can occasionally
  rate-limit or return thin results depending on the topic — if a run comes
  back with no usable sources, try a more specific topic or add sources
  manually in Library instead.

**Case Log**
- A permanent, timestamped list of every claim ever checked in the active
  case, most recent first, with its verdict, the AI's explanation, and the
  source it matched.

**Cases**
- Create a new case, switch your active case, rename one inline, or delete
  one. Deleting a case permanently removes its documents, its check history,
  and its search index — there's a confirmation prompt telling you exactly
  what will be lost before it happens.

**How It Works**
- A plain-English walkthrough of the whole pipeline, for anyone using the
  tool who wants to understand what's happening under the hood.

### Eval

```bash
cd eval
# with the backend running and a corpus already ingested into some case:
pip install requests
python eval.py labeled_set.example.json --base-url http://localhost:8000 --verbose
# if you have more than one case, pass --case-id <id> (see `GET /cases`)
```

## 📊 At-a-Glance

| Component | Purpose |
|---|---|
| 🟢 Supported | Corpus evidence supports the claim |
| 🔴 Contradicted | Corpus evidence conflicts with the claim |
| ⚪ Unverified | Evidence is missing/insufficient |
| 🔎 Hybrid retrieval | Combines semantic + keyword search |
| 💾 SQLite | Durable source of truth |
| 🗂️ Cases | Isolated corpus + index + history |
| 🤖 LLM Judge | Strict evidence-based classification |
| 🌐 Research | Web → cleaned sources → case corpus |

---

## Notes on design decisions

- **Chunking** is sentence-grouped (default 3 sentences/chunk) rather than
  fixed-length token windows, so retrieved evidence reads as coherent claims
  rather than arbitrary text fragments.
- **Hybrid retrieval** min-max normalizes the dense (cosine similarity) and
  sparse (BM25) legs independently before weighting them (`0.6`/`0.4` by
  default, configurable in `.env`), which avoids one signal dominating purely
  because of scale differences between the two scoring systems.
- **Unverified short-circuit**: if the best merged retrieval score falls below
  `MIN_RELEVANCE_SCORE`, the sentence is marked unverified without even
  calling the LLM judge — this keeps latency and API cost down for claims that
  clearly aren't covered by the corpus.
- **The LLM judge** (`app/classifier.py`) is deliberately strict: contradiction
  is checked first and takes priority over support if evidence conflicts, and
  anything ambiguous or off-topic defaults to unverified rather than a false
  "supported".
- **SQLite as the source of truth**: ChromaDB persists its own data to disk,
  but `rank_bm25`'s BM25 index is pure in-memory with no persistence at all.
  Rather than have two different "true" states to keep in sync, SQLite (the
  `documents` and `claim_checks` tables in `app/database.py`) is the single
  durable source of truth for your corpus and your check history. On every
  backend startup, Chroma and BM25 are both rebuilt from SQLite — they're
  treated as disposable, rebuildable *search indexes* over the real data,
  not the real data itself.
- **One search index per case, one embedding model total**: each case gets
  its own Chroma collection and its own in-memory BM25 index, so a claim
  checked in "Thesis Ch. 2" can never accidentally match evidence from "Q3
  Report." Loading the embedding model is slow and memory-hungry, though, so
  `RetrieverRegistry` (in `app/retrieval.py`) loads it exactly once and
  shares it across every case's retriever rather than duplicating it per case.
- **Research runs as a background job, not a long HTTP request**: expanding
  queries, running several web searches, fetching several pages, and drafting
  a paragraph easily takes 20-60+ seconds — too long to hold one HTTP request
  open reliably. `POST /cases/{id}/research` returns a `job_id` immediately;
  the actual work (`app/research.py`) runs via FastAPI's `BackgroundTasks`
  (which Starlette runs in a worker thread automatically for a plain sync
  function), writing progress to the `research_jobs` table in SQLite as it
  goes. The frontend just polls `GET /cases/{id}/research/{job_id}` every
  couple seconds — simple, and it means a job survives you navigating away
  from the Research page mid-run.
- **The research draft is deliberately not trusted**: `draft_from_evidence()`
  is instructed to only state what the fetched evidence actually supports and
  to hedge or omit rather than fill gaps from the model's general knowledge —
  but it's still an LLM completion, not a verified fact. That's why "Insert
  into Draft" doesn't mark anything as pre-approved; it just becomes normal
  draft text, and every sentence in it gets checked from scratch the same way
  anything you typed yourself would be.


---

## 🧠 Verification Pipeline

```text
                 ┌──────────────┐
                 │    Claim     │
                 └──────┬───────┘
                        ↓
          ┌─────────────────────────┐
          │     Hybrid Retrieval    │
          ├────────────┬────────────┤
          │ Dense 0.6  │ Sparse 0.4 │
          └────────────┴────────────┘
                        ↓
               Normalize + merge
                        ↓
                  Top-k evidence
                        ↓
              ┌──────────────────┐
              │ Relevance check  │
              └───────┬──────────┘
                      │
        ┌─────────────┴─────────────┐
        ↓                           ↓
   Low relevance                 Relevant
        ↓                           ↓
 ⚪ Unverified                  LLM Judge
                                    ↓
                         ┌──────────┼──────────┐
                         ↓          ↓          ↓
                    🟢 Supported 🔴 Contrad. ⚪ Unverified
```

---

## 🔬 Why the Architecture Works

<details>
<summary><strong>Sentence-grouped chunking</strong></summary>

The default is three sentences per chunk, keeping retrieved evidence coherent instead of producing arbitrary fragments.

</details>

<details>
<summary><strong>Hybrid retrieval</strong></summary>

Dense cosine similarity and BM25 keyword search are min-max normalized independently and combined with a default `0.6 / 0.4` weighting.

</details>

<details>
<summary><strong>Unverified short-circuit</strong></summary>

Claims below `MIN_RELEVANCE_SCORE` are marked unverified without calling the LLM judge, reducing latency and API cost.

</details>

<details>
<summary><strong>Strict LLM judge</strong></summary>

Contradiction is checked first. Ambiguous or off-topic evidence defaults to unverified rather than being promoted to supported.

</details>

<details>
<summary><strong>SQLite as source of truth</strong></summary>

SQLite stores documents and claim checks permanently. Chroma and BM25 are rebuildable indexes recreated from SQLite on backend startup.

</details>

<details>
<summary><strong>Background research jobs</strong></summary>

Research runs as a background job and exposes a `job_id`, while the frontend polls for progress instead of holding a long HTTP request.

</details>

---

## 📁 Project Map

```text
RAGnarok/
├── backend/      → FastAPI + retrieval + database
├── frontend/     → React/Vite editor UI
├── eval/         → evaluation script + labeled set
└── README.md     → project documentation
```

---

## 🎯 Project Philosophy

> **Evidence should be visible, inspectable, and checkable — not merely generated.**

Evidentia separates:

```text
Retrieval → Evidence → Judgment → Audit Trail
```

so that research-heavy writing can be grounded in an inspectable source corpus.

---

## 🔗 Quick Navigation

[⬆️ Back to top](#-Evidentia--evidence-first-editor) ·
[🎯 What is it?](#-what-is-Evidentia) ·
[🏗️ Architecture](#-architecture) ·
[🚀 Quick Start](#-quick-start) ·
[🖱️ Usage](#-using-it) ·
[🧠 Pipeline](#-verification-pipeline)
