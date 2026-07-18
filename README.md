# RAGnarok — Evidence-First Editor

A writing tool that checks your factual claims against a document corpus **as you write** —
flagging each sentence as **Supported**, **Contradicted**, or **Unverified**, with the exact
source passage shown alongside it.

Not another "chat with your PDF" bot. RAGnarok is grounding-as-you-type: retrieval and
LLM-based claim classification wired directly into the writing loop.

---

## How it works

1. **Ingest** a corpus of trusted documents (txt/markdown files) — they're chunked and indexed
   with both dense embeddings and BM25 keyword search.
2. **Write** in the editor. On a short pause, each finished sentence is sent to the backend.
3. **Hybrid retrieval** pulls the most relevant chunks (dense + keyword, merged and scored).
4. **LLM classification** compares your sentence against the retrieved evidence and returns a
   label (`supported` / `contradicted` / `unverified`), a confidence score, the specific
   evidence span, and a short explanation.
5. **The Evidence Panel** shows live, color-coded results per sentence, plus a running
   "grounding score" for the whole document.

```
┌─────────────┐      chunk+embed      ┌───────────────┐
│  Corpus dir │ ─────────────────────▶│  Chroma (dense)│
│ (txt/md)    │                       │  BM25 (sparse) │
└─────────────┘                       └───────┬───────┘
                                               │ hybrid search
┌─────────────┐   sentence (on pause)          ▼
│   Editor    │ ─────────────────────▶  ┌─────────────┐
│  (frontend) │                         │  Retrieval   │
│             │◀─── label + evidence ── │  + LLM judge │
└─────────────┘                         └─────────────┘
```

---

## Repo structure

```
ragnarok/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI app, endpoints
│   │   ├── config.py        env vars / settings
│   │   ├── models.py        pydantic request/response schemas
│   │   ├── ingest.py        chunking + indexing
│   │   ├── retrieval.py     hybrid (dense + BM25) search
│   │   └── classify.py      LLM claim-vs-evidence classifier
│   ├── corpus/sample_docs/  starter corpus (solar system facts)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── api.js
│   │   └── components/
│   │       ├── Editor.jsx
│   │       └── ClaimBadge.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── eval/
│   ├── eval_set.json        hand-labeled sentence -> expected label pairs
│   └── run_eval.py          scores the pipeline against eval_set.json
└── README.md
```

---

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your ANTHROPIC_API_KEY
```

Start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Ingest the starter corpus (run once, or whenever you add new docs to `backend/corpus/sample_docs/`):

```bash
curl -X POST http://localhost:8000/ingest
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed localhost URL (default `http://localhost:5173`).

### 3. Try it

Type sentences that are true, false, or unrelated to the sample corpus (solar system facts) and
watch the Evidence Panel classify each one as you pause typing.

---

## Evaluation

```bash
cd eval
python run_eval.py
```

This runs every hand-labeled example in `eval_set.json` against the live backend and prints
accuracy, a confusion breakdown, and per-example results. Numbers go straight into your
portfolio writeup — this is the credibility layer that most RAG demos skip.

---

## Design notes / what to highlight in a portfolio writeup

- **Hybrid retrieval, not vector-only**: dense embeddings catch semantic matches, BM25 catches
  exact terms/numbers — important for factual claims where a single number or name matters.
- **Explicit "unverified" state**: the system is built to say "I don't have evidence for this"
  rather than defaulting to either blind trust or a guess. This is the whole point of the name.
- **Every label is traceable**: no claim is flagged without a specific evidence span the user
  can click through to.
- **Eval-first**: the repo ships with a labeled eval set and a scoring script, not just a demo.

## Roadmap / stretch goals (documented, not built in v1)

- In-place highlighting inside a rich-text editor (contentEditable decorations) instead of the
  side-panel view
- Cross-encoder re-ranking stage between hybrid retrieval and classification
- WebSocket-based live updates instead of debounced HTTP polling
- Multi-document contradiction detection *within* the corpus itself
