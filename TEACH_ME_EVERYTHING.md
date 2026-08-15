# RAGnarok, Explained Like You've Never Coded Before

This document walks through **every file** in this project and explains **why
it exists**, **what it does**, and **the concepts behind it** — assuming zero
prior knowledge. Read it top to bottom, or jump to a section using the list
below.

1. [The one-sentence version](#1-the-one-sentence-version)
2. [The problem we're solving](#2-the-problem-were-solving)
3. [Three ideas you need before anything else makes sense](#3-three-ideas-you-need-before-anything-else-makes-sense)
4. [The architecture diagram, decoded](#4-the-architecture-diagram-decoded)
5. [The backend, file by file](#5-the-backend-file-by-file)
6. [The frontend, file by file](#6-the-frontend-file-by-file)
7. [A single sentence's full journey](#7-a-single-sentences-full-journey)
8. [The eval script](#8-the-eval-script)
9. [Glossary](#9-glossary)

---

## 1. The one-sentence version

**RAGnarok is a text editor that checks every sentence you write against a
pile of trusted documents, and tells you whether that sentence is backed up,
contradicted, or unproven — while you're still typing.**

That's it. Everything below is just "how do you actually build that."

---

## 2. The problem we're solving

Imagine you're writing a report. You type: *"The company's revenue grew 40%
last year."* Is that true? You'd normally have to stop, go find the actual
report, search for the number, come back, and keep writing. Most people
don't bother — the claim just goes into the document unchecked.

RAGnarok automates that stop-and-check step. You give it a folder of source
documents you trust (reports, research notes, whatever). As you type, it
silently checks each sentence you finish against those documents and paints
it:

- 🟢 **green** — a source backs this up
- 🔴 **red** — a source says the opposite
- ⚪ **gray** — no source says anything about this either way

---

## 3. Three ideas you need before anything else makes sense

### 3.1 What is an "embedding"?

Computers don't understand words the way you do — they understand numbers.
An **embedding** is a way of turning a sentence into a long list of numbers
(a "vector") such that **sentences with similar meaning end up with similar
numbers**.

Analogy: imagine plotting every sentence as a dot on a giant map, where
sentences about "dogs" cluster in one neighborhood and sentences about
"taxes" cluster in a totally different neighborhood. An embedding model is
the thing that decides *where on the map* a sentence goes. Two sentences
that are close together on that map probably mean similar things — even if
they don't share a single word in common (e.g. "the canine barked" and "the
dog made noise" would land near each other).

We use a pretrained model called **`BAAI/bge-small-en-v1.5`** to do this
(via the `sentence-transformers` Python library). We didn't train it
ourselves — it already knows how to do this from being trained on huge
amounts of text before we ever used it. We just call it like a function:
`text in -> numbers out`.

### 3.2 Dense search vs. sparse search (why we use *both*)

There are two very different ways to find "the most relevant passage" for a
claim:

- **Dense search** (a.k.a. embedding search / semantic search): turn the
  claim into an embedding (see above), turn every chunk of your corpus into
  an embedding too, and find the chunks whose numbers are "closest" to the
  claim's numbers. This is great at catching *meaning* even when the wording
  is totally different. Weak point: it can sometimes miss exact keywords,
  like a specific product name or number.

- **Sparse search** (a.k.a. keyword search, here done with an algorithm
  called **BM25**): this is the classic "does this document contain these
  exact words, and how rare/important are those words" approach — basically
  a smarter version of Ctrl+F. Great at catching exact terms, numbers, and
  names. Weak point: it doesn't understand meaning or paraphrasing at all.

Analogy: dense search is a librarian who read every book and can point you
to something *about the same topic* even if you describe it clumsily.
Sparse search is the library's exact-word index card catalog — dumb, but
extremely precise when you know the exact term you're looking for.

**We use both and blend their scores together** — this is called **hybrid
retrieval**, and it's the "merge + normalize + weight" step in the
architecture diagram. Using both catches more true matches than either one
alone.

### 3.3 What is an "LLM judge"?

Once we've found the most relevant chunks of evidence (using hybrid
retrieval above), we still don't *know* if the claim is supported or
contradicted — we've only found text that's *related* to the claim, not
necessarily text that agrees or disagrees with it.

So the last step hands the claim + the evidence chunks to a large language
model (an AI like the ones behind ChatGPT/Claude/Llama) and asks it,
essentially: *"Given this evidence, does this claim hold up?"* The AI reads
both and returns one of three labels: supported, contradicted, or
unverified. This is called using the LLM **"as a judge"** — not to generate
new text, but to make a yes/no/unclear decision about existing text.

In this project, that judge is an NVIDIA-hosted model (originally it was
Claude; we swapped to NVIDIA's API, which works almost identically — more on
that in the classifier.py section).

---

## 4. The architecture diagram, decoded

```
Source Corpus (.txt/.md)
        |  chunk (sentence-level, 3 sentences/chunk)
        v
   Embed + Index -----------------------------
        |                                     |
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
```

Reading it top to bottom, in plain English:

1. **Source Corpus** — the documents you trust (your uploaded `.txt`/`.md`
   files or pasted text).
2. **chunk** — we cut those documents into small, digestible pieces (a few
   sentences each), because feeding an entire 50-page document to a search
   system at once is both slow and imprecise. Smaller pieces = more precise
   matches.
3. **Embed + Index** — we run every chunk through the embedding model (turns
   it into numbers) and store those numbers in a searchable database
   (ChromaDB). We *also* build a BM25 keyword index of the same chunks at
   the same time — that's the fork into two paths.
4. **Dense Search** / **Sparse Search** — when a new sentence needs
   checking, we search *both* indexes for the chunks most related to it.
5. **merge + normalize + weight** — dense scores and BM25 scores are on
   totally different number scales (like comparing a temperature in Celsius
   to one in Fahrenheit), so we rescale both onto the same 0–1 scale
   ("normalize"), then combine them with a weighted average (60% dense, 40%
   sparse by default) to get one final ranking.
6. **Hybrid Retrieval (top-k)** — we keep just the best few chunks (the
   "k" in "top-k" — by default 5) from that combined ranking.
7. **LLM Judge** — those top chunks + the original sentence get sent to the
   AI model, which decides the verdict.
8. **Verdict** — supported / contradicted / unverified, shown back in the UI.

---

## 5. The backend, file by file

The backend is written in **Python** using a framework called **FastAPI**.
"Backend" means: the part of the app that does the actual thinking (search,
AI calls, data storage) and exposes it over the network so the website
(frontend) can ask it questions. Think of it as a phone number the frontend
calls to say "hey, is this sentence true?" and gets an answer back.

### `backend/app/config.py` — the settings sheet

```python
class Settings(BaseSettings):
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.1-70b-instruct"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 5
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    ...
```

This is a single place that holds every "knob" the app can be tuned with —
your API key, which AI model to use, how many evidence chunks to retrieve
(`top_k`), how much to trust dense vs. sparse search
(`dense_weight`/`sparse_weight`). It uses a library called
**pydantic-settings** which automatically reads these values from your `.env`
file, so you never hard-code secrets like API keys directly into the code.

**Why this matters as a pattern**: almost every real app separates
*settings* from *logic*. If you want to switch AI models or tune search
weights, you edit `.env` — you never have to touch the actual code.

### `backend/app/models.py` — the shape of the data

```python
class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"

class ClaimCheckRequest(BaseModel):
    sentence: str
    context: str | None = None
```

This file defines the **exact shape** every piece of data must have as it
flows through the app — using a library called **Pydantic**. Think of these
as contracts: "a request to check a claim must contain a `sentence` (text)
and may optionally contain `context` (text)." If the frontend sends
something that doesn't match this shape (e.g. sends a number instead of
text), FastAPI automatically rejects it with a clear error — you don't have
to write that validation by hand.

`Verdict` is an **enum** — a fixed list of allowed values. This guarantees a
verdict can *only* ever be `"supported"`, `"contradicted"`, or
`"unverified"` — never a typo like `"suported"`.

### `backend/app/chunking.py` — cutting documents into pieces

This file takes a big block of text and:

1. Splits it into individual sentences (`split_sentences`) using a regex
   (pattern-matching) rule: a sentence boundary is roughly "a period,
   question mark, or exclamation point, followed by a space, followed by a
   capital letter." It also has a small list of abbreviations (`Dr.`,
   `e.g.`, `U.S.`, etc.) it knows *not* to treat as sentence endings, so
   `"Dr. Smith visited."` doesn't get incorrectly cut into `"Dr."` and
   `"Smith visited."`.
2. Groups sentences into chunks of 3 (`chunk_document`), each chunk getting
   a unique ID like `doc1::chunk::0`.

**Why chunk at all, and why 3 sentences?** If you searched over whole
documents, "most relevant document" is a blunt tool — you'd get a 10-page
report back and still have to hunt for the right paragraph. If you searched
over *single* sentences, you'd sometimes lose important context (e.g. "It
grew 40%" — grew *what*? You need the sentence before it). Three sentences
is a reasonable middle ground: small enough to be precise, big enough to
keep context.

### `backend/app/retrieval.py` — the hybrid search engine

This is the most conceptually dense file, so let's go slow. It defines a
class called `HybridRetriever` — think of a "class" as a blueprint for
creating an object that holds both data and the functions that operate on
that data, bundled together.

**On startup**, it:
```python
self._embedder = SentenceTransformer(settings.embedding_model_name)
self._chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
self._collection = self._chroma_client.create_collection(...)
```
- Loads the embedding model into memory (this is the slow first-run step
  that downloads ~130MB).
- Opens a connection to **ChromaDB**, a database purpose-built for storing
  embeddings and finding "nearest neighbor" vectors quickly. This is our
  dense-search engine.

**`add_documents()`** — when you upload a source document:
```python
embeddings = self._embedder.encode([c.text for c in new_chunks], normalize_embeddings=True)
self._collection.add(ids=..., documents=..., embeddings=..., metadatas=...)
tokenized_corpus = [_tokenize(c.text) for c in self._chunks]
self._bm25 = BM25Okapi(tokenized_corpus)
```
- Every new chunk gets embedded (turned into numbers) and stored in Chroma.
- Separately, *all* chunks (old + new) get re-tokenized (split into lowercase
  words) and used to rebuild a `BM25Okapi` index — that's our sparse-search
  engine. BM25 needs the *whole* corpus to compute word rarity, so it's
  rebuilt from scratch each time you add documents rather than updated
  incrementally.

**`search()`** — the actual hybrid retrieval step:
```python
dense_result = self._collection.query(query_embeddings=..., n_results=...)
dense_scores_by_id = {chunk_id: 1.0 - distance for chunk_id, distance in ...}

bm25_scores = self._bm25.get_scores(tokenized_query)
sparse_scores_by_id = {c.chunk_id: s for c, s in zip(self._chunks, bm25_scores)}

dense_norm = _normalize(dense_raw)
sparse_norm = _normalize(sparse_raw)
merged_score = settings.dense_weight * d + settings.sparse_weight * s
```
1. Ask Chroma for the closest chunks by embedding distance (dense).
2. Ask BM25 for the highest keyword-overlap chunks (sparse).
3. `_normalize()` rescales both sets of scores into the same 0–1 range
   (min-max normalization: the lowest score in the batch becomes 0, the
   highest becomes 1, everything else lands proportionally in between).
   This step exists *specifically* because Chroma's cosine-similarity
   numbers and BM25's scores are not comparable on their own — without
   normalizing, whichever score happens to have bigger raw numbers would
   unfairly dominate the blend.
4. Combine them into one `merged_score` using the configured weights
   (default 60% dense / 40% sparse) and return the top results, sorted best
   first.

### `backend/app/classifier.py` — the LLM judge (now NVIDIA)

```python
def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.nvidia_api_key, base_url=settings.nvidia_base_url)

response = client.chat.completions.create(
    model=settings.nvidia_model,
    messages=[
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(claim, evidence)},
    ],
)
```

Here's the interesting bit for you specifically, since you asked about the
NVIDIA swap: **NVIDIA's NIM API speaks the exact same "language" (API
format) as OpenAI's API.** This is called being **"OpenAI-compatible."**
That means instead of needing a totally different code library, we can use
the standard `openai` Python package and just point it (`base_url`) at
NVIDIA's servers instead of OpenAI's. This is a common trick across the AI
industry — lots of providers (NVIDIA, Groq, Together, etc.) offer
"OpenAI-compatible" endpoints so existing code barely has to change.

The `_SYSTEM_PROMPT` is a big block of instructions we send *every single
time* telling the model exactly how to behave: judge strictly, prefer
"contradicted" over "supported" if both seem to apply, respond with *only*
JSON (no extra chit-chat) so our code can reliably parse it.

```python
try:
    parsed = json.loads(raw_text)
    verdict = Verdict(parsed["verdict"])
    ...
except (json.JSONDecodeError, KeyError, ValueError):
    verdict = Verdict.UNVERIFIED
    ...
```
This is a **fail-safe**: AI models occasionally don't perfectly follow
formatting instructions. Rather than crashing the whole app if the response
isn't valid JSON, we catch that error and just default to "unverified" —
the safest possible fallback (never accidentally telling you something is
"supported" due to a parsing bug).

### `backend/app/main.py` — the actual API (the "phone number")

This file uses **FastAPI** to expose functions over the network as **HTTP
endpoints** — URLs the frontend can send requests to. Two kinds of requests
matter here:

- **GET** — "give me information" (like `/corpus/status` — "how many docs
  do you have indexed?")
- **POST** — "here's some data, do something with it" (like `/claims/check`
  — "here's a sentence, check it")

```python
@app.post("/claims/check", response_model=ClaimCheckResponse)
def check_claim(request: ClaimCheckRequest) -> ClaimCheckResponse:
    ...
    raw_evidence = retriever.search(query)
    if not raw_evidence or raw_evidence[0]["merged_score"] < settings.min_relevance_score:
        verdict = Verdict.UNVERIFIED
        ...
    else:
        verdict, explanation, best_idx = classify_claim(sentence, raw_evidence)
        ...
```

The `@app.post(...)` line is a **decorator** — it's Python's way of saying
"whenever a POST request arrives at this URL, run the function right below
me." Inside `check_claim`, notice the shortcut: if the *best* retrieval
score is below a threshold (`min_relevance_score`, default 0.15), we skip
calling the AI model entirely and just say "unverified" — because if even
the closest match wasn't very close, there's clearly nothing relevant in
your corpus, and calling the (paid, slower) AI model would be a waste.

The other endpoints:
- `/corpus/ingest` — receive uploaded documents, hand them to
  `retriever.add_documents()`.
- `/corpus/status` — report how many documents/chunks are indexed (used by
  the frontend to show "X documents indexed" and to decide whether the
  editor should be enabled yet).
- `/document/score` — given a list of verdicts, compute the summary
  percentage (supported ÷ total).

### `backend/requirements.txt` and `.env.example`

`requirements.txt` lists every external Python library the project depends
on, with `pip install -r requirements.txt` installing all of them at once
(that's what took a while and downloaded PyTorch). `.env.example` is a
template for your secrets/config — you copy it to `.env` (which is never
checked into version control, see `.gitignore`) and fill in your real API
key.

---

## 6. The frontend, file by file

The frontend is what runs *in your browser* — it's what you actually see
and click on. It's built with **React** (a library for building UIs out of
reusable pieces called "components") and **Vite** (a fast tool that runs a
local development server and bundles the code).

### `frontend/index.html` — the actual webpage skeleton

Barely anything here on purpose — just a single empty `<div id="root">` and
a `<script>` tag that loads our React code, plus `<link>` tags pulling in
the three fonts (Fraunces, IBM Plex Sans, IBM Plex Mono) from Google Fonts.
React then takes over and fills that empty div with everything you see.

### `frontend/src/main.jsx` — the entry point

```jsx
ReactDOM.createRoot(document.getElementById('root')).render(<App />)
```
This is the one line that says "take my `<App />` component and actually
mount it into the page." Every React app starts here.

### `frontend/src/api.js` — talking to the backend

```js
const client = axios.create({ baseURL: BASE_URL })

export async function checkClaim(sentence, context) {
  const { data } = await client.post('/claims/check', { sentence, context })
  return data
}
```
**Axios** is a library for making HTTP requests (the browser equivalent of
"call that phone number"). This file is a small wrapper around it — instead
of every component needing to know the exact URL and request format,
they just call `checkClaim("some sentence")` and get back the parsed
result. `async`/`await` here means "this takes some time (a network round
trip) — pause this function until the answer comes back, without freezing
the whole page while we wait."

### `frontend/src/sentences.js` — the frontend's sentence splitter

This mirrors the backend's `chunking.py` logic (same abbreviation list,
similar regex) but runs *in the browser* so the live preview can instantly
show sentence boundaries without waiting on a network request just to know
where one sentence ends and the next begins. It also flags whether the
*last* sentence is `complete` (ends with punctuation) or not — this is how
we know not to check a sentence you're still in the middle of typing.

### `frontend/src/components/Editor.jsx` — the core interactive piece

This is the most important component to understand, so let's go
line-by-line on the tricky parts.

```jsx
const [inFlight, setInFlight] = useState(new Set())
```
`useState` is React's way of saying "this component needs to remember a
piece of information across re-renders, and re-render itself whenever that
information changes." Here we're tracking which sentences currently have a
request in progress (`inFlight`), so we don't send the same sentence twice.

```jsx
const sentences = useMemo(() => splitSentences(text), [text])
```
`useMemo` says "only recompute this expensive thing when `text` actually
changes" — a small performance optimization so we're not re-splitting the
entire draft into sentences on every unrelated re-render.

```jsx
useEffect(() => {
  clearTimeout(timerRef.current)
  timerRef.current = setTimeout(() => {
    const complete = sentences.filter((s) => s.complete && s.text.length > 3)
    complete.forEach((s, idx) => {
      if (cache[s.text] || inFlight.has(s.text)) return
      runCheck(s.text, context)
    })
  }, DEBOUNCE_MS)
  return () => clearTimeout(timerRef.current)
}, [text, corpusReady])
```
This is a **debounce** — a very common pattern any time you're reacting to
fast, repeated events (typing, scrolling, resizing). Without it, we'd fire
off a network request to check a sentence after *every single keystroke*,
which would be wasteful and laggy. Instead: every time `text` changes, we
cancel any pending timer and start a new 700ms countdown
(`DEBOUNCE_MS`). Only if you *stop typing* for 700ms does the check actually
fire. If you keep typing, the timer keeps getting reset and canceled.

`useRef` (`timerRef`) is used here because we need to remember "the current
pending timer" *without* triggering a re-render every time we update it —
`useState` would cause unnecessary re-renders for something that's really
just internal bookkeeping.

```jsx
{s.text}
{result && (
  <span className={`sentence-mark verdict-${result.verdict}`}>
    {result.verdict === 'supported' && '✓'}
    ...
```
Once a sentence has a result, we render a small colored ✓/✕/— mark right
after it, and the sentence itself gets a CSS class like `verdict-supported`
that the stylesheet uses to color its underline.

### `frontend/src/components/CorpusUpload.jsx` — feeding it documents

Handles two ways to add source material: typing/pasting text directly, or
uploading `.txt`/`.md` files (read via the browser's built-in `FileReader`
API, which turns an uploaded file into text your JavaScript can use). Both
paths end up calling the same `onIngest()` function passed down from
`App.jsx`, which calls the backend's `/corpus/ingest` endpoint.

### `frontend/src/components/ScorePanel.jsx` — the verification stamp

Pure arithmetic + display: counts how many checked sentences are
supported/contradicted/unverified, turns that into percentages, and renders
the circular "stamp" badge (styling for the rotation/border comes from
`styles.css` — the component itself just decides *which* color class to
apply based on which verdict is most common).

### `frontend/src/components/EvidencePopover.jsx` — showing the source

When you click a checked sentence, this renders a modal (pop-up box)
showing: the verdict, the AI's one-sentence explanation, the best matching
evidence passage and its source title, the three underlying scores (dense,
sparse, merged), and a collapsible list of any *other* retrieved passages
that weren't the top match.

### `frontend/src/App.jsx` — tying it all together

This is the "root" component. It owns the two pieces of state that need to
be shared across multiple components — `text` (what you've typed) and
`cache` (a dictionary mapping `sentence text -> its check result`) — and
passes them down as props to `Editor`, `ScorePanel`, etc. It also owns
`activeEvidence` (which sentence's popover, if any, is currently open).

**Why keep this state up here instead of inside `Editor.jsx` itself?**
Because `ScorePanel` also needs to know every sentence's result to compute
the percentage — if the cache lived only inside `Editor`, `ScorePanel`
couldn't see it. Lifting shared state up to the nearest common parent is one
of the most fundamental React patterns.

### `frontend/src/styles.css` — the design system

Rather than scattering colors and fonts throughout the code, the top of the
file defines **CSS variables** (custom properties) once:
```css
:root {
  --stamp-green: #2f6b4f;
  --stamp-red: #a43b2e;
  --font-display: "Fraunces", serif;
  ...
}
```
Every other rule in the file references these (`color: var(--stamp-green)`)
instead of hardcoding the hex value again and again. This means changing
the whole app's color scheme is a one-line edit per color, not a
find-and-replace across hundreds of lines.

The design direction is what I called an **"evidence ledger / case file"**
look: warm parchment background, navy header, brass accents, and — the
signature idea — verdicts shown as **editorial redline marks** (colored
underline + ✓/✕/— symbol) rather than flat highlighter blocks, because that
visually matches what the app actually *is*: a copyeditor's markup, not a
generic dashboard.

### `frontend/package.json` and `vite.config.js`

`package.json` lists the frontend's dependencies (React, Axios, Vite) the
same way `requirements.txt` does for Python — `npm install` reads this file
and downloads everything listed. `vite.config.js` configures the dev server
(e.g. which port to run on, `5173` by default) and tells Vite to use the
React plugin so it understands `.jsx` files.

---

## 7. A single sentence's full journey

Let's trace exactly what happens when you type a sentence, end to end,
tying every file above together:

1. You type in the `<textarea>` inside `Editor.jsx`. Every keystroke updates
   `text` (state owned by `App.jsx`).
2. `sentences.js` re-splits the current text into sentence objects on every
   change (via `useMemo`).
3. The debounce timer in `Editor.jsx` resets on every keystroke. Once you
   pause typing for 700ms, it fires.
4. For each newly-completed sentence not already cached or in-flight, the
   frontend calls `checkClaim()` from `api.js`.
5. Axios sends a `POST /claims/check` request to the FastAPI backend
   (`main.py`).
6. FastAPI validates the request shape against `ClaimCheckRequest`
   (`models.py`) — if it's malformed, it's rejected before your code even
   runs.
7. `main.py` calls `retriever.search()` (`retrieval.py`), which:
   a. Embeds your sentence and asks ChromaDB for the closest chunks (dense).
   b. Tokenizes your sentence and asks BM25 for the best keyword matches
      (sparse).
   c. Normalizes and blends both sets of scores into one ranked list.
8. If the best score is too low, `main.py` short-circuits straight to
   "unverified" (no AI call).
9. Otherwise, `main.py` calls `classify_claim()` (`classifier.py`), which
   sends your sentence + the retrieved evidence to the NVIDIA-hosted LLM and
   parses its JSON verdict back.
10. FastAPI packages the verdict + evidence + explanation into a
    `ClaimCheckResponse` and sends it back over the network.
11. Back in `Editor.jsx`, the response lands in `cache`, triggering a
    re-render.
12. The sentence's `<span>` picks up a new CSS class (`verdict-supported`,
    etc.) from `styles.css`, so it visually changes color/underline, and the
    ✓/✕/— mark appears.
13. `ScorePanel.jsx` recomputes its percentages from the same `cache` and
    updates the stamp badge.
14. If you click the sentence, `EvidencePopover.jsx` opens, showing the
    exact evidence passage that drove the verdict.

---

## 8. The eval script

`eval/eval.py` is a small, separate script — it's not part of the running
app, it's a *quality-check tool* for you (the developer). You give it a
JSON file of sentences with a human-decided "correct" verdict for each
(`labeled_set.example.json`), and it sends every sentence to your running
backend, compares the AI's answer to the correct one, and prints an
accuracy percentage plus a **confusion matrix** (a table showing exactly
which verdicts got mixed up with which — e.g. "3 sentences that should have
been 'contradicted' were marked 'unverified' instead"). This is how you'd
measure whether tweaking the prompt, the retrieval weights, or the AI model
actually made the system better or worse, instead of just guessing.

---

## 9. Glossary

| Term | Plain-English meaning |
|---|---|
| **API** | A defined way for two pieces of software to talk to each other — here, the frontend "calling" the backend. |
| **Backend** | The server-side code that does the real work (search, AI calls, data storage). |
| **Frontend** | The code that runs in your browser and renders what you see/click. |
| **Embedding** | Turning text into a list of numbers such that similar meaning = similar numbers. |
| **Vector database** | A database built specifically to store embeddings and quickly find the "nearest" ones (ChromaDB, here). |
| **Dense search** | Search by meaning, using embeddings. |
| **Sparse search** | Search by exact keywords (BM25, here). |
| **Hybrid retrieval** | Combining dense + sparse search results into one ranked list. |
| **Normalization** | Rescaling different scoring systems onto the same range so they can be fairly combined. |
| **LLM** | Large Language Model — an AI trained to understand and generate text (Claude, GPT, Llama, etc.). |
| **LLM-as-judge** | Using an LLM not to write new text, but to make a decision/classification about existing text. |
| **RAG** | Retrieval-Augmented Generation — the general pattern of "search for relevant info first, then have an AI use it," which this whole app is a variant of. |
| **OpenAI-compatible API** | A non-OpenAI service (like NVIDIA's NIM) that accepts requests in the exact same format OpenAI uses, so existing OpenAI-SDK code works with minimal changes. |
| **FastAPI** | A Python framework for building web APIs quickly, with automatic request validation. |
| **Pydantic** | A Python library for defining and validating the "shape" of data. |
| **Endpoint** | A specific URL your backend responds to (e.g. `/claims/check`). |
| **GET / POST** | Two common types of web requests — GET asks for data, POST sends data to be processed. |
| **React** | A JavaScript library for building UIs out of reusable "components." |
| **Component** | A self-contained, reusable piece of UI (e.g. `Editor.jsx`, `ScorePanel.jsx`). |
| **State** | Data a component remembers across re-renders (`useState`). |
| **Props** | Data passed from a parent component down into a child component. |
| **Hook** | A special React function (starting with `use`) that lets components remember state, run side effects, etc. (`useState`, `useEffect`, `useMemo`, `useRef`). |
| **Debounce** | Waiting for a pause in rapid-fire events before reacting, to avoid doing work on every single keystroke. |
| **Vite** | A fast local dev server/build tool for frontend projects. |
| **Axios** | A JavaScript library for making network requests. |
| **CSS variable** | A named, reusable value (like a color) defined once and referenced everywhere. |
| **.env file** | A file holding secret configuration (like API keys) that's kept out of your actual code and out of version control. |
