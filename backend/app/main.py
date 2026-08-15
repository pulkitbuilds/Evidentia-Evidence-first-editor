"""
RAGnarok backend entrypoint.

    Source Corpus -> chunk -> Embed+Index -> Hybrid Retrieval -> LLM Judge -> Verdict

Everything above happens *within a case* -- a case is a separate project
(e.g. "Q3 Report", "Thesis Ch. 2") with its own corpus, its own search index,
and its own history. SQLite (app/database.py) is the durable source of truth
for cases, documents, and checks: every ingested document is saved there, and
on startup each case's search index (Chroma + BM25) is rebuilt from it, so
everything survives a server restart.

Run with:  uvicorn app.main:app --reload
"""
import datetime as dt
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import database, research
from app.chunking import chunk_document
from app.classifier import classify_claim
from app.config import settings
from app.models import (
    CaseCreateRequest,
    CaseListResponse,
    CaseRecord,
    CaseRenameRequest,
    ClaimCheckRequest,
    ClaimCheckResponse,
    ClaimLogResponse,
    CorpusStatus,
    DocumentDetail,
    DocumentListResponse,
    DocumentRecord,
    EvidenceChunk,
    IngestRequest,
    IngestResponse,
    ResearchJobListResponse,
    ResearchJobRecord,
    ResearchStartRequest,
    ResearchStartResponse,
    ScoreRequest,
    ScoreResponse,
    Verdict,
)
from app.retrieval import drop_case_retriever, get_retriever

app = FastAPI(title="RAGnarok: Evidence-First Editor", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_case(case_id: str) -> None:
    if not database.case_exists(case_id):
        raise HTTPException(status_code=404, detail="Case not found.")


@app.on_event("startup")
def on_startup() -> None:
    database.init_db()

    cases = database.list_cases()
    if not cases:
        # First-ever run: give the user one case to start in, rather than an
        # empty app with nowhere to put a document.
        default_id = str(uuid.uuid4())
        database.create_case(default_id, "My First Case")
        cases = database.list_cases()

    # Rebuild every case's search index (Chroma + BM25) from SQLite. Done at
    # startup for all cases up front, rather than lazily per-request, so the
    # very first claim check against any case isn't slowed down by this.
    for case in cases:
        stored_docs = database.get_documents_for_case(case["id"])
        if stored_docs:
            get_retriever(case["id"]).rebuild_from_documents(stored_docs)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ---------------- Cases ----------------

def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat()


@app.post("/cases", response_model=CaseRecord)
def create_case(request: CaseCreateRequest) -> CaseRecord:
    case_id = str(uuid.uuid4())
    database.create_case(case_id, request.name.strip())
    return CaseRecord(id=case_id, name=request.name.strip(), created_at=_now_iso(), document_count=0, check_count=0)


@app.get("/cases", response_model=CaseListResponse)
def list_cases() -> CaseListResponse:
    return CaseListResponse(cases=[CaseRecord(**c) for c in database.list_cases()])


@app.patch("/cases/{case_id}", response_model=dict)
def rename_case(case_id: str, request: CaseRenameRequest) -> dict:
    _require_case(case_id)
    database.rename_case(case_id, request.name.strip())
    return {"renamed": True}


@app.delete("/cases/{case_id}")
def delete_case(case_id: str) -> dict:
    _require_case(case_id)
    database.delete_case(case_id)
    drop_case_retriever(case_id)
    return {"deleted": True}


# ---------------- Corpus (scoped to a case) ----------------

@app.post("/cases/{case_id}/corpus/ingest", response_model=IngestResponse)
def ingest_corpus(case_id: str, request: IngestRequest) -> IngestResponse:
    _require_case(case_id)
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    retriever = get_retriever(case_id)
    docs = [(d.id, d.title, d.text) for d in request.documents]
    chunks_indexed = retriever.add_documents(docs)

    # Persist to SQLite so this document survives a backend restart.
    for d in request.documents:
        n_chunks = len(chunk_document(d.id, d.title, d.text, settings.max_chunk_sentences))
        database.save_document(case_id, d.id, d.title, d.text, n_chunks)

    return IngestResponse(documents_ingested=len(request.documents), chunks_indexed=chunks_indexed)


@app.get("/cases/{case_id}/corpus/status", response_model=CorpusStatus)
def corpus_status(case_id: str) -> CorpusStatus:
    _require_case(case_id)
    retriever = get_retriever(case_id)
    return CorpusStatus(
        documents_indexed=retriever.num_documents,
        chunks_indexed=retriever.num_chunks,
        document_titles=retriever.document_titles,
    )


@app.get("/cases/{case_id}/corpus/documents", response_model=DocumentListResponse)
def list_documents(case_id: str) -> DocumentListResponse:
    _require_case(case_id)
    rows = database.list_documents(case_id)
    return DocumentListResponse(documents=[DocumentRecord(**r) for r in rows])


@app.get("/cases/{case_id}/corpus/documents/{doc_id}", response_model=DocumentDetail)
def get_document(case_id: str, doc_id: str) -> DocumentDetail:
    _require_case(case_id)
    row = database.get_document(case_id, doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentDetail(**row)


@app.delete("/cases/{case_id}/corpus/documents/{doc_id}")
def delete_document(case_id: str, doc_id: str) -> dict:
    _require_case(case_id)
    existed = database.delete_document(case_id, doc_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Document not found.")
    removed_chunks = get_retriever(case_id).remove_document(doc_id)
    return {"deleted": True, "chunks_removed": removed_chunks}


# ---------------- Claim checking (scoped to a case) ----------------

@app.post("/cases/{case_id}/claims/check", response_model=ClaimCheckResponse)
def check_claim(case_id: str, request: ClaimCheckRequest) -> ClaimCheckResponse:
    _require_case(case_id)
    sentence = request.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="Empty sentence.")

    retriever = get_retriever(case_id)

    # Retrieve using the sentence itself; fold in light context to disambiguate
    # pronouns ("it", "this") without diluting the query too much.
    query = sentence if not request.context else f"{request.context.strip()} {sentence}"
    raw_evidence = retriever.search(query)

    best_idx: int | None = None
    if not raw_evidence or raw_evidence[0]["merged_score"] < settings.min_relevance_score:
        verdict = Verdict.UNVERIFIED
        explanation = "No sufficiently relevant evidence found in the corpus."
        evidence_chunks: list[dict] = []
    else:
        verdict, explanation, best_idx = classify_claim(sentence, raw_evidence)
        evidence_chunks = raw_evidence

    evidence_models = [EvidenceChunk(**e) for e in evidence_chunks]
    best_evidence = None
    if evidence_models:
        if best_idx is not None and 0 <= best_idx < len(evidence_models):
            best_evidence = evidence_models[best_idx]
        else:
            best_evidence = evidence_models[0]

    # Log every check to SQLite, powering the Case Log / history page.
    database.save_claim_check(
        case_id=case_id,
        sentence=sentence,
        verdict=verdict.value,
        explanation=explanation,
        best_doc_title=best_evidence.doc_title if best_evidence else None,
        best_evidence_text=best_evidence.text if best_evidence else None,
        merged_score=best_evidence.merged_score if best_evidence else None,
    )

    return ClaimCheckResponse(
        sentence=sentence,
        verdict=verdict,
        explanation=explanation,
        best_evidence=best_evidence,
        evidence=evidence_models,
    )


@app.get("/cases/{case_id}/history/checks", response_model=ClaimLogResponse)
def get_history(case_id: str, limit: int = 100) -> ClaimLogResponse:
    _require_case(case_id)
    rows = database.list_claim_checks(case_id, limit=limit)
    return ClaimLogResponse(checks=rows)


# ---------------- Research assistant (scoped to a case) ----------------

@app.post("/cases/{case_id}/research", response_model=ResearchStartResponse)
def start_research(case_id: str, request: ResearchStartRequest, background_tasks: BackgroundTasks) -> ResearchStartResponse:
    _require_case(case_id)
    job_id = str(uuid.uuid4())
    database.create_research_job(job_id, case_id, request.topic.strip())
    # Starlette runs a sync background task in a worker thread automatically,
    # so this doesn't block the response -- the frontend polls the job below.
    background_tasks.add_task(research.run_research_job, job_id, case_id, request.topic.strip())
    return ResearchStartResponse(job_id=job_id)


@app.get("/cases/{case_id}/research/{job_id}", response_model=ResearchJobRecord)
def get_research_job(case_id: str, job_id: str) -> ResearchJobRecord:
    _require_case(case_id)
    row = database.get_research_job(job_id)
    if not row or row["case_id"] != case_id:
        raise HTTPException(status_code=404, detail="Research job not found.")
    return ResearchJobRecord(**row)


@app.get("/cases/{case_id}/research", response_model=ResearchJobListResponse)
def list_research_jobs(case_id: str, limit: int = 20) -> ResearchJobListResponse:
    _require_case(case_id)
    rows = database.list_research_jobs(case_id, limit=limit)
    return ResearchJobListResponse(jobs=[ResearchJobRecord(**r) for r in rows])


# ---------------- Document-level scoring (stateless, not case-scoped) ----------------

@app.post("/document/score", response_model=ScoreResponse)
def score_document(request: ScoreRequest) -> ScoreResponse:
    total = len(request.verdicts)
    supported = sum(1 for v in request.verdicts if v == Verdict.SUPPORTED)
    contradicted = sum(1 for v in request.verdicts if v == Verdict.CONTRADICTED)
    unverified = sum(1 for v in request.verdicts if v == Verdict.UNVERIFIED)

    grounding_score = (supported / total) if total else 0.0

    return ScoreResponse(
        total_claims=total,
        supported=supported,
        contradicted=contradicted,
        unverified=unverified,
        grounding_score=round(grounding_score, 4),
    )
