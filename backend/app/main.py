"""
Application FastAPI exposant la pipeline de "chatbot documentaire" via une
API REST typee. Le corpus de PDF (dossier `backend/data/raw_docs/`) est
parse une seule fois au demarrage (brique 1) puis garde en memoire - c'est
volontairement simple, adapte a un petit corpus de demo (voir README pour ce
qui devrait changer a plus grande echelle).
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .orchestrator import document_qa
from .parsing import parse_corpus
from .schemas import (
    AskRequest,
    AskResponse,
    DocumentsResponse,
    DocumentSummary,
    ParsedDocument,
)

RAW_DOCS_DIR = os.environ.get(
    "RAW_DOCS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "raw_docs"),
)

app = FastAPI(
    title="Chatbot documentaire - Foyer (demo)",
    description=(
        "API REST exposant une pipeline RAG faite main (parsing -> analyse "
        "de question -> retrieval -> generation), sans LangChain ni "
        "LlamaIndex, avec trace d'audit complete."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo publique en lecture seule : pas de donnees sensibles utilisateur
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_corpus: list[ParsedDocument] = []


@app.on_event("startup")
def load_corpus() -> None:
    global _corpus
    _corpus = parse_corpus(RAW_DOCS_DIR)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "n_documents": len(_corpus)}


@app.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    if not _corpus:
        load_corpus()
    summaries = []
    ratios = []
    for doc in _corpus:
        stats = doc.parsing_stats
        summaries.append(
            DocumentSummary(
                doc_id=doc.doc_id,
                filename=doc.filename,
                n_pages=stats.n_pages if stats else 0,
                n_sections=len(doc.sections),
                toc_degraded=doc.toc_degraded,
                parsing_stats=stats,
            )
        )
        if stats:
            ratios.append(stats.duplication_ratio)
    avg_ratio = round(sum(ratios) / len(ratios), 3) if ratios else 0.0
    return DocumentsResponse(documents=summaries, avg_duplication_ratio=avg_ratio)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not _corpus:
        load_corpus()
    if not _corpus:
        raise HTTPException(
            status_code=503,
            detail="Aucun document charge : verifiez backend/data/raw_docs/.",
        )
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas etre vide.")
    answer, trace = document_qa(request.question, _corpus)
    return AskResponse(answer=answer, trace=trace)
