"""
Contrats de donnees (Pydantic) partages par les 4 briques et l'API.

Toute la pipeline (parsing -> analyse de question -> retrieval -> generation)
communique exclusivement via ces schemas typés. C'est ce typage explicite qui
rend la decomposition auditable : chaque brique recoit un objet Pydantic
valide en entree et en produit un en sortie, sans etat cache ni "magie" de
framework RAG.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Brique 1 - Parsing
# ---------------------------------------------------------------------------

class ParsedLine(BaseModel):
    """Une ligne de texte extraite d'une page du PDF source."""
    page_num: int
    line_num: int
    text: str
    section_id: Optional[str] = None


class Section(BaseModel):
    """Une section issue du sommaire / des signets (bookmarks) du PDF."""
    section_id: str
    title: str
    start_page: int
    end_page: int


class ParsingStats(BaseModel):
    """
    Mesures reelles de la brique de parsing, notamment le ratio de
    duplication : la representation "ligne par ligne" que l'on construit
    pour permettre le retrieval est presque une copie integrale du texte
    source. Sur un corpus de quelques documents cela ne coute rien, mais
    c'est un vrai probleme de stockage/versioning a l'echelle (voir README,
    section "Choix assumes et limites").
    """
    n_pages: int
    n_lines: int
    source_bytes: int
    parsed_bytes: int
    duplication_ratio: float


class ParsedDocument(BaseModel):
    doc_id: str
    filename: str
    lines: list[ParsedLine] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    parsing_stats: Optional[ParsingStats] = None
    toc_degraded: bool = False


# ---------------------------------------------------------------------------
# Brique 2 - Analyse de la question
# ---------------------------------------------------------------------------

class ParsedQuestion(BaseModel):
    raw: str
    keywords: list[str] = Field(default_factory=list)
    expanded_keywords: list[str] = Field(default_factory=list)
    shape: Literal["single", "listing"] = "single"


# ---------------------------------------------------------------------------
# Brique 3 - Retrieval
# ---------------------------------------------------------------------------

class PageScore(BaseModel):
    doc_id: str
    page_num: int
    keyword_score: float
    toc_score: float
    embedding_score: float
    combined_score: float


class RetrievalResult(BaseModel):
    anchor_found: bool
    candidate_pages: list[int] = Field(default_factory=list)
    candidate_doc_ids: list[str] = Field(default_factory=list)
    scoring_detail: list[PageScore] = Field(default_factory=list)
    merged_context: str = ""
    toc_degraded: bool = False
    low_confidence_anchor: bool = False
    best_page_num: Optional[int] = None
    best_doc_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Brique 4 - Generation
# ---------------------------------------------------------------------------

class EvidenceSpan(BaseModel):
    doc_id: str
    page_num: int
    quote: str


class AnswerWithEvidence(BaseModel):
    answer_found: bool
    complete_answer_found: bool = True
    value: Union[str, list[str]] = ""
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = 0.0
    caveats: list[str] = Field(default_factory=list)


class AbstentionAnswer(BaseModel):
    """
    Reponse structuree explicite en cas d'abandon (aucune ancre de retrieval
    suffisamment fiable). Emise directement par l'orchestrateur SANS appeler
    le LLM, pour ne pas gaspiller un appel et pour eviter toute invention.
    """
    answer_found: Literal[False] = False
    reason: str
    caveats: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Trace d'audit de la pipeline
# ---------------------------------------------------------------------------

class PipelineTrace(BaseModel):
    question_raw: str
    keywords_matched: list[str] = Field(default_factory=list)
    expanded_keywords: list[str] = Field(default_factory=list)
    question_shape: Literal["single", "listing"] = "single"
    toc_degraded_docs: list[str] = Field(default_factory=list)
    candidate_pages_scores: list[PageScore] = Field(default_factory=list)
    anchor_found: bool = False
    low_confidence_anchor: bool = False
    dispatch_mode: Literal["abstention", "sequential", "batch"] = "abstention"
    dispatch_reason: str = ""
    n_llm_calls: int = 0
    approx_tokens_used: int = 0
    total_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# API - requetes / reponses HTTP
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: Union[AnswerWithEvidence, AbstentionAnswer]
    trace: PipelineTrace


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    n_pages: int
    n_sections: int
    toc_degraded: bool
    parsing_stats: ParsingStats


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]
    avg_duplication_ratio: float
