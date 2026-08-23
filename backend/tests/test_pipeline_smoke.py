"""
Test de fumee ("smoke test") de la pipeline complete (orchestrator.py),
avec le LLM entierement MOCKE : aucune cle API, aucun appel reseau.

On remplace `generation.call_structured` (le seul point d'entree vers
`llm_client.py`, lui-meme le seul fichier du projet qui importe le SDK
`openai`) par une fausse fonction qui renvoie directement un objet
`AnswerWithEvidence` construit a la main. Cela permet de verifier que
l'orchestrateur cable correctement les 4 briques et produit des objets bien
types (reponse + trace d'audit), pour un cas "reponse trouvee" ET un cas
"abstention", sans jamais toucher au reseau.
"""
from __future__ import annotations

from app import generation
from app.llm_client import LlmCallResult
from app.orchestrator import document_qa
from app.question_parsing import parse_question
from app.retrieval import retrieve
from app.schemas import (
    AbstentionAnswer,
    AnswerWithEvidence,
    AskRequest,
    EvidenceSpan,
    ParsedDocument,
    ParsedLine,
    PipelineTrace,
    Section,
)


def _fake_corpus() -> list[ParsedDocument]:
    lines = [
        ParsedLine(page_num=1, line_num=1, text="Objet du contrat assurance auto.", section_id="sec-01"),
        ParsedLine(page_num=2, line_num=1, text="Garanties incluses.", section_id="sec-02"),
        ParsedLine(
            page_num=2,
            line_num=2,
            text="La garantie bris de glace a un plafond de 1500 EUR et une franchise de 75 EUR.",
            section_id="sec-02",
        ),
    ]
    sections = [
        Section(section_id="sec-01", title="Objet du contrat", start_page=1, end_page=1),
        Section(section_id="sec-02", title="Garanties incluses", start_page=2, end_page=2),
    ]
    return [
        ParsedDocument(
            doc_id="assurance_auto",
            filename="assurance_auto.pdf",
            lines=lines,
            sections=sections,
            toc_degraded=False,
        )
    ]


def _install_fake_llm(monkeypatch, fake_answer: AnswerWithEvidence) -> None:
    def _fake_call_structured(system_prompt, user_prompt, response_schema, model=None):
        return LlmCallResult(parsed=fake_answer, latency_ms=1.0, approx_tokens=42)

    monkeypatch.setattr(generation, "call_structured", _fake_call_structured)


def test_pipeline_returns_typed_answer_for_answerable_question(monkeypatch):
    fake_answer = AnswerWithEvidence(
        answer_found=True,
        complete_answer_found=True,
        value="75 EUR",
        evidence=[
            EvidenceSpan(
                doc_id="assurance_auto",
                page_num=2,
                quote="franchise de 75 EUR",
            )
        ],
        confidence=0.9,
        caveats=[],
    )
    _install_fake_llm(monkeypatch, fake_answer)

    corpus = _fake_corpus()
    answer, trace = document_qa(
        "Quelle est la franchise de la garantie bris de glace ?", corpus
    )

    assert isinstance(answer, AnswerWithEvidence)
    assert isinstance(trace, PipelineTrace)
    assert answer.answer_found is True
    assert trace.anchor_found is True
    assert trace.dispatch_mode in {"sequential", "batch"}
    assert trace.n_llm_calls >= 1
    assert trace.total_latency_ms >= 0


def test_pipeline_abstains_without_calling_llm(monkeypatch):
    # meme si le LLM etait appele, il renverrait ceci - le test verifie
    # justement qu'il n'est PAS appele du tout en cas d'abstention.
    calls = {"count": 0}

    def _should_not_be_called(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("le LLM ne doit pas etre appele en cas d'abstention")

    monkeypatch.setattr(generation, "call_structured", _should_not_be_called)

    corpus = _fake_corpus()
    answer, trace = document_qa(
        "Quelle est la recette du gateau au chocolat ?", corpus
    )

    assert isinstance(answer, AbstentionAnswer)
    assert isinstance(trace, PipelineTrace)
    assert answer.answer_found is False
    assert trace.anchor_found is False
    assert trace.dispatch_mode == "abstention"
    assert trace.n_llm_calls == 0
    assert calls["count"] == 0


def test_ask_request_schema_roundtrip():
    """Petit garde-fou sur le contrat API (schemas.py) : le frontend envoie
    un AskRequest, le backend le valide."""
    req = AskRequest(question="Quelle est la franchise ?")
    assert req.question == "Quelle est la franchise ?"
