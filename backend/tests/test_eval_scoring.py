"""
Tests unitaires de la logique de scoring de backend/data/run_eval.py
(fonction `score_case`), en isolation complete : on construit des objets
`AnswerWithEvidence` / `AbstentionAnswer` / `PipelineTrace` a la main, sans
jamais appeler l'orchestrateur ni le LLM. Cela garantit que le script
d'evaluation lui-meme est correct, independamment de la disponibilite d'une
cle OPENAI_API_KEY dans l'environnement d'execution.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from run_eval import score_case  # noqa: E402
from app.schemas import (  # noqa: E402
    AbstentionAnswer,
    AnswerWithEvidence,
    EvidenceSpan,
    PipelineTrace,
)


def _trace(**overrides) -> PipelineTrace:
    base = dict(
        question_raw="q",
        anchor_found=True,
        low_confidence_anchor=False,
        dispatch_mode="sequential",
        dispatch_reason="",
        n_llm_calls=1,
        approx_tokens_used=100,
        total_latency_ms=250.0,
    )
    base.update(overrides)
    return PipelineTrace(**base)


def test_score_case_correct_answer_with_matching_keyword():
    case = {
        "id": "q01",
        "should_abstain": False,
        "expected_keywords": ["75"],
        "expected_doc_id": "assurance_auto",
    }
    answer = AnswerWithEvidence(
        answer_found=True,
        value="La franchise est de 75 EUR.",
        evidence=[EvidenceSpan(doc_id="assurance_auto", page_num=2, quote="franchise de 75 EUR")],
        confidence=0.85,
    )
    trace = _trace()
    result = score_case(case, answer, trace)

    assert result["abstention_correct"] is True
    assert result["keyword_hit"] is True
    assert result["doc_hit"] is True


def test_score_case_flags_wrong_keyword_as_miss():
    case = {
        "id": "q02",
        "should_abstain": False,
        "expected_keywords": ["100 000"],
        "expected_doc_id": "assurance_vie",
    }
    answer = AnswerWithEvidence(
        answer_found=True,
        value="Le capital n'est pas precise.",
        evidence=[],
        confidence=0.4,
    )
    trace = _trace()
    result = score_case(case, answer, trace)

    assert result["keyword_hit"] is False


def test_score_case_correct_abstention():
    case = {"id": "q03", "should_abstain": True, "expected_keywords": []}
    answer = AbstentionAnswer(reason="Aucune information pertinente trouvee.")
    trace = _trace(anchor_found=False, dispatch_mode="abstention", n_llm_calls=0)
    result = score_case(case, answer, trace)

    assert result["abstention_correct"] is True
    assert result["did_abstain"] is True
    assert result["confidence"] is None


def test_score_case_flags_wrongful_abstention():
    """Le systeme aurait du repondre mais s'est abstenu a tort : doit etre
    detecte comme une erreur d'abstention (abstention_correct=False)."""
    case = {"id": "q04", "should_abstain": False, "expected_keywords": ["75"]}
    answer = AbstentionAnswer(reason="Aucune information pertinente trouvee.")
    trace = _trace(anchor_found=False, dispatch_mode="abstention", n_llm_calls=0)
    result = score_case(case, answer, trace)

    assert result["abstention_correct"] is False


def test_score_case_flags_wrongful_answer_when_should_abstain():
    """Le systeme aurait du s'abstenir mais a repondu quand meme : doit
    aussi etre detecte comme une erreur d'abstention."""
    case = {"id": "q05", "should_abstain": True, "expected_keywords": []}
    answer = AnswerWithEvidence(answer_found=True, value="Une reponse inventee.", confidence=0.3)
    trace = _trace()
    result = score_case(case, answer, trace)

    assert result["abstention_correct"] is False
