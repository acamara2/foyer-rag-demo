"""
Orchestrateur : cable les 4 briques (parsing deja fait en amont -> analyse de
question -> retrieval -> generation) et produit, en plus de la reponse, une
`PipelineTrace` complete pour l'audit a posteriori.

C'est cette trace qui rend le systeme explicable : on peut savoir exactement
quels mots-cles ont matche, quelles pages ont ete candidates et avec quels
scores, si le sommaire etait degrade, pourquoi le mode sequentiel ou batch a
ete choisi, combien d'appels LLM ont ete faits et combien de temps la requete
a pris. Rien n'est cache dans une boite noire de framework.
"""
from __future__ import annotations

import time
from typing import Union

from .generation import generate_answer
from .question_parsing import parse_question
from .retrieval import retrieve
from .schemas import AbstentionAnswer, AnswerWithEvidence, ParsedDocument, PipelineTrace


def document_qa(
    question: str, corpus: list[ParsedDocument]
) -> tuple[Union[AnswerWithEvidence, AbstentionAnswer], PipelineTrace]:
    """Point d'entree principal de la pipeline. `corpus` est la liste des
    documents deja parses (brique 1, executee une fois au demarrage de
    l'API, voir main.py)."""
    t0 = time.perf_counter()

    # Brique 2 : analyse deterministe de la question.
    parsed_question = parse_question(question)

    # Brique 3 : retrieval / ancrage.
    retrieval = retrieve(parsed_question, corpus)

    toc_degraded_docs = [doc.doc_id for doc in corpus if doc.toc_degraded]

    if not retrieval.anchor_found:
        latency_ms = (time.perf_counter() - t0) * 1000
        trace = PipelineTrace(
            question_raw=question,
            keywords_matched=parsed_question.keywords,
            expanded_keywords=parsed_question.expanded_keywords,
            question_shape=parsed_question.shape,
            toc_degraded_docs=toc_degraded_docs,
            candidate_pages_scores=retrieval.scoring_detail,
            anchor_found=False,
            low_confidence_anchor=retrieval.low_confidence_anchor,
            dispatch_mode="abstention",
            dispatch_reason=(
                "aucune page du corpus ne depasse le seuil minimal de "
                "pertinence combinee (mots-cles + sommaire + similarite) : "
                "abstention immediate, aucun appel LLM effectue."
            ),
            n_llm_calls=0,
            approx_tokens_used=0,
            total_latency_ms=round(latency_ms, 2),
        )
        answer = AbstentionAnswer(
            reason=(
                "Aucune information pertinente n'a ete trouvee dans la base "
                "documentaire pour repondre a cette question."
            ),
            caveats=(
                ["Le sommaire d'au moins un document est degrade (illisible) - "
                 "certaines pages n'ont pas pu etre ancrees via leur section."]
                if retrieval.low_confidence_anchor
                else []
            ),
        )
        return answer, trace

    # Brique 4 : generation (avec dispatch sequentiel/batch interne).
    answer, gen_meta = generate_answer(parsed_question, retrieval)

    latency_ms = (time.perf_counter() - t0) * 1000
    trace = PipelineTrace(
        question_raw=question,
        keywords_matched=parsed_question.keywords,
        expanded_keywords=parsed_question.expanded_keywords,
        question_shape=parsed_question.shape,
        toc_degraded_docs=toc_degraded_docs,
        candidate_pages_scores=retrieval.scoring_detail,
        anchor_found=True,
        low_confidence_anchor=retrieval.low_confidence_anchor,
        dispatch_mode=gen_meta["dispatch_mode"],
        dispatch_reason=gen_meta["dispatch_reason"],
        n_llm_calls=gen_meta["n_llm_calls"],
        approx_tokens_used=gen_meta["approx_tokens"],
        total_latency_ms=round(latency_ms, 2),
    )
    return answer, trace
