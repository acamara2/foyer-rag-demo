"""
Script d'evaluation sur le jeu de donnees "golden" (golden_dataset.json).

Charge le corpus reel (brique 1, parsing.py), rejoue chaque question du jeu
de donnees a travers `orchestrator.document_qa()` (donc les 4 briques + la
trace d'audit complete), compare la reponse obtenue a ce qui est attendu par
une simple correspondance de mots-cles/sous-chaines (pas de NLP complexe :
suffisant pour une demo, voir README), et affiche un resume :

- taux d'ancrage retrieval (anchor_found)
- justesse de l'abstention (abstention correcte sur les questions qui
  doivent echouer, ET absence d'abstention a tort sur les questions qui
  doivent reussir)
- confiance moyenne (sur les reponses non-abstention)
- latence moyenne
- nombre moyen d'appels LLM par question

Necessite une cle OPENAI_API_KEY valide dans l'environnement pour executer
reellement la pipeline (les appels a la brique 4 - generation - appellent le
vrai LLM). Sans cle, le script s'arrete avec un message explicite AVANT
d'essayer d'appeler le LLM (voir `_check_api_key`) ; la logique de scoring
elle-meme est testee independamment, sans reseau, dans
backend/tests/test_eval_scoring.py.
"""
from __future__ import annotations

import json
import os
import sys
import time

# Important : on ajoute `backend/` (pas la racine du repo) au sys.path, et on
# importe via `app.xxx` - exactement comme le fait main.py (uvicorn
# app.main:app, lance depuis backend/) et les tests (voir tests/conftest.py).
# Importer via `backend.app.xxx` creerait un DEUXIEME module `schemas` avec
# une identite de classe differente (isinstance() casserait silencieusement
# entre les objets construits ici et ceux produits par l'orchestrateur).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.orchestrator import document_qa  # noqa: E402
from app.parsing import parse_corpus  # noqa: E402
from app.schemas import AbstentionAnswer, AnswerWithEvidence  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(HERE, "golden_dataset.json")
RAW_DOCS_DIR = os.path.join(HERE, "raw_docs")


def load_golden_dataset(path: str = GOLDEN_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _answer_text(answer) -> str:
    """Extrait un texte comparable (minuscule) d'une reponse, qu'elle soit
    une AnswerWithEvidence (value str ou list[str]) ou une AbstentionAnswer."""
    if isinstance(answer, AbstentionAnswer):
        return (answer.reason or "").lower()
    value = answer.value
    if isinstance(value, list):
        text = " ".join(value)
    else:
        text = value or ""
    evidence_text = " ".join(e.quote for e in getattr(answer, "evidence", []))
    return f"{text} {evidence_text}".lower()


def score_case(case: dict, answer, trace) -> dict:
    """Compare la reponse obtenue aux attentes du cas de test. Retourne un
    dict de resultats individuels (utilise a la fois par run_eval.py et par
    les tests unitaires de la logique de scoring)."""
    is_abstention = isinstance(answer, AbstentionAnswer)
    should_abstain = bool(case.get("should_abstain", False))

    abstention_correct = is_abstention == should_abstain

    keyword_hit = False
    if not should_abstain and not is_abstention:
        text = _answer_text(answer)
        expected_keywords = [k.lower() for k in case.get("expected_keywords", [])]
        if not expected_keywords:
            keyword_hit = True  # rien de specifique a verifier
        else:
            keyword_hit = any(kw in text for kw in expected_keywords)

    doc_hit = None
    expected_doc_id = case.get("expected_doc_id")
    if expected_doc_id and not is_abstention:
        cited_docs = {e.doc_id for e in getattr(answer, "evidence", [])}
        doc_hit = expected_doc_id in cited_docs

    return {
        "id": case.get("id", "?"),
        "question": case.get("question", ""),
        "is_edge_case": case.get("is_edge_case", False),
        "should_abstain": should_abstain,
        "did_abstain": is_abstention,
        "abstention_correct": abstention_correct,
        "keyword_hit": keyword_hit,
        "doc_hit": doc_hit,
        "anchor_found": trace.anchor_found,
        "low_confidence_anchor": trace.low_confidence_anchor,
        "confidence": getattr(answer, "confidence", None) if not is_abstention else None,
        "n_llm_calls": trace.n_llm_calls,
        "latency_ms": trace.total_latency_ms,
        "dispatch_mode": trace.dispatch_mode,
    }


def _check_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERREUR : la variable d'environnement OPENAI_API_KEY est absente.\n"
            "L'evaluation complete a besoin d'appeler le vrai LLM (brique 4) "
            "pour chaque question du jeu de donnees. Definissez OPENAI_API_KEY "
            "(voir .env.example a la racine du projet) puis relancez :\n\n"
            "    export OPENAI_API_KEY=sk-...\n"
            "    python backend/data/run_eval.py\n\n"
            "La logique de scoring (comparaison reponse attendue / obtenue) "
            "est testee independamment, sans reseau, dans "
            "backend/tests/test_eval_scoring.py.",
            file=sys.stderr,
        )
        sys.exit(1)


def run() -> None:
    _check_api_key()

    print(f"Chargement du corpus depuis {RAW_DOCS_DIR} ...")
    corpus = parse_corpus(RAW_DOCS_DIR)
    print(f"{len(corpus)} document(s) charge(s).\n")

    cases = load_golden_dataset()
    results = []

    for case in cases:
        t0 = time.perf_counter()
        answer, trace = document_qa(case["question"], corpus)
        elapsed = (time.perf_counter() - t0) * 1000
        result = score_case(case, answer, trace)
        results.append(result)

        status = "OK" if result["abstention_correct"] and (result["keyword_hit"] or result["should_abstain"]) else "A VERIFIER"
        print(
            f"[{result['id']}] {status:11s} | abstention correcte={result['abstention_correct']} "
            f"| mots-cles trouves={result['keyword_hit']} | anchor_found={result['anchor_found']} "
            f"| confiance={result['confidence']} | mode={result['dispatch_mode']} "
            f"| appels_llm={result['n_llm_calls']} | latence={elapsed:.0f}ms"
        )

    print("\n" + "=" * 70)
    print("RESUME")
    print("=" * 70)

    n = len(results)
    anchor_rate = sum(1 for r in results if r["anchor_found"]) / n
    abstention_ok = sum(1 for r in results if r["abstention_correct"]) / n

    non_abstained = [r for r in results if not r["should_abstain"]]
    keyword_hit_rate = (
        sum(1 for r in non_abstained if r["keyword_hit"]) / len(non_abstained)
        if non_abstained
        else 0.0
    )

    confidences = [r["confidence"] for r in results if r["confidence"] is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    avg_latency = sum(r["latency_ms"] for r in results) / n
    avg_llm_calls = sum(r["n_llm_calls"] for r in results) / n

    edge_cases = [r for r in results if r["is_edge_case"]]

    print(f"Questions evaluees              : {n}")
    print(f"Taux d'ancrage (anchor_found)    : {anchor_rate:.0%}")
    print(f"Justesse de l'abstention         : {abstention_ok:.0%}")
    print(f"Taux de bonne reponse (mots-cles): {keyword_hit_rate:.0%} (sur {len(non_abstained)} questions non-abstention)")
    print(f"Confiance moyenne (LLM)          : {avg_confidence:.2f}")
    print(f"Latence moyenne                  : {avg_latency:.0f} ms")
    print(f"Appels LLM moyens / question     : {avg_llm_calls:.2f}")
    print(f"Cas limites (is_edge_case)       : {len(edge_cases)} question(s), best-effort (voir golden_dataset.json)")

    out_path = os.path.join(HERE, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultats detailles ecrits dans {out_path}")


if __name__ == "__main__":
    run()
