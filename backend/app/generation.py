"""
Brique 4 - Generation de la reponse avec preuves (evidence).

Dispatch sequentiel/batch (repris du dispatcher decrit dans les articles
source qui inspirent cette demo) :

- question.shape == "single"  -> on tente d'abord un appel LLM "sequentiel"
  sur la SEULE page la mieux notee (`retrieval.best_page_num`). Si le modele
  indique `answer_found=True` ET `complete_answer_found=True`, on s'arrete
  la : un seul appel LLM, un seul appel reseau, latence minimale.
  Sinon, on escalade vers un appel "batch" avec tout le contexte fusionne
  (`merged_context`).
- question.shape == "listing" -> on va direct en mode batch avec tout le
  contexte fusionne, car une question d'enumeration a besoin de voir toutes
  les pages candidates en meme temps pour ne rien oublier.

Discipline d'abstention : le prompt systeme dit explicitement au modele de
poser `answer_found=False` (et de ne rien inventer) si le contexte fourni ne
contient pas la reponse. C'est ce qui permet a la demo de repondre "je ne
sais pas" plutot que d'halluciner.
"""
from __future__ import annotations

from .llm_client import call_structured
from .schemas import AnswerWithEvidence, ParsedQuestion, RetrievalResult

SYSTEM_PROMPT = """Tu es un assistant documentaire pour une compagnie d'assurance.
Tu recois un extrait de contrat d'assurance (conditions generales, grille de
garanties...) et une question posee par un utilisateur.

Regles STRICTES :
1. Reponds UNIQUEMENT a partir du texte fourni dans le contexte. N'utilise
   aucune connaissance externe sur les assurances en general.
2. Si le contexte ne contient PAS la reponse a la question, tu DOIS repondre
   avec answer_found=false. Ne devine jamais, n'invente jamais un chiffre,
   un delai ou une condition qui n'est pas explicitement ecrit dans le texte.
3. Si la question demande une liste (toutes les garanties, tous les cas
   exclus...) et que le contexte ne montre peut-etre pas la liste complete,
   mets complete_answer_found=false meme si answer_found=true, et explique
   pourquoi dans caveats.
4. Chaque element de `evidence` doit contenir une courte citation VERBATIM
   (copiee mot pour mot) du contexte fourni, avec le numero de page exact
   indique dans l'en-tete "[Document: ... | Page N]" juste au-dessus du texte
   dont la citation est extraite. N'invente jamais un numero de page.
5. `confidence` est un nombre entre 0 et 1 refletant ta certitude que la
   reponse est correcte et completement etayee par le contexte.
6. Sois concis. `value` est soit une chaine (question simple), soit une liste
   de chaines (question de type liste/enumeration).
"""


def _build_user_prompt(question: ParsedQuestion, context: str) -> str:
    return (
        f"Question de l'utilisateur : {question.raw}\n\n"
        f"Type de question detecte : {question.shape}\n\n"
        f"Contexte (extraits du/des document(s) de la base documentaire) :\n"
        f"-----\n{context}\n-----\n\n"
        "Reponds au format demande, en respectant strictement les regles du "
        "message systeme."
    )


def _single_page_context(retrieval: RetrievalResult, doc_id: str, page_num: int) -> str:
    """Reconstruit un mini-contexte limite a une seule page, a partir du
    merged_context deja assemble par la brique 3 (on filtre le bloc
    correspondant a la page ciblee)."""
    marker = f"| Page {page_num}]"
    blocks = retrieval.merged_context.split("[Document: ")
    for block in blocks:
        if block.strip() and marker in block:
            return "[Document: " + block
    return retrieval.merged_context


def generate_answer(
    question: ParsedQuestion, retrieval: RetrievalResult
) -> tuple[AnswerWithEvidence, dict]:
    """Retourne (reponse, meta) ou meta contient les infos utiles a la trace
    d'audit : nombre d'appels LLM, mode utilise (sequential/batch), tokens
    approximatifs, raison de la decision de dispatch."""
    meta = {"n_llm_calls": 0, "approx_tokens": 0, "dispatch_mode": "batch", "dispatch_reason": ""}

    if question.shape == "single" and retrieval.best_page_num is not None:
        single_context = _single_page_context(
            retrieval, retrieval.best_doc_id, retrieval.best_page_num
        )
        result = call_structured(
            SYSTEM_PROMPT,
            _build_user_prompt(question, single_context),
            AnswerWithEvidence,
        )
        meta["n_llm_calls"] += 1
        meta["approx_tokens"] += result.approx_tokens
        answer = result.parsed

        if answer.answer_found and answer.complete_answer_found:
            meta["dispatch_mode"] = "sequential"
            meta["dispatch_reason"] = (
                "question de type 'single' : la meilleure page "
                f"(page {retrieval.best_page_num}) contenait une reponse complete, "
                "pas d'escalade necessaire."
            )
            return answer, meta

        meta["dispatch_reason"] = (
            "question de type 'single', mais la page la mieux notee ne "
            "contenait pas de reponse complete -> escalade en mode batch "
            "avec tout le contexte fusionne."
        )
    else:
        meta["dispatch_reason"] = (
            "question de type 'listing' -> mode batch direct avec tout le "
            "contexte fusionne, pour ne pas risquer d'oublier un element "
            "de la liste."
        )

    result = call_structured(
        SYSTEM_PROMPT,
        _build_user_prompt(question, retrieval.merged_context),
        AnswerWithEvidence,
    )
    meta["n_llm_calls"] += 1
    meta["approx_tokens"] += result.approx_tokens
    meta["dispatch_mode"] = "batch"
    return result.parsed, meta
