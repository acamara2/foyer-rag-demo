"""
Fine wrapper autour de l'API OpenAI, isole dans un seul fichier.

Objectif explicite du recruteur : ne pas cacher les appels LLM derriere un
framework (pas de LangChain/LlamaIndex). Chaque appel au modele est ecrit a
la main ici, via `client.responses.parse(..., text_format=<PydanticModel>)`
(API "Structured Outputs" du SDK OpenAI officiel) qui force contractuellement
le modele a repondre selon un schema Pydantic donne.

Isolation deliberee : c'est le SEUL fichier du projet qui importe le SDK
`openai`. Changer de fournisseur de LLM plus tard (Azure OpenAI, Anthropic,
un modele open-weights via un endpoint compatible...) ne demande de modifier
QUE ce fichier - toutes les autres briques manipulent des objets Pydantic
typés, pas des requetes HTTP.

L'import du SDK `openai` est volontairement paresseux (a l'interieur de
`get_client()`), pour que ce module - et donc tout le reste de la pipeline
qui en depend indirectement - reste importable meme dans un environnement ou
le paquet `openai` n'est pas installe (ex: tests unitaires avec un client
factice, ou execution hors-ligne). Le vrai appel reseau n'est tente qu'au
moment ou l'on a effectivement besoin d'un LLM.
"""
from __future__ import annotations

import os
import time
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

_client = None


def get_client():
    """Cree (et met en cache) le client OpenAI officiel. Leve une erreur
    explicite si le SDK n'est pas installe ou si la cle API est absente -
    volontairement, pour ne jamais echouer silencieusement."""
    global _client
    if _client is not None:
        return _client
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depend de l'environnement
        raise RuntimeError(
            "Le paquet 'openai' n'est pas installe. "
            "Executez `pip install -r backend/requirements.txt`."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Variable d'environnement OPENAI_API_KEY absente. "
            "Voir .env.example a la racine du projet."
        )
    _client = OpenAI(api_key=api_key)
    return _client


class LlmCallResult:
    """Enveloppe le resultat d'un appel structure : l'objet Pydantic parse,
    plus quelques metadonnees utiles pour la trace d'audit (latence, tokens
    approximatifs)."""

    def __init__(self, parsed: BaseModel, latency_ms: float, approx_tokens: int):
        self.parsed = parsed
        self.latency_ms = latency_ms
        self.approx_tokens = approx_tokens


def call_structured(
    system_prompt: str,
    user_prompt: str,
    response_schema: type[T],
    model: str | None = None,
) -> LlmCallResult:
    """Appelle l'API OpenAI Responses avec sortie structuree forcee par
    `response_schema` (un modele Pydantic). Ecrit a la main, aucune couche
    d'abstraction type LangChain entre cet appel et le SDK officiel."""
    client = get_client()
    model_name = model or DEFAULT_MODEL

    start = time.perf_counter()
    response = client.responses.parse(
        model=model_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=response_schema,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    usage = getattr(response, "usage", None)
    approx_tokens = getattr(usage, "total_tokens", None)
    if approx_tokens is None:
        # Repli grossier si l'objet usage n'est pas expose par le SDK :
        # ~4 caracteres par token, approximation suffisante pour la trace.
        approx_tokens = (len(system_prompt) + len(user_prompt)) // 4

    parsed = response.output_parsed
    return LlmCallResult(parsed=parsed, latency_ms=latency_ms, approx_tokens=approx_tokens)
