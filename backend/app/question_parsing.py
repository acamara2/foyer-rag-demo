"""
Brique 2 - Analyse de la question utilisateur.

Deterministe, sans appel LLM : extraction de mots-cles, expansion via le
lexique metier (lexicon.py), et classification de la "forme" de la question
(single vs listing) via un petit jeu de regles ecrites a la main. C'est ce
choix (pas de LLM ici) qui rend cette brique totalement auditable et rejouable
a l'identique : meme entree -> meme sortie, toujours.

La classification "shape" pilote ensuite la brique 4 : une question "listing"
(ex: "quelles sont toutes les garanties incluses ?") a besoin de voir tout le
contexte pertinent d'un coup (mode batch), alors qu'une question "single"
peut d'abord tenter une recherche sequentielle sur la seule page la plus
pertinente avant d'escalader.
"""
from __future__ import annotations

import re

from .lexicon import expand_terms
from .schemas import ParsedQuestion

# Mots vides francais courants + mots interrogatifs qui n'apportent pas de
# signal de recherche (liste volontairement courte et lisible, pas de lib NLP).
STOPWORDS: set[str] = {
    "le", "la", "les", "l", "un", "une", "des", "de", "du", "d", "au", "aux",
    "et", "ou", "est", "sont", "a", "en", "sur", "dans", "pour", "par", "avec",
    "ce", "cet", "cette", "ces", "qui", "que", "quoi", "quel", "quelle",
    "quels", "quelles", "quels", "qu", "comment", "combien", "pourquoi",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "mon", "ma",
    "mes", "ton", "ta", "tes", "son", "sa", "ses", "notre", "votre", "leur",
    "y", "se", "s", "ne", "pas", "plus", "si", "mais", "donc", "or", "ni",
    "car", "on", "me", "te", "lui", "leurs", "au", "aux", "toutes", "tous",
    "toute", "tout", "c", "n", "etre", "avoir", "cela", "ca",
}

# Termes declencheurs d'une question de type "listing" (recense/enumere).
LISTING_TRIGGERS: list[str] = [
    "liste", "listez", "listes", "quelles sont", "quels sont",
    "tous les", "toutes les", "l'ensemble des", "ensemble des",
    "enumerez", "enumere", "quelles garanties", "quels sont les",
]

_TOKEN_RE = re.compile(r"[a-zàâäéèêëïîôöùûüç0-9']+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    text = text.lower().replace("'", " ")
    return _TOKEN_RE.findall(text.lower())


def extract_keywords(raw_question: str) -> list[str]:
    """Tokenise, met en minuscule et retire les mots vides. Conserve l'ordre
    d'apparition et supprime les doublons."""
    tokens = _tokenize(raw_question)
    keywords: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        if tok in STOPWORDS or len(tok) <= 1:
            continue
        if tok not in seen:
            seen.add(tok)
            keywords.append(tok)
    return keywords


def classify_shape(raw_question: str) -> str:
    """Classifie la question en 'listing' ou 'single' via des declencheurs
    lexicaux simples. Regle explicite, pas d'appel LLM."""
    lowered = raw_question.lower()
    for trigger in LISTING_TRIGGERS:
        if trigger in lowered:
            return "listing"
    return "single"


def parse_question(raw_question: str) -> ParsedQuestion:
    keywords = extract_keywords(raw_question)
    expanded = expand_terms(keywords)
    shape = classify_shape(raw_question)
    return ParsedQuestion(
        raw=raw_question,
        keywords=keywords,
        expanded_keywords=expanded,
        shape=shape,
    )
