"""
Brique 3 - Retrieval (ancrage de la question dans le corpus).

Point cle demande par le recruteur : ne pas s'arreter a une seule similarite
cosinus. On combine ici TROIS signaux explicites, calcules independamment et
combines de facon deterministe (somme ponderee) :

1. `keyword_score`   : correspondance lexicale directe (mots-cles + synonymes
                        du lexique metier) entre la question et le texte de
                        chaque page. Signal le plus fiable pour du vocabulaire
                        contractuel precis (montants, noms de garanties...),
                        donc le plus fortement pondere.
2. `toc_score`        : bonus si la page appartient a une section du sommaire
                        dont le titre correspond aux mots-cles de la question.
                        Signal structurel, absent si le sommaire est degrade
                        (voir brique 1) -> on le neutralise proprement plutot
                        que de planter.
3. `embedding_score`  : similarite cosinus TF-IDF (calculee a la main avec
                        numpy, sans scikit-learn ni sentence-transformers -
                        voir README pour la justification de ce choix) entre
                        la question et le texte de chaque page. Signal semantique
                        "flou", utile pour rattraper les reformulations, mais
                        volontairement secondaire ici.

Si aucune page ne depasse un seuil minimal de pertinence, `anchor_found=False`
et l'orchestrateur produit une reponse d'abstention SANS appeler le LLM.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .schemas import ParsedDocument, ParsedQuestion, PageScore, RetrievalResult

_TOKEN_RE = re.compile(r"[a-zàâäéèêëïîôöùûüç0-9']+", re.IGNORECASE)

# Ponderation des 3 signaux dans le score combine.
W_KEYWORD = 0.55
W_TOC = 0.20
W_EMBEDDING = 0.25

# Seuil minimal de score combine pour considerer qu'une ancre a ete trouvee.
ANCHOR_THRESHOLD = 0.12
# Seuil abaisse si le sommaire est degrade (on perd le signal toc_score,
# on compense en etant un peu plus permissif sur keyword+embedding, tout en
# signalant explicitement `low_confidence_anchor=True`).
ANCHOR_THRESHOLD_DEGRADED = 0.08

# Budget de contexte transmis au LLM (nombre de caracteres, approximation
# simple d'un budget de tokens - suffisant pour cette demo).
CONTEXT_CHAR_BUDGET = 6000
MAX_CANDIDATE_PAGES = 6


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _page_text(doc: ParsedDocument, page_num: int) -> str:
    return "\n".join(ln.text for ln in doc.lines if ln.page_num == page_num)


def _page_numbers(doc: ParsedDocument) -> list[int]:
    return sorted({ln.page_num for ln in doc.lines})


# ---------------------------------------------------------------------------
# Signal 1 : score lexical (mots-cles + expansion lexique)
# ---------------------------------------------------------------------------

def _keyword_score(query_terms: list[str], page_tokens: list[str]) -> float:
    if not query_terms or not page_tokens:
        return 0.0
    page_token_set = Counter(page_tokens)
    hits = 0
    for term in query_terms:
        term_tokens = term.split()
        if len(term_tokens) == 1:
            if page_token_set.get(term_tokens[0], 0) > 0:
                hits += 1
        else:
            # terme multi-mots (issu du lexique, ex: "responsabilite civile")
            joined = " ".join(page_tokens)
            if term in joined:
                hits += 1
    return hits / max(len(query_terms), 1)


# ---------------------------------------------------------------------------
# Signal 2 : score structurel (sommaire / sections)
# ---------------------------------------------------------------------------

def _toc_score(doc: ParsedDocument, page_num: int, query_terms: list[str]) -> float:
    if doc.toc_degraded or not doc.sections:
        return 0.0
    for sec in doc.sections:
        if sec.start_page <= page_num <= sec.end_page:
            title_tokens = set(_tokenize(sec.title))
            overlap = sum(1 for t in query_terms if t.split()[0] in title_tokens)
            return min(overlap / max(len(query_terms), 1), 1.0)
    return 0.0


# ---------------------------------------------------------------------------
# Signal 3 : similarite "embedding" TF-IDF ecrite a la main (numpy uniquement)
# ---------------------------------------------------------------------------

class TfidfIndex:
    """Index TF-IDF minimal, ecrit a la main (pas de scikit-learn / pas de
    sentence-transformers) : vocabulaire construit sur l'ensemble des pages
    du corpus, poids IDF classiques, similarite cosinus. Volontairement
    simple : le but est d'avoir un deuxieme signal semantique "faible mais
    independant" du score lexical exact, pas un moteur de recherche
    semantique de pointe."""

    def __init__(self, page_texts: list[str]):
        import numpy as np

        self._np = np
        tokenized = [_tokenize(t) for t in page_texts]
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for tok in set(tokens):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab = vocab
        n_docs = len(tokenized)
        df = np.zeros(len(vocab))
        for tokens in tokenized:
            for tok in set(tokens):
                df[vocab[tok]] += 1
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0

        self.doc_vectors = np.zeros((n_docs, len(vocab)))
        for i, tokens in enumerate(tokenized):
            counts = Counter(tokens)
            for tok, c in counts.items():
                self.doc_vectors[i, vocab[tok]] = c * self.idf[vocab[tok]]
        norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors_norm = self.doc_vectors / norms

    def query_similarities(self, query_terms: list[str]) -> list[float]:
        np = self._np
        vec = np.zeros(len(self.vocab))
        counts = Counter()
        for term in query_terms:
            for tok in _tokenize(term):
                counts[tok] += 1
        for tok, c in counts.items():
            if tok in self.vocab:
                vec[self.vocab[tok]] = c * self.idf[self.vocab[tok]]
        norm = np.linalg.norm(vec)
        if norm == 0:
            return [0.0] * self.doc_vectors_norm.shape[0]
        vec_norm = vec / norm
        sims = self.doc_vectors_norm @ vec_norm
        return [float(max(s, 0.0)) for s in sims]


def retrieve(question: ParsedQuestion, corpus: list[ParsedDocument]) -> RetrievalResult:
    query_terms = question.expanded_keywords or question.keywords

    page_index: list[tuple[str, int, str]] = []  # (doc_id, page_num, text)
    for doc in corpus:
        for page_num in _page_numbers(doc):
            page_index.append((doc.doc_id, page_num, _page_text(doc, page_num)))

    if not page_index:
        return RetrievalResult(anchor_found=False, toc_degraded=True)

    page_texts = [t for (_, _, t) in page_index]
    tfidf = TfidfIndex(page_texts)
    embedding_scores = tfidf.query_similarities(query_terms)

    any_toc_degraded = any(doc.toc_degraded for doc in corpus)
    docs_by_id = {doc.doc_id: doc for doc in corpus}

    scores: list[PageScore] = []
    for (doc_id, page_num, text), emb_score in zip(page_index, embedding_scores):
        page_tokens = _tokenize(text)
        kw_score = _keyword_score(query_terms, page_tokens)
        t_score = _toc_score(docs_by_id[doc_id], page_num, query_terms)
        combined = W_KEYWORD * kw_score + W_TOC * t_score + W_EMBEDDING * emb_score
        scores.append(
            PageScore(
                doc_id=doc_id,
                page_num=page_num,
                keyword_score=round(kw_score, 4),
                toc_score=round(t_score, 4),
                embedding_score=round(emb_score, 4),
                combined_score=round(combined, 4),
            )
        )

    scores.sort(key=lambda s: s.combined_score, reverse=True)

    doc_has_degraded_toc = {s.doc_id: docs_by_id[s.doc_id].toc_degraded for s in scores}
    threshold = ANCHOR_THRESHOLD
    low_confidence = False
    if scores and doc_has_degraded_toc.get(scores[0].doc_id, False):
        threshold = ANCHOR_THRESHOLD_DEGRADED
        low_confidence = True

    top = [s for s in scores if s.combined_score >= threshold][:MAX_CANDIDATE_PAGES]

    if not top:
        return RetrievalResult(
            anchor_found=False,
            scoring_detail=scores[:MAX_CANDIDATE_PAGES],
            toc_degraded=any_toc_degraded,
            low_confidence_anchor=low_confidence,
        )

    context_parts = []
    used_chars = 0
    for s in top:
        doc = docs_by_id[s.doc_id]
        text = _page_text(doc, s.page_num)
        header = f"[Document: {doc.filename} | Page {s.page_num}]"
        block = f"{header}\n{text}\n"
        if used_chars + len(block) > CONTEXT_CHAR_BUDGET and context_parts:
            break
        context_parts.append(block)
        used_chars += len(block)

    return RetrievalResult(
        anchor_found=True,
        candidate_pages=[s.page_num for s in top],
        candidate_doc_ids=[s.doc_id for s in top],
        scoring_detail=scores[:MAX_CANDIDATE_PAGES],
        merged_context="\n".join(context_parts),
        toc_degraded=any_toc_degraded,
        low_confidence_anchor=low_confidence,
        best_page_num=top[0].page_num,
        best_doc_id=top[0].doc_id,
    )
