"""
Tests unitaires de la brique 1 (parsing.py), contre les VRAIS PDF generes
par backend/data/generate_synthetic_docs.py (pas de mock ici : on verifie
le comportement reel de pypdf sur nos documents synthetiques).
"""
from __future__ import annotations

import os

import pytest

from app.parsing import parse_pdf

RAW_DOCS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw_docs")
)


def _path(filename: str) -> str:
    return os.path.join(RAW_DOCS_DIR, filename)


@pytest.fixture(scope="module")
def auto_doc():
    path = _path("assurance_auto.pdf")
    if not os.path.exists(path):
        pytest.skip(
            "assurance_auto.pdf absent : lancez d'abord "
            "`python backend/data/generate_synthetic_docs.py`."
        )
    return parse_pdf(path)


@pytest.fixture(scope="module")
def degraded_doc():
    path = _path("rapport_sans_sommaire.pdf")
    if not os.path.exists(path):
        pytest.skip(
            "rapport_sans_sommaire.pdf absent : lancez d'abord "
            "`python backend/data/generate_synthetic_docs.py`."
        )
    return parse_pdf(path)


def test_pages_and_lines_are_computed(auto_doc):
    assert auto_doc.parsing_stats is not None
    assert auto_doc.parsing_stats.n_pages > 0
    assert auto_doc.parsing_stats.n_lines > 0
    # une ligne par ligne parsee : coherent avec le nombre reel de ParsedLine
    assert auto_doc.parsing_stats.n_lines == len(auto_doc.lines)


def test_sections_extracted_from_clean_toc(auto_doc):
    assert auto_doc.toc_degraded is False
    titles = [s.title for s in auto_doc.sections]
    assert "Garanties incluses" in titles
    assert "Exclusions" in titles
    # les sections doivent couvrir des plages de pages valides et ordonnees
    for sec in auto_doc.sections:
        assert sec.start_page <= sec.end_page
        assert sec.start_page >= 1
        assert sec.end_page <= auto_doc.parsing_stats.n_pages


def test_duplication_ratio_is_sane(auto_doc):
    stats = auto_doc.parsing_stats
    assert stats.source_bytes > 0
    assert stats.parsed_bytes > 0
    # ratio mesure reellement (pas invente) : borne large mais pas absurde
    # pour un petit document texte (quelques dizaines de Ko).
    assert 0.05 < stats.duplication_ratio < 3.0
    assert stats.duplication_ratio == round(stats.parsed_bytes / stats.source_bytes, 3)


def test_degraded_toc_document_is_flagged(degraded_doc):
    assert degraded_doc.toc_degraded is True
    assert degraded_doc.sections == []
    # le document reste parsable malgre l'absence de sommaire : le texte
    # est toujours extrait ligne par ligne.
    assert degraded_doc.parsing_stats.n_lines > 0
    assert len(degraded_doc.lines) > 0
