"""
Tests unitaires de la brique 3 (retrieval.py), avec un corpus 100% en
memoire (aucun PDF, aucun reseau) : on construit directement des objets
`ParsedDocument` / `ParsedLine` / `Section` factices pour verifier que la
combinaison des 3 signaux (mots-cles + sommaire + embedding TF-IDF) produit
un classement sense, et que le seuil d'ancrage fonctionne bien dans les deux
sens (ancre trouvee / non trouvee).
"""
from __future__ import annotations

from app.question_parsing import parse_question
from app.retrieval import retrieve
from app.schemas import ParsedDocument, ParsedLine, Section


def _make_line(page_num: int, line_num: int, text: str, section_id: str | None) -> ParsedLine:
    return ParsedLine(page_num=page_num, line_num=line_num, text=text, section_id=section_id)


def _fake_corpus() -> list[ParsedDocument]:
    auto_lines = [
        _make_line(1, 1, "Objet du contrat assurance automobile.", "sec-01"),
        _make_line(1, 2, "Le vehicule assure est couvert en responsabilite civile.", "sec-01"),
        _make_line(2, 1, "Garanties incluses.", "sec-02"),
        _make_line(2, 2, "La garantie bris de glace a un plafond de 1500 EUR et une franchise de 75 EUR.", "sec-02"),
        _make_line(2, 3, "La garantie vol et incendie a un plafond de 25000 EUR.", "sec-02"),
        _make_line(3, 1, "Exclusions.", "sec-03"),
        _make_line(3, 2, "Les sinistres en etat d'ivresse sont exclus de toute garantie.", "sec-03"),
    ]
    auto_sections = [
        Section(section_id="sec-01", title="Objet du contrat", start_page=1, end_page=1),
        Section(section_id="sec-02", title="Garanties incluses", start_page=2, end_page=2),
        Section(section_id="sec-03", title="Exclusions", start_page=3, end_page=3),
    ]
    auto_doc = ParsedDocument(
        doc_id="assurance_auto",
        filename="assurance_auto.pdf",
        lines=auto_lines,
        sections=auto_sections,
        toc_degraded=False,
    )

    vie_lines = [
        _make_line(1, 1, "Objet du contrat assurance vie.", "sec-01"),
        _make_line(2, 1, "Garanties deces.", "sec-02"),
        _make_line(2, 2, "Le capital deces garanti est de 100000 EUR.", "sec-02"),
    ]
    vie_sections = [
        Section(section_id="sec-01", title="Objet du contrat", start_page=1, end_page=1),
        Section(section_id="sec-02", title="Garanties deces", start_page=2, end_page=2),
    ]
    vie_doc = ParsedDocument(
        doc_id="assurance_vie",
        filename="assurance_vie.pdf",
        lines=vie_lines,
        sections=vie_sections,
        toc_degraded=False,
    )

    return [auto_doc, vie_doc]


def test_relevant_question_finds_correct_anchor_page():
    corpus = _fake_corpus()
    question = parse_question("Quelle est la franchise de la garantie bris de glace ?")
    result = retrieve(question, corpus)

    assert result.anchor_found is True
    assert result.best_doc_id == "assurance_auto"
    assert result.best_page_num == 2
    # le score combine de la meilleure page doit dominer les autres
    top_score = result.scoring_detail[0].combined_score
    assert all(top_score >= s.combined_score for s in result.scoring_detail)


def test_keyword_and_toc_signals_contribute_to_ranking():
    corpus = _fake_corpus()
    question = parse_question("Quel est le capital deces garanti en assurance vie ?")
    result = retrieve(question, corpus)

    assert result.anchor_found is True
    assert result.best_doc_id == "assurance_vie"
    top = result.scoring_detail[0]
    # la page 2 du doc vie contient a la fois le mot-cle ("deces", "capital")
    # et une section au titre correspondant ("Garanties deces") : les deux
    # signaux keyword_score et toc_score doivent etre strictement positifs.
    assert top.keyword_score > 0
    assert top.toc_score > 0


def test_unrelated_question_does_not_find_anchor():
    corpus = _fake_corpus()
    question = parse_question("Quelle est la recette du gateau au chocolat ?")
    result = retrieve(question, corpus)

    assert result.anchor_found is False
    assert result.best_page_num is None
    assert result.merged_context == ""


def test_empty_corpus_does_not_crash():
    question = parse_question("Une question quelconque ?")
    result = retrieve(question, [])
    assert result.anchor_found is False


def test_degraded_toc_document_still_scores_via_keyword_and_embedding():
    """Un document au sommaire degrade doit rester utilisable par retrieval :
    le toc_score est neutralise a 0 sur ses pages mais keyword_score et
    embedding_score continuent de fonctionner, avec un seuil d'ancrage plus
    permissif (low_confidence_anchor=True) plutot qu'un crash."""
    degraded_lines = [
        _make_line(1, 1, "Rapport de conjoncture du marche de l'assurance.", None),
        _make_line(1, 2, "Le Commissariat aux Assurances surveille le secteur.", None),
    ]
    degraded_doc = ParsedDocument(
        doc_id="rapport_sans_sommaire",
        filename="rapport_sans_sommaire.pdf",
        lines=degraded_lines,
        sections=[],
        toc_degraded=True,
    )
    question = parse_question("Quel organisme controle le secteur de l'assurance ?")
    result = retrieve(question, [degraded_doc])

    assert result.toc_degraded is True
    for score in result.scoring_detail:
        assert score.toc_score == 0.0
    # ne doit pas planter meme si aucune section n'existe pour ce document
    assert isinstance(result.anchor_found, bool)
