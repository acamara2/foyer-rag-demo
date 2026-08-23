"""
Brique 1 - Parsing des documents PDF.

Choix technique (a documenter honnetement) : la consigne initiale demandait
PyMuPDF (`fitz`). Dans l'environnement de construction/test de cette demo
(bac a sable sans acces reseau sortant), `pymupdf` ne pouvait pas etre
installe. On utilise donc `pypdf` (extraction de texte + signets/outline du
PDF) qui offre les memes capacites pour des PDF texte (non scannes) generes
programmatiquement comme les notres : extraction ligne par ligne et lecture
du sommaire/bookmarks natif du fichier. `pymupdf` reste liste comme piste
d'amelioration dans le README si l'on veut un jour un parsing plus robuste
(mise en page, colonnes, tableaux).

Sortie : un `ParsedDocument` par PDF, avec :
- les lignes de texte (page, numero de ligne, section rattachee),
- les sections issues du sommaire natif du PDF (si disponible),
- des statistiques de parsing, en particulier le `duplication_ratio` :
  le volume de donnees "ligne par ligne" que l'on construit pour permettre
  le retrieval est presque une copie integrale du texte source. C'est une
  vraie question de stockage/versioning a l'echelle, mesuree ici pour de
  vrai (pas une valeur inventee), et affichee dans l'UI.
"""
from __future__ import annotations

import json
import os
import re

from pypdf import PdfReader

from .schemas import ParsedDocument, ParsedLine, ParsingStats, Section

_PLACEHOLDER_TITLE_RE = re.compile(r"^[\s\-_.]*$")


def _is_blank_title(title: str | None) -> bool:
    return title is None or bool(_PLACEHOLDER_TITLE_RE.match(title))


def _extract_sections(reader: PdfReader, n_pages: int) -> tuple[list[Section], bool]:
    """Construit les sections a partir du sommaire natif (bookmarks/outline)
    du PDF. Retourne (sections, toc_degraded)."""
    try:
        outline = reader.outline
    except Exception:
        outline = []

    flat_items = []

    def _walk(items):
        for item in items:
            if isinstance(item, list):
                _walk(item)
            else:
                flat_items.append(item)

    _walk(outline or [])

    if not flat_items:
        return [], True

    entries = []
    for item in flat_items:
        title = getattr(item, "title", None)
        try:
            page_idx = reader.get_destination_page_number(item)
        except Exception:
            page_idx = None
        if page_idx is None:
            continue
        entries.append((title, page_idx + 1))

    if not entries or all(_is_blank_title(t) for t, _ in entries):
        return [], True

    entries.sort(key=lambda t: t[1])

    sections: list[Section] = []
    for i, (title, start_page) in enumerate(entries):
        end_page = entries[i + 1][1] - 1 if i + 1 < len(entries) else n_pages
        end_page = max(end_page, start_page)
        section_id = f"sec-{i + 1:02d}"
        sections.append(
            Section(
                section_id=section_id,
                title=(title or "").strip() or f"Section {i + 1}",
                start_page=start_page,
                end_page=end_page,
            )
        )
    return sections, False


def _section_id_for_page(sections: list[Section], page_num: int) -> str | None:
    for sec in sections:
        if sec.start_page <= page_num <= sec.end_page:
            return sec.section_id
    return None


def parse_pdf(path: str, doc_id: str | None = None) -> ParsedDocument:
    """Parse un seul fichier PDF en `ParsedDocument`."""
    filename = os.path.basename(path)
    doc_id = doc_id or os.path.splitext(filename)[0]
    source_bytes = os.path.getsize(path)

    reader = PdfReader(path)
    n_pages = len(reader.pages)

    sections, toc_degraded = _extract_sections(reader, n_pages)

    lines: list[ParsedLine] = []
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        section_id = _section_id_for_page(sections, page_num)
        line_num = 0
        for raw_line in text.split("\n"):
            stripped = raw_line.strip()
            if not stripped:
                continue
            line_num += 1
            lines.append(
                ParsedLine(
                    page_num=page_num,
                    line_num=line_num,
                    text=stripped,
                    section_id=section_id,
                )
            )

    parsed_payload = [ln.model_dump() for ln in lines]
    parsed_bytes = len(json.dumps(parsed_payload, ensure_ascii=False).encode("utf-8"))
    duplication_ratio = round(parsed_bytes / source_bytes, 3) if source_bytes else 0.0

    stats = ParsingStats(
        n_pages=n_pages,
        n_lines=len(lines),
        source_bytes=source_bytes,
        parsed_bytes=parsed_bytes,
        duplication_ratio=duplication_ratio,
    )

    return ParsedDocument(
        doc_id=doc_id,
        filename=filename,
        lines=lines,
        sections=sections,
        parsing_stats=stats,
        toc_degraded=toc_degraded,
    )


def parse_corpus(raw_docs_dir: str) -> list[ParsedDocument]:
    """Parse tous les PDF d'un dossier (ordre alphabetique, deterministe)."""
    docs: list[ParsedDocument] = []
    if not os.path.isdir(raw_docs_dir):
        return docs
    for filename in sorted(os.listdir(raw_docs_dir)):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(raw_docs_dir, filename)
            docs.append(parse_pdf(path))
    return docs
