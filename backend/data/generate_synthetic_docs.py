"""
Generateur du corpus PDF synthetique utilise par la demo.

Produit 5 PDF "reels" (pas des scans, du texte + une vraie table des matieres
native / bookmarks PDF) avec du contenu francais invente (assurance), via
`reportlab` (platypus). Les bookmarks sont poses au niveau des titres de
section (style "Heading1") grace a un hook `afterFlowable` qui appelle
`canvas.bookmarkPage()` + `canvas.addOutlineEntry()` - ce sont des bookmarks
PDF standards, lisibles ensuite par `pypdf` (voir backend/app/parsing.py)
exactement comme le sommaire natif d'un vrai PDF.

Volontairement AUCUNE dependance a PyMuPDF ici : le meme constat que dans
parsing.py s'applique (pas d'acces reseau dans le bac a sable de
construction pour installer `pymupdf`). `reportlab` suffit amplement pour
generer des bookmarks PDF standards.

5 documents generes dans backend/data/raw_docs/ :
1. assurance_auto.pdf         - TOC propre, table de garanties.
2. assurance_habitation.pdf   - TOC propre, UNE clause qui chevauche
                                 deliberement deux sections (Degats des eaux
                                 / Responsabilite civile).
3. assurance_sante.pdf        - TOC propre, tableau comparatif deux
                                 "formules" cote a cote (ambiguite d'ordre de
                                 lecture a l'extraction).
4. assurance_vie.pdf          - TOC propre, sections standards.
5. rapport_sans_sommaire.pdf  - AUCUN bookmark exploitable (toc_degraded),
                                 contenu plausible mais hors-sujet assurance
                                 contractuelle, pour exercer l'abstention.
"""
from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "raw_docs")

STYLES = getSampleStyleSheet()
STYLES.add(
    ParagraphStyle(
        name="BodyJustify",
        parent=STYLES["BodyText"],
        alignment=TA_JUSTIFY,
        spaceAfter=10,
        leading=14,
    )
)
STYLES.add(
    ParagraphStyle(
        name="DocTitle",
        parent=STYLES["Title"],
        spaceAfter=18,
    )
)


class BookmarkedDocTemplate(SimpleDocTemplate):
    """SimpleDocTemplate qui pose automatiquement un bookmark PDF natif
    (utilisable comme sommaire par pypdf) a chaque paragraphe de style
    'Heading1' rencontre pendant la construction du document."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bookmark_counter = 0

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "Heading1":
            text = flowable.getPlainText()
            key = f"bm-{self._bookmark_counter}"
            self._bookmark_counter += 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=0, closed=0)


def _heading(text: str) -> Paragraph:
    return Paragraph(text, STYLES["Heading1"])


def _sub(text: str) -> Paragraph:
    return Paragraph(text, STYLES["Heading2"])


def _p(text: str) -> Paragraph:
    return Paragraph(text, STYLES["BodyJustify"])


def _garanties_table(rows: list[tuple[str, str, str]]) -> Table:
    data = [["Garantie", "Plafond", "Franchise"]] + rows
    t = Table(data, colWidths=[7 * cm, 5 * cm, 4 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


# ---------------------------------------------------------------------------
# 1. assurance_auto.pdf
# ---------------------------------------------------------------------------

def build_assurance_auto(path: str) -> None:
    doc = BookmarkedDocTemplate(
        path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
    )
    story = [
        Paragraph("Conditions Generales - Assurance Automobile", STYLES["DocTitle"]),
        _p(
            "Le present document decrit les conditions generales applicables au "
            "contrat d'assurance automobile souscrit aupres de notre compagnie. "
            "Il precise l'objet du contrat, les garanties incluses et optionnelles, "
            "les exclusions, les franchises applicables ainsi que les modalites de "
            "declaration de sinistre."
        ),
        PageBreak(),
        _heading("Objet du contrat"),
        _p(
            "Le present contrat a pour objet de garantir le souscripteur, en sa "
            "qualite de conducteur habituel du vehicule assure designe aux "
            "conditions particulieres, contre les consequences pecuniaires de sa "
            "responsabilite civile ainsi que, selon les garanties souscrites, "
            "contre certains dommages subis par le vehicule lui-meme. Le "
            "vehicule assure doit etre immatricule et utilise dans le cadre d'un "
            "usage prive ou professionnel declare aux conditions particulieres."
        ),
        _p(
            "Le souscripteur s'engage a declarer avec exactitude tout changement "
            "de circonstances susceptible d'aggraver le risque (changement de "
            "vehicule, changement de conducteur habituel, usage professionnel "
            "nouveau) dans un delai de quinze jours."
        ),
        PageBreak(),
        _heading("Garanties incluses"),
        _p(
            "Les garanties suivantes sont incluses de plein droit dans tout "
            "contrat d'assurance automobile, quelle que soit la formule "
            "souscrite. Elles ne peuvent faire l'objet d'aucune exclusion "
            "contractuelle contraire aux dispositions legales en vigueur au "
            "Grand-Duche de Luxembourg."
        ),
        _garanties_table(
            [
                ("Responsabilite civile", "Illimite (dommages corporels)", "Aucune"),
                ("Defense penale et recours", "10 000 EUR", "Aucune"),
                ("Assistance 0 km", "Illimite (remorquage inclus)", "Aucune"),
            ]
        ),
        Spacer(1, 10),
        _p(
            "La garantie Responsabilite civile couvre les dommages corporels et "
            "materiels causes a des tiers du fait de la circulation du vehicule "
            "assure. La garantie Assistance 0 km s'applique des le lieu de "
            "l'immobilisation du vehicule, y compris au pied du domicile du "
            "souscripteur."
        ),
        PageBreak(),
        _heading("Garanties optionnelles"),
        _p(
            "Les garanties suivantes peuvent etre ajoutees au contrat de base, "
            "moyennant une prime complementaire indiquee aux conditions "
            "particulieres."
        ),
        _garanties_table(
            [
                ("Bris de glace", "1 500 EUR", "75 EUR"),
                ("Vol et incendie", "25 000 EUR", "300 EUR"),
                ("Dommages tous accidents", "Valeur venale du vehicule", "450 EUR"),
            ]
        ),
        Spacer(1, 10),
        _p(
            "La garantie Bris de glace couvre le remplacement ou la reparation "
            "du pare-brise, des vitres laterales et de la lunette arriere. La "
            "garantie Dommages tous accidents s'applique meme en l'absence de "
            "tiers identifie, y compris en cas de collision avec un animal ou "
            "un obstacle fixe."
        ),
        PageBreak(),
        _heading("Exclusions"),
        _p("Sont exclus de toutes les garanties du present contrat, sans exception :"),
        _p(
            "- les sinistres survenus alors que le conducteur se trouvait en "
            "etat d'ivresse manifeste ou sous l'emprise de stupefiants ;<br/>"
            "- les sinistres resultant d'un usage du vehicule non conforme a "
            "l'usage declare aux conditions particulieres (par exemple un usage "
            "professionnel intensif non declare) ;<br/>"
            "- les dommages resultant d'un defaut d'entretien manifeste et "
            "connu du souscripteur ;<br/>"
            "- les dommages survenus lors de la participation a des competitions "
            "sportives motorisees, courses ou rallyes, meme amicaux ;<br/>"
            "- les dommages resultant directement ou indirectement d'un fait de "
            "guerre, d'emeute ou de mouvement populaire."
        ),
        PageBreak(),
        _heading("Franchises"),
        _p(
            "Une franchise est le montant qui reste a la charge du souscripteur "
            "lors de l'indemnisation d'un sinistre relevant d'une garantie "
            "optionnelle. Les montants de franchise applicables a chaque "
            "garantie sont rappeles dans les tableaux des sections Garanties "
            "incluses et Garanties optionnelles ci-avant. Aucune franchise n'est "
            "appliquee sur la garantie Responsabilite civile ni sur la garantie "
            "Assistance 0 km."
        ),
        _p(
            "En cas de sinistres multiples relevant de la meme garantie au "
            "cours d'une meme annee d'assurance, la franchise s'applique "
            "independamment a chaque sinistre declare."
        ),
        PageBreak(),
        _heading("Declaration de sinistre (delais)"),
        _p(
            "Tout sinistre doit etre declare a l'assureur dans un delai de cinq "
            "jours ouvres a compter de sa survenance ou de sa connaissance par "
            "le souscripteur. Ce delai est reduit a deux jours ouvres en cas de "
            "vol du vehicule."
        ),
        _p(
            "La declaration doit etre accompagnee, dans la mesure du possible, "
            "d'un constat amiable signe par les parties impliquees, de photos "
            "des dommages et, en cas de vol, du recepisse de depot de plainte "
            "aupres des services de police. Tout retard injustifie dans la "
            "declaration peut entrainer une reduction de l'indemnite due, "
            "proportionnelle au prejudice que ce retard a cause a l'assureur."
        ),
    ]
    doc.build(story)


# ---------------------------------------------------------------------------
# 2. assurance_habitation.pdf
# ---------------------------------------------------------------------------

def build_assurance_habitation(path: str) -> None:
    doc = BookmarkedDocTemplate(
        path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
    )
    story = [
        Paragraph("Conditions Generales - Assurance Habitation", STYLES["DocTitle"]),
        _p(
            "Le present document decrit les conditions generales applicables au "
            "contrat d'assurance habitation couvrant la residence principale ou "
            "secondaire designee aux conditions particulieres."
        ),
        PageBreak(),
        _heading("Objet du contrat"),
        _p(
            "Le present contrat a pour objet de garantir le souscripteur contre "
            "les dommages materiels affectant le bien immobilier assure ainsi "
            "que son contenu (mobilier, objets personnels), et contre les "
            "consequences pecuniaires de sa responsabilite civile en tant "
            "qu'occupant du logement."
        ),
        PageBreak(),
        _heading("Garanties incluses"),
        _garanties_table(
            [
                ("Incendie et evenements assimiles", "300 000 EUR", "150 EUR"),
                ("Degats des eaux", "150 000 EUR", "100 EUR"),
                ("Vol et vandalisme", "20 000 EUR", "200 EUR"),
                ("Bris de glace", "5 000 EUR", "50 EUR"),
            ]
        ),
        PageBreak(),
        _heading("Degats des eaux"),
        _p(
            "La garantie Degats des eaux couvre les dommages resultant de "
            "fuites, infiltrations, ruptures ou debordements de canalisations, "
            "d'appareils a effet d'eau ou de toitures, survenus dans le logement "
            "assure. Elle prend en charge les frais de recherche de fuite ainsi "
            "que la remise en etat des locaux et du mobilier endommage."
        ),
        _p(
            "Lorsque le degat des eaux survenu dans le logement assure cause "
            "egalement un prejudice a un tiers, par exemple un voisin ou la "
            "copropriete (infiltration chez le voisin du dessous, degats aux "
            "parties communes de l'immeuble), la garantie Responsabilite civile "
            "du present contrat s'applique en complement de la garantie Degats "
            "des eaux, dans les limites et conditions precisees a la section "
            "Responsabilite civile ci-apres. Le souscripteur doit dans ce cas "
            "declarer un seul et meme sinistre, en mentionnant explicitement "
            "l'existence d'un tiers lese."
        ),
        PageBreak(),
        _heading("Responsabilite civile"),
        _p(
            "La garantie Responsabilite civile couvre les consequences "
            "pecuniaires de la responsabilite que le souscripteur peut encourir "
            "en raison des dommages corporels, materiels et immateriels causes "
            "a des tiers du fait de son habitation, dans la limite de 500 000 "
            "EUR par sinistre. Comme indique a la section Degats des eaux, cette "
            "garantie intervient notamment lorsqu'un degat des eaux imputable au "
            "logement assure cause un prejudice a un voisin ou a la copropriete."
        ),
        PageBreak(),
        _heading("Exclusions"),
        _p(
            "Sont exclus de toutes les garanties du present contrat : les "
            "dommages resultant d'un defaut d'entretien manifeste du logement, "
            "les dommages survenus alors que le logement est demeure inoccupe "
            "plus de quatre-vingt-dix jours consecutifs sans declaration "
            "prealable, les dommages resultant d'un usage professionnel non "
            "declare des locaux, ainsi que les dommages resultant directement "
            "ou indirectement d'un fait de guerre ou d'emeute."
        ),
        PageBreak(),
        _heading("Franchises"),
        _p(
            "Les montants de franchise applicables a chaque garantie sont "
            "rappeles dans le tableau de la section Garanties incluses "
            "ci-avant. Aucune franchise n'est appliquee sur la garantie "
            "Responsabilite civile."
        ),
        PageBreak(),
        _heading("Sinistres"),
        _p(
            "Tout sinistre doit etre declare a l'assureur dans un delai de cinq "
            "jours ouvres, reduit a deux jours ouvres en cas de vol ou de "
            "vandalisme. La declaration doit preciser la nature du sinistre, sa "
            "date de survenance estimee, et etre accompagnee de toute piece "
            "justificative disponible (photos, factures, constat amiable)."
        ),
    ]
    doc.build(story)


# ---------------------------------------------------------------------------
# 3. assurance_sante.pdf
# ---------------------------------------------------------------------------

def build_assurance_sante(path: str) -> None:
    doc = BookmarkedDocTemplate(
        path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
    )
    comparatif = Table(
        [
            ["Critere", "Formule Essentielle", "Formule Confort"],
            ["Hospitalisation", "Remboursement 100% frais reels", "Remboursement 100% frais reels + chambre particuliere"],
            ["Consultations medecin generaliste", "Remboursement a 80%", "Remboursement a 100%"],
            ["Optique (monture + verres)", "150 EUR / 2 ans", "400 EUR / 2 ans"],
            ["Dentaire (soins et prothese)", "Remboursement a 70%", "Remboursement a 90%"],
            ["Medecines douces (osteopathie...)", "Non couvert", "3 seances/an, 40 EUR/seance"],
        ],
        colWidths=[6.5 * cm, 5 * cm, 5.5 * cm],
    )
    comparatif.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story = [
        Paragraph("Conditions Generales - Assurance Sante", STYLES["DocTitle"]),
        _p(
            "Le present document decrit les conditions generales applicables au "
            "contrat d'assurance sante complementaire, disponible en deux "
            "formules : Formule Essentielle et Formule Confort."
        ),
        PageBreak(),
        _heading("Objet du contrat"),
        _p(
            "Le present contrat a pour objet de completer les remboursements de "
            "l'assurance maladie legale obligatoire pour les frais medicaux, "
            "d'hospitalisation, dentaires et optiques du souscripteur et des "
            "personnes a charge designees aux conditions particulieres."
        ),
        PageBreak(),
        _heading("Formules et garanties"),
        _p(
            "Le tableau ci-dessous presente, cote a cote pour faciliter la "
            "comparaison, le detail des garanties offertes par chacune des deux "
            "formules disponibles."
        ),
        comparatif,
        Spacer(1, 10),
        _p(
            "Le passage d'une formule a l'autre peut etre demande a tout moment "
            "par le souscripteur, avec effet au premier jour du mois suivant la "
            "demande. La Formule Confort inclut par ailleurs une garantie "
            "medecines douces absente de la Formule Essentielle."
        ),
        PageBreak(),
        _heading("Exclusions"),
        _p(
            "Sont exclus de toutes les garanties, quelle que soit la formule "
            "souscrite : la chirurgie et les actes a visee esthetique non "
            "reconstructrice, les cures thermales non prescrites medicalement, "
            "les traitements de fertilite au-dela de trois tentatives, ainsi "
            "que les frais engages hors de l'Union europeenne sans accord "
            "prealable de l'assureur."
        ),
        PageBreak(),
        _heading("Franchises"),
        _p(
            "Aucune franchise n'est appliquee sur les garanties de la Formule "
            "Essentielle ni sur celles de la Formule Confort : les montants "
            "indiques dans le tableau de la section Formules et garanties "
            "s'entendent nets, sans reste a charge supplementaire, hors depassements "
            "d'honoraires non conventionnes."
        ),
        PageBreak(),
        _heading("Sinistres"),
        _p(
            "Les demandes de remboursement doivent etre transmises a l'assureur "
            "dans un delai de deux ans a compter de la date des soins, "
            "accompagnees des decomptes de l'assurance maladie legale et des "
            "factures acquittees originales."
        ),
    ]
    doc.build(story)


# ---------------------------------------------------------------------------
# 4. assurance_vie.pdf
# ---------------------------------------------------------------------------

def build_assurance_vie(path: str) -> None:
    doc = BookmarkedDocTemplate(
        path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
    )
    story = [
        Paragraph("Conditions Generales - Assurance Vie", STYLES["DocTitle"]),
        _p(
            "Le present document decrit les conditions generales applicables au "
            "contrat d'assurance vie individuel, combinant une garantie en cas "
            "de deces et une garantie en cas de vie a l'echeance."
        ),
        PageBreak(),
        _heading("Objet du contrat"),
        _p(
            "Le present contrat a pour objet le versement d'un capital ou "
            "d'une rente au beneficiaire designe, en cas de deces du "
            "souscripteur survenant pendant la duree du contrat, ou au "
            "souscripteur lui-meme si celui-ci est en vie a la date "
            "d'echeance contractuelle."
        ),
        PageBreak(),
        _heading("Garanties deces"),
        _p(
            "En cas de deces du souscripteur pendant la duree du contrat, un "
            "capital deces de 100 000 EUR est verse au beneficiaire designe aux "
            "conditions particulieres, dans un delai de trente jours a compter "
            "de la reception de l'ensemble des pieces justificatives (acte de "
            "deces, piece d'identite du beneficiaire, releve d'identite "
            "bancaire)."
        ),
        _p(
            "Si aucun beneficiaire n'a ete designe ou si le beneficiaire "
            "designe est deja decede, le capital est verse aux ayants droit du "
            "souscripteur selon les regles de devolution successorale."
        ),
        PageBreak(),
        _heading("Garanties en cas de vie"),
        _p(
            "Si le souscripteur est en vie a la date d'echeance du contrat, "
            "l'assureur verse le capital constitue par les primes versees, "
            "revalorise annuellement, au choix du souscripteur sous forme de "
            "capital unique ou de rente viagere."
        ),
        PageBreak(),
        _heading("Rachat"),
        _p(
            "Le souscripteur peut demander le rachat total ou partiel de son "
            "contrat a tout moment apres un delai minimal de deux ans a compter "
            "de la souscription. Le rachat partiel ne peut ramener la valeur du "
            "contrat en dessous de 1 000 EUR, sous peine d'entrainer le rachat "
            "total automatique du contrat."
        ),
        PageBreak(),
        _heading("Exclusions"),
        _p(
            "Sont exclus de la garantie deces : le suicide du souscripteur "
            "survenant au cours de la premiere annee du contrat, le deces "
            "resultant d'une fausse declaration intentionnelle sur l'etat de "
            "sante du souscripteur a la souscription, ainsi que le deces "
            "resultant directement de la participation du souscripteur a un "
            "fait de guerre a l'etranger."
        ),
    ]
    doc.build(story)


# ---------------------------------------------------------------------------
# 5. rapport_sans_sommaire.pdf (PAS de TOC exploitable, delibere)
# ---------------------------------------------------------------------------

def build_rapport_sans_sommaire(path: str) -> None:
    """Document deliberement SANS sommaire exploitable : aucun bookmark
    n'est pose (on utilise un SimpleDocTemplate standard, pas la variante
    a bookmarks). Contenu plausible mais qui n'est PAS un contrat
    d'assurance (rapport de conjoncture marche), pour que les questions
    posees contre la base documentaire n'y trouvent pas d'ancre fiable et
    declenchent le chemin d'abstention de retrieval.py / orchestrator.py."""
    doc = SimpleDocTemplate(
        path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
    )
    story = [
        Paragraph("Perspectives du marche de l'assurance au Luxembourg - 2026", STYLES["DocTitle"]),
        _p(
            "Ce rapport interne presente une analyse des tendances macro-"
            "economiques susceptibles d'affecter le secteur de l'assurance au "
            "Grand-Duche de Luxembourg au cours des douze prochains mois. Il ne "
            "constitue en aucun cas un document contractuel et ne decrit "
            "aucune garantie, exclusion ou franchise opposable a un "
            "souscripteur."
        ),
        PageBreak(),
        _sub("Contexte macroeconomique"),
        _p(
            "La croissance economique du Luxembourg devrait rester moderee en "
            "2026, portee principalement par le secteur financier et les "
            "services aux entreprises. L'inflation, en repli par rapport aux "
            "pics observes precedemment, continue neanmoins de peser sur le "
            "cout moyen des sinistres, en particulier dans les branches "
            "automobile et habitation ou le cout des reparations et des "
            "materiaux de construction reste eleve."
        ),
        _p(
            "Le Commissariat aux Assurances (CAA), autorite de controle "
            "prudentiel du secteur, a publie plusieurs recommandations "
            "invitant les compagnies a renforcer leur gouvernance des risques "
            "climatiques, notamment pour les branches habitation face a la "
            "multiplication des episodes d'inondation observee ces dernieres "
            "annees dans la Grande Region."
        ),
        PageBreak(),
        _sub("Tendances par branche"),
        _p(
            "La branche assurance vie continue de beneficier de l'attractivite "
            "fiscale du Luxembourg pour les produits transfrontaliers, avec une "
            "collecte nette en hausse sur les trois dernieres annees. La "
            "branche sante complementaire connait une demande croissante liee "
            "au vieillissement de la population active frontaliere."
        ),
        _p(
            "A l'inverse, la branche automobile fait face a une pression sur "
            "les marges techniques, la frequence des sinistres corporels graves "
            "restant stable alors que le cout unitaire moyen des sinistres "
            "materiels a progresse d'environ huit pour cent sur la periode "
            "consideree, principalement du fait du renoncherissement des "
            "pieces detachees electroniques dans les vehicules recents."
        ),
        PageBreak(),
        _sub("Enjeux reglementaires et technologiques"),
        _p(
            "L'adoption croissante d'outils d'intelligence artificielle "
            "generative dans la relation client (chatbots documentaires, "
            "assistants de souscription) souleve des questions de gouvernance "
            "et de tracabilite des reponses fournies aux assures, en particulier "
            "sur la capacite de ces outils a s'abstenir plutot qu'a fournir une "
            "reponse incertaine sur un point contractuel precis."
        ),
        _p(
            "Ce rapport ne fait etat d'aucun montant de garantie, de plafond ou "
            "de franchise applicable a un contrat individuel : pour toute "
            "question relative aux conditions d'un contrat specifique, se "
            "referer aux conditions generales du produit concerne."
        ),
    ]
    doc.build(story)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    builders = {
        "assurance_auto.pdf": build_assurance_auto,
        "assurance_habitation.pdf": build_assurance_habitation,
        "assurance_sante.pdf": build_assurance_sante,
        "assurance_vie.pdf": build_assurance_vie,
        "rapport_sans_sommaire.pdf": build_rapport_sans_sommaire,
    }
    for filename, builder in builders.items():
        path = os.path.join(OUT_DIR, filename)
        builder(path)
        print(f"genere : {path}")


if __name__ == "__main__":
    main()
