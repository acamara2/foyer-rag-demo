"""
Petit lexique metier (assurance, francais) ecrit a la main.

Objectif : permettre a la brique 2 (analyse de question) d'elargir les
mots-cles de l'utilisateur avec des synonymes/acronymes du domaine, sans
dependre d'un modele d'embeddings ou d'un framework NLP. C'est volontairement
simple et 100% auditable : chaque entree est une decision explicite, visible
et modifiable par un humain (contrairement a un espace vectoriel appris).

Format : terme -> liste de termes equivalents/associes (dans les deux sens,
l'expansion est appliquee de maniere symetrique par question_parsing.py).
"""

INSURANCE_LEXICON: dict[str, list[str]] = {
    "garantie": ["couverture", "protection"],
    "couverture": ["garantie", "protection"],
    "sinistre": ["incident declare", "incident", "declaration de sinistre"],
    "franchise": ["montant restant a charge", "reste a charge"],
    "rc": ["responsabilite civile"],
    "responsabilite civile": ["rc"],
    "plafond": ["montant maximum", "limite de remboursement", "maximum garanti"],
    "exclusion": ["risque non couvert", "non couvert", "cas exclu"],
    "prime": ["cotisation", "montant a payer"],
    "cotisation": ["prime"],
    "souscripteur": ["assure", "preneur d'assurance", "titulaire du contrat"],
    "assure": ["souscripteur", "beneficiaire"],
    "degat des eaux": ["fuite d'eau", "infiltration", "degat d'eau"],
    "vol": ["cambriolage", "effraction"],
    "bris de glace": ["vitres brisees", "pare-brise casse"],
    "resiliation": ["annulation du contrat", "fin de contrat"],
    "delai de declaration": ["delai de declaration de sinistre", "delai pour declarer"],
    "capital deces": ["somme versee au deces", "prestation deces"],
    "beneficiaire": ["ayant droit"],
    "avenant": ["modification du contrat"],
    "expertise": ["expertise contradictoire", "evaluation du sinistre"],
}


def _lexicon_lookup(term: str) -> list[str]:
    """Recherche tolerante au pluriel simple (garanties -> garantie), car le
    lexique est ecrit au singulier. Pas de vrai lemmatiseur : une regle
    suffit pour ce petit corpus et reste totalement lisible."""
    if term in INSURANCE_LEXICON:
        return INSURANCE_LEXICON[term]
    if term.endswith("s") and term[:-1] in INSURANCE_LEXICON:
        return INSURANCE_LEXICON[term[:-1]]
    return []


def expand_terms(terms: list[str]) -> list[str]:
    """Retourne la liste `terms` enrichie des synonymes connus, sans doublons,
    en preservant l'ordre d'apparition."""
    expanded: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for candidate in [term] + _lexicon_lookup(term):
            if candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded
