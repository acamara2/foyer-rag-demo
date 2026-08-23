"""
Interface Streamlit (francais) du "chatbot documentaire" - demo.

Interface volontairement mince : toute la logique (parsing, retrieval,
generation, dispatch sequentiel/batch, abstention) vit cote backend
(FastAPI, voir backend/app/). Ce fichier ne fait qu'appeler l'API REST via
`requests` et afficher la reponse, la preuve citee, et le detail technique
de la trace d'audit - rien n'est recalcule ici.

URL du backend configurable via la variable d'environnement BACKEND_URL
(par defaut http://localhost:8000, cf. .env.example).
"""
from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Chatbot documentaire - Foyer (demo)",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Appels API
# ---------------------------------------------------------------------------

def call_ask(question: str) -> dict | None:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/ask", json={"question": question}, timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        st.error(
            f"Impossible de contacter le backend ({BACKEND_URL}) : {exc}\n\n"
            "Verifiez que l'API FastAPI est bien lancee (voir README, section "
            "'Lancer en local')."
        )
        return None


@st.cache_data(ttl=30)
def call_documents() -> dict | None:
    try:
        resp = requests.get(f"{BACKEND_URL}/documents", timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException:
        return None


# ---------------------------------------------------------------------------
# Aide a l'interpretation metier du signal technique
# ---------------------------------------------------------------------------

def _business_guidance(answer: dict, trace: dict) -> tuple[str, str]:
    """Retourne (niveau_streamlit, message) traduisant le signal technique
    (answer_found / confidence / low_confidence_anchor) en action metier
    concrete pour un conseiller ou un utilisateur final."""
    is_abstention = "reason" in answer

    if is_abstention:
        return (
            "warning",
            "Information non trouvee dans la base documentaire -> "
            "a transferer a un conseiller. Ne pas deviner ni extrapoler "
            "une reponse a partir de ce systeme.",
        )

    confidence = answer.get("confidence", 0.0)
    complete = answer.get("complete_answer_found", True)
    low_conf_anchor = trace.get("low_confidence_anchor", False)

    if not answer.get("answer_found", False):
        return (
            "warning",
            "Le modele n'a pas trouve de reponse dans le contexte fourni -> "
            "a transferer a un conseiller.",
        )

    if low_conf_anchor:
        return (
            "warning",
            "Reponse trouvee mais l'ancrage documentaire est de faible "
            "confiance (sommaire degrade sur le document source) -> a "
            "verifier manuellement avant de la transmettre telle quelle.",
        )

    if not complete:
        return (
            "warning",
            "Reponse partielle seulement (le modele indique que la liste "
            "ou l'information pourrait etre incomplete) -> a completer par "
            "un conseiller avant transmission a l'assure.",
        )

    if confidence >= 0.7:
        return (
            "success",
            "Reponse fiable : voir la source citee ci-dessous. Peut etre "
            "transmise telle quelle, avec reference a la page source.",
        )
    if confidence >= 0.4:
        return (
            "info",
            "Reponse plausible mais confiance moyenne -> a relire avant "
            "transmission a l'assure.",
        )
    return (
        "warning",
        "Confiance faible -> a verifier par un conseiller avant toute "
        "utilisation.",
    )


# ---------------------------------------------------------------------------
# Barre laterale : sante du corpus documentaire
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Corpus documentaire")
    st.caption(f"Backend : {BACKEND_URL}")

    docs_payload = call_documents()
    if docs_payload is None:
        st.warning(
            "Backend indisponible : impossible de recuperer les statistiques "
            "du corpus."
        )
    else:
        n_docs = len(docs_payload.get("documents", []))
        avg_ratio = docs_payload.get("avg_duplication_ratio", 0.0)
        st.metric("Documents charges", n_docs)
        st.metric(
            "Ratio de duplication moyen (parsing)",
            f"{avg_ratio:.0%}",
            help=(
                "Volume des donnees 'ligne par ligne' construites par la "
                "brique de parsing pour permettre le retrieval, rapporte au "
                "volume du PDF source. Mesure reelle sur ce corpus, pas une "
                "valeur theorique : illustre le cout de stockage/versioning "
                "du parsing a plus grande echelle (voir README)."
            ),
        )
        with st.expander("Detail par document"):
            for doc in docs_payload.get("documents", []):
                stats = doc.get("parsing_stats", {})
                toc_state = "dégradé" if doc.get("toc_degraded") else "OK"
                st.write(
                    f"**{doc.get('filename')}** - {doc.get('n_pages')} pages, "
                    f"{doc.get('n_sections')} sections (sommaire {toc_state}), "
                    f"duplication {stats.get('duplication_ratio', 0):.0%}"
                )

    st.divider()
    st.caption(
        "Demo technique - pipeline RAG faite main (sans LangChain ni "
        "LlamaIndex), 4 briques + orchestrateur + trace d'audit complete."
    )


# ---------------------------------------------------------------------------
# Corps principal : question / reponse
# ---------------------------------------------------------------------------

st.title("Chatbot documentaire - Foyer (demo)")
st.write(
    "Posez une question sur les contrats d'assurance du corpus de demo "
    "(auto, habitation, sante, vie). Le systeme repond exclusivement a "
    "partir des documents charges, avec preuve citee, ou s'abstient "
    "explicitement s'il ne trouve pas l'information."
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("ask_form", clear_on_submit=False):
    question = st.text_input(
        "Votre question",
        placeholder="Ex. : Quelle est la franchise de la garantie bris de glace en assurance auto ?",
    )
    submitted = st.form_submit_button("Poser la question")

if submitted and question.strip():
    with st.spinner("Recherche dans le corpus documentaire..."):
        result = call_ask(question.strip())
    if result is not None:
        st.session_state.history.insert(0, {"question": question.strip(), "result": result})

if not st.session_state.history:
    st.info("Aucune question posee pour l'instant.")

for entry in st.session_state.history:
    q = entry["question"]
    result = entry["result"]
    answer = result.get("answer", {})
    trace = result.get("trace", {})
    is_abstention = "reason" in answer

    st.markdown(f"### Q : {q}")

    col_answer, col_meta = st.columns([3, 1])

    with col_answer:
        if is_abstention:
            st.warning(f"**Reponse non trouvee.** {answer.get('reason', '')}")
        else:
            value = answer.get("value", "")
            if isinstance(value, list):
                st.markdown("**Reponse :**")
                for item in value:
                    st.markdown(f"- {item}")
            else:
                st.markdown(f"**Reponse :** {value}")

            confidence = answer.get("confidence", 0.0)
            st.progress(min(max(confidence, 0.0), 1.0), text=f"Confiance : {confidence:.0%}")

            if answer.get("caveats"):
                for caveat in answer["caveats"]:
                    st.caption(f"⚠️ {caveat}")

        level, message = _business_guidance(answer, trace)
        guidance_fn = {"success": st.success, "warning": st.warning, "info": st.info}[level]
        guidance_fn(f"**Que faire de cette reponse ?** {message}")

    with col_meta:
        st.metric("Ancre trouvee", "Oui" if trace.get("anchor_found") else "Non")
        st.metric("Appels LLM", trace.get("n_llm_calls", 0))
        st.metric("Latence", f"{trace.get('total_latency_ms', 0):.0f} ms")

    if not is_abstention and answer.get("evidence"):
        st.markdown("**Extraits cites (preuve) :**")
        for span in answer["evidence"]:
            st.markdown(
                f"> {span.get('quote', '')}\n\n"
                f"— *{span.get('doc_id', '')}, page {span.get('page_num', '?')}*"
            )

    with st.expander("Détail technique (trace d'audit de la pipeline)"):
        st.write(
            f"**Type de question detecte :** {trace.get('question_shape', '?')} "
            f"| **Mode de dispatch :** {trace.get('dispatch_mode', '?')}"
        )
        st.caption(trace.get("dispatch_reason", ""))

        st.write(f"**Mots-cles extraits :** {', '.join(trace.get('keywords_matched', [])) or '—'}")
        st.write(
            f"**Mots-cles apres expansion lexique :** "
            f"{', '.join(trace.get('expanded_keywords', [])) or '—'}"
        )

        if trace.get("toc_degraded_docs"):
            st.write(
                "**Documents au sommaire degrade :** "
                + ", ".join(trace["toc_degraded_docs"])
            )
        if trace.get("low_confidence_anchor"):
            st.write("**Ancre de faible confiance** (sommaire degrade sur le document source retenu).")

        scores = trace.get("candidate_pages_scores", [])
        if scores:
            st.write("**Scores par page candidate (mots-cles / sommaire / embedding / combine) :**")
            st.dataframe(
                [
                    {
                        "document": s.get("doc_id"),
                        "page": s.get("page_num"),
                        "score_mots_cles": s.get("keyword_score"),
                        "score_sommaire": s.get("toc_score"),
                        "score_embedding": s.get("embedding_score"),
                        "score_combine": s.get("combined_score"),
                    }
                    for s in scores
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.write(
            f"**Appels LLM :** {trace.get('n_llm_calls', 0)} "
            f"| **Tokens approx. :** {trace.get('approx_tokens_used', 0)} "
            f"| **Latence totale :** {trace.get('total_latency_ms', 0):.0f} ms"
        )

    st.divider()
