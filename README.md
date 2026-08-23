# Chatbot documentaire - demo (Foyer)

Demo technique d'un "chatbot documentaire" (question/reponse sur des
contrats d'assurance PDF), construite pour illustrer une candidature
AI Engineer / Senior Data Scientist. Elle repond au cas d'usage typique
d'un assureur : un conseiller ou un assure pose une question en langage
naturel sur des conditions generales, et le systeme doit soit repondre
avec une preuve citee (page, extrait verbatim), soit s'abstenir
explicitement plutot que d'inventer un montant, un delai ou une exclusion.

Point de depart volontaire : **pas de LangChain, pas de LlamaIndex**. La
pipeline est ecrite a la main, brique par brique, avec des contrats de
donnees Pydantic explicites entre chaque etape. L'objectif est de montrer
une comprehension fine de ce qu'un framework RAG fait habituellement "a
notre place" (chunking, retrieval hybride, orchestration des appels LLM,
gestion de l'abstention) - et de le rendre entierement auditable.

## Pourquoi ce projet

Foyer, comme beaucoup d'assureurs, doit repondre a des questions precises
sur des documents contractuels denses (conditions generales, grilles de
garanties) tout en minimisant le risque d'hallucination : un chatbot qui
invente un plafond de garantie ou un delai de declaration de sinistre est
un risque reglementaire et commercial direct. Cette demo montre une
approche RAG qui privilegie explicitement :

- la **tracabilite** (chaque reponse cite sa source exacte : document +
  page + citation verbatim) ;
- la **discipline d'abstention** (le systeme dit "je ne sais pas" plutot
  que d'halluciner, sans meme appeler le LLM quand aucune ancre fiable
  n'est trouvee) ;
- un **retrieval hybride** (mots-cles + sommaire structurel + similarite
  semantique), pas une simple recherche vectorielle boite noire ;
- un **dispatch cout/latence explicite** (un seul appel LLM sequentiel
  quand une seule page suffit, escalade en mode batch sinon) ;
- une **mesure reelle** (pas theorique) du cout de stockage du parsing
  ligne par ligne (`duplication_ratio`), affichee dans l'UI.

## Architecture

```
                         ┌───────────────────────────┐
                         │   Frontend (Streamlit)     │
                         │   frontend/streamlit_app.py│
                         │   - question utilisateur   │
                         │   - reponse + confiance    │
                         │   - preuves citees         │
                         │   - detail technique (trace)│
                         └──────────────┬─────────────┘
                                        │ HTTP (requests)
                                        │ GET /documents, POST /ask
                                        ▼
                         ┌───────────────────────────┐
                         │  Backend API (FastAPI)     │
                         │  backend/app/main.py       │
                         └──────────────┬─────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────┐
                    │        Orchestrateur               │
                    │        backend/app/orchestrator.py │
                    │  document_qa(question, corpus)     │
                    │  -> (Answer | Abstention, Trace)   │
                    └───────────────────┬─────────────────┘
                                        │
      ┌──────────────┬──────────────────┼──────────────────┬────────────┐
      ▼              ▼                  ▼                  ▼            │
┌───────────┐  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐    │
│ Brique 1  │  │  Brique 2    │   │  Brique 3     │   │  Brique 4    │    │
│ Parsing   │  │  Question    │   │  Retrieval    │   │  Generation  │    │
│ parsing.py│  │  parsing.py  │   │  retrieval.py │   │  generation.py│   │
│           │  │              │   │               │   │  llm_client.py│  │
│ PDF ->    │  │ mots-cles +  │   │ mots-cles +   │   │ dispatch      │   │
│ lignes +  │  │ lexique +    │   │ sommaire +    │   │ sequentiel/   │   │
│ sections  │  │ shape        │   │ TF-IDF        │   │ batch, appel  │   │
│ (sommaire)│  │ (single/     │   │ (numpy)       │   │ OpenAI        │   │
│ + stats   │  │  listing)    │   │ -> ancrage    │   │ Structured    │   │
│ parsing   │  │              │   │  ou abstention│   │ Outputs, ou   │   │
│ (une fois │  │              │   │               │   │ abstention    │   │
│ au demarr.)│ │              │   │               │   │ SANS appel LLM│   │
└───────────┘  └─────────────┘   └──────────────┘   └──────────────┘    │
                                                                          │
                    ┌─────────────────────────────────────────────────┐ │
                    │  PipelineTrace (schemas.py) - audit complet :    │◄┘
                    │  mots-cles matches, scores par page, sommaire    │
                    │  degrade ?, mode dispatch, nb appels LLM,        │
                    │  tokens approx., latence totale                 │
                    └─────────────────────────────────────────────────┘
```

Chaque brique communique exclusivement via des schemas Pydantic typés
(`backend/app/schemas.py`) : aucune "magie" de framework, aucun etat
cache. La trace d'audit (`PipelineTrace`) est produite pour **chaque**
question, reussie ou non, et affichee integralement dans l'onglet
"Detail technique" du frontend.

## Structure du depot

```
foyer-rag-demo/
├── backend/
│   ├── app/                      # les 4 briques + orchestrateur + API FastAPI
│   │   ├── schemas.py            # contrats Pydantic partages
│   │   ├── lexicon.py            # lexique metier assurance (synonymes)
│   │   ├── parsing.py            # brique 1 : PDF -> ParsedDocument
│   │   ├── question_parsing.py   # brique 2 : question -> ParsedQuestion
│   │   ├── retrieval.py          # brique 3 : ancrage (mots-cles+sommaire+TF-IDF)
│   │   ├── generation.py         # brique 4 : dispatch + prompt + abstention
│   │   ├── llm_client.py         # SEUL fichier qui importe le SDK openai
│   │   ├── orchestrator.py       # cable les 4 briques + trace d'audit
│   │   └── main.py               # API FastAPI (/health, /documents, /ask)
│   ├── data/
│   │   ├── generate_synthetic_docs.py  # genere les 5 PDF de demo (reportlab)
│   │   ├── raw_docs/                   # les 5 PDF generes
│   │   ├── golden_dataset.json         # 18 questions/reponses attendues
│   │   └── run_eval.py                 # evaluation bout-en-bout (besoin d'une cle API)
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_parsing.py
│   │   ├── test_retrieval.py
│   │   ├── test_pipeline_smoke.py      # orchestrateur avec LLM MOCKE
│   │   └── test_eval_scoring.py        # logique de run_eval.py, isolee
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── streamlit_app.py
│   └── requirements.txt
├── render.yaml
├── .env.example
├── .gitignore
└── README.md
```

## Lancer en local

Deux terminaux : un pour le backend, un pour le frontend.

### Terminal 1 - backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example ../.env         # puis renseigner OPENAI_API_KEY dans .env
export OPENAI_API_KEY=sk-...       # ou charge automatiquement via python-dotenv
uvicorn app.main:app --reload --port 8000
```

L'API est alors disponible sur `http://localhost:8000` (`/health`,
`/documents`, `/ask`, doc interactive auto-generee sur `/docs`).

### Terminal 2 - frontend

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000
streamlit run streamlit_app.py
```

L'interface s'ouvre sur `http://localhost:8501`.

### Regenerer le corpus PDF de demo

Les 5 PDF sont deja commites dans `backend/data/raw_docs/`, mais peuvent
etre regeneres a l'identique (contenu invente, deterministe) :

```bash
cd backend
pip install reportlab
python data/generate_synthetic_docs.py
```

### Lancer l'evaluation (golden dataset)

```bash
cd backend
export OPENAI_API_KEY=sk-...
python data/run_eval.py
```

Necessite une cle `OPENAI_API_KEY` valide car chaque question du jeu de
donnees passe reellement par la brique 4 (generation, appel LLM). Le
script affiche un resume (taux d'ancrage, justesse de l'abstention,
confiance moyenne, latence moyenne, appels LLM moyens/question) et ecrit
le detail dans `backend/data/eval_results.json`. Sans cle API, le script
s'arrete immediatement avec un message explicite ; la logique de scoring
elle-meme (comparaison reponse attendue/obtenue) est testee
independamment, sans reseau, dans `backend/tests/test_eval_scoring.py`.

### Lancer les tests

```bash
cd backend
pip install -r requirements.txt   # inclut pytest
pytest tests/ -v
```

`test_parsing.py` et `test_retrieval.py` tournent sans cle API (le premier
utilise les vrais PDF generes, le second un corpus 100% en memoire).
`test_pipeline_smoke.py` fait tourner l'orchestrateur complet avec le
client LLM **mocke** (aucun appel reseau). `test_eval_scoring.py` teste la
logique du script d'evaluation en isolation.

## Deployer sur Render

1. Pousser ce depot sur GitHub (le candidat connecte son propre compte -
   aucun remote n'est configure par cette demo).
2. Sur [render.com](https://render.com), cliquer **New +** -> **Blueprint**,
   selectionner le depot GitHub : Render lit `render.yaml` a la racine et
   propose de creer les 2 services (`foyer-rag-backend` en Docker,
   `foyer-rag-frontend` en environnement Python).
3. Avant de confirmer le deploiement (ou juste apres, dans l'onglet
   **Environment** du service `foyer-rag-backend`), renseigner :
   - `OPENAI_API_KEY` = votre cle API OpenAI (secret, jamais commite -
     marque `sync: false` dans `render.yaml`) ;
   - `OPENAI_MODEL` = `gpt-4o-mini` (deja pre-rempli, modifiable).
4. Une fois le backend deploye, noter son URL publique (ex.
   `https://foyer-rag-backend.onrender.com`) et la renseigner comme valeur
   de `BACKEND_URL` dans l'onglet **Environment** du service
   `foyer-rag-frontend` (Render ne propage pas automatiquement une URL
   `https://` complete entre deux services web independants via le
   blueprint - c'est la seule etape manuelle necessaire).
5. Redeployer le frontend (ou attendre le redeploiement automatique lie au
   changement de variable d'environnement).

Si vous preferez tout faire a la main plutot que via le blueprint : creer
un service **Web Service (Docker)** pour le backend en pointant
`dockerfilePath` sur `backend/Dockerfile` et `dockerContext` sur `.`
(racine du repo), et un service **Web Service (Python)** pour le frontend
avec `buildCommand: pip install -r frontend/requirements.txt` et
`startCommand: streamlit run frontend/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`.

## Choix assumes et limites

- **Pas de LangChain / LlamaIndex, par choix.** L'objectif de cette demo
  est de montrer une comprehension explicite de chaque etape d'un
  pipeline RAG (chunking par page + section, scoring de retrieval
  multi-signal, dispatch sequentiel/batch, discipline d'abstention), pas
  d'assembler des composants de framework. Le cout : plus de code ecrit a
  la main ; le benefice : zero boite noire, chaque decision est visible et
  testable independamment (voir `backend/tests/`).

- **`pypdf` plutot que `PyMuPDF` pour le parsing.** La consigne initiale
  visait `PyMuPDF` (`fitz`). Le bac a sable utilise pour construire et
  tester cette demo n'a aucun acces reseau sortant (pip/PyPI bloques), et
  `pymupdf` n'a donc pas pu etre installe pour verifier le code. `pypdf`
  offre les memes capacites necessaires ici (extraction de texte +
  lecture des signets/outline natifs du PDF) pour des PDF texte generes
  programmatiquement comme ceux de ce corpus. Le code de
  `backend/app/parsing.py` est isole derriere le schema `ParsedDocument` :
  remplacer `pypdf` par `PyMuPDF` (utile en production pour une mise en
  page plus complexe - colonnes, tableaux scannes) ne demanderait de
  modifier qu'un seul fichier.

- **Embedding "fait main" (TF-IDF + numpy), pas de sentence-transformers
  ni d'API d'embeddings externe.** Choix deliberé pour ce corpus de demo :
  (1) zero dependance lourde a telecharger/heberger, (2) signal
  semantique volontairement *secondaire* dans le score combine
  (poids 0.25 contre 0.55 pour les mots-cles) car le vocabulaire
  contractuel (montants, noms de garanties exacts) se preche mieux par
  correspondance lexicale exacte que par similarite vectorielle floue,
  (3) transparence totale du calcul (vocabulaire, IDF, cosinus - tout est
  lisible dans `retrieval.py`, aucun modele opaque). A plus grande echelle
  ou avec des reformulations plus variees, un vrai modele d'embeddings
  (API OpenAI embeddings, ou `sentence-transformers` local) remplacerait
  avantageusement ce TF-IDF - le point d'insertion est isole dans la
  classe `TfidfIndex`.

- **Mise en page multi-colonnes / tableaux : non resolue, juste
  documentee.** Le document `assurance_sante.pdf` contient un tableau
  comparatif "Formule Essentielle" vs "Formule Confort" dont l'extraction
  ligne par ligne (`page.extract_text()`) linearise les cellules sans
  garder l'alignement colonne-par-colonne : une extraction naive
  ligne-a-ligne peut associer une valeur a la mauvaise formule. Le
  contexte complet de page est neanmoins transmis au LLM (qui peut
  generalement recomposer le tableau depuis le texte lineaire), et ce cas
  est marque `is_edge_case: true` dans `golden_dataset.json` avec un
  credit partiel assume plutot qu'une pretention a l'avoir resolu
  proprement. Une vraie solution demanderait une extraction consciente de
  la structure tabulaire (ex. `pdfplumber.extract_tables()`).

- **Pas d'authentification, pas de multi-tenant, pas d'historique
  persistant.** L'API est ouverte (CORS `*`), sans notion d'utilisateur ni
  de session serveur, et le frontend garde l'historique de conversation
  uniquement en memoire de session Streamlit (`st.session_state`, perdu au
  rechargement). Volontaire pour une demo publique en lecture seule sans
  donnee sensible reelle - une vraie mise en production necessiterait au
  minimum une authentification (SSO Foyer), une isolation par
  utilisateur/role, et une tracabilite persistante des questions/reponses
  (audit reglementaire).

- **Corpus volontairement petit (5 documents).** Le corpus entier est
  parse une fois au demarrage de l'API et garde en memoire
  (`backend/app/main.py`) - adapte a une demo, pas a un corpus de
  production. Ce qui devrait changer a plus grande echelle :
  - **revalider les poids keyword/TOC/embedding** (`W_KEYWORD=0.55`,
    `W_TOC=0.20`, `W_EMBEDDING=0.25` dans `retrieval.py`) sur un vrai
    corpus et un vrai jeu de questions utilisateurs, pas sur 5 PDF
    synthetiques ;
  - **gerer la latence et la charge** : indexation TF-IDF recalculee a
    chaque requete (`O(n_pages)`), acceptable pour un corpus de demo,
    a remplacer par un index precalcule et rafraichi de facon incrementale
    pour un vrai volume de documents et de trafic concurrent ;
  - **surveiller le ratio de duplication** mesure ici pour de vrai
    (`duplication_ratio`, affiche dans l'UI et dans `/documents`) : sur ce
    corpus de demo il tourne autour de 79% en moyenne (representation
    ligne-par-ligne quasi aussi volumineuse que le PDF source) ; a
    l'echelle de centaines de contrats, ce chiffre devient un vrai enjeu
    de stockage et de versioning qu'il faut suivre dans le temps, pas
    seulement mesurer une fois.

## Contact

Candidature AI Engineer / Senior Data Scientist - Groupe Foyer (Luxembourg).
