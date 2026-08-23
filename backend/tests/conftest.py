"""
Rend le package `app` (backend/app) importable depuis les tests, quel que
soit le repertoire depuis lequel `pytest` est lance (racine du repo ou
backend/). Ajoute simplement `backend/` au sys.path.
"""
from __future__ import annotations

import os
import sys

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
