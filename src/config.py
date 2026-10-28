"""Central configuration for the UPSC PYQ chatbot project.

Values are read from environment variables with sensible
defaults. All data — raw PDFs, the SQLite metadata store, and the Chroma
vector index — lives under DATA_DIR on the local machine. Google Cloud
(Vertex AI) is used only for embeddings and, later, the agent's reasoning
calls; both authenticate via Application Default Credentials, no API key
is stored in this project.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Google Cloud / Vertex AI ---
GCP_PROJECT = os.environ["GCP_PROJECT"]  # set in .env
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")
EMBEDDING_DIMENSIONS = 768  # text-embedding-005 default output size

# Reasoning model for the agent layer (Phase 5). Confirmed available and
# working on this Vertex AI project as of the Phase 3 extraction build.
REASONING_MODEL = os.getenv("REASONING_MODEL", "gemini-2.5-flash")

# Search fallback: Gemini's built-in Google Search grounding tool, used only
# when a pasted article lacks enough context for the agent to work with.
# SerpApi (free tier) was considered as a cheaper alternative, but grounding
# returns Google's own pre-filtered, model-ready results in one integrated
# call rather than raw snippets the pipeline would have to filter itself —
# worth the ~$0.01-0.04/call cost given this branch is rare. See PLAN.md §6a.

# --- Local storage (fully local — no cloud storage/database) ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
SQLITE_PATH = DATA_DIR / "questions.db"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION_NAME = "upsc_questions"

# --- Scope (per project plan) ---
# 2016 is the confirmed floor of upsc.gov.in's own archive for Civil Services
# Prelims/Mains — 2011-2015 does not exist there under any search tried.
EXAM_BOARD = "UPSC"
YEAR_START = 2016
YEAR_END = 2025
