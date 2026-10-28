# UPSC PYQ Chatbot

Paste a news article (text and/or an image) and get the UPSC previous-year questions (Prelims and Mains) that relate to it. An optional final step generates new, clearly labelled UPSC-style practice questions.

**This repository contains code only.** The question corpus, source PDFs, SQLite database and vector index are not included, so a fresh clone has nothing to retrieve until you build your own data (see [Bring your own data](#bring-your-own-data)).

## How it works

A fixed, controllable pipeline (not a free-roaming agent loop), orchestrated in `src/agent/pipeline.py`:

1. **Context extraction** (`context.py`): pull themes, entities and policy angles out of the article.
2. **Query expansion** (`query_expansion.py`): turn them into several retrieval queries; broaden and retry if too few matches come back.
3. **Retrieval** (`retrieval.py`): multi-query semantic search over a Chroma vector store.
4. **Relevance filtering** (`relevance.py`): an LLM judges which candidates actually relate to the article.
5. **Lookup and formatting** (`formatting.py`): fetch full records from SQLite and format them.
6. **Practice question generation** (`generate_questions.py`, opt-in): authors new questions in the style of the matches. These are always returned and displayed separately from real past questions.

## Layout

```
app.py                 Streamlit UI
src/config.py          Settings, read from environment / .env
src/agent/             The pipeline steps above, plus the Gemini client
src/db/                SQLite schema and init, Chroma vector store wrapper
src/embed/             Vertex AI embedding function for Chroma
src/ingest/            Download papers, extract questions with Gemini, load into the DBs
```

## Setup

Requires Python 3.13 and a Google Cloud project with Vertex AI enabled. Embeddings and reasoning go through Vertex AI using Application Default Credentials; no API key is used.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

gcloud auth application-default login
echo 'GCP_PROJECT=your-gcp-project-id' > .env   # optional: GCP_LOCATION, EMBEDDING_MODEL, REASONING_MODEL
```

## Bring your own data

All data lives under `data/` (git-ignored). The ingest scripts in `src/ingest/` build it in this order:

1. `download_papers.py`: fetch the official papers into `data/raw_pdfs/`.
2. `gemini_extract.py`: render pages and extract structured questions with Gemini into `data/extracted/`.
3. `format_questions.py`: clean and normalise the extracted questions.
4. `load_to_db.py`: load them into `data/questions.db` and the Chroma index in `data/chroma/` (create the schema first with `src/db/init_db.py`).

## Run

```bash
streamlit run app.py
```

## Notes

- Past-paper questions are returned without an answer key, since UPSC does not publish an official Prelims key.
- Generated practice questions are fabricated by design and always labelled as AI-generated.
