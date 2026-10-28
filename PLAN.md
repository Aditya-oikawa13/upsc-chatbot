# UPSC PYQ Chatbot — Project Plan (v2, local-first)

## 0. Goal

An agentic chatbot that, given a pasted news article, finds and returns **UPSC previous-year questions** (Prelims + Mains, 2016–2025) that relate to it — question text, options and year for Prelims; question text, marks, and word limit for Mains — drawn from a locally-stored, richly-structured database of official past papers. No correct-answer key is included (UPSC publishes none for Prelims — see §2). Optional subject papers are excluded. Built to expand later into more agent capabilities and additional exam boards (state PSCs); the first such expansion is a practice-question generator (§6c) that authors new, clearly-labeled UPSC-style questions rather than retrieving real ones.

## 1. Guiding principles (updated)

1. **Local-first for data and storage; cloud for embeddings and reasoning.** Raw data, the structured metadata store, and the vector index all live persistently on your machine — no cloud storage, no cloud database, no cloud vector store. Both embedding generation and the agent's reasoning go through Vertex AI: at this corpus's scale the cost is negligible (see §5, cost estimate) and it avoids spending local CPU/RAM/time running an embedding model yourself.
2. **Extraction stays deterministic in spirit, even though it ended up using an LLM.** The original plan was regex/layout parsing with zero LLM calls; once the actual PDFs turned out to have corrupted text layers (Prelims) and no text layer at all (Mains — see §3), that was replaced with schema-constrained Gemini extraction after confirming the cost was trivial (~$2 for the whole corpus) and it measurably fixed OCR's failure modes. The principle that survives: no fabrication — every field is either read off the page or left `null`, never guessed or enriched.
3. **Official source only, 2016–2025.** No third-party compilations. upsc.gov.in publishes clean, machine-parseable Prelims and Mains papers from 2016 onward — that's the actual floor of its archive (confirmed directly, not assumed), so that range alone is used.
4. **Agentic, not just RAG.** The news-matching feature is the first *tool* an agent uses, not the whole system. The agent does theme extraction → multi-query retrieval → self-check/retry → synthesis, so future tools (mains-answer grading, mock tests, revision planning) register as new tools on the same agent later.
5. **Exam-board-agnostic schema.** Every record carries `exam_board` (`UPSC` now, state PSCs later) so expansion is a new local connector + new Chroma collection, not a redesign.
6. **One explicit, disclosed exception to "no fabrication" (added post-launch, §6c).** Everything above is either read off an official paper or left `null` — the practice-question generator (`src/agent/generate_questions.py`) is the sole deliberate exception: it authors new UPSC-style questions that exist nowhere in any real paper. Kept honest by staying opt-in, never merging into `prelims_matches`/`mains_matches`, and always rendering under its own visibly-distinct "AI-generated" label — the exception is disclosed on screen, not smuggled into the archive results.

## 2. Data sourcing

| Exam | Source | Range |
|---|---|---|
| Prelims — General Studies + CSAT | upsc.gov.in previous papers | 2016–2025 |
| Mains — GS I–IV, Essay | upsc.gov.in previous papers | 2016–2025 |
| Excluded | Optional subject papers (Mains) | — |
| Excluded | Pre-2016 papers, any third-party site | — (dropped per scope; upsc.gov.in's archive genuinely doesn't go back further — confirmed by direct search, not a scraping gap) |

**Two things confirmed by directly checking upsc.gov.in (Phase 2 recon), not assumed from the original plan:**
1. The official "Previous Question Papers" archive covers **2016 onward only** — 2011–2015 doesn't exist in it under any search variant tried.
2. **UPSC does not publish an official Prelims answer key**, for any year — confirmed via the site's own Answer Keys page, its archive, and its site search, all returning nothing for Civil Services. (UPSC did commit to starting this, via a September 2025 Supreme Court affidavit, but nothing has actually appeared on the site as of this writing.) Per your decision, `correct_option` stays `null` for all Prelims records rather than reaching for a coaching-site source — this is the norm for this dataset, not an edge case.
| Future | State PSC official sites | New connector per board when we get there |

All downloaded PDFs are saved under a local folder structure, e.g.:

```
data/
  raw_pdfs/
    prelims/{year}/gs_paper1.pdf, csat_paper2.pdf, answer_key.pdf
    mains/{year}/essay.pdf, gs1.pdf, gs2.pdf, gs3.pdf, gs4.pdf
```

This folder *is* the `source_doc_url` reference and the permanent archive — nothing gets deleted after extraction, since re-parsing later (e.g. once we add LLM enrichment) means going back to the original file.

## 3. Extraction — Gemini-based (OCR tested first, replaced after real quality comparison)

**Reality check done before writing any parsing code** (per your instinct to actually look at the PDFs first): the assumption that these are clean text-layer PDFs was wrong, in two different ways per exam type —

- **Prelims**: born-digital PDFs *do* have a text layer, but it has genuine character-level corruption baked into the embedded font's encoding — e.g. `Kamaraj~` for `Kamarajar`, `Artoca7pus` for `Artocarpus`. Confirmed identical in both `pdfplumber` and `PyMuPDF`, so it's a broken ToUnicode mapping in the PDF itself, not a library bug. The pages are also two-column, which naive extraction interleaves.
- **Mains**: these are **scanned photocopies with no text layer at all** — `get_text()` returns nothing. Full OCR is mandatory, not a fallback case.

**Fix, tested and working**: render every page to an image (PyMuPDF, 300 DPI) and OCR it with Tesseract, for *both* exam types — this sidesteps the Prelims encoding bug entirely (OCR reads the rendered glyph shapes, not the broken character codes) and is the only option for Mains anyway. Tested side by side: OCR correctly reads `Kamarajar Port` and `Artocarpus heterophyllus` where the text layer had them corrupted.

- **Prelims pages**: split each page image into left/right column halves before OCR (`lang=eng`), to avoid column interleaving; question blocks identified by numbering + an options pattern (`(a)/(b)/(c)/(d)`). `correct_option` is `null` for every record — no official answer key exists (§2).
- **Mains pages**: OCR with `lang=hin+eng` (Hindi and English are printed as paired lines per question, not separate sections); keep only the English line per question block, detected via Devanagari-Unicode-range filtering.
- **Known OCR rough edge**: digit/letter confusion, specifically `1`/`l` (e.g. "150 words" OCR'd as "l50 words", marks "10" as "l0"). This mostly affects `marks`/`word_limit`, not question text. Mitigation: regex normalization for the common confusions, cross-checked against each paper's printed instruction block (UPSC states the marks/word-limit pattern once per paper, e.g. "Q1–10 in 150 words," which is more reliable to parse once than trusting per-question OCR digits) — and `null` rather than a guessed value when a figure still doesn't parse cleanly, consistent with how `correct_option` is already handled.
- **Always captured, never optional**: `year`, `source_doc_url` (local path), `exam_type`, `paper`, `question_number`, `question_text`.
- **No enrichment fields are populated in this phase** — no LLM-derived subject/topic/keyword tagging. Those columns still exist in the schema (nullable) so a future enrichment pass can fill them in without a schema migration.

**Decision reversed mid-build**: extraction ended up using Gemini (Vertex AI) directly, not Tesseract. Once actually tested, Gemini fixed Tesseract's exact weak spots (digit confusion in marks/word-limits, reconstructing split/tabular layouts) for about $2 total across the whole corpus — cheap enough that consistency won out over a hybrid pipeline. See §3a for what this actually took to get right.

### 3a. What extraction actually took (Phase 3 build notes)

Per-page Gemini calls, schema-constrained JSON output (`response_schema`), one call per page for Prelims/Mains-GS1-3/Essay. Issues hit and fixed, in order:

1. **Concurrent-request network errors** (`Connection reset by peer`, `Broken pipe`) silently dropped pages under 6-way concurrency. Fixed with retry-with-backoff (5 attempts) and lower concurrency (3 workers); documents with any page still failing after retries aren't saved, so a re-run automatically retries only the incomplete ones.
2. **Essay papers**: "Section A" and "Section B" both print topics numbered 1–4, so the printed number isn't unique within the document — deduping by number silently discarded half the topics for at least one year before this was caught. Fixed with an Essay-specific schema (plain topic strings, no forced number) and sequential renumbering at aggregation time.
3. **A required-schema-field trap**: the Mains schema requiring `marks`/`word_limit` as integers made Gemini return an *empty* result for Essay pages entirely (topics don't have per-topic marks/word-limits — those are fixed paper-level constants: 125 marks × 2 essays, 1000–1200 words). Fixed by removing those fields from the Essay schema entirely and hardcoding the known constants instead of trying to read them.
4. **A genuine source-scan defect**: one document (2017 Mains GS2) had its left margin clipped during the original government scan, truncating "15/16/17" to "5/6/7" — confirmed by reading the rendered page directly, not a model error. Fixed with a one-off manual correction using question content to identify the true numbers; not worth a general-purpose detector for a single occurrence across 69 documents.
5. **GS4 (Ethics) case studies span page boundaries** — a scenario's Hindi and English versions can be on entirely different pages, and either can itself span multiple pages, so per-page extraction fragmented them (lost sub-parts, wrong marks). Fixed by sending the *entire* document as one multi-image call for GS4 specifically, so the model has full context — this took total marks per paper from wildly inconsistent (9–16 questions, marks not summing to the fixed 250 total) to sequential and correct in 8 of 10 years exactly at 250; two years (2016, 2022) are still slightly over (280, 260) and left as a known minor discrepancy rather than chased further, since it doesn't affect question text or the retrieval feature.

**Final corpus**: 2,601 questions (1,799 Prelims + 794 Mains) across 2016–2025, fully sequential numbering in every document except the two GS4 years noted above.

## 4. Storage — fully local

| Layer | Choice | Why |
|---|---|---|
| Raw documents | Local filesystem (`data/raw_pdfs/`) | Permanent source of truth, free |
| Structured metadata | **SQLite** (`data/questions.db`) | Zero-infra, file-based, full relational queries, free, trivially backed up/versioned |
| Vector index | **ChromaDB**, persistent client (`data/chroma/`) | Free, local, on-disk persistence, built-in metadata filtering alongside vectors — the natural fit for "vector DB that lives on my device" |

`SQLite` is the source of truth for everything structured; `Chroma`'s metadata mirrors the filterable subset (`question_id`, `exam_type`, `year`, `paper`) so retrieval can combine semantic search with hard filters like "Prelims only" or "2018–2023."

### `questions` table (SQLite)

| Field | Filled now? | Notes |
|---|---|---|
| `question_id` | ✅ | Primary key |
| `exam_board` | ✅ | `UPSC` |
| `exam_type` | ✅ | `prelims` \| `mains` |
| `is_objective` | ✅ | Derived from options-pattern detection |
| `paper` | ✅ | GS1–4, Essay, CSAT, GS-Prelims |
| `year` | ✅ | **Always populated** |
| `question_number` | ✅ | |
| `question_text` | ✅ | |
| `options` (JSON) | ✅ (Prelims only) | |
| `correct_option` | ✅ where answer key parses cleanly, else `null` | |
| `marks`, `word_limit` | ✅ (Mains, where printed) | |
| `source_doc_url` | ✅ | **Always populated** — local file path |
| `extraction_method` | ✅ | `rule-based` |
| `extracted_at` | ✅ | |
| `subject`, `topic`, `subtopic` | ⏳ null for now | Reserved for future enrichment phase |
| `keywords[]`, `linked_theme`, `static_or_current` | ⏳ null for now | Reserved for future enrichment phase |

## 5. Embeddings & retrieval

- **Embedding model: Vertex AI, cloud-hosted** — `text-embedding-005` via the Vertex AI API. Running a local embedding model was the original plan, but at this corpus size the cloud cost is trivial (below) and it avoids spending your machine's CPU/RAM/time on an otherwise one-time embedding job.
- Each question is embedded as `question_text` (+ options text for Prelims, since the options often carry disambiguating context).
- At inference time, the pasted news article and its reformulated sub-queries are embedded the same way, via the same API.

### 5a. Cost estimate

Corpus size for 2016–2025, UPSC only: roughly **~1,800 Prelims questions** (GS Paper I ~100/yr + CSAT Paper II ~80/yr × 10 years) + **~850 Mains questions** (GS I–IV + Essay, ~85/yr × 10 years) ≈ **~2,650 questions total**. Average record length (question + options where applicable) ≈ 400–600 characters → roughly **~1.3 million characters** for the entire one-time corpus embedding job.

| Model | Rate | One-time full-corpus embed | Per news-article query (~500–1000 chars incl. reformulations) |
|---|---|---|---|
| `text-embedding-005` (character-based) | ~$0.000025 / 1K chars online, ~$0.00002 / 1K chars batch | **≈ $0.03** | **< $0.0001** |
| `gemini-embedding-001` (token-based, alternative) | $0.15 / 1M input tokens | **≈ $0.05** | **< $0.0002** |

So: embedding the *entire* 10-year corpus once costs on the order of **a few cents**, and every subsequent news-article lookup costs a small fraction of a cent — negligible against free-trial credit, and not worth optimizing away with a local model. (Rates per Google's [Vertex AI pricing page](https://cloud.google.com/vertex-ai/pricing) and the [Gemini Embedding pricing announcement](https://developers.googleblog.com/gemini-embedding-available-gemini-api/) — worth a quick re-check against current pricing before relying on it long-term, since Google revises these periodically.)

Recommendation: `text-embedding-005` for the corpus batch job (cheapest, character-based, well-proven for retrieval); revisit only if Phase 7 evaluation shows match quality issues.

## 6. Agent architecture (Phase 5, built)

**Framework decision reversed from the original sketch**: no Google ADK. The pipeline is a fixed sequence of steps (only one conditional branch, one bounded retry) — not an autonomous tool-calling loop deciding its own plan each time. ADK's value is orchestrating agents that need to *decide* their own sequence; ours doesn't, so it would have added a dependency and a learning curve without buying anything. Built instead as plain Python calling `google-genai` directly, in `src/agent/`, matching the pattern already proven in Phase 3's extraction pipeline (schema-constrained structured output, retry-with-backoff for the transient connection resets this environment has shown against Vertex AI throughout the project).

```
USER INPUT (text and/or image of a news article)
        │
        ▼
┌──────────────────────────┐
│ 1. CONTEXT (context.py)   │  Gemini, multimodal — reads text or image directly,
│    → topic, entities,     │  extracts topic/entities/key_facts, judges sufficiency
│      key_facts, sufficient│
└─────────┬──────────────────┘
          │ insufficient
          ▼
┌──────────────────────────┐
│ Google Search grounding   │  built-in Gemini tool (§6a: chosen over the free
│ (gemini_client.grounded_  │  SerpApi alternative — grounding returns pre-filtered,
│  call)                    │  model-ready results in one call, not raw snippets
└─────────┬──────────────────┘  the pipeline would have to filter itself)
          │ (re-structured back into the same context shape)
          ▼
┌──────────────────────────┐
│ 2. QUERY EXPANSION        │  Gemini (query_expansion.py) — 5-8 diverse queries
│    (direct/category/      │  tagged by angle, e.g. an endangered-bird article →
│     policy/international/ │  direct (the species), category (endangered species
│     pattern)               │  broadly), policy (Wildlife Protection Act), pattern
│                            │  ("X in the news", UPSC's actual Prelims phrasing)
└─────────┬──────────────────┘
          ▼
┌──────────────────────────┐
│ 3. RETRIEVAL               │  Deterministic, NOT an LLM call (retrieval.py) —
│    (tool, no LLM)          │  every query run against Chroma, both exam types
│    per query → top-k        │  pooled unfiltered (the point is finding both),
│    pool + dedupe by id     │  deduped keeping the closest match per question
└─────────┬──────────────────┘
          ▼
┌──────────────────────────┐
│ 4. RELEVANCE FILTER         │  Gemini, structured output (relevance.py) — judges
│    + STRUCTURING           │  genuine thematic relevance (a superficially-similar
│    → relevant question_ids │  but irrelevant hit gets dropped here), tiers matches
│      + reason + tier       │  direct/thematic/broader, drops redundant near-dupes
└─────────┬──────────────────┘
          │ < 3 total matches? (§6b — raised from 2)
          ├─── retry once with broadened queries, merged not replaced (§6b) ─┐
          ▼                                                                  │
┌──────────────────────────┐                                                │
│ 5. LOOKUP + FORMAT          │◄───────────────────────────────────────────────┘
│    (pipeline.py,           │  fetch full records from SQLite by id — the model
│     deterministic)         │  never re-writes question text, so what's displayed
│    group by exam_type      │  is always byte-for-byte what UPSC printed
└─────────┬──────────────────┘
          ▼
find_matches() returns: context, queries used, prelims_matches, mains_matches
  (each match = full DB record + reason + tier, ready to render — real PYQs only)
          │
          │ one-way dependency only — step 6 reads this output for style
          │ exemplars, nothing above needs step 6 to have run (§6d)
          ▼
┌──────────────────────────┐
│ 6. GENERATE PRACTICE        │  Gemini, structured output (generate_questions.py,
│    QUESTIONS                │  §6c) — opt-in, fabricated by design (§1.6). Authors
│    (generate_practice_      │  new UPSC-style Prelims/Mains questions on the same
│     questions())            │  topic, using ≤1 real match per exam type as a
│                              │  style-only exemplar — never as source facts.
└──────────────────────────┘
run_pipeline() runs steps 1-6 in sequence and returns everything together.
app.py instead calls find_matches() and generate_practice_questions()
separately (§6d): real PYQs render as soon as find_matches() returns, and
step 6 runs concurrently in a background thread rather than blocking on it.
```

**Where Google Cloud is used:** embeddings (Phase 4, one-time + per-query) and every reasoning step above, all via Vertex AI using your already-verified ADC credentials. Storage, extraction, and the vector index itself remain local and free (§5a for why this is cheap enough not to matter).

### 6a. Search fallback: grounding vs. SerpApi

Considered SerpApi (genuinely free tier, 100-250 searches/month, same underlying Google results) against Gemini's built-in Google Search grounding tool (~$0.01-0.04/call, since it's billed per underlying search query rather than per API call). Chose grounding: it returns synthesized, pre-filtered, model-ready text in one integrated call, where SerpApi would return raw snippets needing a second Gemini call to filter/synthesize anyway — at which point the token cost of processing noisy raw results narrows most of the apparent savings. This branch only fires when a pasted article is judged too sparse to work with, so the cost either way is rare and small in absolute terms.

**Interface**: Streamlit, run locally (`streamlit run app.py`) — no Cloud Run needed unless you later want it reachable outside your machine, at which point deploying the same app is a small add-on step.

### 6b. Reliability pass — determinism, retrieval breadth, retry-merge fix (post-launch)

**Problem observed**: match counts were wildly inconsistent — 0 matches on one run, 13 on a similarly-substantive article, sometimes on literal re-runs of the same article. Root cause: nothing in the pipeline set `temperature` or `seed` anywhere, so every Gemini call — including relevance filtering, the one step that actually decides which candidates get kept — ran at the model's default temperature (1.0) with `thinking_budget=0` (chain-of-thought fully off). A binary keep/drop judgment made with no reasoning at temperature 1.0 on the same candidate pool will not return the same answer twice.

Fixes made in `gemini_client.py` / `query_expansion.py` / `relevance.py` / `retrieval.py` / `pipeline.py`:
- `structured_call()` now defaults to `temperature=0.0` + a fixed `seed=42` (`DETERMINISTIC_SEED`) for every call — same input now reliably gives the same output.
- `thinking_budget=-1` (dynamic thinking) turned on for query expansion, relevance filtering, and practice-question generation (§6c) — the three steps that make an actual judgment/authoring call, as opposed to context extraction's mechanical field-parsing (left at `thinking_budget=0` for speed).
- Retrieval widened: `TOP_K_PER_QUERY` 6 → 12, `MAX_CANDIDATES` 40 → 60 (`retrieval.py`).
- Query-expansion schema given `"minItems": 6` — the prompt asked for "5-8 queries" but nothing previously stopped a bad roll from returning fewer.
- **Retry-merge bug**: the broaden-and-retry path (triggered when too few matches are found) previously *replaced* the first pass's candidates/matches outright — a genuine match found in pass 1 could be silently dropped if the broadened pass 2 didn't independently rediscover it. Fixed to merge both candidate pools (dedupe by `question_id`, keep the lowest distance) before re-judging relevance once over the union. `MIN_MATCHES_BEFORE_RETRY` raised 2 → 3 so the retry triggers more readily.

**Trade-off named honestly**: dynamic thinking measurably improves judgment quality but adds real latency — relevance filtering ~53s, practice-question generation ~26s, observed directly in `logs/`. Addressed separately, not by dialing thinking back down — see §6d.

### 6c. Practice-question generation (opt-in, fabricated by design)

New file `src/agent/generate_questions.py` — a different kind of output from everything else in this project (§1.6): it *authors* new UPSC-style Prelims/Mains questions on the article's topic, not questions read off a real paper.

- 3 Prelims + 2 Mains by default (`NUM_PRELIMS`/`NUM_MAINS`), schema-constrained JSON, `temperature=0.5` (some variety is fine/desirable here, unlike the deterministic steps in §6b) with dynamic thinking.
- Uses ≤1 real match per exam type (already found by `find_matches()`) as a **style-only** exemplar — the prompt explicitly forbids reusing their facts, only their phrasing/structure (the "consider the following statements" idiom, UPSC's compressed multi-clause Mains directive style).
- Prompted for genuine depth, not headline paraphrase: tests the static-syllabus concept/institution/law behind the news trigger; Prelims statements mix a subtly-wrong statement in with correct ones (not one obviously-fake outlier) and must include ≥1 static-syllabus angle beyond the article itself; told explicitly not to fabricate a statistic it isn't confident of.
- Includes an answer key + explanation for Prelims and a brief answer-writing approach for Mains — genuinely useful additions real PYQs can't have, since UPSC publishes no official Prelims key at all (§2).
- **Real bug hit and fixed**: Vertex's structured-output schema (via the `google-genai` SDK's `types.Schema` pydantic model) only accepts **string** enum values, even for an integer-typed field. `marks`/`word_limit` were originally declared `"type": "integer", "enum": [10, 15]` — a client-side `pydantic.ValidationError` raised before any network request is even made, on every attempt, so the existing retry-with-backoff loop wastefully retried a guaranteed-to-fail call 5 times and surfaced only a generic "failed after 5 attempts." Fixed by declaring them as string enums (`"10"`/`"15"`) and casting to `int` after the response returns.
- UI: never merged into `prelims_matches`/`mains_matches`. Rendered in its own "Practice" tab in `app.py`, deliberately styled *unlike* the real-PYQ stamp cards (dashed borders, an un-rotated "AI-generated" outline badge instead of the rotated stamp), with an on-screen disclaimer and a form checkbox to opt out (default on).

### 6d. Pipeline split & background execution (responsiveness pass)

**Problem observed**: the full run (`find_matches` ~75s + generation ~25s, both sequential) showed the user one static spinner for the entire ~100s — felt slow even though the underlying work wasn't wasted, because there was zero live feedback and the two stages were blocking each other despite generation only having a one-way dependency on `find_matches`'s output (style exemplars — nothing in `find_matches` needs generation to have run).

- `pipeline.py` split into `find_matches()` (steps 1-5) and `generate_practice_questions()` (step 6); `run_pipeline()` kept as a thin synchronous wrapper of both, unchanged for any caller that doesn't need the split.
- `app.py`: `find_matches()` runs in the main thread inside a live `st.status(..., expanded=True)` block — `on_log` now writes straight to the status widget as each step completes (Streamlit pushes each `.write()` to the browser as the script executes, not only once it finishes), replacing the old single static spinner with a step-by-step feed. Real PYQ matches render the moment this call returns.
- `generate_practice_questions()` is then submitted to a `ThreadPoolExecutor` — a `st.cache_resource` singleton, since a bare module-level executor would respawn (and leak a thread) on every Streamlit rerun — so it runs concurrently instead of blocking the real-match display.
- The Practice tab shows a live "writing in the background" placeholder via `st.fragment(run_every=2)`, polling the future every 2s; once resolved, it stashes the result in `session_state` and does one full `st.rerun()` to permanently switch to the static results view, rather than continuing to poll forever after there's nothing left to wait for. The background thread's own logging writes straight to the log file only — Streamlit UI calls aren't safe to make from a worker thread without the main script's context.
- Net effect: real PYQ matches now appear roughly 20-25s sooner than under the old single-block design, and the wait for the rest is visibly explained rather than silent.

## 7. Roadmap

- **Phase 0** — ✅ Done. Python environment: `pdfplumber`/`PyMuPDF`, `pymupdf`, `chromadb`, `sqlite3` (stdlib), `google-genai`/Vertex SDK for embedding + extraction + agent reasoning calls. ADC verified.
- **Phase 1** — ✅ Done. `questions.db` schema created; Chroma persistent collection configured (embedding function backed by Vertex AI `text-embedding-005`).
- **Phase 2** — ✅ Done. All 69 official Prelims (GS1/GS2 × 10 years) + Mains (GS1–4 + Essay × 10 years, minus one genuine gap: no 2017 Essay link on the official site) PDFs downloaded into `data/raw_pdfs/{examType}/{year}/`, ~430MB total. Link discovery required browser automation (`data/manifest.json`) since upsc.gov.in's page routes sit behind bot-detection that blocks plain HTTP clients — but the static PDF files themselves download fine with plain `requests` once the URLs are known (`src/ingest/download_papers.py`).
- **Phase 3** — ✅ Done. Gemini-based extraction (not the originally planned Tesseract OCR — see §3/§3a for why and what it took); 2,601 structured question records in `data/extracted/{examType}/{year}/{paper}.json`, ready for Phase 4 loading.
- **Phase 4** — ✅ Done. All 2,601 records loaded into `questions.db` (SQLite) and embedded + upserted into Chroma (`text-embedding-005`, retry-with-backoff added here too after hitting the same transient connection-reset issue as Phase 3). Option labels normalized (`(a)`/`A.`/`a` → `a`) at load time. Verified: SQLite and Chroma both report 2,601 records, spot-checked IDs match across both stores, and semantic search returns genuinely relevant results with metadata filters working (e.g. a `paper: GS4` filtered query for "ethical dilemma of a civil servant facing corruption" correctly surfaced matching case studies).
- **Phase 5** — ✅ Done. Agent pipeline built in `src/agent/` (not ADK — see §6): context extraction (with grounding fallback) → query expansion → retrieval → relevance filtering → lookup/format. Tested end-to-end with both a self-sufficient article (Great Indian Bustard conservation — found 9 well-reasoned Prelims/Mains matches spanning Wildlife Protection Act, Biological Diversity Act, and the Supreme Court's environmental law role, not just bird-specific hits) and a deliberately sparse one (a bare headline that correctly triggered the grounding fallback and still found 6 relevant matches).
- **Phase 6** — ✅ Done. `app.py` — paste article text and/or upload a photo, form submit runs the full pipeline with a spinner, results shown in Prelims/Mains tabs (question, options or marks/word-limit, year, paper, tier badge, match reason), with an expander showing what the agent understood and which queries it used. Verified live in a real browser end-to-end (not just import-checked): submitted an India-EU FTA article, got 9 Prelims + 5 Mains matches rendering correctly across both tabs, including a genuinely clever thematic catch (a "Common Carbon Metric" Prelims question correctly linked to the article's carbon border adjustment mechanism).
- **Phase 6b** — ✅ Done. Reliability pass: determinism (`temperature`/`seed`), dynamic thinking on judgment steps, wider retrieval, and the retry-merge bug — see §6b for the full list and the "0 to 13 matches on the same article" symptom that motivated it.
- **Phase 6c** — ✅ Done. Practice-question generation (`generate_questions.py`) — opt-in, clearly labeled AI-generated Prelims/Mains questions with an answer key/approach, styled distinctly from real PYQ results — see §6c, including the Vertex string-enum schema bug hit and fixed along the way.
- **Phase 6d** — ✅ Done. Responsiveness pass: `find_matches()`/`generate_practice_questions()` split, live `st.status` progress feed, and background-threaded generation with a polling Practice tab — see §6d. Real PYQ matches now render ~20-25s sooner than the old single-block design.
- **Phase 7** — Partially done via §6b (determinism + retry-merge fixed; retrieval widened). Still open: a versioned golden query set (known news → expected question_ids) to actually measure precision/recall instead of eyeballing runs, hybrid search (BM25 + dense via RRF), and reranking. See `RAG_BEST_PRACTICES.md` (project root) for the researched options and trade-offs on each before picking one.
- **Phase 8 (future)** — Optional LLM-based enrichment pass to fill `subject`/`topic`/`keywords`/`linked_theme` (opt-in, not required for the current scope); state PSC connectors (same local pattern, new raw_pdfs subfolder + new Chroma collection); additional agent tools (mains-answer grader, mock-test generator, spaced-repetition planner).

## 8. Decisions locked in (updated)

| Decision | Choice |
|---|---|
| Data location | Fully local — raw PDFs, SQLite, Chroma all on-device |
| Data sourcing | Official upsc.gov.in only, 2016–2025 (confirmed archive floor), no third-party sites |
| Prelims correct_option | Always `null` — no official answer key exists for this exam, confirmed directly |
| Metadata extraction | Rule-based (no LLM); `year` and `source_doc_url` always captured |
| Structured metadata store | SQLite |
| Vector store | ChromaDB (persistent, local) |
| Embedding model | Vertex AI `text-embedding-005` (cloud) — ≈$0.05 one-time for the full corpus, negligible per-query; local model dropped to save machine resources/time |
| Agent reasoning | Gemini via Vertex AI (ADC) — the only cloud dependency |
| Agent framework | Plain Python calling `google-genai` directly (§6) — no Google ADK; corrected here, this row previously contradicted §6's own reasoning |
| Interface | Streamlit, run locally |
| Determinism policy (§6b) | `temperature=0.0` + fixed `seed` by default for every agent-pipeline Gemini call; `thinking_budget=-1` (dynamic) for query expansion, relevance filtering, and practice-question generation specifically — not context extraction |
| Practice-question generation (§6c) | Opt-in (checkbox, default on), clearly labeled "AI-generated," rendered in its own tab — never merged into real-PYQ results; the one disclosed exception to §1's no-fabrication principle |
| Structured-output enums (§6c) | Vertex's schema validator only accepts **string** enum values, even for an integer-typed field — declare as a string enum and cast at the boundary, never `"type": "integer"` with an int `enum` |
| Background generation (§6d) | `generate_practice_questions()` runs in a `ThreadPoolExecutor` (`st.cache_resource` singleton) so real PYQ matches render before practice questions finish; the Practice tab polls via `st.fragment(run_every=2)` |

## 9. Caveat worth flagging

With everything local, your raw PDFs + `questions.db` + `data/chroma/` are now the *only* copy of this dataset — there's no cloud backup by default. Worth putting `data/` under your own backup routine (external drive, git, cloud sync folder of your choice) once Phase 2–4 data collection is done, since re-downloading/re-parsing from scratch is avoidable effort.
