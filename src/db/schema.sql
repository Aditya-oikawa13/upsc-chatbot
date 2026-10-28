-- Structured metadata store for previous-year questions.
-- This is the source of truth; Chroma's metadata is a filterable mirror of a
-- subset of these columns. See PLAN.md §4 for field-by-field rationale.

CREATE TABLE IF NOT EXISTS questions (
    question_id        TEXT PRIMARY KEY,
    exam_board          TEXT NOT NULL DEFAULT 'UPSC',
    exam_type           TEXT NOT NULL CHECK (exam_type IN ('prelims', 'mains')),
    is_objective        INTEGER NOT NULL CHECK (is_objective IN (0, 1)),
    paper               TEXT NOT NULL,
    year                INTEGER NOT NULL,
    question_number     INTEGER,
    question_text       TEXT NOT NULL,

    -- Prelims only
    options_json        TEXT,       -- JSON array of {"label": "a", "text": "..."}
    correct_option       TEXT,       -- always NULL for now: UPSC publishes no official Prelims answer key (confirmed)

    -- Mains only
    marks               INTEGER,
    word_limit           INTEGER,

    -- Always populated
    source_doc_url       TEXT NOT NULL,   -- local file path to the source PDF
    extraction_method    TEXT NOT NULL DEFAULT 'rule-based',
    extracted_at         TEXT NOT NULL,   -- ISO timestamp

    -- Reserved for a future enrichment phase — NULL until then, no LLM calls yet
    subject             TEXT,
    topic               TEXT,
    subtopic             TEXT,
    keywords_json        TEXT,
    linked_theme          TEXT,
    static_or_current      TEXT,

    -- NULL until src/ingest/format_questions.py backfills it. JSON
    -- {"intro": ..., "statements": [...], "outro": ...} — a display-only
    -- reflow of question_text for PYQs that pack numbered statements into
    -- one run-on paragraph. Verified word-for-word against question_text
    -- before being cached (see agent/formatting.py:is_verbatim); questions
    -- without a statement list, or that fail verification, stay NULL and
    -- the app falls back to a local regex splitter at render time.
    formatted_json        TEXT
);

CREATE INDEX IF NOT EXISTS idx_questions_year ON questions (year);
CREATE INDEX IF NOT EXISTS idx_questions_exam_type ON questions (exam_type);
CREATE INDEX IF NOT EXISTS idx_questions_paper ON questions (paper);
CREATE INDEX IF NOT EXISTS idx_questions_exam_board ON questions (exam_board);
