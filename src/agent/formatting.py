"""Display formatting for question text pulled verbatim from PDF source —
shared by the live app (as a free, local fallback) and the one-time backfill
script (src/ingest/format_questions.py) that caches LLM-formatted versions
in SQLite. See PLAN.md and format_questions.py's module docstring for why
this is a cached batch step, not a per-request LLM call.

Some PYQs pack numbered statements ("1. ... 2. ... 3. ...") into one run-on
paragraph — verbatim from the source PDFs, no line breaks between them.
`split_statement_question` splits those into (intro, statements, outro)
using regex; it's fast and free but only catches phrasing already in
_OUTRO_PHRASES. The backfill script asks an LLM to do the same job more
generally, and `is_verbatim` is the free, local check that guards against
the model rewording anything in the process — no re-prompting on failure,
callers just don't cache that row and this regex splitter remains the
fallback at render time.
"""
from __future__ import annotations

import re

_STATEMENT_MARKER_RE = re.compile(r"(?<!\d)([1-9]\d?)\.\s+(?!\d)")
_OUTRO_PHRASES = [
    "select the correct answer",
    "select the correct code",
    "select the correct combination",
    "which of the statements given above",
    "which of the statement given above",
    "which of the above statements",
    "which of the statements above",
    "which one of the statements given above",
    "how many of the statements given above",
    "how many of the above statements",
    "which of the pairs given above",
    "which of these assumptions",
    "which of the following conclusions",
    "which one of the following inferences",
    "which one of the following is the correct",
    "what is/are the valid conclusion",
    "which of the above",
]


def split_statement_question(text: str) -> tuple[str, list[str], str] | None:
    """If `text` contains a run of sequentially numbered statements (1. 2.
    3. ...), split it into (intro, statements, outro). Returns None if no
    such run is found, so the caller can fall back to a plain paragraph."""
    run = []
    for m in _STATEMENT_MARKER_RE.finditer(text):
        n = int(m.group(1))
        if n == 1:
            run = [m]
        elif run and n == int(run[-1].group(1)) + 1:
            run.append(m)
    if len(run) < 2:
        return None

    intro = text[: run[0].start()].strip()
    bounds = [m.start() for m in run[1:]] + [None]
    statements = [text[run[i].end() : bounds[i]].strip() for i in range(len(run))]

    # The last statement and the trailing instruction sentence ("Select the
    # correct answer...") run together with no marker between them — split
    # on the earliest known instruction phrase found in that tail.
    last_statement = statements[-1]
    lower = last_statement.lower()
    split_at = min((lower.find(p) for p in _OUTRO_PHRASES if p in lower), default=None)
    outro = ""
    if split_at is not None:
        outro = last_statement[split_at:].strip()
        last_statement = last_statement[:split_at].strip()
    statements[-1] = last_statement

    if not intro or not all(statements):
        return None
    return intro, statements, outro


def has_statement_list(text: str) -> bool:
    """Cheap local pre-filter: does this question need LLM formatting at
    all? Used by the backfill script to skip the ~85% of questions that
    have no numbered-statement run and are already fine as plain text."""
    return split_statement_question(text) is not None


_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def is_verbatim(original: str, formatted: dict, slack: int = 2) -> bool:
    """Zero-cost local guard — no extra LLM calls. Confirms the model only
    restructured the text (plus the one explicitly-allowed dehyphenation
    fix) rather than rewording it, by comparing word sets. The "N. " list
    markers are expected to disappear (that's the point of formatting), so
    they're stripped from `original` the same way split_statement_question
    would before comparing. A small amount of remaining slack absorbs a
    legitimate hyphen-break word merge/split; anything bigger means the
    model changed real words, so the caller should not trust this row."""
    reconstructed = " ".join(
        [formatted.get("intro", ""), *formatted.get("statements", []), formatted.get("outro", "")]
    )
    orig_words = _words(_STATEMENT_MARKER_RE.sub(" ", original))
    new_words = _words(reconstructed)
    missing = orig_words - new_words
    added = new_words - orig_words
    return len(missing) <= slack and len(added) <= slack
