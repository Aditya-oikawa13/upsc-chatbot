"""One-time (re-runnable) backfill: LLM-format the previous-year questions
that pack numbered statements ("1. ... 2. ... 3. ...") into one run-on
paragraph, and cache the result in SQLite (questions.formatted_json).

Why cache instead of formatting live in the agent pipeline: question text
is static — reformatting it on every search would add latency and API cost
to every user request for zero new benefit. This runs once, over just the
~15% of questions that actually need it (agent.formatting.has_statement_list
pre-filters locally, for free), batching many questions into each LLM call
rather than one call per question.

Trust, not verify-and-regenerate: the prompt is strict (verbatim only, no
rewording) and each result is checked locally against the original text
with agent.formatting.is_verbatim — a free word-set comparison, no second
LLM call. A row that fails the check, or that the model didn't return at
all, is simply left NULL; the app already has a local regex fallback
(agent.formatting.split_statement_question) for that case, so nothing ever
regresses to unformatted text — it just doesn't get the LLM's finer
handling of phrasing regex doesn't recognize.

Usage:
    python -m src.ingest.format_questions [--batch-size 25] [--force]
"""
from __future__ import annotations

import argparse
import json

from ..agent.formatting import has_statement_list, is_verbatim
from ..agent.gemini_client import structured_call
from ..db import init_db

DEFAULT_BATCH_SIZE = 25

FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "intro": {"type": "string"},
                    "statements": {"type": "array", "items": {"type": "string"}},
                    "outro": {"type": "string"},
                },
                "required": ["question_id", "intro", "statements", "outro"],
            },
        }
    },
    "required": ["questions"],
}

FORMAT_PROMPT = """The questions below were extracted from PDF exam booklets, \
and each one packs numbered statements ("1. ... 2. ... 3. ...") together \
with its lead-in and closing sentences into one run-on paragraph, with no \
line breaks. Restructure each one for display.

Do NOT reword, paraphrase, summarize, add, or remove any words, and do not \
change the meaning — reproduce the original wording exactly. The only \
edits allowed: undo an obvious PDF line-wrap hyphenation break in the \
middle of a word (e.g. "conser- vation" -> "conservation"), and fix an \
unambiguous single-word OCR/extraction garble. Never touch a real \
hyphenated word (e.g. "policy-makers" stays exactly as is).

For each question, split it into:
- intro: the lead-in sentence/stem, verbatim, up to (not including) the \
first numbered statement
- statements: each numbered statement's text, verbatim, without its \
leading number
- outro: the closing instruction/question sentence that follows the \
statements, verbatim (e.g. "Select the correct answer using the code \
given below.")

Return one entry per question_id given below, in any order.

Questions:
{questions}"""


def _format_one(question_id: str, text: str) -> str:
    return f"[{question_id}]: {text}"


def _fetch_candidates(conn, force: bool) -> list[tuple[str, str]]:
    where = "" if force else "WHERE formatted_json IS NULL"
    cur = conn.execute(f"SELECT question_id, question_text FROM questions {where}")
    return [(qid, text) for qid, text in cur.fetchall() if has_statement_list(text)]


def _format_batch(batch: list[tuple[str, str]]) -> dict[str, dict]:
    prompt = FORMAT_PROMPT.format(questions="\n\n".join(_format_one(qid, text) for qid, text in batch))
    result = structured_call([prompt], FORMAT_SCHEMA)
    return {q["question_id"]: q for q in result["questions"]}


def run(batch_size: int = DEFAULT_BATCH_SIZE, force: bool = False) -> None:
    init_db.init_db()
    conn = init_db.get_connection()

    candidates = _fetch_candidates(conn, force)
    print(f"{len(candidates)} question(s) need formatting" + (" (--force: re-checking all)" if force else ""))
    if not candidates:
        conn.close()
        return

    originals = dict(candidates)
    cached = skipped_unverified = skipped_missing = failed_batches = 0

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        print(f"  formatting {i + 1}-{i + len(batch)} of {len(candidates)}...")
        try:
            results = _format_batch(batch)
        except Exception as e:
            print(f"    batch failed, leaving these {len(batch)} unformatted: {e}")
            failed_batches += 1
            continue

        rows = []
        for qid, _ in batch:
            item = results.get(qid)
            if item is None:
                skipped_missing += 1
                continue
            if not is_verbatim(originals[qid], item):
                skipped_unverified += 1
                continue
            formatted = {"intro": item["intro"], "statements": item["statements"], "outro": item["outro"]}
            rows.append((json.dumps(formatted, ensure_ascii=False), qid))

        if rows:
            with conn:
                conn.executemany("UPDATE questions SET formatted_json = ? WHERE question_id = ?", rows)
            cached += len(rows)

    conn.close()
    print(
        f"Done — cached {cached}, skipped {skipped_unverified} (failed verbatim check), "
        f"{skipped_missing} (not returned by model), {failed_batches} batch(es) errored."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true", help="Re-check questions that already have formatted_json")
    args = parser.parse_args()
    run(batch_size=args.batch_size, force=args.force)
