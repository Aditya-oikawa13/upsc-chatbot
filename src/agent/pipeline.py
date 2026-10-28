"""Orchestrates the full news-article -> matching-PYQs flow.

Fixed sequence of steps (not a free-roaming agent loop) — context extraction
-> query expansion -> retrieval -> relevance filtering -> lookup/format, then
an optional final step that *generates* new UPSC-style practice questions
rather than retrieving real ones. See PLAN.md §6 for the full architecture
and the reasoning behind a controllable pipeline over an autonomous
ReAct-style agent.

The generation step is a different kind of output from everything before it
— fabricated by design, not read off a real paper — so it's opt-in and comes
back under its own `generated_prelims`/`generated_mains` keys, never merged
into `prelims_matches`/`mains_matches`. Keep it that way in any caller too.

Split into two functions, not one, on purpose: `find_matches` (steps 1-5)
and `generate_practice_questions` (step 6) are independently useful because
the dependency between them is one-way — generation only *reads*
`find_matches`'s output (for style exemplars), nothing in `find_matches`
needs generation to have run. A caller that wants to show real PYQ matches
immediately and let the slower generation step finish in the background
(e.g. a UI — see app.py) should call these separately rather than going
through `run_pipeline`, which just runs both in sequence for callers that
don't care about that distinction.

Usage:
    from src.agent.pipeline import run_pipeline
    result = run_pipeline(article_text="...")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..db import init_db
from .context import extract_context
from .generate_questions import generate_questions
from .query_expansion import expand_queries
from .relevance import filter_relevant
from .retrieval import retrieve_candidates

MIN_MATCHES_BEFORE_RETRY = 3

LogFn = Callable[[str], None]


def _noop_log(_message: str) -> None:
    pass

BROADEN_SUFFIX = (
    " The previous search was too narrow and found almost nothing — make "
    "these queries noticeably broader and more general than a first attempt "
    "would be."
)


def _lookup_records(question_ids: list[str]) -> dict[str, dict]:
    if not question_ids:
        return {}
    conn = init_db.get_connection()
    placeholders = ",".join("?" for _ in question_ids)
    cur = conn.execute(f"SELECT * FROM questions WHERE question_id IN ({placeholders})", question_ids)
    cols = [d[0] for d in cur.description]
    records = {}
    for row in cur.fetchall():
        r = dict(zip(cols, row))
        if r.get("options_json"):
            r["options"] = json.loads(r["options_json"])
        if r.get("formatted_json"):
            r["formatted"] = json.loads(r["formatted_json"])
        records[r["question_id"]] = r
    return records


def find_matches(
    article_text: str | None = None,
    image_path: str | Path | None = None,
    on_log: LogFn = _noop_log,
) -> dict:
    """Steps 1-5: context extraction -> query expansion -> retrieval ->
    relevance filtering (with one bounded broaden-and-retry) -> SQLite
    lookup. Returns real PYQ matches only — no generated practice
    questions; see generate_practice_questions for that."""
    if not article_text and not image_path:
        raise ValueError("Provide article_text and/or image_path")

    on_log("Reading the article" + (" and image" if image_path else "") + "...")
    context = extract_context(article_text=article_text, image_path=image_path)
    on_log(f'Extracted topic — "{context["topic"]}"')
    on_log(f"Entities identified — {', '.join(context['entities']) or 'none'}")
    on_log(f"Key facts — {context['key_facts']}")
    if context.get("augmented_with_search"):
        on_log("Article was too sparse on its own — falling back to a Google Search lookup for more context.")
        on_log(f'Re-extracted topic after search — "{context["topic"]}"')

    on_log("Expanding into search queries (direct / category / policy / international / pattern angles)...")
    queries = expand_queries(context)
    for q in queries:
        on_log(f"  query [{q['angle']}] — {q['query']}")

    on_log("Searching the previous-year-question vector database...")
    candidates = retrieve_candidates(queries)
    on_log(f"Retrieved {len(candidates)} candidate question(s) across all queries.")

    on_log("Judging genuine relevance of each candidate against the article...")
    matches = filter_relevant(context, candidates)
    on_log(f"Kept {len(matches)} relevant match(es) after filtering.")

    retried = False
    if len(matches) < MIN_MATCHES_BEFORE_RETRY and not retried:
        retried = True
        on_log(f"Only {len(matches)} match(es) found — broadening the queries and retrying once...")
        broader_prompt_context = dict(context)
        broader_prompt_context["key_facts"] += BROADEN_SUFFIX
        queries = expand_queries(broader_prompt_context)
        for q in queries:
            on_log(f"  broadened query [{q['angle']}] — {q['query']}")
        on_log("Re-searching the vector database with the broadened queries...")
        broadened_candidates = retrieve_candidates(queries)
        on_log(f"Retrieved {len(broadened_candidates)} candidate question(s) on retry.")

        # Merge with the first pass rather than replace it — a genuine match
        # already found should never be lost just because the broadened
        # queries alone didn't independently rediscover it.
        combined = {c["question_id"]: c for c in candidates}
        for c in broadened_candidates:
            existing = combined.get(c["question_id"])
            if existing is None or c["distance"] < existing["distance"]:
                combined[c["question_id"]] = c
        candidates = sorted(combined.values(), key=lambda r: r["distance"])
        on_log(f"Combined candidate pool: {len(candidates)} unique question(s) across both passes.")

        on_log("Re-judging relevance over the combined candidate pool...")
        matches = filter_relevant(context, candidates)
        on_log(f"Kept {len(matches)} relevant match(es) after retry.")

    on_log("Looking up full question records (text, options, year) from SQLite...")
    records = _lookup_records([m["question_id"] for m in matches])

    prelims_matches, mains_matches = [], []
    for m in matches:
        record = records.get(m["question_id"])
        if not record:
            on_log(f"  dropped {m['question_id']} — matched but no record found in the database")
            continue  # shouldn't happen, but never surface a match we can't back with real data
        entry = {**record, "reason": m["reason"], "tier": m["tier"]}
        (prelims_matches if record["exam_type"] == "prelims" else mains_matches).append(entry)

    on_log(f"Done — {len(prelims_matches)} Prelims match(es), {len(mains_matches)} Mains match(es).")

    return {
        "context": context,
        "queries": queries,
        "retried": retried,
        "prelims_matches": prelims_matches,
        "mains_matches": mains_matches,
    }


def generate_practice_questions(
    context: dict,
    prelims_matches: list[dict] | None = None,
    mains_matches: list[dict] | None = None,
    on_log: LogFn = _noop_log,
) -> dict:
    """Step 6 (optional): author new UPSC-style practice questions — see
    generate_questions.py. Only needs find_matches's output as (optional)
    style exemplars, so it's safe to run this after find_matches has
    already returned to the caller (e.g. in a background thread while real
    matches are already on screen)."""
    on_log("Authoring new UPSC-style practice questions on this topic (not real PYQs)...")
    generated = generate_questions(context, prelims_matches, mains_matches)
    on_log(
        f"Generated {len(generated['prelims'])} Prelims + {len(generated['mains'])} Mains "
        "practice question(s)."
    )
    return {"generated_prelims": generated["prelims"], "generated_mains": generated["mains"]}


def run_pipeline(
    article_text: str | None = None,
    image_path: str | Path | None = None,
    on_log: LogFn = _noop_log,
    generate_practice: bool = True,
) -> dict:
    """Convenience wrapper: runs find_matches then, if requested,
    generate_practice_questions, synchronously in sequence. A caller that
    wants the two run with different timing (e.g. generation in the
    background) should call them separately instead."""
    result = find_matches(article_text=article_text, image_path=image_path, on_log=on_log)
    if generate_practice:
        generated = generate_practice_questions(
            result["context"], result["prelims_matches"], result["mains_matches"], on_log=on_log
        )
    else:
        generated = {"generated_prelims": [], "generated_mains": []}
    return {**result, **generated}
