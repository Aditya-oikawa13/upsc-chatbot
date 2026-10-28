"""Step 4: judge genuine relevance over the pooled candidate pool, and
structure the result. Vector similarity alone over-includes — a question
can be semantically close in wording while being thematically irrelevant.
This is the step that actually reasons about whether a match makes sense
for an aspirant connecting this article to what UPSC tends to ask.

The model only ever returns question_ids + reasons, never re-written
question text — the real text/options/year always come from SQLite
afterward (see pipeline.py), so there's no risk of the model drifting from
what was actually printed.
"""
from __future__ import annotations

from .gemini_client import structured_call

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "tier": {
                        "type": "string",
                        "enum": ["direct", "thematic", "broader"],
                    },
                },
                "required": ["question_id", "reason", "tier"],
            },
        }
    },
    "required": ["matches"],
}

RELEVANCE_PROMPT = """A UPSC Civil Services aspirant pasted a news article. \
Here is what it's about:

Topic: {topic}
Key facts: {key_facts}

Below is a pool of candidate previous-year questions (Prelims + Mains), \
retrieved by semantic search on several angles related to this article. \
Some are genuinely relevant (the aspirant would find it useful to see this \
question given the article); many are not — they may be superficially \
similar in wording while being thematically unrelated, or may be redundant \
near-duplicates of another match already kept.

For each genuinely relevant question, classify how it relates:
- direct: about the same specific topic/entity as the article
- thematic: about the same underlying category, mechanism, or issue
- broader: connects to a wider governance/policy theme the article touches on

Only include questions that are actually useful to show. Drop anything \
irrelevant, and drop near-duplicate repeats of a question already included \
unless each genuinely adds something. Give a one-sentence reason per match, \
in the aspirant's terms (why this is worth reviewing given the article).

Candidates:
{candidates}"""


def _format_candidate(c: dict) -> str:
    label = f"[{c['question_id']}] ({c['exam_type']}, {c['paper']}, {c['year']})"
    return f"{label}: {c['document'][:400]}"


def filter_relevant(context: dict, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    prompt = RELEVANCE_PROMPT.format(
        topic=context["topic"],
        key_facts=context["key_facts"],
        candidates="\n\n".join(_format_candidate(c) for c in candidates),
    )
    result = structured_call([prompt], RELEVANCE_SCHEMA, thinking_budget=-1)
    return result["matches"]
