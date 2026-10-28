"""Step 2: turn the article's context into a diverse set of search queries.

The point isn't to find the same event asked about before — it's to find
questions on the same *trend*. A news story about an endangered bird should
surface search queries about endangered species broadly, wildlife protection
law, international conservation treaties, and the "X in the news" pattern
UPSC actually uses for Prelims — not just queries about that one bird.
"""
from __future__ import annotations

from .gemini_client import structured_call

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 6,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "angle": {
                        "type": "string",
                        "enum": ["direct", "category", "policy", "international", "pattern"],
                    },
                },
                "required": ["query", "angle"],
            },
        }
    },
    "required": ["queries"],
}

QUERY_PROMPT = """You are helping find UPSC Civil Services previous-year \
questions (Prelims + Mains) related to a news topic — not questions about \
the exact same event, but questions on the same underlying trend, category, \
or governance mechanism that UPSC tends to test around topics like this.

Topic: {topic}
Entities: {entities}
Key facts: {key_facts}

Generate 5-8 diverse search queries a previous-year-question database should \
be searched with, covering different angles:
- direct: the specific topic/entity itself
- category: the broader class it belongs to (e.g. a species -> endangered \
species; a scheme -> government welfare schemes)
- policy: the relevant law, act, policy, or institutional mechanism in India
- international: the relevant international treaty, convention, or \
comparative angle, if any applies
- pattern: phrased the way UPSC actually asks about current-affairs-linked \
static topics, e.g. "X in the news", "regions/species/organizations \
frequently in the news"

Not every angle necessarily applies to every topic — skip one rather than \
force a bad query, but favor breadth over narrowly re-querying the same \
thing. The database being searched is already entirely UPSC previous-year \
questions, so write plain topical search phrases — do not include "UPSC" \
or exam-related words in the query text itself."""


def expand_queries(context: dict) -> list[dict]:
    prompt = QUERY_PROMPT.format(
        topic=context["topic"],
        entities=", ".join(context["entities"]),
        key_facts=context["key_facts"],
    )
    result = structured_call([prompt], QUERY_SCHEMA, thinking_budget=-1)
    return result["queries"]
