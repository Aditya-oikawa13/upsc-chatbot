"""Step 1: read the user's pasted article (text and/or image), extract the
substantive content the rest of the pipeline reasons over, and judge whether
there's enough to work with. If not, fall back to Gemini's Google Search
grounding tool to fill in the gaps — see PLAN.md §6/§6a for why grounding
was chosen over a free alternative (SerpApi) despite the cost difference.
"""
from __future__ import annotations

from pathlib import Path

from google.genai import types

from .gemini_client import grounded_call, structured_call

CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "key_facts": {"type": "string"},
        "sufficient": {"type": "boolean"},
    },
    "required": ["topic", "entities", "key_facts", "sufficient"],
}

CONTEXT_PROMPT = """You are helping a UPSC Civil Services aspirant who just \
pasted a news article (as text and/or an image) into a study tool. Read it \
and extract:

- topic: a short (5-10 word) descriptive label for what this article is about
- entities: the key named entities, places, organizations, acts/policies, \
species, events etc. mentioned that a question-setter might reference
- key_facts: a dense paragraph (3-6 sentences) summarizing the substantive \
facts — what happened, who/what is involved, why it matters — enough for \
someone to write exam questions connecting this to broader topics
- sufficient: true if this article has enough concrete substance (specific \
facts, entities, a real event) to reason about; false if it's too short, \
vague, or just a headline/fragment with nothing to actually analyze"""

RESTRUCTURE_PROMPT = """The following is background information gathered \
about a news topic. Extract the same fields as before from it. Since this \
was gathered specifically to fill in a gap, set sufficient to true unless \
it is genuinely still too thin to reason about.

---
{text}
---"""


def extract_context(article_text: str | None = None, image_path: str | Path | None = None) -> dict:
    parts = []
    if image_path:
        image_path = Path(image_path)
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        parts.append(types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime))
    if article_text:
        parts.append(article_text)
    parts.append(CONTEXT_PROMPT)

    context = structured_call(parts, CONTEXT_SCHEMA)

    if not context["sufficient"]:
        query_hint = article_text or context["topic"]
        grounded_text = grounded_call(
            f"Find recent, factual news context about: {query_hint}. "
            f"Summarize the key facts, entities, and background."
        )
        context = structured_call([RESTRUCTURE_PROMPT.format(text=grounded_text)], CONTEXT_SCHEMA)
        context["augmented_with_search"] = True
    else:
        context["augmented_with_search"] = False

    return context
