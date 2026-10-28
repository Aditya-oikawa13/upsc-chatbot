"""Step 6 (optional, opt-in): author new, original UPSC-style practice
questions on the article's topic — not retrieval, generation.

This is a different kind of output from the rest of the pipeline on purpose.
Every other field in this project is either read off an official PDF or
left null (see PLAN.md §0/§1.2 — "no fabrication"). These questions are the
one deliberate exception: they don't exist in any real paper. The caller
(app.py) must always label them as AI-generated practice material, never
render them inside the real-PYQ match cards, and never persist them to
SQLite/Chroma alongside genuine archive records.

Style grounding: if the pipeline already found real matches for this
article, a couple of them are passed in as few-shot exemplars purely for
phrasing/structure (the exact "consider the following statements" /
"which of the statements given above is/are correct?" idiom UPSC uses, and
Mains' compressed multi-clause directive style) — never as source facts to
copy. The model is explicitly told to draw its content only from the
article's own context, not from the exemplars.
"""
from __future__ import annotations

from .gemini_client import structured_call

NUM_PRELIMS = 3
NUM_MAINS = 2


def _build_schema(n_prelims: int, n_mains: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "prelims": {
                "type": "array",
                "minItems": n_prelims,
                "maxItems": n_prelims,
                "items": {
                    "type": "object",
                    "properties": {
                        "intro": {
                            "type": "string",
                            "description": 'Lead-in stem, e.g. "Consider the following statements:"',
                        },
                        "statements": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 4,
                            "items": {"type": "string"},
                        },
                        "outro": {
                            "type": "string",
                            "description": (
                                'Closing instruction, e.g. "Which of the statements given above '
                                'is/are correct?" or "How many of the above statements are correct?"'
                            ),
                        },
                        "options": {
                            "type": "array",
                            "minItems": 4,
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "text": {"type": "string"},
                                },
                                "required": ["label", "text"],
                            },
                        },
                        "correct_option": {"type": "string"},
                        "explanation": {
                            "type": "string",
                            "description": "1-3 sentences: why the correct option is correct and the others aren't.",
                        },
                    },
                    "required": ["intro", "statements", "outro", "options", "correct_option", "explanation"],
                },
            },
            "mains": {
                "type": "array",
                "minItems": n_mains,
                "maxItems": n_mains,
                "items": {
                    "type": "object",
                    "properties": {
                        "paper": {"type": "string", "enum": ["GS1", "GS2", "GS3", "GS4"]},
                        "question_text": {"type": "string"},
                        # Vertex's structured-output schema only accepts string enum
                        # values (a google-genai client-side validation rule, not a
                        # prompt issue) — enumerate as strings and cast to int below
                        # rather than declaring "type": "integer" with an int enum,
                        # which raises a pydantic ValidationError before any request
                        # is even sent, on every attempt, every retry.
                        "marks": {"type": "string", "enum": ["10", "15"]},
                        "word_limit": {"type": "string", "enum": ["150", "250"]},
                        "approach": {
                            "type": "string",
                            "description": (
                                "4-6 sentence answer-writing approach: the dimensions/angles a strong "
                                "answer would cover, not a full model answer."
                            ),
                        },
                    },
                    "required": ["paper", "question_text", "marks", "word_limit", "approach"],
                },
            },
        },
        "required": ["prelims", "mains"],
    }

GENERATE_PROMPT = """You are a UPSC Civil Services question-setter creating \
NEW original practice questions inspired by a news article — in UPSC's own \
style, depth, and difficulty, not questions about the same superficial \
headline.

Topic: {topic}
Entities: {entities}
Key facts: {key_facts}

What "genuine UPSC depth" means, and what to avoid:
- UPSC almost never tests the raw news event itself. It uses the event as a \
trigger to test the static concept, institution, law, or mechanism behind \
it. A question should require connecting the article to that broader \
static portion of the syllabus, not just restating the article's facts.
- Prelims statements should be individually plausible and require real \
conceptual discrimination — mix a subtly wrong statement in with correct \
ones rather than one obviously-false outlier. At least one statement \
should test a related static-syllabus fact (a constitutional provision, an \
institution's mandate, a treaty, a numeric detail) rather than only facts \
from the article. Do not fabricate a statistic or fact you are not \
confident is true — if unsure, phrase it as a general/qualitative claim \
instead of inventing a number.
- Mains questions should be multi-faceted the way UPSC actually compresses \
demands into one sentence — a directive verb (Discuss / Examine / \
Critically analyze / Elucidate) applied to the core issue, often paired \
with a second clause asking for implications, challenges, or a way \
forward (e.g. "Discuss X. Also suggest measures to address Y."). Avoid a \
single flat "what do you think about X" question.

{exemplar_block}
Generate exactly {n_prelims} Prelims questions and {n_mains} Mains \
questions. Prelims: 4 options, exactly one correct (mark its label in \
correct_option), plus a short explanation. Mains: assign whichever GS \
paper (GS1-4) the question actually belongs to given its content — not all \
in one paper — with standard marks/word-limit pairing (10 marks/150 words \
or 15 marks/250 words), plus a brief answer-writing approach (dimensions to \
cover, not a full essay).

These are original questions you are authoring, not reproductions of any \
real paper — do not copy wording, statements, or specific facts from the \
style examples below; they're for tone and structure only."""

EXEMPLAR_BLOCK = """For tone and structure only — do NOT reuse their content, facts, or wording:

{exemplars}

"""


def _format_exemplars(prelims_matches: list[dict], mains_matches: list[dict]) -> str:
    lines = []
    for m in (prelims_matches or [])[:1]:
        lines.append(f"[Prelims style] {m['question_text'][:400]}")
    for m in (mains_matches or [])[:1]:
        lines.append(f"[Mains style] {m['question_text'][:400]}")
    if not lines:
        return ""
    return EXEMPLAR_BLOCK.format(exemplars="\n\n".join(lines))


def generate_questions(
    context: dict,
    prelims_matches: list[dict] | None = None,
    mains_matches: list[dict] | None = None,
    n_prelims: int = NUM_PRELIMS,
    n_mains: int = NUM_MAINS,
) -> dict:
    """Returns {"prelims": [...], "mains": [...]} of newly-authored,
    original practice questions. Caller must label these as AI-generated —
    they are not, and must never be presented as, real PYQs."""
    prompt = GENERATE_PROMPT.format(
        topic=context["topic"],
        entities=", ".join(context["entities"]),
        key_facts=context["key_facts"],
        exemplar_block=_format_exemplars(prelims_matches, mains_matches),
        n_prelims=n_prelims,
        n_mains=n_mains,
    )
    schema = _build_schema(n_prelims, n_mains)
    result = structured_call([prompt], schema, temperature=0.5, thinking_budget=-1)
    for q in result["mains"]:
        q["marks"] = int(q["marks"])
        q["word_limit"] = int(q["word_limit"])
    return result
