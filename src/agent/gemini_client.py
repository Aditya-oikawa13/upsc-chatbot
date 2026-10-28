"""Shared Gemini client + a small helper for schema-constrained calls, used
by every step in the agent pipeline (context, query expansion, relevance
filtering). Retry-with-backoff matches the pattern used in ingestion —
this environment has shown real transient connection resets against
Vertex AI, not a hypothetical concern.
"""
from __future__ import annotations

import json
import time

from google import genai
from google.genai import types

from .. import config

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=config.GCP_PROJECT, location=config.GCP_LOCATION)
    return _client


DETERMINISTIC_SEED = 42


def structured_call(
    contents: list,
    schema: dict,
    model: str = config.REASONING_MODEL,
    max_retries: int = 5,
    temperature: float = 0.0,
    thinking_budget: int = 0,
) -> dict:
    """A schema-constrained, tool-free call. Returns the parsed JSON dict.

    temperature=0.0 + a fixed seed by default — these calls make judgment
    calls (what to search for, what counts as relevant) that should be
    reproducible for the same input, not re-rolled on every request. Callers
    doing a genuinely broader/creative pass (e.g. the retry's broadened
    queries) can override temperature explicitly.
    """
    client = get_client()
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                    seed=DETERMINISTIC_SEED,
                    thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                ),
            )
            return json.loads(resp.text)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"Gemini structured call failed after {max_retries} attempts") from last_error


def grounded_call(prompt: str, model: str = config.REASONING_MODEL, max_retries: int = 5) -> str:
    """A free-text call with Google Search grounding enabled. Tools and
    schema-constrained output can't be combined in one call, so this
    returns plain text — callers re-structure it with a follow-up
    structured_call if they need fields out of it."""
    client = get_client()
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            return resp.text
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"Gemini grounded call failed after {max_retries} attempts") from last_error
