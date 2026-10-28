"""Step 3: run every generated query against Chroma and pool the results.

Deliberately not an LLM call — this is a plain deterministic tool. Both
exam types are searched together and unfiltered by design: the point of
this feature is surfacing both Prelims and Mains matches for one article.
"""
from __future__ import annotations

from ..db import vector_store

TOP_K_PER_QUERY = 12
MAX_CANDIDATES = 60


def retrieve_candidates(queries: list[dict], top_k: int = TOP_K_PER_QUERY) -> list[dict]:
    collection = vector_store.get_collection()
    best_distance: dict[str, float] = {}
    best_record: dict[str, dict] = {}

    for q in queries:
        results = collection.query(query_texts=[q["query"]], n_results=top_k)
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for qid, doc, meta, dist in zip(ids, docs, metas, dists):
            if qid not in best_distance or dist < best_distance[qid]:
                best_distance[qid] = dist
                best_record[qid] = {
                    "question_id": qid,
                    "document": doc,
                    "distance": dist,
                    "matched_query": q["query"],
                    "matched_angle": q["angle"],
                    **meta,
                }

    candidates = sorted(best_record.values(), key=lambda r: r["distance"])
    return candidates[:MAX_CANDIDATES]
