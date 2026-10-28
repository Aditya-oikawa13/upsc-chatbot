"""One-off smoke test for Phase 1: proves SQLite + Chroma + Vertex embeddings
work together end to end, using a couple of hand-written dummy records (not
real PYQs). Safe to delete once Phase 3/4 ingestion is in place and tested
for real.

Usage:
    python -m src.db.smoke_test
"""
from __future__ import annotations

import datetime as dt
import json

from . import init_db, vector_store

DUMMY_RECORDS = [
    {
        "question_id": "SMOKE-PRELIMS-2020-1",
        "exam_board": "UPSC",
        "exam_type": "prelims",
        "is_objective": 1,
        "paper": "GS-Prelims",
        "year": 2020,
        "question_number": 1,
        "question_text": "Consider the following statements about the Reserve Bank of India's monetary policy committee.",
        "options_json": json.dumps(
            [
                {"label": "a", "text": "It has 6 members"},
                {"label": "b", "text": "It has 4 members"},
                {"label": "c", "text": "It has 8 members"},
                {"label": "d", "text": "It has 10 members"},
            ]
        ),
        "correct_option": "a",
        "marks": None,
        "word_limit": None,
        "source_doc_url": "SMOKE_TEST — not a real source",
    },
    {
        "question_id": "SMOKE-MAINS-2019-GS3-1",
        "exam_board": "UPSC",
        "exam_type": "mains",
        "is_objective": 0,
        "paper": "GS3",
        "year": 2019,
        "question_number": 1,
        "question_text": "Discuss the role of the Monetary Policy Committee in controlling inflation in India.",
        "options_json": None,
        "correct_option": None,
        "marks": 15,
        "word_limit": 250,
        "source_doc_url": "SMOKE_TEST — not a real source",
    },
]


def run() -> None:
    init_db.init_db()
    conn = init_db.get_connection()
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    with conn:
        for r in DUMMY_RECORDS:
            conn.execute(
                """
                INSERT OR REPLACE INTO questions (
                    question_id, exam_board, exam_type, is_objective, paper, year,
                    question_number, question_text, options_json, correct_option,
                    marks, word_limit, source_doc_url, extraction_method, extracted_at
                ) VALUES (
                    :question_id, :exam_board, :exam_type, :is_objective, :paper, :year,
                    :question_number, :question_text, :options_json, :correct_option,
                    :marks, :word_limit, :source_doc_url, 'rule-based', :extracted_at
                )
                """,
                {**r, "extracted_at": now},
            )
    print(f"Inserted {len(DUMMY_RECORDS)} dummy rows into SQLite.")

    collection = vector_store.get_collection()
    collection.upsert(
        ids=[r["question_id"] for r in DUMMY_RECORDS],
        documents=[r["question_text"] for r in DUMMY_RECORDS],
        metadatas=[
            {
                "exam_type": r["exam_type"],
                "year": r["year"],
                "paper": r["paper"],
            }
            for r in DUMMY_RECORDS
        ],
    )
    print(f"Upserted into Chroma. Collection now has {collection.count()} records.")

    query = "RBI interest rate policy and inflation control"
    results = collection.query(query_texts=[query], n_results=2)
    print(f"\nQuery: {query!r}")
    for qid, doc, dist in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        print(f"  [{dist:.4f}] {qid}: {doc}")


if __name__ == "__main__":
    run()
