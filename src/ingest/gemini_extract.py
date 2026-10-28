"""Extract structured questions from UPSC PDFs using Gemini (Vertex AI).

Why Gemini instead of a text-layer/OCR pipeline: see PLAN.md §3. Short version —
Prelims PDFs have a text layer with genuine character-level corruption baked
into the font encoding, and Mains PDFs are scanned images with no text layer
at all. Local OCR (Tesseract) was tested and works, but Gemini measurably
fixed Tesseract's two weak spots (digit confusion in marks/word-limits, and
reconstructing split/tabular layouts) at a cost of about $2 for the entire
10-year corpus — trivial enough that consistency won out over a hybrid
Tesseract+Gemini pipeline.

Each page is rendered to an image and sent to Gemini with a schema-constrained
prompt asking it to extract only the English-language question content
(Hindi is ignored per project scope — this bot is English-only). Output is
written per-document to data/extracted/{examType}/{year}/{paper}.json for
inspection before Phase 4 loads it into SQLite + Chroma.

Usage:
    python -m src.ingest.gemini_extract                # all documents
    python -m src.ingest.gemini_extract --year 2023     # one year
    python -m src.ingest.gemini_extract --limit 1       # smoke test
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf
from google import genai
from google.genai import types
from PIL import Image

from .. import config

MAX_IMAGE_DIM = 2000
MODEL = "gemini-2.5-flash"
MAX_WORKERS = 3

PRELIMS_PROMPT = """This is one page from an Indian UPSC Civil Services \
Preliminary exam question paper, printed bilingually in Hindi and English \
(two-column layout, one language per column or per page). Extract ONLY the \
English-language multiple-choice questions on this page, verbatim as \
printed. Ignore all Hindi text completely. For each question give its exact \
printed question number, the question text, and its four options (a)-(d) \
with their text. Do not translate, paraphrase, solve, or answer the \
questions. If this page has no English question content (a Hindi-only page, \
or a cover/instructions page), return an empty questions list."""

MAINS_PROMPT = """This is one page from an Indian UPSC Civil Services Main \
exam question paper (General Studies or Essay), scanned from a printed \
booklet where each question appears in Hindi followed immediately by its \
English translation. Extract ONLY the English-language version of each \
question on this page, verbatim. Ignore the Hindi text completely. For each \
question give: its exact printed question number, the English question \
text, the marks value printed for it (an integer; use 0 if genuinely not \
visible), and the word limit stated for it, e.g. "150 words" -> 150 (an \
integer; use 0 if genuinely not stated). If this page has no English \
question content (an instructions-only page), return an empty questions \
list."""

ESSAY_PROMPT = """This is one page from an Indian UPSC Civil Services Main \
exam Essay question paper, printed bilingually in Hindi and English. Each \
essay topic is a short quotation or statement to write an essay on — NOT a \
question with marks or a word limit printed next to it (those are fixed for \
the whole paper and stated once in the instructions, not per topic). \
Extract ONLY the English-language version of each topic on this page, \
verbatim, ignoring the Hindi text completely and ignoring which numbered \
section (e.g. "Section A", "Section B") it falls under. If this page has no \
English topic content (an instructions-only page), return an empty topics \
list."""

ESSAY_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["topics"],
}

# Essay marks/word-limit are fixed paper-level constants, not printed per
# topic — verified against the instructions page (e.g. "125x2=250" marks,
# "1000-1200 words each"), stable across the 2016-2025 GS-pattern era.
ESSAY_MARKS = 125
ESSAY_WORD_LIMIT = None  # printed as a 1000-1200 range, not a single figure

PRELIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_number": {"type": "integer"},
                    "question_text": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["label", "text"],
                        },
                    },
                },
                "required": ["question_number", "question_text", "options"],
            },
        }
    },
    "required": ["questions"],
}

MAINS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_number": {"type": "integer"},
                    "question_text": {"type": "string"},
                    "marks": {"type": "integer"},
                    "word_limit": {"type": "integer"},
                },
                "required": ["question_number", "question_text", "marks", "word_limit"],
            },
        }
    },
    "required": ["questions"],
}

# GS4 (Ethics) mixes short conceptual questions with case studies: a Hindi
# scenario spanning one or more pages, its English translation spanning one
# or more DIFFERENT pages, then lettered sub-questions (a)/(b) with combined
# marks. Extracting page-by-page fragments these across page boundaries —
# tested and confirmed to lose content (see PLAN.md §3). Fix: send every
# page of the document in a single call so the model has full context.
GS4_PROMPT = """These are all the pages, in order, of one Indian UPSC Civil \
Services Main exam General Studies Paper IV (Ethics) question paper. It is \
printed bilingually: some questions are short, with a Hindi paragraph \
immediately followed by its English translation. Others are case studies — \
a scenario (a paragraph or several) followed by one or more lettered \
sub-questions like (a)/(b) — where the Hindi version of the whole case study \
may be on entirely different pages from its English version, and either \
version may itself span multiple pages.

Extract ONLY the English-language version of each numbered item, verbatim, \
ignoring all Hindi text completely. For a case study, concatenate its full \
scenario text and all of its lettered sub-questions into one single \
question_text string, in the order printed, preserving the (a)/(b) labels \
inline as printed. Use the exact printed top-level question number (do not \
renumber). For marks: if a single value is printed for the whole item, use \
that; if separate values are printed per lettered sub-part, sum them into \
one total. For word_limit: use the value printed for that item if a single \
one applies; if sub-parts have different word limits, or none is stated, \
use 0."""

GS4_SCHEMA = MAINS_SCHEMA


def render_page_png(pdf_path: Path, page_index: int, max_dim: int = MAX_IMAGE_DIM) -> bytes:
    doc = pymupdf.open(pdf_path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(dpi=300)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        im.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()


def page_count(pdf_path: Path) -> int:
    doc = pymupdf.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def extract_page(
    client: genai.Client, pdf_path: Path, page_index: int, exam_type: str, paper: str, max_retries: int = 5
) -> list[dict]:
    if paper == "Essay":
        prompt, schema = ESSAY_PROMPT, ESSAY_SCHEMA
    elif exam_type == "prelims":
        prompt, schema = PRELIMS_PROMPT, PRELIMS_SCHEMA
    else:
        prompt, schema = MAINS_PROMPT, MAINS_SCHEMA
    image_bytes = render_page_png(pdf_path, page_index)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            data = json.loads(resp.text)
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1, 2, 4, 8s backoff
    else:
        raise RuntimeError(f"page {page_index} failed after {max_retries} attempts") from last_error

    if paper == "Essay":
        # Normalize topics -> the same question-shaped dicts the rest of the
        # pipeline expects, with a placeholder number (real numbering is
        # assigned in process_document, since section-local numbers on the
        # page aren't unique within the document).
        questions = [{"question_number": 0, "question_text": t} for t in data.get("topics", [])]
    else:
        questions = data.get("questions", [])
    for q in questions:
        q["_page_index"] = page_index
    return questions


def extract_gs4_document(client: genai.Client, pdf_path: Path, max_retries: int = 5) -> list[dict]:
    n_pages = page_count(pdf_path)
    parts = [
        types.Part.from_bytes(data=render_page_png(pdf_path, i), mime_type="image/png")
        for i in range(n_pages)
    ]

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[*parts, GS4_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GS4_SCHEMA,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            data = json.loads(resp.text)
            return data.get("questions", [])
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"GS4 whole-document extraction failed after {max_retries} attempts") from last_error


def process_document(client: genai.Client, pdf_path: Path, exam_type: str, year: int, paper: str) -> list[dict]:
    all_questions: list[dict] = []
    failed_pages: list[int] = []

    if paper == "GS4":
        # Whole-document call, not per-page — case studies span page
        # boundaries in both languages, so per-page extraction fragments
        # them (see extract_gs4_document / PLAN.md §3). Numbers are printed
        # correctly and uniquely here, so no dedup/renumbering needed below.
        try:
            all_questions = extract_gs4_document(client, pdf_path)
        except Exception as e:
            print(f"    whole-document extraction FAILED (after retries): {e}")
            failed_pages.append(-1)  # sentinel: whole doc, not a specific page
        numbered = {q["question_number"]: q for q in all_questions}
    else:
        n_pages = page_count(pdf_path)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(extract_page, client, pdf_path, i, exam_type, paper): i
                for i in range(n_pages)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    all_questions.extend(fut.result())
                except Exception as e:
                    print(f"    page {i + 1}/{n_pages} FAILED (after retries): {e}")
                    failed_pages.append(i)
        numbered = _merge_questions(all_questions, paper)

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    records = []
    for n, q in sorted(numbered.items()):
        record = {
            "question_id": f"UPSC-{exam_type}-{year}-{paper}-{n}",
            "exam_board": "UPSC",
            "exam_type": exam_type,
            "is_objective": 1 if exam_type == "prelims" else 0,
            "paper": paper,
            "year": year,
            "question_number": n,
            "question_text": q["question_text"],
            "source_doc_url": str(pdf_path),
            "extraction_method": "gemini-2.5-flash",
            "extracted_at": now,
        }
        if exam_type == "prelims":
            record["options"] = q.get("options", [])
            record["correct_option"] = None
        elif paper == "Essay":
            record["marks"] = ESSAY_MARKS
            record["word_limit"] = ESSAY_WORD_LIMIT
        else:
            record["marks"] = q.get("marks") or None
            record["word_limit"] = q.get("word_limit") or None
        records.append(record)
    return records, failed_pages


def _merge_questions(all_questions: list[dict], paper: str) -> dict[int, dict]:
    if paper == "Essay":
        # Essay papers print two independent sections ("Section A", "Section
        # B") that BOTH restart numbering at 1 — so the printed
        # question_number isn't unique within the document and can't be used
        # as a dedup key (it collided and silently dropped half the topics
        # for at least one year before this was caught). Each topic appears
        # exactly once per page-call here, so just keep everything and
        # renumber sequentially in extraction order.
        ordered = sorted(all_questions, key=lambda q: (q["_page_index"], q["question_number"]))
        numbered = {i + 1: q for i, q in enumerate(ordered)}
    else:
        # Dedupe by question_number: a question occasionally gets picked up
        # on two adjacent pages (e.g. options spilling onto the next page).
        # Keep whichever extraction has the longer question_text + more
        # options, on the assumption that's the more complete one.
        numbered = {}
        for q in all_questions:
            n = q["question_number"]
            if n not in numbered:
                numbered[n] = q
                continue
            existing = numbered[n]
            existing_len = len(existing["question_text"]) + len(str(existing.get("options", "")))
            new_len = len(q["question_text"]) + len(str(q.get("options", "")))
            if new_len > existing_len:
                numbered[n] = q

    return numbered


def run(year_filter: int | None = None, limit: int | None = None) -> None:
    client = genai.Client(vertexai=True, project=config.GCP_PROJECT, location=config.GCP_LOCATION)
    out_dir = config.DATA_DIR / "extracted"

    jobs = []
    for exam_type in ("prelims", "mains"):
        exam_dir = config.RAW_PDF_DIR / exam_type
        for year_dir in sorted(exam_dir.iterdir()):
            year = int(year_dir.name)
            if year_filter and year != year_filter:
                continue
            for pdf_path in sorted(year_dir.glob("*.pdf")):
                paper = pdf_path.stem  # GS1, GS2, GS3, GS4, Essay
                jobs.append((exam_type, year, paper, pdf_path))

    if limit:
        jobs = jobs[:limit]

    for exam_type, year, paper, pdf_path in jobs:
        dest = out_dir / exam_type / str(year) / f"{paper}.json"
        if dest.exists():
            print(f"[{year} {exam_type} {paper}] skipped (already extracted)")
            continue
        print(f"[{year} {exam_type} {paper}] extracting from {pdf_path.name} ({page_count(pdf_path)} pages)...")
        records, failed_pages = process_document(client, pdf_path, exam_type, year, paper)

        if failed_pages:
            print(
                f"[{year} {exam_type} {paper}] NOT SAVED — {len(failed_pages)} page(s) still failed "
                f"after retries ({[p + 1 for p in failed_pages]}); re-run this script to retry this document."
            )
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        flag = " *** LOW COUNT, CHECK MANUALLY ***" if is_suspicious(exam_type, paper, len(records)) else ""
        print(f"[{year} {exam_type} {paper}] -> {len(records)} questions -> {dest.relative_to(config.DATA_DIR)}{flag}")


def is_suspicious(exam_type: str, paper: str, count: int) -> bool:
    if exam_type == "prelims":
        return count < (90 if paper == "GS1" else 70)
    if paper == "Essay":
        return count < 6
    if paper == "GS4":
        return count < 8
    return count < 18  # GS1-GS3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N documents (smoke test)")
    args = parser.parse_args()
    run(year_filter=args.year, limit=args.limit)
