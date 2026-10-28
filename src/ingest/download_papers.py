"""Download official UPSC question papers listed in data/manifest.json.

The manifest was built by scraping upsc.gov.in's "Previous Question Papers"
pages (current + archives) via browser automation, since those page routes
sit behind bot-detection that blocks plain HTTP clients — see PLAN.md §2.
The actual PDF files, however, are static assets under
upsc.gov.in/sites/default/files/ and download fine with plain `requests`.

Usage:
    python -m src.ingest.download_papers
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from .. import config

MANIFEST_PATH = config.DATA_DIR / "manifest.json"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Roman-numeral -> arabic, for normalizing "General Studies Paper - III" etc.
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


def canonical_paper(exam_type: str, label: str) -> str:
    """Map a raw label like 'General Studies - III' or 'Essay (Compulsory)'
    to a canonical paper code: GS1/GS2/GS3/GS4/Essay."""
    if re.search(r"essay", label, re.IGNORECASE):
        return "Essay"
    m = re.search(r"\b(I|II|III|IV)\b\s*$", label.strip())
    if m:
        return f"GS{_ROMAN[m.group(1)]}"
    raise ValueError(f"Could not canonicalize label {label!r} for {exam_type}")


def local_path(entry: dict) -> Path:
    paper = canonical_paper(entry["examType"], entry["label"])
    return config.RAW_PDF_DIR / entry["examType"] / str(entry["year"]) / f"{paper}.pdf"


def download_one(entry: dict, session: requests.Session) -> tuple[Path, str]:
    dest = local_path(entry)
    if dest.exists() and dest.stat().st_size > 0:
        return dest, "skipped (already present)"

    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = session.get(entry["href"], timeout=60)
    resp.raise_for_status()

    if not resp.content.startswith(b"%PDF"):
        raise ValueError(
            f"Response for {entry['href']} is not a PDF (likely a bot-detection "
            f"soft-404) — got {len(resp.content)} bytes, content-type="
            f"{resp.headers.get('content-type')}"
        )

    dest.write_bytes(resp.content)
    return dest, f"downloaded ({len(resp.content):,} bytes)"


def run() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest = [e for e in manifest if config.YEAR_START <= e["year"] <= config.YEAR_END]
    manifest.sort(key=lambda e: (e["examType"], e["year"], e["label"]))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    ok, failed = 0, []
    for entry in manifest:
        try:
            dest, status = download_one(entry, session)
            print(f"[{entry['year']} {entry['examType']:7s}] {dest.relative_to(config.DATA_DIR)}: {status}")
            ok += 1
        except Exception as e:
            print(f"[{entry['year']} {entry['examType']:7s}] FAILED — {entry['href']}: {e}")
            failed.append(entry)
        time.sleep(0.5)  # be polite to the server

    print(f"\n{ok}/{len(manifest)} papers OK, {len(failed)} failed.")
    if failed:
        for e in failed:
            print(f"  - {e['year']} {e['examType']} {e['label']}: {e['href']}")


if __name__ == "__main__":
    run()
