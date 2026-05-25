"""RAG enrichment orchestrator.

Reads bengaluru_data _gemini.md files from GCS, chunks page-wise,
extracts contextual metadata via Gemini, uploads results back to GCS.

Usage:
    python -m rag_enrichment.orchestrator --source jll --limit 1
    python -m rag_enrichment.orchestrator --source jll --category bengaluru_data --resume
    python -m rag_enrichment.orchestrator --source cushman --dry-run
"""

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

from gemini_ocr.config import GCS_BUCKET
from gemini_ocr.gcs_client import _client, download_blob, upload_file
from rag_enrichment.chunker import split_into_pages
from rag_enrichment.processor import process_report


def _list_gemini_mds(source: str, category: str) -> list[dict]:
    """List _gemini.md blobs in extracted_pdfs/<category>/."""
    client = _client()
    prefix = f"{source}/extracted_pdfs/{category}/"
    results = []
    for blob in client.list_blobs(GCS_BUCKET, prefix=prefix):
        if blob.name.endswith("_gemini.md"):
            stem = blob.name.split("/")[-1].replace("_gemini.md", "")
            results.append({
                "stem": stem,
                "category": category,
                "blob_name": blob.name,
            })
    return results


def _result_exists(source: str, category: str, stem: str) -> bool:
    """Check if _rag_metadata.json already exists."""
    blob_name = f"{source}/extracted_pdfs/{category}/{stem}_rag_metadata.json"
    client = _client()
    blobs = list(client.list_blobs(GCS_BUCKET, prefix=blob_name, max_results=1))
    return len(blobs) > 0


def _process_one(source: str, entry: dict) -> dict:
    """Download md → chunk → process → upload results."""
    stem = entry["stem"]
    category = entry["category"]

    with tempfile.TemporaryDirectory(prefix="rag_") as td:
        tmp = Path(td)

        local_md = download_blob(entry["blob_name"], tmp / f"{stem}_gemini.md")
        markdown = local_md.read_text(encoding="utf-8")
        print(f"  [{stem}] downloaded ({len(markdown):,} chars)")

        chunks = split_into_pages(markdown)
        print(f"  [{stem}] {len(chunks)} pages")

        if not chunks:
            print(f"  [{stem}] no pages found, skipping")
            return {"stem": stem, "pages": 0, "records": 0}

        records = process_report(stem, markdown, chunks)
        print(f"  [{stem}] {len(records)} records extracted")

        if records:
            json_path = tmp / f"{stem}_rag_metadata.json"
            json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            json_blob = f"{source}/extracted_pdfs/{category}/{stem}_rag_metadata.json"
            upload_file(json_path, json_blob, content_type="application/json")

            xlsx_path = tmp / f"{stem}_rag_metadata.xlsx"
            df = pd.DataFrame(records)
            df.to_excel(xlsx_path, index=False, engine="openpyxl")
            xlsx_blob = f"{source}/extracted_pdfs/{category}/{stem}_rag_metadata.xlsx"
            upload_file(
                xlsx_path, xlsx_blob,
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            print(f"  [{stem}] uploaded → {category}/")

    return {"stem": stem, "pages": len(chunks), "records": len(records)}


def main():
    ap = argparse.ArgumentParser(description="RAG enrichment: chunk + metadata extraction")
    ap.add_argument("--source", required=True, help="GCS source folder (e.g. jll)")
    ap.add_argument("--category", default="bengaluru_data",
                    choices=["bengaluru_data", "india_data"])
    ap.add_argument("--limit", type=int, help="Process only first N reports")
    ap.add_argument("--resume", action="store_true",
                    help="Skip reports that already have _rag_metadata.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entries = _list_gemini_mds(args.source, args.category)
    print(f"Found {len(entries)} _gemini.md files in {args.source}/{args.category}")

    if args.resume:
        before = len(entries)
        entries = [
            e for e in entries
            if not _result_exists(args.source, args.category, e["stem"])
        ]
        print(f"Resume: {before - len(entries)} already done, {len(entries)} remaining")

    if args.limit and args.limit < len(entries):
        entries = entries[:args.limit]
        print(f"Limited to first {args.limit}")

    if not entries:
        print("Nothing to do.")
        return

    if args.dry_run:
        print(f"\nDry run — {len(entries)} report(s):")
        for e in entries:
            print(f"  {e['stem']}")
        return

    print(f"\nProcessing {len(entries)} report(s)\n")

    grand = {"reports": 0, "pages": 0, "records": 0}

    for entry in entries:
        print(f"\n=== {entry['stem']} ===")
        try:
            stats = _process_one(args.source, entry)
            grand["reports"] += 1
            grand["pages"] += stats["pages"]
            grand["records"] += stats["records"]
        except Exception:
            print(f"  FAILED {entry['stem']}:", file=sys.stderr)
            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print("RAG ENRICHMENT SUMMARY")
    print(f"  Reports processed : {grand['reports']}/{len(entries)}")
    print(f"  Total pages       : {grand['pages']}")
    print(f"  Total records     : {grand['records']}")


if __name__ == "__main__":
    main()
