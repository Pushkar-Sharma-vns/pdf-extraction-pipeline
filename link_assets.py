"""Link classified images to RAG metadata records by matching PDF stem + page number.

For each bengaluru_data PDF that has both _rag_metadata.json and classification_results.json:
  1. Parse page numbers from image filenames (page<N>_<label>_idx<I>.jpg)
  2. For each metadata record, find all RELEVANT images on that page
  3. Replace linked_assets with actual GCS URIs + classification data
  4. Re-upload updated _rag_metadata.json and _rag_metadata.xlsx

Usage:
    python link_assets.py --source jll
    python link_assets.py --source cushman --dry-run
    python link_assets.py --source jll --limit 1
"""

import argparse
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from gemini_ocr.config import GCS_BUCKET
from gemini_ocr.gcs_client import _client, upload_file

IMAGE_NAME_RE = re.compile(r"page(\d+)_([^_]+)_idx(\d+)\.jpg")


def _list_linkable_pdfs(source: str, category: str) -> list[dict]:
    """Find PDFs that have BOTH _rag_metadata.json AND classification_results.json."""
    client = _client()
    prefix = f"{source}/extracted_pdfs/{category}/"

    # Find all _rag_metadata.json stems
    meta_stems = {}
    for blob in client.list_blobs(GCS_BUCKET, prefix=prefix):
        if blob.name.endswith("_rag_metadata.json"):
            stem = blob.name.split("/")[-1].replace("_rag_metadata.json", "")
            meta_stems[stem] = blob.name

    # Check which have classification_results.json
    results = []
    for stem, meta_blob in sorted(meta_stems.items()):
        cls_blob = f"{prefix}{stem}_extracted_images/classification_results.json"
        blobs = list(client.list_blobs(GCS_BUCKET, prefix=cls_blob, max_results=1))
        if blobs:
            results.append({
                "stem": stem,
                "meta_blob": meta_blob,
                "classify_blob": cls_blob,
                "category": category,
            })

    return results


def _load_json_blob(blob_name: str) -> list:
    client = _client()
    return json.loads(client.bucket(GCS_BUCKET).blob(blob_name).download_as_text())


def _parse_image_page(filename: str) -> tuple[int, str, int] | None:
    m = IMAGE_NAME_RE.match(filename)
    if m:
        return int(m.group(1)), m.group(2), int(m.group(3))
    return None


def _build_page_image_map(classify_results: list[dict]) -> dict[int, list[dict]]:
    """Group relevant images by page number."""
    by_page = defaultdict(list)
    for img in classify_results:
        if not img.get("is_relevant"):
            continue
        uri = img.get("gcs_uri", "")
        fname = uri.split("/")[-1]
        parsed = _parse_image_page(fname)
        if parsed:
            page_num, label, idx = parsed
            by_page[page_num].append({
                "image_url": uri,
                "asset_type": img.get("category", "other"),
                "caption": img.get("reasoning", ""),
                "summary": img.get("reasoning", ""),
            })
    return dict(by_page)


def _process_one(source: str, entry: dict) -> dict:
    stem = entry["stem"]
    category = entry["category"]

    meta_records = _load_json_blob(entry["meta_blob"])
    classify_results = _load_json_blob(entry["classify_blob"])

    page_images = _build_page_image_map(classify_results)

    updated = 0
    for record in meta_records:
        page = record.get("page_number")
        images = page_images.get(page, [])
        if images:
            record["linked_assets_json"] = json.dumps(images)
            updated += 1
        elif record.get("linked_assets_json") in ("[]", "", None):
            pass  # leave as-is

    # Re-upload
    with tempfile.TemporaryDirectory(prefix="link_") as td:
        tmp = Path(td)

        json_path = tmp / f"{stem}_rag_metadata.json"
        json_path.write_text(json.dumps(meta_records, indent=2), encoding="utf-8")
        json_blob = f"{source}/extracted_pdfs/{category}/{stem}_rag_metadata.json"
        upload_file(json_path, json_blob, content_type="application/json")

        xlsx_path = tmp / f"{stem}_rag_metadata.xlsx"
        df = pd.DataFrame(meta_records)
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        xlsx_blob = f"{source}/extracted_pdfs/{category}/{stem}_rag_metadata.xlsx"
        upload_file(xlsx_path, xlsx_blob,
                    content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    total_images = sum(len(v) for v in page_images.values())
    return {"stem": stem, "pages_linked": updated, "images_linked": total_images,
            "total_pages": len(meta_records)}


def main():
    ap = argparse.ArgumentParser(description="Link classified images to RAG metadata")
    ap.add_argument("--source", required=True)
    ap.add_argument("--category", default="bengaluru_data",
                    choices=["bengaluru_data", "india_data"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entries = _list_linkable_pdfs(args.source, args.category)
    print(f"Found {len(entries)} PDFs with both metadata + classification in {args.source}/{args.category}")

    if args.limit and args.limit < len(entries):
        entries = entries[:args.limit]
        print(f"Limited to {args.limit}")

    if args.dry_run:
        for e in entries:
            print(f"  {e['stem']}")
        return

    if not entries:
        print("Nothing to link.")
        return

    t0 = time.time()
    grand = {"pdfs": 0, "pages_linked": 0, "images_linked": 0}
    log_lines = []

    for entry in entries:
        print(f"\n=== {entry['stem']} ===")
        try:
            stats = _process_one(args.source, entry)
            print(f"  {stats['pages_linked']}/{stats['total_pages']} pages linked, "
                  f"{stats['images_linked']} images attached")
            grand["pdfs"] += 1
            grand["pages_linked"] += stats["pages_linked"]
            grand["images_linked"] += stats["images_linked"]
            log_lines.append(f"  OK  {stats['stem'][:55]:55s}  "
                             f"pages:{stats['pages_linked']}/{stats['total_pages']}  "
                             f"imgs:{stats['images_linked']}")
        except Exception as e:
            print(f"  FAILED — {e}")
            log_lines.append(f"  FAIL  {entry['stem'][:55]:55s}  {str(e)[:60]}")

    elapsed = time.time() - t0

    log_path = Path(f"link_assets_log_{args.source}_{args.category}.txt")
    log_content = [
        f"Link Assets Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source: {args.source}  Category: {args.category}",
        f"Total: {grand['pdfs']} PDFs, {grand['pages_linked']} pages linked, "
        f"{grand['images_linked']} images attached, {elapsed:.0f}s",
        "",
    ] + log_lines + [""]
    log_path.write_text("\n".join(log_content))

    print(f"\n{'=' * 50}")
    print("LINK ASSETS SUMMARY")
    print(f"  PDFs processed  : {grand['pdfs']}")
    print(f"  Pages linked    : {grand['pages_linked']}")
    print(f"  Images attached : {grand['images_linked']}")
    print(f"  Time            : {elapsed:.0f}s")
    print(f"  Log             : {log_path}")


if __name__ == "__main__":
    main()
