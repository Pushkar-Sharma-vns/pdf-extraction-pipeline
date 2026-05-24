"""GCS → Gemini extraction → GCS pipeline.

Output routing:
  If any page mentions Bengaluru / Bangalore / Karnataka →
      <source>/extracted_pdfs/bengaluru_data/<stem>_gemini.md
  Otherwise →
      <source>/extracted_pdfs/india_data/<stem>_gemini.md

Usage:
    python -m gemini_ocr.pipeline                                  # all sources, sequential
    python -m gemini_ocr.pipeline --source jll --workers 3         # parallel (3 PDFs at once)
    python -m gemini_ocr.pipeline --blob jll/pdfs/x.pdf            # one PDF
    python -m gemini_ocr.pipeline --dry-run                        # list only
    python -m gemini_ocr.pipeline --classify-only                  # classify without extracting
"""

import argparse
import re
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # PyMuPDF

from gemini_ocr.config import GCS_BUCKET
from gemini_ocr.gcs_client import list_sources, list_pdfs, list_existing_outputs, download_blob, upload_file
from gemini_ocr.extractor import extract_pdf

BENGALURU_RE = re.compile(
    r"bengaluru|bangalore|karnataka",
    re.IGNORECASE,
)

_print_lock = threading.Lock()


def _tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def classify_pdf(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    try:
        for page in doc:
            text = page.get_text()
            if BENGALURU_RE.search(text):
                return "bengaluru_data"
    finally:
        doc.close()
    return "india_data"


def _out_blob_name(pdf_blob: str, category: str) -> str:
    source = pdf_blob.split("/")[0]
    stem = Path(pdf_blob).stem
    return f"{source}/extracted_pdfs/{category}/{stem}_gemini.md"


def process_blob(pdf_blob: str) -> dict:
    stem = Path(pdf_blob).stem
    with tempfile.TemporaryDirectory(prefix="gemini_gcs_") as td:
        tmp = Path(td)
        local_pdf = download_blob(pdf_blob, tmp / f"{stem}.pdf")
        _tprint(f"  [{stem}] downloaded")

        category = classify_pdf(local_pdf)
        _tprint(f"  [{stem}] classified: {category}")

        local_md = tmp / f"{stem}_gemini.md"
        stats = extract_pdf(local_pdf, local_md)

        out_blob = _out_blob_name(pdf_blob, category)
        gs_uri = upload_file(local_md, out_blob)
        _tprint(f"  [{stem}] uploaded → {category}/")
        stats["output_blob"] = out_blob
        stats["category"] = category

    return stats


def _run_sequential(pdf_blobs: list[str]) -> dict:
    grand = _empty_grand()
    for blob in pdf_blobs:
        print(f"\n=== {blob} ===")
        try:
            stats = process_blob(blob)
            _accumulate(grand, stats)
        except Exception:
            print(f"  FAILED {blob}:", file=sys.stderr)
            traceback.print_exc()
    return grand


def _run_parallel(pdf_blobs: list[str], workers: int) -> dict:
    grand = _empty_grand()
    print(f"Running with {workers} parallel workers\n")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_blob, blob): blob for blob in pdf_blobs}
        for future in as_completed(futures):
            blob = futures[future]
            try:
                stats = future.result()
                _accumulate(grand, stats)
                _tprint(f"  DONE  {blob}  ({stats['category']})")
            except Exception:
                _tprint(f"  FAILED {blob}:", file=sys.stderr)
                traceback.print_exc()
    return grand


def _empty_grand() -> dict:
    return {
        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        "chunks": 0, "pdfs": 0,
        "bengaluru_data": 0, "india_data": 0,
    }


def _accumulate(grand: dict, stats: dict):
    for k in ("input_tokens", "output_tokens", "cost_usd", "chunks"):
        grand[k] += stats[k]
    grand["pdfs"] += 1
    grand[stats["category"]] += 1


def main():
    ap = argparse.ArgumentParser(description="GCS → Gemini extraction → GCS")
    ap.add_argument("--source", help="Process only this source folder (e.g. knight_frank)")
    ap.add_argument("--blob", help="Process a single PDF blob (e.g. knight_frank/pdfs/report.pdf)")
    ap.add_argument("--dry-run", action="store_true", help="List PDFs without processing")
    ap.add_argument("--classify-only", action="store_true",
                    help="Download and classify PDFs without running Gemini extraction")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel PDF workers (default: 1 = sequential)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip PDFs whose output already exists in GCS")
    args = ap.parse_args()

    if args.blob:
        pdf_blobs = [args.blob]
    else:
        sources = [args.source] if args.source else list_sources()
        if not sources:
            sys.exit(f"No source folders found in gs://{GCS_BUCKET}/")
        print(f"Sources: {sources}")

        pdf_blobs = []
        for s in sources:
            found = list_pdfs(s)
            print(f"  {s}/pdfs/: {len(found)} PDF(s)")
            pdf_blobs.extend(found)

    if not pdf_blobs:
        sys.exit("No PDFs found.")

    if args.dry_run:
        print(f"\nDry run — {len(pdf_blobs)} PDF(s) would be processed:")
        for b in pdf_blobs:
            print(f"  {b}")
        return

    if args.classify_only:
        print(f"\nClassify-only — downloading and checking {len(pdf_blobs)} PDF(s)\n")
        counts = {"bengaluru_data": 0, "india_data": 0}
        for blob in pdf_blobs:
            stem = Path(blob).stem
            with tempfile.TemporaryDirectory() as td:
                local = download_blob(blob, Path(td) / f"{stem}.pdf")
                cat = classify_pdf(local)
            counts[cat] += 1
            print(f"  {cat:16s}  {blob}")
        print(f"\nTotals: {counts}")
        return

    if args.resume:
        sources_in_play = {b.split("/")[0] for b in pdf_blobs}
        existing = set()
        for s in sources_in_play:
            existing |= list_existing_outputs(s)
        before = len(pdf_blobs)
        pdf_blobs = [
            b for b in pdf_blobs
            if _out_blob_name(b, "bengaluru_data") not in existing
            and _out_blob_name(b, "india_data") not in existing
        ]
        skipped = before - len(pdf_blobs)
        print(f"Resume: {skipped} already done, {len(pdf_blobs)} remaining")
        if not pdf_blobs:
            print("Nothing to do.")
            return

    print(f"\nProcessing {len(pdf_blobs)} PDF(s)\n")

    if args.workers > 1:
        grand = _run_parallel(pdf_blobs, args.workers)
    else:
        grand = _run_sequential(pdf_blobs)

    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print(f"  PDFs processed   : {grand['pdfs']}/{len(pdf_blobs)}")
    print(f"  bengaluru_data   : {grand['bengaluru_data']}")
    print(f"  india_data       : {grand['india_data']}")
    print(f"  Chunks           : {grand['chunks']}")
    print(f"  Input tokens     : {grand['input_tokens']:,}")
    print(f"  Output tokens    : {grand['output_tokens']:,}")
    print(f"  Total cost       : ${grand['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
