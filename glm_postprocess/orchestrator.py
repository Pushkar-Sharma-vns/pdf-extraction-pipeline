"""GLM-OCR post-processing orchestrator.

GCS flow (Phase 1):
  1. List _gemini.md files in <source>/extracted_pdfs/{bengaluru_data|india_data}/
     to discover which PDFs have already been Gemini-extracted
  2. For each, download the original PDF from <source>/pdfs/<pdfname>.pdf
  3. Run glmocr parse → image extraction → upload images back to the same
     category folder as <pdfname>_extracted_images/

Usage:
    # GCS mode — process a source (reads from extracted_pdfs to find PDFs)
    python -m glm_postprocess.orchestrator --source jll
    python -m glm_postprocess.orchestrator --source cushman --resume --workers 3

    # Local mode — process existing test_results/
    python -m glm_postprocess.orchestrator --local --min-size 100

    # With Phase 2 classification chained
    python -m glm_postprocess.orchestrator --source jll --classify

    # Phase 2 only (classify already-uploaded images)
    python -m glm_postprocess.orchestrator --classify-only --source jll
"""

import argparse
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from glm_postprocess.config import (
    GLMOCR_BIN, CONFIG_YAML, RESULTS_ROOT, MIN_SIZE_DEFAULT,
)
from glm_postprocess.image_extractor import find_pdf_dirs, process_pdf_dir

_print_lock = threading.Lock()


def _tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def run_glmocr(pdf_path: Path, results_root: Path) -> Path:
    stem = pdf_path.stem
    out_dir = results_root / stem
    model_json = out_dir / f"{stem}_model.json"

    if model_json.exists():
        return model_json

    cmd = [GLMOCR_BIN, "parse", str(pdf_path), "--output", str(results_root)]
    if CONFIG_YAML.exists():
        cmd += ["--config", str(CONFIG_YAML)]

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0

    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"glmocr failed (exit {res.returncode}) for {pdf_path.name}")

    _tprint(f"    glmocr: done in {dt:.1f}s")

    if not model_json.exists():
        raise FileNotFoundError(f"Expected {model_json} not found after glmocr run")
    return model_json


# ── GCS helpers ──────────────────────────────────────────────────────

def _list_extracted_pdfs(source: str) -> list[dict]:
    """List PDFs that have _gemini.md files in extracted_pdfs/.

    Returns list of dicts: {stem, category, pdf_blob, gemini_blob}
    """
    from gemini_ocr.gcs_client import _client, GCS_BUCKET

    client = _client()
    results = []
    for category in ("bengaluru_data", "india_data"):
        prefix = f"{source}/extracted_pdfs/{category}/"
        for blob in client.list_blobs(GCS_BUCKET, prefix=prefix):
            if blob.name.endswith("_gemini.md"):
                stem = blob.name.split("/")[-1].replace("_gemini.md", "")
                results.append({
                    "stem": stem,
                    "category": category,
                    "pdf_blob": f"{source}/pdfs/{stem}.pdf",
                    "gemini_blob": blob.name,
                })
    return results


def _process_gcs_entry(entry: dict, source: str, min_size: int, results_root: Path) -> dict:
    """Download PDF → glmocr → extract images → upload to same category folder."""
    from gemini_ocr.gcs_client import download_blob
    from glm_postprocess.gcs_upload import upload_images

    stem = entry["stem"]
    category = entry["category"]
    pdf_blob = entry["pdf_blob"]
    t0 = time.time()
    timings = {}

    with tempfile.TemporaryDirectory(prefix="glm_gcs_") as td:
        tmp = Path(td)

        t1 = time.time()
        local_pdf = download_blob(pdf_blob, tmp / f"{stem}.pdf")
        timings["download"] = round(time.time() - t1, 1)
        _tprint(f"  [{stem}] downloaded ({timings['download']}s)")

        t1 = time.time()
        run_glmocr(local_pdf, results_root)
        timings["glmocr"] = round(time.time() - t1, 1)
        _tprint(f"  [{stem}] glmocr complete ({timings['glmocr']}s)")

        t1 = time.time()
        pdf_dir = results_root / stem
        stats = process_pdf_dir(pdf_dir, min_size=min_size)
        timings["extract"] = round(time.time() - t1, 1)
        _tprint(f"  [{stem}] extracted {stats['saved']} images ({timings['extract']}s)")

        if stats["saved"] > 0:
            t1 = time.time()
            uploaded = upload_images(stats["out_dir"], source, category, stem)
            timings["upload"] = round(time.time() - t1, 1)
            _tprint(f"  [{stem}] uploaded {uploaded} images → {category}/ ({timings['upload']}s)")
            stats["uploaded"] = uploaded
        else:
            stats["uploaded"] = 0

        stats["category"] = category
        stats["stem"] = stem
        stats["timings"] = timings
        stats["total_seconds"] = round(time.time() - t0, 1)

    return stats


# ── Run modes ────────────────────────────────────────────────────────

def _run_classify_only(args):
    from glm_postprocess.classifier import (
        list_image_folders, classify_folder, upload_results, folder_has_results,
    )

    if not args.source:
        sys.exit("--classify-only requires --source")

    folders = list_image_folders(args.source)
    print(f"Found {len(folders)} image folder(s) for '{args.source}'")

    if args.resume:
        before = len(folders)
        folders = [f for f in folders if not folder_has_results(f)]
        print(f"Resume: {before - len(folders)} already classified, {len(folders)} remaining")

    if args.dry_run:
        print(f"\nDry run — {len(folders)} folder(s)")
        for f in folders:
            print(f"  {f}")
        return

    if not folders:
        print("Nothing to do.")
        return

    print(f"\nClassifying {len(folders)} folder(s) with {args.classify_workers} async workers\n")
    for folder in folders:
        print(f"\n=== {folder.split('/')[-1]} ===")
        results = classify_folder(folder, workers=args.classify_workers)
        if results:
            uri = upload_results(folder, results)
            relevant = sum(1 for r in results if r.get("is_relevant"))
            print(f"  {len(results)} classified: {relevant} relevant, "
                  f"{len(results) - relevant} not relevant → {uri}")


def _run_local(args):
    pdf_dirs = find_pdf_dirs(args.results_root)
    if not pdf_dirs:
        sys.exit(f"No valid PDF folders found in {args.results_root}")

    print(f"Found {len(pdf_dirs)} PDF folder(s) (min_size={args.min_size})\n")
    grand = {"saved": 0, "skipped_small": 0, "pdfs": 0}

    for d in pdf_dirs:
        print(f"\n=== {d.name} ===")
        stats = _process_local(d, args.min_size)
        print(f"  saved: {stats['saved']}  by_label: {stats['by_label']}")
        if stats["skipped_small"]:
            print(f"  filtered {stats['skipped_small']} crops < {args.min_size}px")
        print(f"  → {stats['out_dir']}")
        grand["saved"] += stats["saved"]
        grand["skipped_small"] += stats["skipped_small"]
        grand["pdfs"] += 1

    print(f"\n{'=' * 50}")
    print(f"LOCAL SUMMARY: {grand['pdfs']} PDFs, {grand['saved']} images extracted")


def _process_local(pdf_dir: Path, min_size: int) -> dict:
    return process_pdf_dir(pdf_dir, min_size=min_size)


def _run_gcs(args):
    from glm_postprocess.gcs_upload import images_already_uploaded

    if not args.source:
        sys.exit("--source is required for GCS mode")

    source = args.source

    entries = _list_extracted_pdfs(source)

    if args.category:
        entries = [e for e in entries if e["category"] == args.category]

    print(f"Found {len(entries)} Gemini-extracted PDFs for '{source}'")
    for cat in ("bengaluru_data", "india_data"):
        n = sum(1 for e in entries if e["category"] == cat)
        if n:
            print(f"  {cat}: {n}")

    if not entries:
        sys.exit("No extracted PDFs found. Run gemini_ocr pipeline first.")

    if args.resume:
        before = len(entries)
        entries = [
            e for e in entries
            if not images_already_uploaded(source, e["stem"])
        ]
        print(f"Resume: {before - len(entries)} already done, {len(entries)} remaining")
        if not entries:
            print("Nothing to do.")
            return

    if args.limit and args.limit < len(entries):
        entries = entries[:args.limit]
        print(f"Limited to first {args.limit} PDF(s)")

    if args.dry_run:
        print(f"\nDry run — {len(entries)} PDF(s):")
        for e in entries:
            print(f"  {e['category']:16s}  {e['stem']}")
        return

    print(f"\nProcessing {len(entries)} PDF(s)  (min_size={args.min_size})\n")

    from datetime import datetime

    grand = {"saved": 0, "uploaded": 0, "pdfs": 0, "bengaluru_data": 0, "india_data": 0}
    pdf_logs = []
    t_batch_start = time.time()

    def _do(entry):
        return entry, _process_gcs_entry(entry, source, args.min_size, args.results_root)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_do, e): e for e in entries}
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    _, stats = future.result()
                    grand["saved"] += stats["saved"]
                    grand["uploaded"] += stats["uploaded"]
                    grand["pdfs"] += 1
                    if stats["category"] in grand:
                        grand[stats["category"]] += 1
                    pdf_logs.append(stats)
                except Exception as e:
                    _tprint(f"  FAILED {entry['stem']}:", file=sys.stderr)
                    traceback.print_exc()
                    pdf_logs.append({"stem": entry["stem"], "saved": 0, "uploaded": 0,
                                     "total_seconds": 0, "timings": {}, "error": str(e)})
    else:
        for entry in entries:
            print(f"\n=== {entry['stem']} ({entry['category']}) ===")
            try:
                _, stats = _do(entry)
                grand["saved"] += stats["saved"]
                grand["uploaded"] += stats["uploaded"]
                grand["pdfs"] += 1
                if stats["category"] in grand:
                    grand[stats["category"]] += 1
                pdf_logs.append(stats)
            except Exception as e:
                print(f"  FAILED {entry['stem']}:", file=sys.stderr)
                traceback.print_exc()
                pdf_logs.append({"stem": entry["stem"], "saved": 0, "uploaded": 0,
                                 "total_seconds": 0, "timings": {}, "error": str(e)})

    # Phase 2: classify if requested
    if args.classify and grand["uploaded"] > 0:
        from glm_postprocess.classifier import (
            classify_folder, upload_results, list_images_in_folder,
        )
        print(f"\n{'=' * 50}")
        print("PHASE 2: Classifying uploaded images...\n")
        for entry in entries:
            folder = f"{source}/extracted_pdfs/{entry['category']}/{entry['stem']}_extracted_images"
            if list_images_in_folder(folder):
                out = classify_folder(folder, workers=args.classify_workers)
                results = out["results"] if isinstance(out, dict) else out
                if results:
                    upload_results(folder, results)
                    relevant = sum(1 for r in results if r.get("is_relevant"))
                    classify_time = out.get("elapsed_seconds", 0) if isinstance(out, dict) else 0
                    _tprint(f"  [{entry['stem']}] {len(results)} classified: "
                            f"{relevant} relevant ({classify_time}s)")
                    for pl in pdf_logs:
                        if pl.get("stem") == entry["stem"]:
                            pl["classify_images"] = len(results)
                            pl["classify_relevant"] = relevant
                            pl["classify_seconds"] = classify_time

    total_elapsed = time.time() - t_batch_start

    # Write log file
    log_dir = Path("glm_postprocess/output")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"orchestrator_log_{source}_{args.category or 'all'}.txt"
    log_lines = [
        f"GLM Postprocess Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source: {source}  Category: {args.category or 'all'}",
        f"Total: {grand['pdfs']}/{len(entries)} PDFs, {grand['saved']} images extracted, "
        f"{grand['uploaded']} uploaded, {total_elapsed:.0f}s",
        "",
    ]
    for pl in pdf_logs:
        status = "FAIL" if pl.get("error") else "OK"
        t = pl.get("timings", {})
        line = (f"  {status:4s}  {pl.get('stem','?')[:55]:55s}  "
                f"imgs:{pl.get('saved',0):3d}  up:{pl.get('uploaded',0):3d}  "
                f"{pl.get('total_seconds',0)}s")
        if t:
            line += f"  [dl:{t.get('download',0)}s ocr:{t.get('glmocr',0)}s ext:{t.get('extract',0)}s up:{t.get('upload',0)}s]"
        if pl.get("classify_images"):
            line += f"  cls:{pl['classify_relevant']}/{pl['classify_images']} {pl.get('classify_seconds',0)}s"
        log_lines.append(line)
        if pl.get("error"):
            log_lines.append(f"         ERROR: {pl['error'][:100]}")
    log_lines.append("")
    log_path.write_text("\n".join(log_lines))
    print(f"\nLog written → {log_path}")

    print(f"\n{'=' * 50}")
    print("GCS PIPELINE SUMMARY")
    print(f"  PDFs processed   : {grand['pdfs']}/{len(entries)}")
    print(f"  Images extracted : {grand['saved']}")
    print(f"  Images uploaded  : {grand['uploaded']}")
    print(f"  bengaluru_data   : {grand['bengaluru_data']}")
    print(f"  india_data       : {grand['india_data']}")
    print(f"  Total time       : {total_elapsed:.0f}s")


def main():
    ap = argparse.ArgumentParser(
        description="GLM-OCR post-processing: glmocr → extract images → GCS upload"
    )
    ap.add_argument("--local", action="store_true",
                    help="Local mode: process existing test_results/ folders (no GCS)")
    ap.add_argument("--source", help="GCS source folder (e.g. cushman)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="Skip PDFs with existing images in GCS")
    ap.add_argument("--min-size", type=int, default=MIN_SIZE_DEFAULT,
                    help=f"Skip crops smaller than NxN pixels (default: {MIN_SIZE_DEFAULT})")
    ap.add_argument("--category", choices=["bengaluru_data", "india_data"],
                    help="Filter to only this category")
    ap.add_argument("--limit", type=int,
                    help="Process only the first N PDFs")
    ap.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--classify", action="store_true",
                    help="Run Phase 2 classification after extraction + upload")
    ap.add_argument("--classify-only", action="store_true",
                    help="Run Phase 2 classification only on already-uploaded images")
    ap.add_argument("--classify-workers", type=int, default=20,
                    help="Async concurrency limit for classification (default: 20)")
    args = ap.parse_args()

    if args.classify_only:
        _run_classify_only(args)
    elif args.local:
        _run_local(args)
    else:
        _run_gcs(args)


if __name__ == "__main__":
    main()
