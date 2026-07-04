"""RAG enrichment orchestrator.

Reads bengaluru_data _gemini.md files from GCS, chunks page-wise,
extracts contextual metadata via Gemini (async parallel), uploads results back.

Usage:
    python -m rag_enrichment.orchestrator --source jll --limit 1 --local-output rag_enrichment/output
    python -m rag_enrichment.orchestrator --source jll --category bengaluru_data --resume
    python -m rag_enrichment.orchestrator --source cushman --dry-run
"""

import argparse
import json
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

from gemini_ocr.config import GCS_BUCKET
from gemini_ocr.gcs_client import _client, download_blob, upload_file
from rag_enrichment.chunker import split_into_pages
from rag_enrichment.processor import process_report


def _list_gemini_mds(source: str, category: str) -> list[dict]:
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
    blob_name = f"{source}/extracted_pdfs/{category}/{stem}_rag_metadata.json"
    client = _client()
    blobs = list(client.list_blobs(GCS_BUCKET, prefix=blob_name, max_results=1))
    return len(blobs) > 0


def _process_one(source: str, entry: dict, local_output: Path | None = None) -> dict:
    stem = entry["stem"]
    category = entry["category"]

    with tempfile.TemporaryDirectory(prefix="rag_") as td:
        tmp = Path(td)

        local_md = download_blob(entry["blob_name"], tmp / f"{stem}_gemini.md")
        markdown = local_md.read_text(encoding="utf-8")
        print(f"  [{stem}] downloaded ({len(markdown):,} chars)")

        chunks = split_into_pages(markdown)
        print(f"  [{stem}] {len(chunks)} pages, processing async...")

        if not chunks:
            print(f"  [{stem}] no pages found, skipping")
            return {"stem": stem, "pages": 0, "succeeded": 0, "failed": 0,
                    "failed_pages": [], "elapsed_seconds": 0}

        stats = process_report(stem, markdown, chunks)
        records = stats["records"]
        cost = stats.get("cost", {})

        cost_str = ""
        if cost.get("total_usd"):
            cost_str = (
                f" | ${cost['total_usd']:.4f} "
                f"(in={cost['uncached_input_tokens']} "
                f"cached={cost['cached_read_tokens']} "
                f"out={cost['output_tokens']})"
            )
        print(f"  [{stem}] {stats['succeeded']}/{stats['total_pages']} OK, "
              f"{stats['failed']} failed, {stats['elapsed_seconds']}s{cost_str}")

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

            if local_output:
                local_output.mkdir(parents=True, exist_ok=True)
                local_json = local_output / f"{stem}_rag_metadata.json"
                local_xlsx = local_output / f"{stem}_rag_metadata.xlsx"
                local_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
                df.to_excel(local_xlsx, index=False, engine="openpyxl")
                print(f"  [{stem}] saved locally → {local_output}/")

    return {
        "stem": stem,
        "pages": stats["total_pages"],
        "succeeded": stats["succeeded"],
        "failed": stats["failed"],
        "failed_pages": stats["failed_pages"],
        "elapsed_seconds": stats["elapsed_seconds"],
        "cost": stats.get("cost", {}),
    }


def _write_log(log_path: Path, source: str, category: str, pdf_logs: list[dict],
               grand: dict, total_elapsed: float):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"RAG Enrichment Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source: {source}  Category: {category}",
        f"Total: {grand['reports']}/{grand['total_reports']} reports, "
        f"{grand['succeeded']} pages OK, {grand['failed']} failed, "
        f"{total_elapsed:.0f}s",
        f"Cost: ${grand['cost_usd']:.4f} "
        f"(uncached_in={grand['uncached_input_tokens']} "
        f"cached={grand['cached_read_tokens']} "
        f"out={grand['output_tokens']})",
        "",
    ]
    for p in pdf_logs:
        status = "OK" if p["failed"] == 0 else "PARTIAL"
        cost = p.get("cost", {})
        cost_str = f"  ${cost.get('total_usd', 0):.4f}" if cost.get("total_usd") else ""
        lines.append(f"  {status:8s}  {p['stem'][:60]:60s}  "
                      f"pages:{p['pages']}  ok:{p['succeeded']}  "
                      f"fail:{p['failed']}  {p['elapsed_seconds']}s{cost_str}")
        for fp in p.get("failed_pages", []):
            lines.append(f"           page {fp['page']}: {fp['error'][:100]}")
    lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLog written → {log_path}")


def main():
    ap = argparse.ArgumentParser(description="RAG enrichment: chunk + metadata extraction")
    ap.add_argument("--source", required=True, help="GCS source folder (e.g. jll)")
    ap.add_argument("--category", default="bengaluru_data",
                    choices=["bengaluru_data", "india_data"])
    ap.add_argument("--limit", type=int, help="Process only first N reports")
    ap.add_argument("--resume", action="store_true",
                    help="Skip reports that already have _rag_metadata.json")
    ap.add_argument("--local-output", type=Path, default=None,
                    help="Also save JSON + Excel locally to this folder")
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

    print(f"\nProcessing {len(entries)} report(s) (async, 10 concurrent chunks)\n")

    grand = {"reports": 0, "total_reports": len(entries),
             "pages": 0, "succeeded": 0, "failed": 0, "cost_usd": 0.0,
             "uncached_input_tokens": 0, "cached_read_tokens": 0, "output_tokens": 0}
    pdf_logs = []
    t_start = time.time()

    for entry in entries:
        print(f"\n=== {entry['stem']} ===")
        try:
            stats = _process_one(args.source, entry, local_output=args.local_output)
            grand["reports"] += 1
            grand["pages"] += stats["pages"]
            grand["succeeded"] += stats["succeeded"]
            grand["failed"] += stats["failed"]
            cost = stats.get("cost", {})
            grand["cost_usd"] += cost.get("total_usd", 0)
            grand["uncached_input_tokens"] += cost.get("uncached_input_tokens", 0)
            grand["cached_read_tokens"] += cost.get("cached_read_tokens", 0)
            grand["output_tokens"] += cost.get("output_tokens", 0)
            pdf_logs.append(stats)
        except Exception:
            print(f"  FAILED {entry['stem']}:", file=sys.stderr)
            traceback.print_exc()
            pdf_logs.append({
                "stem": entry["stem"], "pages": 0, "succeeded": 0,
                "failed": 0, "failed_pages": [], "elapsed_seconds": 0,
            })

    total_elapsed = time.time() - t_start

    log_dir = args.local_output or Path("rag_enrichment/output")
    log_path = log_dir / f"rag_log_{args.source}_{args.category}.txt"
    _write_log(log_path, args.source, args.category, pdf_logs, grand, total_elapsed)

    print(f"\n{'=' * 50}")
    print("RAG ENRICHMENT SUMMARY")
    print(f"  Reports processed : {grand['reports']}/{len(entries)}")
    print(f"  Total pages       : {grand['pages']}")
    print(f"  Succeeded         : {grand['succeeded']}")
    print(f"  Failed            : {grand['failed']}")
    print(f"  Total time        : {total_elapsed:.0f}s")
    print(f"  Total cost        : ${grand['cost_usd']:.4f}")
    print(f"  Tokens            : uncached_in={grand['uncached_input_tokens']} "
          f"cached={grand['cached_read_tokens']} out={grand['output_tokens']}")


if __name__ == "__main__":
    main()
