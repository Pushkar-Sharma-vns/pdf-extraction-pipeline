"""Phase 2 — Async image classification using Gemini.

Reads extracted images from GCS, classifies each as relevant to real estate
or not, and uploads classification_results.json back to the same GCS folder.

Usage:
    python -m glm_postprocess.classifier --source jll --category bengaluru_data
    python -m glm_postprocess.classifier --source cushman --workers 20 --resume
    python -m glm_postprocess.classifier --folder cushman/extracted_pdfs/bengaluru_data/report_extracted_images
    python -m glm_postprocess.classifier --dry-run --source jll
"""

import argparse
import asyncio
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from gemini_ocr.config import GEMINI_API_KEY, GCS_BUCKET
from gemini_ocr.gcs_client import _client, upload_file

CLASSIFY_MODEL = "gemini-2.5-flash"
CONCURRENT_DEFAULT = 20
MAX_RETRIES = 2
RETRY_DELAY = 2

PROMPT = (
    "Analyze this image extracted from a real estate market report. "
    "Classify whether it is relevant to real estate.\n\n"
    "RELEVANT: charts/graphs showing market data, property photos, building images, "
    "city skylines, area maps, floor plans, infographics with real estate metrics, "
    "construction site photos, architectural renderings.\n\n"
    "NOT RELEVANT: company logos, headshot photos, generic icons, decorative borders, "
    "stock imagery unrelated to property/buildings, page numbers, watermarks."
)


class ImageClassification(BaseModel):
    is_relevant: bool = Field(description="True if the image is relevant to real estate")
    confidence_score: float = Field(description="Confidence between 0.0 and 1.0")
    category: str = Field(description="Single-word category: chart, property_photo, map, logo, decoration, icon, infographic, other")
    reasoning: str = Field(description="Brief sentence explaining the classification")


def list_image_folders(source: str, category: str = None) -> list[str]:
    client = _client()
    folders = set()
    categories = [category] if category else ["bengaluru_data", "india_data"]
    for cat in categories:
        prefix = f"{source}/extracted_pdfs/{cat}/"
        blobs = client.list_blobs(GCS_BUCKET, prefix=prefix, delimiter="/")
        list(blobs)
        for p in blobs.prefixes:
            if p.endswith("_extracted_images/"):
                folders.add(p.rstrip("/"))
    return sorted(folders)


def list_images_in_folder(folder_prefix: str) -> list[str]:
    client = _client()
    return [
        b.name
        for b in client.list_blobs(GCS_BUCKET, prefix=folder_prefix + "/")
        if b.name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def folder_has_results(folder_prefix: str) -> bool:
    client = _client()
    blobs = list(client.list_blobs(
        GCS_BUCKET,
        prefix=f"{folder_prefix}/classification_results.json",
        max_results=1,
    ))
    return len(blobs) > 0


def _download_images_parallel(blob_names: list[str], max_threads: int = 10) -> list[tuple[str, bytes]]:
    """Download images from GCS in parallel using threads."""
    def _dl(name):
        bucket = _client().bucket(GCS_BUCKET)
        data = bucket.blob(name).download_as_bytes()
        return (f"gs://{GCS_BUCKET}/{name}", data)

    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        return list(pool.map(_dl, blob_names))


async def classify_single_image(aclient, semaphore, gcs_uri: str, image_bytes: bytes) -> dict:
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        async with semaphore:
            try:
                image_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                )
                response = await aclient.models.generate_content(
                    model=CLASSIFY_MODEL,
                    contents=[image_part, PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ImageClassification,
                        temperature=0.1,
                    ),
                )
                result = json.loads(response.text)
                result["gcs_uri"] = gcs_uri
                return result
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)

    return {"gcs_uri": gcs_uri, "error": last_error, "is_relevant": False}


async def classify_folder_async(folder_prefix: str, workers: int = CONCURRENT_DEFAULT) -> dict:
    image_blobs = list_images_in_folder(folder_prefix)
    if not image_blobs:
        return {"results": [], "elapsed_seconds": 0}

    t0 = time.time()

    print(f"  {len(image_blobs)} images, downloading (parallel)...")
    images = _download_images_parallel(image_blobs)

    print(f"  dispatching with {workers} async workers...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    aclient = client.aio

    semaphore = asyncio.Semaphore(workers)
    tasks = [classify_single_image(aclient, semaphore, uri, data) for uri, data in images]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    return {"results": list(results), "elapsed_seconds": round(elapsed, 1)}


def classify_folder(folder_prefix: str, workers: int = CONCURRENT_DEFAULT) -> dict:
    return asyncio.run(classify_folder_async(folder_prefix, workers))


def upload_results(folder_prefix: str, results: list[dict]):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="classify_"
    ) as f:
        json.dump(results, f, indent=2)
        tmp_path = Path(f.name)

    blob_name = f"{folder_prefix}/classification_results.json"
    uri = upload_file(tmp_path, blob_name, content_type="application/json")
    tmp_path.unlink(missing_ok=True)
    return uri


def _write_log(log_path: Path, source: str, category: str, folder_logs: list[dict],
               grand: dict, total_elapsed: float):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Image Classification Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source: {source}  Category: {category or 'all'}",
        f"Total: {grand['folders']} folders, {grand['images']} images, "
        f"{grand['relevant']} relevant, {grand['not_relevant']} not relevant, "
        f"{grand['errors']} errors, {total_elapsed:.0f}s",
        "",
    ]
    for fl in folder_logs:
        name = fl["folder"].split("/")[-1][:55]
        lines.append(f"  {name:55s}  imgs:{fl['images']:3d}  rel:{fl['relevant']:3d}  "
                      f"err:{fl['errors']}  {fl['elapsed_seconds']}s")
        for err in fl.get("error_uris", []):
            lines.append(f"    ERROR: {err}")
    lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLog written → {log_path}")


def main():
    ap = argparse.ArgumentParser(description="Classify extracted images using Gemini (async)")
    ap.add_argument("--source", help="GCS source folder (e.g. cushman)")
    ap.add_argument("--category", choices=["bengaluru_data", "india_data"],
                    help="Filter to only this category")
    ap.add_argument("--folder", help="Specific extracted_images folder prefix in GCS")
    ap.add_argument("--workers", type=int, default=CONCURRENT_DEFAULT,
                    help=f"Max concurrent Gemini calls (default: {CONCURRENT_DEFAULT})")
    ap.add_argument("--resume", action="store_true",
                    help="Skip folders that already have classification_results.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.folder:
        folders = [args.folder]
    elif args.source:
        folders = list_image_folders(args.source, category=args.category)
        print(f"Found {len(folders)} image folder(s) for '{args.source}'"
              f"{f' ({args.category})' if args.category else ''}")
    else:
        sys.exit("Provide --source or --folder")

    if not folders:
        sys.exit("No image folders found.")

    if args.resume:
        before = len(folders)
        folders = [f for f in folders if not folder_has_results(f)]
        print(f"Resume: {before - len(folders)} already classified, {len(folders)} remaining")

    if args.dry_run:
        print(f"\nDry run — {len(folders)} folder(s):")
        for f in folders:
            imgs = list_images_in_folder(f)
            print(f"  {f}  ({len(imgs)} images)")
        return

    if not folders:
        print("Nothing to do.")
        return

    print(f"\nClassifying {len(folders)} folder(s) with {args.workers} async workers\n")

    grand = {"folders": 0, "images": 0, "relevant": 0, "not_relevant": 0, "errors": 0}
    folder_logs = []
    t_start = time.time()

    for folder in folders:
        print(f"\n=== {folder.split('/')[-1]} ===")
        out = classify_folder(folder, workers=args.workers)
        results = out["results"]

        if results:
            uri = upload_results(folder, results)
            relevant = sum(1 for r in results if r.get("is_relevant"))
            errors = sum(1 for r in results if "error" in r)
            not_rel = len(results) - relevant - errors
            error_uris = [r["gcs_uri"] for r in results if "error" in r]

            print(f"  {len(results)} classified: {relevant} relevant, "
                  f"{not_rel} not relevant, {errors} errors  ({out['elapsed_seconds']}s)")
            print(f"  → {uri}")

            grand["folders"] += 1
            grand["images"] += len(results)
            grand["relevant"] += relevant
            grand["not_relevant"] += not_rel
            grand["errors"] += errors
            folder_logs.append({
                "folder": folder, "images": len(results), "relevant": relevant,
                "errors": errors, "elapsed_seconds": out["elapsed_seconds"],
                "error_uris": error_uris,
            })

    total_elapsed = time.time() - t_start

    log_dir = Path("glm_postprocess/output")
    log_path = log_dir / f"classify_log_{args.source}_{args.category or 'all'}.txt"
    _write_log(log_path, args.source, args.category, folder_logs, grand, total_elapsed)

    print(f"\n{'=' * 50}")
    print("CLASSIFICATION SUMMARY")
    print(f"  Folders processed : {grand['folders']}")
    print(f"  Total images      : {grand['images']}")
    print(f"  Relevant          : {grand['relevant']}")
    print(f"  Not relevant      : {grand['not_relevant']}")
    if grand["errors"]:
        print(f"  Errors            : {grand['errors']}")
    print(f"  Total time        : {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
