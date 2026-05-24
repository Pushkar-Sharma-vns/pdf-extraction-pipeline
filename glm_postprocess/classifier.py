"""Phase 2 — Async image classification using Gemini.

Reads extracted images from GCS via gs:// URIs, classifies each as
relevant to real estate or not, and uploads classification_results.json
back to the same GCS folder.

Usage:
    python -m glm_postprocess.classifier --source jll --dry-run
    python -m glm_postprocess.classifier --source cushman --workers 20
    python -m glm_postprocess.classifier --folder cushman/extracted_pdfs/bengaluru_data/report_extracted_images
    python -m glm_postprocess.classifier --source savills --resume --workers 20
"""

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from gemini_ocr.config import GEMINI_API_KEY, GCS_BUCKET
from gemini_ocr.gcs_client import _client, upload_file

CLASSIFY_MODEL = "gemini-2.5-flash"
CONCURRENT_DEFAULT = 20

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


def list_image_folders(source: str) -> list[str]:
    """List all <pdfname>_extracted_images/ prefixes under a source."""
    client = _client()
    folders = set()
    for cat in ("bengaluru_data", "india_data"):
        prefix = f"{source}/extracted_pdfs/{cat}/"
        blobs = client.list_blobs(GCS_BUCKET, prefix=prefix, delimiter="/")
        list(blobs)
        for p in blobs.prefixes:
            if p.endswith("_extracted_images/"):
                folders.add(p.rstrip("/"))
    return sorted(folders)


def list_images_in_folder(folder_prefix: str) -> list[str]:
    """List all image blob names in a folder prefix."""
    client = _client()
    return [
        b.name
        for b in client.list_blobs(GCS_BUCKET, prefix=folder_prefix + "/")
        if b.name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def folder_has_results(folder_prefix: str) -> bool:
    """Check if classification_results.json already exists in the folder."""
    client = _client()
    blobs = list(client.list_blobs(
        GCS_BUCKET,
        prefix=f"{folder_prefix}/classification_results.json",
        max_results=1,
    ))
    return len(blobs) > 0


def _download_image_bytes(blob_name: str) -> bytes:
    """Download image from GCS as raw bytes."""
    bucket = _client().bucket(GCS_BUCKET)
    return bucket.blob(blob_name).download_as_bytes()


async def classify_single_image(aclient, semaphore, gcs_uri: str, image_bytes: bytes) -> dict:
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
            return {"gcs_uri": gcs_uri, "error": str(e), "is_relevant": False}


async def classify_folder_async(folder_prefix: str, workers: int = CONCURRENT_DEFAULT) -> list[dict]:
    image_blobs = list_images_in_folder(folder_prefix)
    if not image_blobs:
        return []

    print(f"  {len(image_blobs)} images, downloading from GCS...")
    images = []
    for blob_name in image_blobs:
        gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
        img_bytes = _download_image_bytes(blob_name)
        images.append((gcs_uri, img_bytes))

    print(f"  dispatching with {workers} async workers...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    aclient = client.aio

    semaphore = asyncio.Semaphore(workers)
    tasks = [classify_single_image(aclient, semaphore, uri, data) for uri, data in images]
    results = await asyncio.gather(*tasks)
    return list(results)


def classify_folder(folder_prefix: str, workers: int = CONCURRENT_DEFAULT) -> list[dict]:
    """Sync wrapper around async classification."""
    return asyncio.run(classify_folder_async(folder_prefix, workers))


def upload_results(folder_prefix: str, results: list[dict]):
    """Upload classification_results.json to the same GCS folder."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="classify_"
    ) as f:
        json.dump(results, f, indent=2)
        tmp_path = Path(f.name)

    blob_name = f"{folder_prefix}/classification_results.json"
    uri = upload_file(tmp_path, blob_name, content_type="application/json")
    tmp_path.unlink(missing_ok=True)
    return uri


def main():
    ap = argparse.ArgumentParser(description="Classify extracted images using Gemini (async)")
    ap.add_argument("--source", help="GCS source folder (e.g. cushman)")
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
        folders = list_image_folders(args.source)
        print(f"Found {len(folders)} image folder(s) for source '{args.source}'")
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

    for folder in folders:
        print(f"\n=== {folder.split('/')[-1]} ===")
        results = classify_folder(folder, workers=args.workers)

        if results:
            uri = upload_results(folder, results)
            relevant = sum(1 for r in results if r.get("is_relevant"))
            errors = sum(1 for r in results if "error" in r)
            print(f"  {len(results)} classified: {relevant} relevant, "
                  f"{len(results) - relevant - errors} not relevant, {errors} errors")
            print(f"  → {uri}")
            grand["folders"] += 1
            grand["images"] += len(results)
            grand["relevant"] += relevant
            grand["not_relevant"] += len(results) - relevant - errors
            grand["errors"] += errors

    print(f"\n{'=' * 50}")
    print("CLASSIFICATION SUMMARY")
    print(f"  Folders processed : {grand['folders']}")
    print(f"  Total images      : {grand['images']}")
    print(f"  Relevant          : {grand['relevant']}")
    print(f"  Not relevant      : {grand['not_relevant']}")
    if grand["errors"]:
        print(f"  Errors            : {grand['errors']}")


if __name__ == "__main__":
    main()
