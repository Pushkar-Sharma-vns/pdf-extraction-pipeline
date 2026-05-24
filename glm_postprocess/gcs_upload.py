"""GCS upload helpers for extracted images."""

from pathlib import Path

from gemini_ocr.gcs_client import upload_file, _client, GCS_BUCKET
from gemini_ocr.pipeline import classify_pdf


def upload_images(local_dir: Path, source: str, category: str, pdf_stem: str) -> int:
    """Upload all JPGs from local_dir to GCS.

    Target: <source>/extracted_pdfs/<category>/<pdf_stem>_extracted_images/<filename>
    Returns number of files uploaded.
    """
    jpgs = sorted(local_dir.glob("*.jpg"))
    prefix = f"{source}/extracted_pdfs/{category}/{pdf_stem}_extracted_images"
    count = 0
    for jpg in jpgs:
        blob_name = f"{prefix}/{jpg.name}"
        upload_file(jpg, blob_name, content_type="image/jpeg")
        count += 1
    return count


def images_already_uploaded(source: str, pdf_stem: str) -> bool:
    """Check if extracted images already exist in GCS (either category)."""
    for cat in ("bengaluru_data", "india_data"):
        prefix = f"{source}/extracted_pdfs/{cat}/{pdf_stem}_extracted_images/"
        blobs = list(_client().list_blobs(GCS_BUCKET, prefix=prefix, max_results=1))
        if blobs:
            return True
    return False
