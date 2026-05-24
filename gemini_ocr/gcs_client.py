from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

from gemini_ocr.config import GCS_BUCKET, GCS_CREDENTIALS


def _client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_file(str(GCS_CREDENTIALS))
    return storage.Client(credentials=creds, project=creds.project_id)


def _bucket() -> storage.Bucket:
    return _client().bucket(GCS_BUCKET)


def list_sources() -> list[str]:
    """Return top-level 'source' folder names under the bucket root.

    Structure expected: gs://re_reports/<source>/pdfs/*.pdf
    """
    client = _client()
    blobs = client.list_blobs(GCS_BUCKET, delimiter="/")
    list(blobs)  # consume iterator so prefixes populate
    return [p.rstrip("/") for p in blobs.prefixes]


def list_pdfs(source: str) -> list[str]:
    """Return blob names (full path) of all PDFs under <source>/pdfs/."""
    prefix = f"{source}/pdfs/"
    return [
        b.name
        for b in _client().list_blobs(GCS_BUCKET, prefix=prefix)
        if b.name.lower().endswith(".pdf")
    ]


def download_blob(blob_name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _bucket().blob(blob_name).download_to_filename(str(dest))
    return dest


def blob_exists(blob_name: str) -> bool:
    return _bucket().blob(blob_name).exists()


def list_existing_outputs(source: str) -> set[str]:
    """Return set of blob names already in extracted_pdfs/ for a source."""
    prefix = f"{source}/extracted_pdfs/"
    return {
        b.name
        for b in _client().list_blobs(GCS_BUCKET, prefix=prefix)
    }


def upload_file(local_path: Path, blob_name: str, content_type: str = "text/markdown") -> str:
    blob = _bucket().blob(blob_name)
    blob.upload_from_filename(str(local_path), content_type=content_type)
    return f"gs://{GCS_BUCKET}/{blob_name}"
