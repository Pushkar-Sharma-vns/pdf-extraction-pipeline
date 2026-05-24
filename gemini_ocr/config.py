import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GCS_CREDENTIALS = Path(os.environ.get(
    "GCS_CREDENTIALS",
    Path(__file__).resolve().parent.parent / "portal-data-bucket.json",
))
GCS_BUCKET = os.environ.get("GCS_BUCKET", "re_reports")

GEMINI_MODEL = "gemini-3-flash-preview"
CHUNK_PAGES = 10

PRICE_INPUT_1M = 0.50
PRICE_OUTPUT_1M = 3.00
