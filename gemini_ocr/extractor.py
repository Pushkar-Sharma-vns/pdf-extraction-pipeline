"""Core Gemini extraction logic — split PDF into chunks, extract via Gemini, return markdown."""

import sys
import time
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
from google import genai
from google.genai import types

from gemini_ocr.config import GEMINI_API_KEY, GEMINI_MODEL, CHUNK_PAGES, PRICE_INPUT_1M, PRICE_OUTPUT_1M

PROMPT = (
    "Perform a high-fidelity extraction of this PDF chunk.\n"
    "For each page in order:\n"
    "- Transcribe narrative text faithfully.\n"
    "- Convert every table to clean Markdown format.\n"
    "- For each chart/graph, convert visual data points to a Markdown table; "
    "  include axis labels, units, and the chart's title as a caption.\n"
    "- Preserve reading order; mark page breaks with an HTML comment "
    "  like <!-- page N -->.\n"
    "Do not invent data. If a value is unreadable, write [unreadable]."
)


def _get_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def split_pdf(pdf_path: Path, chunk_size: int, tmp_dir: Path) -> list[tuple[Path, int, int]]:
    src = fitz.open(str(pdf_path))
    n = src.page_count
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n) - 1
        dst = fitz.open()
        dst.insert_pdf(src, from_page=start, to_page=end)
        out = tmp_dir / f"{pdf_path.stem}_p{start + 1}-{end + 1}.pdf"
        dst.save(str(out))
        dst.close()
        chunks.append((out, start + 1, end + 1))
    src.close()
    return chunks


MAX_RETRIES = 5
BACKOFF_BASE = 2


def _retry(fn, *args, label="", **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = "429" in err_str or "rate" in err_str or "resource_exhausted" in err_str
            if not is_retryable or attempt == MAX_RETRIES - 1:
                raise
            wait = BACKOFF_BASE ** attempt
            print(f"    {label} rate-limited, retry {attempt + 1}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)


def _upload_and_wait(client, chunk_path: Path):
    f = _retry(client.files.upload, file=str(chunk_path), label="upload")
    while f.state.name == "PROCESSING":
        time.sleep(1)
        f = client.files.get(name=f.name)
    if f.state.name == "FAILED":
        raise RuntimeError(f"Gemini upload failed for {chunk_path.name}")
    return f


def extract_pdf(pdf_path: Path, out_md: Path) -> dict:
    """Extract a single PDF with Gemini, writing results to out_md.

    Returns token/cost stats.
    """
    client = _get_client()
    stem = pdf_path.stem
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(f"# Gemini extraction — {stem}\n\n")

    totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "chunks": 0}

    with tempfile.TemporaryDirectory(prefix="gemini_chunks_") as td:
        tmp_dir = Path(td)
        chunks = split_pdf(pdf_path, CHUNK_PAGES, tmp_dir)
        print(f"  {stem}: {len(chunks)} chunk(s) of up to {CHUNK_PAGES} pages")

        with out_md.open("a", encoding="utf-8") as f:
            for chunk_path, p_start, p_end in chunks:
                try:
                    file_obj = _upload_and_wait(client, chunk_path)
                    resp = _retry(
                        client.models.generate_content,
                        model=GEMINI_MODEL,
                        contents=[file_obj, PROMPT],
                        config=types.GenerateContentConfig(temperature=0.1),
                        label=f"p{p_start}-{p_end}",
                    )
                    usage = resp.usage_metadata
                    cost = (
                        (usage.prompt_token_count / 1_000_000) * PRICE_INPUT_1M
                        + (usage.candidates_token_count / 1_000_000) * PRICE_OUTPUT_1M
                    )
                    f.write(f"\n\n<!-- chunk: pages {p_start}-{p_end} -->\n\n")
                    f.write(resp.text or "")
                    totals["input_tokens"] += usage.prompt_token_count
                    totals["output_tokens"] += usage.candidates_token_count
                    totals["cost_usd"] += cost
                    totals["chunks"] += 1
                    print(f"    pages {p_start}-{p_end}: "
                          f"in={usage.prompt_token_count:,} out={usage.candidates_token_count:,} "
                          f"${cost:.4f}")
                except Exception as e:
                    f.write(f"\n\n<!-- chunk: pages {p_start}-{p_end} FAILED: {e} -->\n\n")
                    print(f"    pages {p_start}-{p_end}: FAILED — {e}", file=sys.stderr)

    return totals
