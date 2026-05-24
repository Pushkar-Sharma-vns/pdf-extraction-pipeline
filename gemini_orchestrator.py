"""Batch Gemini extraction across the stress-test corpus.

For each PDF in PDF_DIR:
  - split into 10-page chunks (Gemini hallucinates more on long PDFs)
  - upload each chunk, run extraction prompt
  - append response to test_results/<pdfname>/<pdfname>_gemini.md
"""

import os
import sys
import time
import tempfile
import traceback
from pathlib import Path

import fitz  # PyMuPDF
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCeju1K1ik-wxQvC9ErLI4xW-ZIVI3PpWc")
GEMINI_MODEL = "gemini-3-flash-preview"

PDF_DIR = Path("15_StressTest_Docs_for_Parser")
RESULTS_ROOT = Path("test_results")
CHUNK_PAGES = 10

PRICE_INPUT_1M = 0.50
PRICE_OUTPUT_1M = 3.00

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


def split_pdf_to_chunks(pdf_path: Path, chunk_size: int, tmp_dir: Path) -> list[tuple[Path, int, int]]:
    """Return list of (chunk_path, start_page_1based, end_page_1based)."""
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


def upload_and_wait(client, chunk_path: Path):
    f = client.files.upload(file=str(chunk_path))
    while f.state.name == "PROCESSING":
        time.sleep(1)
        f = client.files.get(name=f.name)
    if f.state.name == "FAILED":
        raise RuntimeError(f"Gemini upload failed for {chunk_path.name}")
    return f


def extract_chunk(client, file_obj, prompt: str):
    return client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[file_obj, prompt],
        config=types.GenerateContentConfig(temperature=0.1),
    )


def process_pdf(client, pdf_path: Path) -> dict:
    stem = pdf_path.stem
    out_dir = RESULTS_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / f"{stem}_gemini.md"

    out_md.write_text(f"# Gemini extraction — {stem}\n\n")

    totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "chunks": 0}

    with tempfile.TemporaryDirectory(prefix="gemini_chunks_") as td:
        tmp_dir = Path(td)
        chunks = split_pdf_to_chunks(pdf_path, CHUNK_PAGES, tmp_dir)
        print(f"  {stem}: {len(chunks)} chunk(s) of up to {CHUNK_PAGES} pages")

        with out_md.open("a", encoding="utf-8") as f:
            for chunk_path, p_start, p_end in chunks:
                try:
                    file_obj = upload_and_wait(client, chunk_path)
                    resp = extract_chunk(client, file_obj, PROMPT)
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
                    print(
                        f"    pages {p_start}-{p_end}: "
                        f"in={usage.prompt_token_count:,} out={usage.candidates_token_count:,} "
                        f"${cost:.4f}"
                    )
                except Exception as e:
                    f.write(f"\n\n<!-- chunk: pages {p_start}-{p_end} FAILED: {e} -->\n\n")
                    print(f"    pages {p_start}-{p_end}: FAILED — {e}", file=sys.stderr)

    return totals


def main():
    if not PDF_DIR.exists():
        sys.exit(f"PDF directory not found: {PDF_DIR}")

    client = genai.Client(api_key=GEMINI_API_KEY)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDF(s) in {PDF_DIR}")

    grand = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "chunks": 0, "pdfs": 0}
    for pdf in pdfs:
        print(f"\n=== {pdf.name} ===")
        try:
            stats = process_pdf(client, pdf)
            for k in stats:
                grand[k] += stats[k]
            grand["pdfs"] += 1
        except Exception:
            print(f"  FAILED {pdf.name}:", file=sys.stderr)
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("BATCH SUMMARY (Gemini)")
    print(f"  PDFs processed : {grand['pdfs']}/{len(pdfs)}")
    print(f"  Chunks         : {grand['chunks']}")
    print(f"  Input tokens   : {grand['input_tokens']:,}")
    print(f"  Output tokens  : {grand['output_tokens']:,}")
    print(f"  Total cost     : ${grand['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
