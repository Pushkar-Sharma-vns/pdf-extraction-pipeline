"""Batch GLM-OCR + Qwen2.5-VL enrichment across the stress-test corpus.

For each PDF in PDF_DIR:
  1. Run `glmocr parse` to produce test_results/<pdfname>/<pdfname>_model.json
     (skipped if the JSON already exists, so reruns are cheap)
  2. Call enrich_pdf() to crop chart/figure regions, run Qwen2.5-VL via mlx-vlm,
     and write test_results/<pdfname>/<pdfname>_ocr.md

The Qwen model is loaded ONCE for the whole batch — reused across all 15 PDFs.
"""

import subprocess
import sys
import time
import traceback
from pathlib import Path

from enrich_report import enrich_pdf, load_model_ctx

PDF_DIR = Path("15_StressTest_Docs_for_Parser")
RESULTS_ROOT = Path("test_results")
CONFIG = Path("config.yaml")
GLMOCR_BIN = str(Path(sys.executable).parent / "glmocr")  # use this venv's glmocr


def run_glmocr(pdf_path: Path) -> Path:
    """Run glmocr on the PDF and return the path to the generated _model.json.

    Skips the run if the JSON already exists.
    """
    stem = pdf_path.stem
    out_dir = RESULTS_ROOT / stem
    model_json = out_dir / f"{stem}_model.json"

    if model_json.exists():
        print(f"  glmocr: cached → {model_json}")
        return model_json

    cmd = [GLMOCR_BIN, "parse", str(pdf_path), "--output", str(RESULTS_ROOT)]
    if CONFIG.exists():
        cmd += ["--config", str(CONFIG)]

    print(f"  glmocr: running {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        sys.stderr.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"glmocr failed (exit {res.returncode}) for {pdf_path.name}")
    print(f"  glmocr: done in {dt:.1f}s")

    if not model_json.exists():
        raise FileNotFoundError(f"Expected {model_json} not found after glmocr run")
    return model_json


def process_pdf(pdf_path: Path, ctx) -> dict:
    stem = pdf_path.stem
    out_md = RESULTS_ROOT / stem / f"{stem}_ocr.md"

    model_json = run_glmocr(pdf_path)
    print(f"  enrich: {model_json.name} → {out_md.name}")
    t0 = time.time()
    stats = enrich_pdf(model_json, pdf_path, out_md, ctx)
    dt = time.time() - t0
    print(f"  enrich: done in {dt:.1f}s  {stats}")
    return stats


def main():
    if not PDF_DIR.exists():
        sys.exit(f"PDF directory not found: {PDF_DIR}")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDF(s) in {PDF_DIR}")

    ctx = load_model_ctx()      # load Qwen ONCE for the whole batch

    grand = {"enriched": 0, "skipped": 0, "kept_structured": 0, "pdfs": 0}
    for pdf in pdfs:
        print(f"\n=== {pdf.name} ===")
        try:
            stats = process_pdf(pdf, ctx)
            for k in stats:
                grand[k] += stats[k]
            grand["pdfs"] += 1
        except Exception:
            print(f"  FAILED {pdf.name}:", file=sys.stderr)
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("BATCH SUMMARY (Hybrid GLM-OCR + Qwen2.5-VL)")
    print(f"  PDFs processed         : {grand['pdfs']}/{len(pdfs)}")
    print(f"  Chart/figure enriched  : {grand['enriched']}")
    print(f"  Chart/figure SKIP'd    : {grand['skipped']}")
    print(f"  Already-structured kept: {grand['kept_structured']}")


if __name__ == "__main__":
    main()
