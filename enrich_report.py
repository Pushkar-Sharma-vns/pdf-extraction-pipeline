import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

VLM_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
NORM = 1000                       # GLM-OCR normalizes each bbox axis to ~1000
CROP_DPI = 300
VLM_TARGETS = {"chart", "figure"}
DROP_LABELS = {"header", "footer", "number", "vision_footnote"}


@dataclass
class ModelCtx:
    model: object
    processor: object
    config: object


def load_model_ctx(model_id: str = VLM_MODEL) -> ModelCtx:
    print(f"Loading {model_id} on Apple GPU (mlx-vlm, Unified Memory)…")
    model, processor = load(model_id)
    config = load_config(model_id)
    return ModelCtx(model=model, processor=processor, config=config)


def looks_structured(content: str) -> bool:
    if not content:
        return False
    s = content.strip()
    return "|" in s and "---" in s


def to_pdf_rect(page, bbox):
    w, h = page.rect.width, page.rect.height
    x1, y1, x2, y2 = (max(0, min(NORM, v)) for v in bbox)
    return fitz.Rect(x1 / NORM * w, y1 / NORM * h, x2 / NORM * w, y2 / NORM * h)


def crop_region(page, bbox, out_path: Path) -> Path:
    rect = to_pdf_rect(page, bbox)
    zoom = CROP_DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect)
    pix.save(str(out_path))
    return out_path


def extract_chart_md(ctx: ModelCtx, image_path: Path, hint: str) -> str:
    instruction = (
        "The previous OCR model identified this as a chart. "
        "Please extract all numerical data points and labels into a Markdown table. "
        "Include axis titles, units, and the chart title as a caption above the table. "
        "If the image is not a data chart (e.g. a logo, photo, or pure decoration), "
        "reply with exactly: SKIP"
    )
    if hint:
        instruction += f"\n\nPartial OCR text from this region (reference only):\n{hint}"

    formatted = apply_chat_template(ctx.processor, ctx.config, instruction, num_images=1)
    result = generate(
        ctx.model, ctx.processor, formatted,
        image=[str(image_path)],
        max_tokens=1024,
        verbose=False,
    )
    return result.text if hasattr(result, "text") else str(result)


def render_default(item) -> Optional[str]:
    label = item.get("label", "")
    content = (item.get("content") or "").strip()
    if not content or label in DROP_LABELS:
        return None
    if label == "doc_title":
        return f"# {content}\n"
    if label == "paragraph_title":
        return f"## {content}\n"
    if label == "figure_title":
        return f"**{content}**\n"
    return f"{content}\n"


def enrich_pdf(json_path: Path, pdf_path: Path, out_path: Path, ctx: ModelCtx) -> dict:
    """Build an enriched Markdown report from a GLM-OCR pdf_model.json + the original PDF.

    Returns a small summary dict (counts of enrichments/skips) for orchestrator logging.
    """
    pages = json.loads(json_path.read_text())
    pdf = fitz.open(str(pdf_path))
    tmp = Path(tempfile.mkdtemp(prefix="enrich_crops_"))
    parts: list[str] = []
    enriched = skipped = kept = 0

    try:
        for pi, page_items in enumerate(pages):
            if not page_items:
                continue
            parts.append(f"\n<!-- page {pi + 1} -->\n\n")
            page = pdf[pi]

            for it in page_items:
                label = it.get("label", "")
                content = (it.get("content") or "").strip()

                if label in VLM_TARGETS:
                    if looks_structured(content):
                        parts.append(f"{content}\n\n")
                        kept += 1
                        continue

                    bbox = it.get("bbox_2d")
                    if not bbox:
                        if content:
                            parts.append(f"{content}\n\n")
                        continue

                    crop_path = crop_region(
                        page, bbox, tmp / f"p{pi}_{it.get('index', 0)}.png"
                    )
                    md = extract_chart_md(ctx, crop_path, hint=content).strip()

                    if md.upper().startswith("SKIP"):
                        skipped += 1
                        continue

                    parts.append(
                        f"\n> **[VLM-enriched {label} — p{pi + 1}]**\n\n{md}\n\n"
                    )
                    enriched += 1
                else:
                    chunk = render_default(it)
                    if chunk:
                        parts.append(chunk)
    finally:
        pdf.close()
        shutil.rmtree(tmp, ignore_errors=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts))
    return {"enriched": enriched, "skipped": skipped, "kept_structured": kept}


def main():
    DEFAULT_PDF_NAME = "india-real-estate-office-and-residential-market-jan-mar-2026-12790"
    DEFAULT_RESULTS = Path("test_results") / DEFAULT_PDF_NAME

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path,
                    default=DEFAULT_RESULTS / f"{DEFAULT_PDF_NAME}_model.json")
    ap.add_argument("--pdf", type=Path, default=Path(f"{DEFAULT_PDF_NAME}.pdf"))
    ap.add_argument("--out", type=Path,
                    default=DEFAULT_RESULTS / f"{DEFAULT_PDF_NAME}_ocr.md")
    args = ap.parse_args()

    ctx = load_model_ctx()
    stats = enrich_pdf(args.json, args.pdf, args.out, ctx)
    print(f"Wrote {args.out}  ({stats})")


if __name__ == "__main__":
    main()
