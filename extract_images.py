"""Extract pictorial regions (image / chart / figure_title / etc.) from GLM-OCR results.

For each PDF folder in test_results/:
  - reads <stem>_model.json (per-page item list with bbox_2d in 1000-normalized space)
  - reads layout_vis/<stem>_page<N>.jpg  (one rendered page per outer-list index)
  - crops every pictorial item and saves it to <folder>/extracted_images/

Output filename: page<N>_<label>_idx<I>.jpg
  - page index is 1-based (matches what humans expect)
  - (page, label, index) is the stable mapping key for downstream content extraction
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

RESULTS_ROOT = Path("test_results")
PICTORIAL_LABELS = {
    "image", "chart", "figure_title", "figure",
    "graph", "map", "picture", "diagram", "photo",
}
NORM = 1000


def find_pdf_dirs(root: Path) -> list[Path]:
    dirs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / f"{d.name}_model.json").exists() and (d / "layout_vis").is_dir():
            dirs.append(d)
    return dirs


def page_jpg_path(pdf_dir: Path, page_idx: int) -> Path | None:
    stem = pdf_dir.name
    # layout_vis page indexing is 0-based: <stem>_page0.jpg, page1.jpg, ...
    p = pdf_dir / "layout_vis" / f"{stem}_page{page_idx}.jpg"
    return p if p.exists() else None


def bbox_to_pixels(bbox, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (max(0, min(NORM, v)) for v in bbox)
    px = lambda v, dim: int(round(v / NORM * dim))
    left, top, right, bottom = px(x1, img_w), px(y1, img_h), px(x2, img_w), px(y2, img_h)
    # Pillow requires left < right, top < bottom, and >0 area
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1
    return left, top, right, bottom


_slug_re = re.compile(r"[^A-Za-z0-9._-]+")


def label_slug(label: str) -> str:
    return _slug_re.sub("_", (label or "unknown")).strip("_") or "unknown"


def process_pdf_dir(pdf_dir: Path, min_size: int = 0) -> dict:
    stem = pdf_dir.name
    model_json = pdf_dir / f"{stem}_model.json"
    out_dir = pdf_dir / "extracted_images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    pages = json.loads(model_json.read_text())
    saved = skipped_no_bbox = skipped_no_jpg = skipped_small = 0
    by_label: dict[str, int] = {}

    for page_idx, items in enumerate(pages):
        if not items:
            continue
        jpg = page_jpg_path(pdf_dir, page_idx)
        if jpg is None:
            skipped_no_jpg += sum(1 for it in items if it.get("label") in PICTORIAL_LABELS)
            continue

        page_img = None  # lazy-load: only open the JPG if this page has pictorial items

        for item in items:
            label = item.get("label", "")
            if label not in PICTORIAL_LABELS:
                continue

            bbox = item.get("bbox_2d")
            if not bbox:
                skipped_no_bbox += 1
                continue

            if page_img is None:
                page_img = Image.open(jpg)

            box = bbox_to_pixels(bbox, page_img.width, page_img.height)
            left, top, right, bottom = box
            if min_size and (right - left < min_size or bottom - top < min_size):
                skipped_small += 1
                continue

            crop = page_img.crop(box)
            idx = item.get("index", 0)
            name = f"page{page_idx + 1}_{label_slug(label)}_idx{idx}.jpg"
            crop.convert("RGB").save(out_dir / name, "JPEG", quality=92)
            saved += 1
            by_label[label] = by_label.get(label, 0) + 1

        if page_img is not None:
            page_img.close()

    return {
        "saved": saved,
        "skipped_no_bbox": skipped_no_bbox,
        "skipped_no_jpg": skipped_no_jpg,
        "skipped_small": skipped_small,
        "by_label": by_label,
        "out_dir": out_dir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=RESULTS_ROOT,
                    help="Directory containing one folder per PDF (default: test_results/)")
    ap.add_argument("--pdf", help="Process only this PDF folder name (skip the rest)")
    ap.add_argument("--min-size", type=int, default=0,
                    help="Skip crops whose width or height (JPG pixels) is below this. Default 0 = no filter.")
    args = ap.parse_args()

    if not args.root.exists():
        sys.exit(f"Root not found: {args.root}")

    pdf_dirs = find_pdf_dirs(args.root)
    if args.pdf:
        pdf_dirs = [d for d in pdf_dirs if d.name == args.pdf]
        if not pdf_dirs:
            sys.exit(f"No matching PDF folder for --pdf {args.pdf}")

    print(f"Found {len(pdf_dirs)} PDF folder(s) to process under {args.root}  (min_size={args.min_size})")
    grand = {"saved": 0, "skipped_no_bbox": 0, "skipped_no_jpg": 0, "skipped_small": 0, "pdfs": 0}
    grand_labels: dict[str, int] = {}

    for d in pdf_dirs:
        print(f"\n=== {d.name} ===")
        stats = process_pdf_dir(d, min_size=args.min_size)
        print(f"  saved: {stats['saved']}  by_label: {stats['by_label']}")
        if stats["skipped_small"]:
            print(f"  Filtered  {stats['skipped_small']} crops smaller than {args.min_size}px")
        if stats["skipped_no_bbox"]:
            print(f"  WARN  {stats['skipped_no_bbox']} pictorial items had no bbox")
        if stats["skipped_no_jpg"]:
            print(f"  WARN  {stats['skipped_no_jpg']} items skipped (missing page JPG)")
        print(f"  -> {stats['out_dir']}")

        grand["saved"] += stats["saved"]
        grand["skipped_no_bbox"] += stats["skipped_no_bbox"]
        grand["skipped_no_jpg"] += stats["skipped_no_jpg"]
        grand["skipped_small"] += stats["skipped_small"]
        grand["pdfs"] += 1
        for lbl, n in stats["by_label"].items():
            grand_labels[lbl] = grand_labels.get(lbl, 0) + n

    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print(f"  PDFs processed       : {grand['pdfs']}/{len(pdf_dirs)}")
    print(f"  Images saved         : {grand['saved']}")
    print(f"  By label             : {grand_labels}")
    if grand["skipped_small"]:
        print(f"  Filtered (small)     : {grand['skipped_small']}")
    if grand["skipped_no_bbox"]:
        print(f"  Skipped (no bbox)    : {grand['skipped_no_bbox']}")
    if grand["skipped_no_jpg"]:
        print(f"  Skipped (no page jpg): {grand['skipped_no_jpg']}")


if __name__ == "__main__":
    main()
