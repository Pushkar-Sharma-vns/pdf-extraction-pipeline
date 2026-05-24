"""Image extraction from GLM-OCR results using polygon coordinates.

Crops pictorial regions (image, chart, figure_title, etc.) from layout_vis page JPGs
using bbox_2d coordinates from the _model.json file.
"""

import json
import re
import shutil
from pathlib import Path

from PIL import Image

from glm_postprocess.config import PICTORIAL_LABELS, NORM, MIN_SIZE_DEFAULT

_slug_re = re.compile(r"[^A-Za-z0-9._-]+")


def _label_slug(label: str) -> str:
    return _slug_re.sub("_", (label or "unknown")).strip("_") or "unknown"


def _bbox_to_pixels(bbox, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (max(0, min(NORM, v)) for v in bbox)
    px = lambda v, dim: int(round(v / NORM * dim))
    left, top, right, bottom = px(x1, img_w), px(y1, img_h), px(x2, img_w), px(y2, img_h)
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1
    return left, top, right, bottom


def _page_jpg_path(pdf_dir: Path, stem: str, page_idx: int) -> Path | None:
    p = pdf_dir / "layout_vis" / f"{stem}_page{page_idx}.jpg"
    return p if p.exists() else None


def find_pdf_dirs(root: Path) -> list[Path]:
    dirs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / f"{d.name}_model.json").exists() and (d / "layout_vis").is_dir():
            dirs.append(d)
    return dirs


def process_pdf_dir(pdf_dir: Path, min_size: int = MIN_SIZE_DEFAULT) -> dict:
    """Crop pictorial items from a GLM-OCR output folder.

    Returns dict with keys: saved, skipped_no_bbox, skipped_no_jpg, skipped_small,
    by_label, out_dir.
    """
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
        jpg = _page_jpg_path(pdf_dir, stem, page_idx)
        if jpg is None:
            skipped_no_jpg += sum(1 for it in items if it.get("label") in PICTORIAL_LABELS)
            continue

        page_img = None

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

            box = _bbox_to_pixels(bbox, page_img.width, page_img.height)
            left, top, right, bottom = box
            if min_size and (right - left < min_size or bottom - top < min_size):
                skipped_small += 1
                continue

            crop = page_img.crop(box)
            idx = item.get("index", 0)
            name = f"page{page_idx + 1}_{_label_slug(label)}_idx{idx}.jpg"
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
