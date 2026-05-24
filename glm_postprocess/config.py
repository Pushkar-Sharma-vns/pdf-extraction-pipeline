import sys
from pathlib import Path

PICTORIAL_LABELS = {
    "image", "chart", "figure_title", "figure",
    "graph", "map", "picture", "diagram", "photo",
}
NORM = 1000
MIN_SIZE_DEFAULT = 100
RESULTS_ROOT = Path("test_results")
CONFIG_YAML = Path("config.yaml")
GLMOCR_BIN = str(Path(sys.executable).parent / "glmocr")
