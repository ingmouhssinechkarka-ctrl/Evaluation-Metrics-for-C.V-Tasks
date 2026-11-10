"""Run the three demos and save figures to ./figures."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("ocr", "object_detection", "image_generation"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import detection_metrics  # noqa: E402
import generation_metrics  # noqa: E402
import ocr_metrics  # noqa: E402

if __name__ == "__main__":
    fig_dir = os.path.join(ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    ocr_metrics.demo()
    print()
    detection_metrics.demo(fig_dir)
    print()
    generation_metrics.demo(fig_dir)
