"""Backward-compatible entry point for the D04 detection-embedded evaluation.

The implementation historically duplicated two other evaluation scripts and
contained a syntax error in its image-id setup.  It now delegates to the shared
COCO evaluator while preserving the original default model and output paths.
"""

from __future__ import annotations

try:
    from .D04_best_small_object_eval import main
except ImportError:
    from D04_best_small_object_eval import main


if __name__ == "__main__":
    main()
