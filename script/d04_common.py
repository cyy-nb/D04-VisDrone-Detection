"""Shared utilities for the D04 VisDrone project.

The historical scripts assumed a hard-coded ``F:\\D04`` layout.  New code should
resolve that layout through :func:`project_path` so the same commands work on
other machines by setting the ``D04_HOME`` environment variable.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

CLASS_NAMES: tuple[str, ...] = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
DEFAULT_D04_HOME = Path(os.environ.get("D04_HOME", r"F:\D04")).expanduser()


def project_root(root: str | Path | None = None) -> Path:
    """Return the project data root, allowing ``D04_HOME`` and callers to override it."""

    return Path(root).expanduser() if root is not None else DEFAULT_D04_HOME


def project_path(*parts: str | Path, root: str | Path | None = None) -> Path:
    """Build a path below the configured D04 project root."""

    return project_root(root).joinpath(*parts)


def image_files(image_dir: str | Path) -> list[Path]:
    """Return supported image files in deterministic, case-insensitive order."""

    directory = Path(image_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def read_yolo_label(label_path: str | Path, width: int, height: int) -> list[dict[str, Any]]:
    """Read one YOLO label file and return validated pixel-space boxes.

    Labels are intentionally validated instead of silently skipped.  A missing
    label, malformed row, non-finite value, invalid class or invalid normalized
    coordinate indicates a broken dataset and should be fixed at conversion time.
    """

    path = Path(label_path)
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    objects: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{path}:{line_number}: expected 5 columns, got {len(parts)}")
            try:
                class_id, x_center, y_center, box_width, box_height = map(float, parts)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric label value") from exc
            values = (class_id, x_center, y_center, box_width, box_height)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_number}: non-finite label value")
            if not class_id.is_integer() or not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"{path}:{line_number}: invalid class id {class_id!r}")
            if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
                raise ValueError(f"{path}:{line_number}: center must be inside [0, 1]")
            if not 0 < box_width <= 1 or not 0 < box_height <= 1:
                raise ValueError(f"{path}:{line_number}: box size must be inside (0, 1]")

            x = (x_center - box_width / 2) * width
            y = (y_center - box_height / 2) * height
            pixel_width = box_width * width
            pixel_height = box_height * height
            objects.append(
                {
                    "class_id": int(class_id),
                    "bbox": [x, y, pixel_width, pixel_height],
                    "area": pixel_width * pixel_height,
                }
            )
    return objects


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text atomically on the same filesystem and return the final path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return target


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Serialize JSON deterministically and replace the destination atomically."""

    text = json.dumps(payload, ensure_ascii=False, indent=indent) + "\n"
    return atomic_write_text(path, text)


def require_files(paths: Iterable[str | Path]) -> None:
    """Raise one useful error listing all missing input files."""

    missing = [str(Path(path)) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))


def format_metric(value: float) -> str:
    """Format a COCO metric while preserving the conventional ``-1`` sentinel."""

    return f"{value:.6f}"


def coerce_sequence(value: Sequence[Any] | None) -> list[Any]:
    """Normalize optional Ultralytics-like sequences for straightforward iteration."""

    return [] if value is None else list(value)
