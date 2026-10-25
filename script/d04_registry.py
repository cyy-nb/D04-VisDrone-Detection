"""Register project-specific Ultralytics modules before model loading."""

from __future__ import annotations


def ensure_d04_registered():
    """Expose :class:`D04EnhanceStem` to Ultralytics YAML parsing and checkpoints.

    Custom modules embedded in ``best.pt`` are resolved from
    ``ultralytics.nn.tasks``.  Registering explicitly makes training and
    evaluation reproducible without requiring a sitecustomize patch.
    """

    try:
        from ultralytics.nn import tasks
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required to load D04 models. "
            "Install it with: python -m pip install -r requirements-train.txt"
        ) from exc

    try:
        from .d04_enhance import D04EnhanceStem
    except ImportError:
        from d04_enhance import D04EnhanceStem

    existing = getattr(tasks, "D04EnhanceStem", None)
    if existing is not None and existing is not D04EnhanceStem:
        # Do not silently replace an incompatible class already loaded by the
        # environment; the checkpoint may otherwise fail with opaque errors.
        raise RuntimeError(
            "A different D04EnhanceStem is already registered in ultralytics.nn.tasks."
        )
    tasks.D04EnhanceStem = D04EnhanceStem
    return D04EnhanceStem
