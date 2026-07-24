"""Decision-surface grid helpers.

A `Grid` is a regular sampling of the visible plane.  Every classifier /
clusterer is asked to predict on that grid, and the resulting labels are
shipped to the browser as a base64 encoded byte array which the frontend
paints straight into an ImageData buffer.

Index convention: cell `(row, col)` lives at flat index `row * resolution +
col`, where `row = 0` is the *bottom* of the plot (`y_min`) and `col = 0` is
the *left* (`x_min`).  The frontend flips rows when drawing because canvas
pixel coordinates grow downwards.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np

# Keeping the surface small matters: payload size is resolution^2 bytes per
# animation frame, so 56x56 over 40 frames is ~125 kB before base64.
MIN_RESOLUTION = 16
MAX_RESOLUTION = 96
DEFAULT_RESOLUTION = 56

# Number of samples used to draw a regression curve across the x range.
CURVE_SAMPLES = 160


@dataclass(frozen=True)
class Viewport:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def as_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }


@dataclass
class Grid:
    """Sampling of the viewport, plus the flattened coordinates to predict on."""

    viewport: Viewport
    resolution: int
    points: np.ndarray  # shape (resolution**2, 2), row-major, bottom-up

    @classmethod
    def build(cls, viewport: Viewport, resolution: int = DEFAULT_RESOLUTION) -> "Grid":
        resolution = int(np.clip(resolution, MIN_RESOLUTION, MAX_RESOLUTION))
        xs = np.linspace(viewport.x_min, viewport.x_max, resolution)
        ys = np.linspace(viewport.y_min, viewport.y_max, resolution)
        xx, yy = np.meshgrid(xs, ys)  # yy varies along rows -> bottom-up rows
        points = np.column_stack([xx.ravel(), yy.ravel()])
        return cls(viewport=viewport, resolution=resolution, points=points)

    def curve_x(self) -> np.ndarray:
        """x positions used when sampling a regression function."""
        return np.linspace(self.viewport.x_min, self.viewport.x_max, CURVE_SAMPLES)

    def as_dict(self) -> dict:
        payload = self.viewport.as_dict()
        payload["resolution"] = self.resolution
        return payload


def encode_bytes(values: np.ndarray) -> str:
    """Base64-encode an array that is already in the uint8 range."""
    return base64.b64encode(np.ascontiguousarray(values, dtype=np.uint8).tobytes()).decode("ascii")


def encode_classes(labels: np.ndarray) -> str:
    """Encode integer class predictions (0..254) for one grid."""
    labels = np.asarray(labels).ravel()
    return encode_bytes(np.clip(labels, 0, 254))


def encode_confidence(values: np.ndarray) -> str:
    """Encode a 0..1 confidence/probability field as bytes."""
    values = np.clip(np.asarray(values, dtype=float).ravel(), 0.0, 1.0)
    return encode_bytes(np.round(values * 255.0))


def class_surface(
    labels: np.ndarray,
    n_classes: int,
    confidence: np.ndarray | None = None,
) -> dict:
    """Build the JSON payload describing one decision-surface frame."""
    surface = {
        "kind": "classes",
        "n_classes": int(n_classes),
        "classes": encode_classes(labels),
        "confidence": None,
    }
    if confidence is not None:
        surface["confidence"] = encode_confidence(confidence)
    return surface


def confidence_from_scores(scores: np.ndarray) -> np.ndarray:
    """Turn per-class scores into a 0..1 'how sure is the model here' field.

    For two classes this is |p - 0.5| * 2; for more it is the gap between the
    best and the runner-up class.  Both land in [0, 1] and both go to zero on
    the decision boundary, which is exactly where we want the shading to fade.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        return np.clip(np.abs(scores - 0.5) * 2.0, 0.0, 1.0)
    if scores.shape[1] == 1:
        return np.clip(np.abs(scores[:, 0] - 0.5) * 2.0, 0.0, 1.0)
    ordered = np.sort(scores, axis=1)
    return np.clip(ordered[:, -1] - ordered[:, -2], 0.0, 1.0)


def margin_confidence(decision: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Squash an unbounded decision function into a 0..1 confidence field."""
    decision = np.asarray(decision, dtype=float)
    if decision.ndim > 1:
        if decision.shape[1] == 1:
            decision = decision.ravel()
        else:
            ordered = np.sort(decision, axis=1)
            decision = ordered[:, -1] - ordered[:, -2]
    magnitude = np.abs(decision)
    if scale is None:
        scale = float(np.percentile(magnitude, 95)) or 1.0
    return np.clip(magnitude / max(scale, 1e-9), 0.0, 1.0)
