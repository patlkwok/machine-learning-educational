"""Synthetic dataset generators.

Everything lives in a roughly [-5, 5] x [-5, 5] box so the plot never has to
rescale dramatically between datasets.  Classification and clustering sets
return integer labels; regression sets return `None` labels because there the
vertical axis *is* the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

PLOT_RADIUS = 5.0

MIN_SAMPLES = 4
MAX_SAMPLES = 1200


def _rescale(xy: np.ndarray, radius: float = 4.4) -> np.ndarray:
    """Centre a point cloud and scale it to fill a box of the given radius."""
    xy = np.asarray(xy, dtype=float)
    centre = (xy.max(axis=0) + xy.min(axis=0)) / 2.0
    xy = xy - centre
    extent = np.abs(xy).max()
    if extent > 1e-9:
        xy = xy * (radius / extent)
    return xy


def _blobs(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    xy, labels = make_blobs(
        n_samples=n_samples,
        centers=classes,
        cluster_std=0.4 + 2.2 * noise,
        center_box=(-4.0, 4.0),
        random_state=seed,
    )
    return _rescale(xy), labels


def _anisotropic(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    xy, labels = make_blobs(
        n_samples=n_samples,
        centers=classes,
        cluster_std=0.4 + 1.6 * noise,
        center_box=(-4.0, 4.0),
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, np.pi)
    stretch = np.array([[1.0, 0.0], [0.0, 0.32]])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return _rescale(xy @ stretch @ rotation.T), labels


def _moons(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    xy, labels = make_moons(n_samples=n_samples, noise=0.02 + 0.28 * noise, random_state=seed)
    return _rescale(xy), labels


def _circles(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    xy, labels = make_circles(
        n_samples=n_samples,
        noise=0.02 + 0.18 * noise,
        factor=0.45,
        random_state=seed,
    )
    return _rescale(xy), labels


def _spirals(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    per_arm = max(2, n_samples // classes)
    xs, ys, labels = [], [], []
    for arm in range(classes):
        t = np.sqrt(rng.uniform(0.06, 1.0, per_arm)) * 2.6 * np.pi
        offset = 2.0 * np.pi * arm / classes
        radius = t * 0.55
        jitter = rng.normal(scale=0.06 + 0.7 * noise, size=(per_arm, 2))
        xs.append(radius * np.cos(t + offset) + jitter[:, 0])
        ys.append(radius * np.sin(t + offset) + jitter[:, 1])
        labels.append(np.full(per_arm, arm))
    xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    return _rescale(xy), np.concatenate(labels)


def _xor(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    per_quadrant = max(1, n_samples // 4)
    centres = [(-2.2, 2.2), (2.2, 2.2), (-2.2, -2.2), (2.2, -2.2)]
    quadrant_class = [0, 1, 1, 0]
    xs, labels = [], []
    for (cx, cy), cls in zip(centres, quadrant_class):
        pts = rng.normal(loc=(cx, cy), scale=0.35 + 1.1 * noise, size=(per_quadrant, 2))
        xs.append(pts)
        labels.append(np.full(per_quadrant, cls))
    return _rescale(np.vstack(xs)), np.concatenate(labels)


def _uniform(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform positions with random labels: structure-free by construction.

    For clustering it shows what k-means does when there are no clusters; for
    classification the labels carry no signal at all, so any training accuracy
    above chance is pure memorisation.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-4.4, 4.4, size=(n_samples, 2))
    labels = rng.integers(0, classes, size=n_samples)
    # Guarantee every class shows up even in tiny samples.
    for cls in range(min(classes, n_samples)):
        labels[cls] = cls
    return xy, labels


def _regression_x(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(-4.4, 4.4, n_samples))


def _linear_regression(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, None]:
    rng = np.random.default_rng(seed + 1)
    x = _regression_x(n_samples, seed)
    slope = rng.uniform(-1.1, 1.1)
    intercept = rng.uniform(-1.2, 1.2)
    y = slope * x + intercept + rng.normal(scale=0.15 + 2.4 * noise, size=n_samples)
    return np.column_stack([x, y]), None


def _wave_regression(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, None]:
    rng = np.random.default_rng(seed + 2)
    x = _regression_x(n_samples, seed)
    y = 2.6 * np.sin(x * 0.9) + rng.normal(scale=0.1 + 1.8 * noise, size=n_samples)
    return np.column_stack([x, y]), None


def _cubic_regression(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, None]:
    rng = np.random.default_rng(seed + 3)
    x = _regression_x(n_samples, seed)
    y = 0.09 * x**3 - 0.45 * x + rng.normal(scale=0.1 + 1.8 * noise, size=n_samples)
    return np.column_stack([x, y]), None


def _step_regression(n_samples: int, noise: float, seed: int, classes: int) -> tuple[np.ndarray, None]:
    rng = np.random.default_rng(seed + 4)
    x = _regression_x(n_samples, seed)
    y = np.where(x < -1.5, -2.5, np.where(x < 1.5, 0.5, 3.0))
    y = y + rng.normal(scale=0.1 + 1.4 * noise, size=n_samples)
    return np.column_stack([x, y]), None


@dataclass(frozen=True)
class GeneratorSpec:
    id: str
    name: str
    kind: str  # "labelled" (classification / clustering) or "regression"
    description: str
    fn: Callable[[int, float, int, int], tuple[np.ndarray, np.ndarray | None]] = field(repr=False)
    min_classes: int = 2
    max_classes: int = 2

    @property
    def supports_classes(self) -> bool:
        return self.max_classes > self.min_classes

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "supports_classes": self.supports_classes,
            "min_classes": self.min_classes,
            "max_classes": self.max_classes,
        }


GENERATORS: dict[str, GeneratorSpec] = {
    spec.id: spec
    for spec in [
        GeneratorSpec(
            id="blobs",
            name="Gaussian blobs",
            kind="labelled",
            description="Round, well-separated clusters. The easy case that almost every algorithm gets right.",
            fn=_blobs,
            min_classes=2,
            max_classes=6,
        ),
        GeneratorSpec(
            id="anisotropic",
            name="Stretched blobs",
            kind="labelled",
            description="Elongated, rotated clusters. k-means struggles here because it assumes round clusters.",
            fn=_anisotropic,
            min_classes=2,
            max_classes=5,
        ),
        GeneratorSpec(
            id="moons",
            name="Two moons",
            kind="labelled",
            description="Interleaving crescents. Not linearly separable, so linear models top out around 85%.",
            fn=_moons,
        ),
        GeneratorSpec(
            id="circles",
            name="Concentric circles",
            kind="labelled",
            description="A ring inside a ring. Needs a kernel or a non-linear model to separate at all.",
            fn=_circles,
        ),
        GeneratorSpec(
            id="spirals",
            name="Spirals",
            kind="labelled",
            description="Intertwined arms. The hardest built-in set: only flexible models untangle it.",
            fn=_spirals,
            min_classes=2,
            max_classes=4,
        ),
        GeneratorSpec(
            id="xor",
            name="XOR quadrants",
            kind="labelled",
            description="Diagonally opposite quadrants share a class. The classic example a single line cannot solve.",
            fn=_xor,
        ),
        GeneratorSpec(
            id="uniform",
            name="Uniform noise",
            kind="labelled",
            description="No structure at all: random positions, random labels. Shows what clustering does when there are no clusters, and what a classifier does when there is nothing to learn.",
            fn=_uniform,
            min_classes=2,
            max_classes=6,
        ),
        GeneratorSpec(
            id="linear_regression",
            name="Noisy line",
            kind="regression",
            description="A straight relationship plus noise. Linear regression is the right tool.",
            fn=_linear_regression,
        ),
        GeneratorSpec(
            id="wave_regression",
            name="Sine wave",
            kind="regression",
            description="A smooth curve. A straight line underfits badly; polynomial degree 5+ tracks it.",
            fn=_wave_regression,
        ),
        GeneratorSpec(
            id="cubic_regression",
            name="Cubic curve",
            kind="regression",
            description="An S-shaped trend. Degree 3 fits it exactly; higher degrees start chasing noise.",
            fn=_cubic_regression,
        ),
        GeneratorSpec(
            id="step_regression",
            name="Step function",
            kind="regression",
            description="Sharp jumps. No polynomial fits this cleanly, which is the point.",
            fn=_step_regression,
        ),
    ]
}


def generator_specs() -> list[dict]:
    return [spec.as_dict() for spec in GENERATORS.values()]


def generate(
    generator: str,
    n_samples: int = 200,
    noise: float = 0.2,
    seed: int = 0,
    classes: int = 2,
) -> list[dict]:
    """Produce a list of `{x, y, label}` points for the frontend."""
    spec = GENERATORS.get(generator)
    if spec is None:
        raise KeyError(f"Unknown generator '{generator}'")

    n_samples = int(np.clip(n_samples, MIN_SAMPLES, MAX_SAMPLES))
    noise = float(np.clip(noise, 0.0, 1.0))
    classes = int(np.clip(classes, spec.min_classes, spec.max_classes))
    seed = int(seed) % (2**31 - 1)

    xy, labels = spec.fn(n_samples, noise, seed, classes)
    xy = np.asarray(xy, dtype=float)
    xy = np.clip(xy, -PLOT_RADIUS + 0.15, PLOT_RADIUS - 0.15)

    if labels is None:
        return [{"x": float(px), "y": float(py), "label": None} for px, py in xy]

    labels = np.asarray(labels, dtype=int)
    return [
        {"x": float(px), "y": float(py), "label": int(lab)}
        for (px, py), lab in zip(xy, labels)
    ]
