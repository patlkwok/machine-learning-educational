"""Shared plumbing for the algorithm modules.

Every algorithm module exposes:

* ``SPEC`` - an :class:`AlgorithmSpec` describing it and its hyperparameters,
  which the frontend turns into a control panel and an explanation card.
* ``fit(points, params, grid)`` - returns a :class:`FitResult` holding one
  :class:`Step` per frame of the animation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from ..grid import Grid

# Upper bound on animation frames, to keep responses small and playback sane.
MAX_STEPS = 48


class DataError(ValueError):
    """Raised when the supplied points cannot be used by this algorithm.

    The message is shown verbatim to the user, so it should say what to do.
    """


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    type: str  # "float" | "int" | "select" | "bool"
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[dict] | None = None
    help: str = ""
    # "linear" or "log". A log slider spends equal travel on each decade, which
    # is the only usable way to offer a range like 1e-5 to 10. Requires min > 0.
    scale: str = "linear"
    # Carried in the request and validated as usual, but never drawn. For values
    # a user should influence without reading, such as a random seed.
    hidden: bool = False

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "options": self.options,
            "help": self.help,
            "scale": self.scale,
            "hidden": self.hidden,
        }

    def coerce(self, raw: Any) -> Any:
        if raw is None:
            return self.default
        try:
            if self.type == "int":
                value = int(round(float(raw)))
            elif self.type == "float":
                value = float(raw)
            elif self.type == "bool":
                if isinstance(raw, str):
                    return raw.strip().lower() in {"1", "true", "yes", "on"}
                return bool(raw)
            else:  # select
                allowed = [opt["value"] for opt in (self.options or [])]
                value = raw if raw in allowed else self.default
                return value
        except (TypeError, ValueError):
            return self.default

        if self.min is not None:
            value = max(value, type(value)(self.min))
        if self.max is not None:
            value = min(value, type(value)(self.max))
        return value


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    name: str
    task: str  # "regression" | "classification" | "clustering"
    tagline: str
    description: list[str]
    watch_for: list[str]
    params: list[Param]
    step_unit: str = "step"
    step_hint: str = ""
    # Optional button that randomises one hidden parameter, e.g. an
    # initialisation seed: {"param": ..., "label": ..., "help": ...}.
    reseed: dict | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "task": self.task,
            "tagline": self.tagline,
            "description": self.description,
            "watch_for": self.watch_for,
            "params": [p.as_dict() for p in self.params],
            "step_unit": self.step_unit,
            "step_hint": self.step_hint,
            "reseed": self.reseed,
        }

    def resolve(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        raw = raw or {}
        return {p.name: p.coerce(raw.get(p.name)) for p in self.params}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class Step:
    """One frame of the animation."""

    label: str
    description: str
    metrics: dict[str, float] = field(default_factory=dict)
    surface: dict | None = None
    curve: list[list[float]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, index: int) -> dict:
        return {
            "index": index,
            "label": self.label,
            "description": self.description,
            "metrics": {k: _jsonable(v) for k, v in self.metrics.items()},
            "surface": self.surface,
            "curve": self.curve,
            "extras": _jsonable(self.extras),
        }


@dataclass
class FitResult:
    task: str
    steps: list[Step]
    metric_labels: dict[str, str] = field(default_factory=dict)
    chart_metrics: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    split: "Split | None" = None

    def as_dict(self, algorithm: str, grid: Grid) -> dict:
        formats = {
            key: ("percent" if _is_ratio(key) else "number") for key in self.metric_labels
        }
        series = []
        for key in self.chart_metrics:
            values = [step.metrics.get(key) for step in self.steps]
            if all(v is None for v in values):
                continue
            series.append(
                {
                    "key": key,
                    "label": self.metric_labels.get(key, key),
                    "values": [_jsonable(v) for v in values],
                }
            )
        return {
            "algorithm": algorithm,
            "task": self.task,
            "grid": grid.as_dict(),
            "steps": [step.as_dict(i) for i, step in enumerate(self.steps)],
            "metric_labels": self.metric_labels,
            "metric_formats": formats,
            "metric_series": series,
            "summary": _format_summary(self.summary),
            "notes": self.notes,
            "extras": _jsonable(self.extras),
            "split": self.split.as_dict() if self.split is not None else None,
        }


RATIO_WORDS = ("accuracy", "silhouette")


def _is_ratio(key: str) -> bool:
    """True for metrics that read better as a percentage than a raw 0–1 number."""
    lowered = key.lower()
    return any(word in lowered for word in RATIO_WORDS) and "silhouette" not in lowered


def _format_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Render 0–1 accuracies in the summary as percentages."""
    out: dict[str, Any] = {}
    for key, value in summary.items():
        if _is_ratio(key) and isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
            out[key] = fmt_pct(float(value))
        else:
            out[key] = _jsonable(value)
    return out


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays into plain JSON-safe Python values."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return round(value, 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------


@dataclass
class LabelledData:
    X: np.ndarray  # (n, 2)
    y: np.ndarray  # (n,) contiguous 0..k-1
    class_values: list[int]  # original label of each remapped class

    @property
    def n_classes(self) -> int:
        return len(self.class_values)


def _xy_array(points: Sequence[dict]) -> np.ndarray:
    return np.array([[p["x"], p["y"]] for p in points], dtype=float)


def prepare_labelled(points: Sequence[dict], min_per_class: int = 1) -> LabelledData:
    """Validate and pack points for a supervised classifier."""
    if len(points) < 4:
        raise DataError("Add at least 4 points (or generate a dataset) before training.")

    missing = sum(1 for p in points if p.get("label") is None)
    if missing:
        raise DataError(
            "This dataset has no class labels. Pick a classification dataset, "
            "or switch to a classification algorithm and click to add labelled points."
        )

    X = _xy_array(points)
    raw = np.array([int(p["label"]) for p in points], dtype=int)
    class_values = sorted(set(raw.tolist()))

    if len(class_values) < 2:
        raise DataError(
            "Only one class is present. Add points of at least one more class "
            "using the class buttons above the plot."
        )

    lookup = {value: index for index, value in enumerate(class_values)}
    y = np.array([lookup[v] for v in raw], dtype=int)

    counts = np.bincount(y, minlength=len(class_values))
    if counts.min() < min_per_class:
        short = class_values[int(np.argmin(counts))]
        raise DataError(
            f"Class {short} has only {int(counts.min())} point(s); this algorithm "
            f"needs at least {min_per_class} per class."
        )

    return LabelledData(X=X, y=y, class_values=class_values)


def prepare_regression(points: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Pack points for a regressor: horizontal axis is the feature, vertical the target."""
    if len(points) < 3:
        raise DataError("Add at least 3 points (or generate a dataset) before fitting.")
    xy = _xy_array(points)
    if np.ptp(xy[:, 0]) < 1e-9:
        raise DataError(
            "All points share the same x value, so there is nothing to regress. "
            "Spread the points out horizontally."
        )
    return xy[:, :1], xy[:, 1]


# --------------------------------------------------------------------------
# Train / validation split
# --------------------------------------------------------------------------

# Below this many training points a model has nothing to learn from, so the
# validation set is given up rather than the training set starved.
MIN_TRAIN_POINTS = 4


@dataclass(frozen=True)
class SplitSpec:
    """What to hold back, and which partition to draw.

    The seed is separate from the dataset seed on purpose: it lets you redraw
    the partition while the data stays fixed, which is the only way to tell a
    model's quality apart from a lucky split.
    """

    fraction: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class Split:
    """Which points train the model, and which are held back to score it."""

    train: np.ndarray
    val: np.ndarray
    note: str | None = None

    @property
    def active(self) -> bool:
        return len(self.val) > 0

    def as_dict(self) -> dict:
        return {
            "validation_indices": self.val.tolist(),
            "n_train": int(len(self.train)),
            "n_validation": int(len(self.val)),
        }


def make_split(n: int, spec: "SplitSpec | float", y: np.ndarray | None = None) -> Split:
    """Hold back part of the points, stratified by class where possible.

    `spec` is a :class:`SplitSpec`, or a bare fraction for callers that do not
    care which partition they get.

    Returns an empty validation set — rather than raising — whenever holding
    points back would leave the model untrainable. The caller surfaces the note.
    """
    if not isinstance(spec, SplitSpec):
        spec = SplitSpec(fraction=float(spec))
    fraction, seed = spec.fraction, spec.seed

    indices = np.arange(n)
    empty = np.empty(0, dtype=int)

    if fraction <= 0:
        return Split(indices, empty)

    n_val = min(int(round(n * fraction)), n - MIN_TRAIN_POINTS)
    if n_val < 1:
        return Split(indices, empty, note=(
            f"Too few points to hold any back: at least {MIN_TRAIN_POINTS + 1} are needed "
            f"before a validation split is possible."
        ))

    # Stratifying keeps every class represented on both sides, which matters
    # most exactly when the data is small and the split is large.
    stratify = None
    if y is not None and len(y) == n and np.bincount(y).min() >= 2:
        stratify = y
    try:
        train, val = train_test_split(
            indices, test_size=n_val, random_state=seed, shuffle=True, stratify=stratify
        )
    except ValueError:
        train, val = train_test_split(indices, test_size=n_val, random_state=seed, shuffle=True)

    if y is not None and len(np.unique(y[train])) < 2:
        return Split(indices, empty, note=(
            "A validation split would leave only one class in the training set, so "
            "every point was kept for training."
        ))

    return Split(np.sort(train), np.sort(val))


# Shared so every algorithm names the same quantity the same way; extend with
# `{**METRIC_LABELS, "n_support": "Support vectors"}` for algorithm-specific ones.
METRIC_LABELS = {
    "train_accuracy": "Training accuracy",
    "val_accuracy": "Validation accuracy",
    "train_log_loss": "Training log loss",
    "val_log_loss": "Validation log loss",
    "train_loss": "Training loss",
    "val_loss": "Validation loss",
    "train_mse": "Training MSE",
    "val_mse": "Validation MSE",
    "train_r2": "Training R²",
    "val_r2": "Validation R²",
}

# Gaps smaller than this are noise on a few hundred points, not a lesson.
OVERFIT_ACCURACY_GAP = 0.12
OVERFIT_ERROR_RATIO = 2.0


def split_notes(split: Split, metrics: dict[str, Any] | None = None) -> list[str]:
    """Explain the split, and call out a train/validation gap when there is one."""
    if split.note:
        return [split.note]

    total = len(split.train) + len(split.val)
    if not split.active:
        return [
            "No points are held back, so every number here is measured on the data the model "
            "trained on and none of them can reveal overfitting. Raise the validation split to "
            "hold some points back."
        ]

    notes = [
        f"{len(split.val)} of {total} points ({len(split.val) / total * 100:.0f}%) were held out "
        f"of training and scored separately; the model never saw them. Held-out points are "
        f"ringed in the plot."
    ]

    metrics = metrics or {}
    train_acc, val_acc = metrics.get("train_accuracy"), metrics.get("val_accuracy")
    if train_acc is not None and val_acc is not None and train_acc - val_acc > OVERFIT_ACCURACY_GAP:
        notes.append(
            f"Training accuracy beats validation accuracy by "
            f"{(train_acc - val_acc) * 100:.0f} percentage points. That gap is overfitting: the "
            f"model is fitting noise it will never see again."
        )

    train_err, val_err = metrics.get("train_mse"), metrics.get("val_mse")
    if train_err and val_err and val_err > train_err * OVERFIT_ERROR_RATIO:
        notes.append(
            f"Validation error is {val_err / train_err:.1f}x the training error, which means the "
            f"curve is chasing individual points rather than the underlying trend."
        )
    return notes


def prepare_unlabelled(points: Sequence[dict], min_points: int = 3) -> np.ndarray:
    if len(points) < min_points:
        raise DataError(f"Add at least {min_points} points (or generate a dataset) before clustering.")
    return _xy_array(points)


# --------------------------------------------------------------------------
# Misc helpers
# --------------------------------------------------------------------------


def thin(values: Sequence[Any], max_items: int = MAX_STEPS) -> list[Any]:
    """Evenly subsample a sequence, always keeping the first and last entry."""
    values = list(values)
    if len(values) <= max_items:
        return values
    idx = np.unique(np.linspace(0, len(values) - 1, max_items).round().astype(int))
    return [values[i] for i in idx]


def geometric_schedule(limit: int, count: int) -> list[int]:
    """Increasing integers from 1..limit, dense at the start, sparse at the end."""
    if limit <= count:
        return list(range(1, limit + 1))
    raw = np.geomspace(1, limit, count)
    return sorted(set(int(round(v)) for v in raw)) or [limit]


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"

