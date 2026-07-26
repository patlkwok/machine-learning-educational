"""Support vector machine, animated by capping the solver's iteration count."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..grid import Grid, class_surface, margin_confidence
from .base import (
    METRIC_LABELS,
    AlgorithmSpec,
    FitResult,
    Param,
    Split,
    Step,
    make_split,
    prepare_labelled,
    split_notes,
    geometric_schedule,
)

# Dataset size at which the requested frame count is delivered in full; above
# it the budget is scaled down in proportion. Only the multiclass path needs
# this — see the note in fit().
FRAME_REFERENCE_POINTS = 600

SPEC = AlgorithmSpec(
    id="svm",
    name="Support Vector Machine",
    task="classification",
    tagline="Find the boundary with the widest possible margin — and bend it with a kernel.",
    description=[
        "An SVM does not just look for <em>a</em> separating boundary, it looks for the one with the "
        "widest <strong>margin</strong>: the largest empty corridor between the classes. Only the "
        "points on or inside that corridor matter. Those are the <strong>support vectors</strong>, "
        "circled in the plot; delete every other point and you would get exactly the same boundary.",
        "<code>C</code> controls how much violating the margin costs. Small <code>C</code> means a wide, "
        "soft corridor that tolerates mistakes; large <code>C</code> means a narrow, hard corridor that "
        "contorts to classify every training point.",
        "The <strong>kernel trick</strong> lets the same algorithm draw curved boundaries: an RBF kernel "
        "measures similarity by distance, so the boundary can wrap around clusters. This animation caps "
        "the number of solver iterations, so you can watch the optimiser tighten the margin frame by frame.",
    ],
    watch_for=[
        "The number of support vectors drops as the solver converges — it is discarding points that turn out not to matter.",
        "Switch to the RBF kernel on the circles or moons data: a boundary a straight line could never draw.",
        "Push gamma very high and each point grows its own little island — that is RBF overfitting.",
        "Compare C = 0.1 with C = 100 on noisy blobs: soft margin ignores the noise, hard margin chases it.",
    ],
    step_unit="iteration",
    step_hint="Each frame gives the underlying solver a few more iterations to work with.",
    params=[
        Param(
            name="kernel",
            label="Kernel",
            type="select",
            default="rbf",
            options=[
                {"value": "linear", "label": "Linear (straight boundary)"},
                {"value": "rbf", "label": "RBF (curved, distance based)"},
                {"value": "poly", "label": "Polynomial"},
            ],
            help="How similarity between points is measured.",
        ),
        Param(
            name="C",
            label="C (margin hardness)",
            type="float",
            default=1.0,
            min=0.01,
            max=100.0,
            step=0.01,
            help="Cost of misclassifying a training point. Large C = narrow, unforgiving margin.",
        ),
        Param(
            name="gamma",
            label="Gamma (RBF / poly reach)",
            type="float",
            default=0.5,
            min=0.01,
            max=20.0,
            step=0.01,
            help="How far a single point's influence reaches. High gamma = tight, local bubbles.",
        ),
        Param(
            name="degree",
            label="Polynomial degree",
            type="int",
            default=3,
            min=2,
            max=6,
            step=1,
            help="Only used by the polynomial kernel.",
        ),
        Param(
            name="frames",
            label="Animation frames",
            type="int",
            default=16,
            min=4,
            max=40,
            step=1,
            help="How many intermediate solver states to capture.",
        ),
    ],
)


def _margin_lines(model: SVC, scaler: StandardScaler, grid: Grid) -> list[dict]:
    """Boundary and ±1 margin lines for a linear kernel, in plot coordinates."""
    if model.kernel != "linear" or model.coef_.shape[0] != 1:
        return []
    w = model.coef_[0] / scaler.scale_
    b = float(model.intercept_[0] - np.dot(model.coef_[0], scaler.mean_ / scaler.scale_))
    vp = grid.viewport
    lines = []
    for level, kind in ((0.0, "boundary"), (1.0, "margin"), (-1.0, "margin")):
        if abs(w[1]) > 1e-9:
            xs = np.array([vp.x_min, vp.x_max])
            ys = (level - b - w[0] * xs) / w[1]
        elif abs(w[0]) > 1e-9:
            ys = np.array([vp.y_min, vp.y_max])
            xs = np.full(2, (level - b) / w[0])
        else:
            continue
        lines.append(
            {"kind": kind, "points": [[float(xs[0]), float(ys[0])], [float(xs[1]), float(ys[1])]]}
        )
    return lines


def _hinge_loss(decision: np.ndarray, y: np.ndarray, n_classes: int) -> float | None:
    if n_classes != 2 or decision.ndim != 1:
        return None
    signed = np.where(y == 1, 1.0, -1.0)
    return float(np.mean(np.maximum(0.0, 1.0 - signed * decision)))


def _kernel_matrix(A, B, kernel: str, gamma: float, degree: int) -> np.ndarray:
    """K(A, B), matching SVC's own kernel definitions exactly.

    ``coef0`` has to be passed for the polynomial kernel: SVC defaults it to
    0.0 while ``polynomial_kernel`` defaults it to 1, and the mismatch is
    silent — it just returns a different function.

    float32 halves a matrix that reaches 74 MB at the sharpest grid, and costs
    nothing visible: the surface is quantised to a byte per cell downstream.
    """
    if kernel == "linear":
        K = linear_kernel(A, B)
    elif kernel == "poly":
        K = polynomial_kernel(A, B, degree=degree, gamma=gamma, coef0=0.0)
    else:
        K = rbf_kernel(A, B, gamma=gamma)
    return K.astype(np.float32, copy=False)


def _decision(K: np.ndarray, model: SVC, n_train: int) -> np.ndarray:
    """``model.decision_function(Z)`` for a binary SVC, from a cached K(Z, X_train).

    The dual coefficients are scattered into a full-length vector — zeros for
    every point that is not a support vector — so the cache needs no per-frame
    column slicing; that fancy-index copy costs far more than the matvec saves.

    alpha takes K's dtype deliberately: a float32 K against a float64 alpha
    makes numpy upcast the whole matrix, and the upcast alone is ~18x the
    matvec.
    """
    alpha = np.zeros(n_train, dtype=K.dtype)
    alpha[model.support_] = model.dual_coef_[0]
    return K @ alpha + model.intercept_[0]


def _labels_from(decision: np.ndarray, model: SVC) -> np.ndarray:
    """The class each decision value votes for — what predict() returns for binary SVC."""
    return model.classes_[(decision > 0).astype(int)]


@dataclass(frozen=True)
class _Scoring:
    """Everything a frame needs to score itself, built once before the loop."""

    Xs: np.ndarray
    y_train: np.ndarray
    Xs_val: np.ndarray | None
    y_val: np.ndarray
    split: Split
    scaler: StandardScaler
    grid_s: np.ndarray
    # K(rows, X_train) for the grid, training and validation rows, or None when
    # the cached decision formula does not apply. See _build_scoring().
    K_grid: np.ndarray | None = None
    K_train: np.ndarray | None = None
    K_val: np.ndarray | None = None

    @property
    def cached(self) -> bool:
        return self.K_grid is not None


def _build_scoring(data, X_train, y_train, X_val, y_val, split, grid, kernel, gamma, degree):
    """Fit the scaler and, where possible, cache the kernel once for all frames.

    Support vectors are always a subset of the training points, so K(rows,
    X_train) is valid for every frame no matter which points that frame's
    solver settled on — and each frame collapses to one matrix-vector product
    against it, instead of re-evaluating the kernel between every support
    vector and every grid cell.

    Only a binary SVC has the single decision function this relies on. Above
    two classes libsvm goes one-vs-one: ``dual_coef_`` becomes
    (n_classes - 1, n_SV) and ``intercept_`` carries one entry per *pair*, so
    the formula does not apply and those frames fall back to asking the model.
    """
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    Xs_val = scaler.transform(X_val) if split.active else None
    grid_s = scaler.transform(grid.points)

    caches: dict[str, np.ndarray | None] = {}
    if data.n_classes == 2:
        caches["K_grid"] = _kernel_matrix(grid_s, Xs, kernel, gamma, degree)
        caches["K_train"] = _kernel_matrix(Xs, Xs, kernel, gamma, degree)
        if split.active:
            caches["K_val"] = _kernel_matrix(Xs_val, Xs, kernel, gamma, degree)

    return _Scoring(
        Xs=Xs,
        y_train=y_train,
        Xs_val=Xs_val,
        y_val=y_val,
        split=split,
        scaler=scaler,
        grid_s=grid_s,
        **caches,
    )


def fit(points, params, grid: Grid, validation: float = 0.0) -> FitResult:
    data = prepare_labelled(points)
    kernel = params["kernel"]
    C = float(params["C"])
    gamma = float(params["gamma"])
    degree = int(params["degree"])
    frames = int(params["frames"])

    split = make_split(len(data.y), validation, data.y)
    X_train, y_train = data.X[split.train], data.y[split.train]
    X_val, y_val = data.X[split.val], data.y[split.val]

    scoring = _build_scoring(
        data, X_train, y_train, X_val, y_val, split, grid, kernel, gamma, degree
    )
    Xs = scoring.Xs

    # Refitting the solver is cheap (~5 ms at 1000 points); what costs is
    # drawing the decision surface, which evaluates the kernel between every
    # support vector and every grid cell. With two classes the cache above
    # reduces that to a matvec and the user gets every frame they asked for.
    # The one-vs-one path still pays it in full, so past a few hundred points
    # trim its budget rather than let a full-size dataset take seconds:
    # consecutive solver checkpoints look near-identical at that scale anyway.
    if scoring.cached:
        frame_budget = frames
    else:
        frame_budget = int(
            np.clip(frames * FRAME_REFERENCE_POINTS / max(len(y_train), 1), 6, frames)
        )

    def make(max_iter: int) -> SVC:
        return SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            max_iter=max_iter,
            cache_size=200,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        converged = make(-1).fit(Xs, y_train)
        n_iter = getattr(converged, "n_iter_", None)
        total_iters = int(np.max(n_iter)) if n_iter is not None else 200
        total_iters = int(np.clip(total_iters, 2, 5000))
        schedule = geometric_schedule(total_iters, frame_budget)

        steps: list[Step] = []
        for iteration in schedule:
            model = make(iteration).fit(Xs, y_train)
            steps.append(_step(model, data, grid, iteration, total_iters, False, scoring))
        steps.append(_step(converged, data, grid, total_iters, total_iters, True, scoring))

    n_sv = int(converged.support_.shape[0])
    notes = [
        f"{n_sv} of {len(y_train)} training points ended up as support vectors "
        f"({n_sv / len(y_train) * 100:.0f}% of them define the entire boundary)."
    ]
    if kernel == "linear":
        notes.append("The dashed lines are the ±1 margins; the solid line is the decision boundary.")
    else:
        notes.append(
            "With a non-linear kernel the margin lives in a higher-dimensional space, so only the "
            "boundary itself is drawn here."
        )
    if n_sv > 0.8 * len(y_train):
        notes.append(
            "Almost every point is a support vector, which usually means C or gamma is too small "
            "for this data — the model is barely committing to a boundary."
        )
    if frame_budget < frames:
        notes.append(
            f"Showing {frame_budget} of the {frames} requested frames: with more than two classes "
            f"every frame has to re-measure the kernel across the whole plot for all "
            f"{len(y_train)} training points, and neighbouring frames are nearly identical on a "
            f"dataset this size."
        )
    notes.extend(split_notes(split, steps[-1].metrics))

    final = steps[-1]
    return FitResult(
        task="classification",
        steps=steps,
        metric_labels={
            **METRIC_LABELS,
            "n_support": "Support vectors",
            "hinge_loss": "Hinge loss",
        },
        chart_metrics=["train_accuracy", "val_accuracy", "n_support"],
        summary={
            "Training accuracy": final.metrics["train_accuracy"],
            "Validation accuracy": final.metrics["val_accuracy"],
            "Support vectors": final.metrics["n_support"],
            "Kernel": kernel,
            "Solver iterations": total_iters,
        },
        notes=notes,
        split=split,
        extras={"class_values": data.class_values, "kernel": kernel},
    )


def _step(model, data, grid, iteration, total, final, s: _Scoring) -> Step:
    y_train, y_val, split = s.y_train, s.y_val, s.split
    if s.cached:
        n_train = len(y_train)
        decision = _decision(s.K_grid, model, n_train)
        labels = _labels_from(decision, model)
        train_decision = _decision(s.K_train, model, n_train)
        acc = float((_labels_from(train_decision, model) == y_train).mean())
        val_acc = (
            float(
                (_labels_from(_decision(s.K_val, model, n_train), model) == y_val).mean()
            )
            if split.active
            else None
        )
        hinge = _hinge_loss(train_decision, y_train, data.n_classes)
    else:
        decision = model.decision_function(s.grid_s)
        labels = model.predict(s.grid_s)
        acc = float((model.predict(s.Xs) == y_train).mean())
        val_acc = float((model.predict(s.Xs_val) == y_val).mean()) if split.active else None
        # Only two classes have a hinge loss, and those always take the cached
        # path above — so there is nothing here to spend a kernel walk on.
        hinge = None
    n_sv = int(model.support_.shape[0])

    label = "Converged" if final else f"Iteration {iteration}"
    if final:
        description = (
            f"Solver converged after {total} iterations: training accuracy {acc * 100:.1f}%"
            + (f", validation {val_acc * 100:.1f}%." if val_acc is not None else ".")
            + " The circled points are the only ones that matter."
        )
    else:
        description = (
            f"Solver capped at {iteration} of {total} iterations: training accuracy "
            f"{acc * 100:.1f}%, {n_sv} support vectors so far."
        )

    return Step(
        label=label,
        description=description,
        metrics={
            "train_accuracy": acc,
            "val_accuracy": val_acc,
            "n_support": n_sv,
            "hinge_loss": hinge,
            "iteration": iteration,
        },
        surface=class_surface(
            labels, n_classes=data.n_classes, confidence=margin_confidence(decision)
        ),
        extras={
            # model.support_ indexes the training subset; map back so the
            # frontend can ring the right points in the caller's full list.
            "support_indices": split.train[model.support_].tolist(),
            "margin_lines": _margin_lines(model, s.scaler, grid),
            "class_values": data.class_values,
        },
    )
