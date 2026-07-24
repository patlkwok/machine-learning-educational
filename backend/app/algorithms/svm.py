"""Support vector machine, animated by capping the solver's iteration count."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..grid import Grid, class_surface, margin_confidence
from .base import AlgorithmSpec, FitResult, Param, Step, geometric_schedule, prepare_labelled

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


def fit(points, params, grid: Grid) -> FitResult:
    data = prepare_labelled(points)
    kernel = params["kernel"]
    C = float(params["C"])
    gamma = float(params["gamma"])
    degree = int(params["degree"])
    frames = int(params["frames"])

    scaler = StandardScaler().fit(data.X)
    Xs = scaler.transform(data.X)
    grid_s = scaler.transform(grid.points)

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
        converged = make(-1).fit(Xs, data.y)
        n_iter = getattr(converged, "n_iter_", None)
        total_iters = int(np.max(n_iter)) if n_iter is not None else 200
        total_iters = int(np.clip(total_iters, 2, 5000))
        schedule = geometric_schedule(total_iters, frames)

        steps: list[Step] = []
        for iteration in schedule:
            model = make(iteration).fit(Xs, data.y)
            steps.append(_step(model, scaler, data, grid, grid_s, iteration, total_iters, False))
        steps.append(
            _step(converged, scaler, data, grid, grid_s, total_iters, total_iters, True)
        )

    n_sv = int(converged.support_.shape[0])
    notes = [
        f"{n_sv} of {len(data.y)} points ended up as support vectors "
        f"({n_sv / len(data.y) * 100:.0f}% of the data defines the entire boundary)."
    ]
    if kernel == "linear":
        notes.append("The dashed lines are the ±1 margins; the solid line is the decision boundary.")
    else:
        notes.append(
            "With a non-linear kernel the margin lives in a higher-dimensional space, so only the "
            "boundary itself is drawn here."
        )
    if n_sv > 0.8 * len(data.y):
        notes.append(
            "Almost every point is a support vector, which usually means C or gamma is too small "
            "for this data — the model is barely committing to a boundary."
        )

    final = steps[-1]
    return FitResult(
        task="classification",
        steps=steps,
        metric_labels={
            "accuracy": "Training accuracy",
            "n_support": "Support vectors",
            "hinge_loss": "Hinge loss",
        },
        chart_metrics=["accuracy", "n_support", "hinge_loss"],
        summary={
            "Accuracy": final.metrics["accuracy"],
            "Support vectors": final.metrics["n_support"],
            "Kernel": kernel,
            "Solver iterations": total_iters,
        },
        notes=notes,
        extras={"class_values": data.class_values, "kernel": kernel},
    )


def _step(model, scaler, data, grid, grid_s, iteration, total, final) -> Step:
    decision = model.decision_function(grid_s)
    labels = model.predict(grid_s)
    acc = float((model.predict(scaler.transform(data.X)) == data.y).mean())
    n_sv = int(model.support_.shape[0])
    hinge = _hinge_loss(model.decision_function(scaler.transform(data.X)), data.y, data.n_classes)

    label = "Converged" if final else f"Iteration {iteration}"
    if final:
        description = (
            f"Solver converged after {total} iterations: accuracy {acc * 100:.1f}%, "
            f"{n_sv} support vectors. The circled points are the only ones that matter."
        )
    else:
        description = (
            f"Solver capped at {iteration} of {total} iterations: accuracy {acc * 100:.1f}%, "
            f"{n_sv} support vectors so far."
        )

    return Step(
        label=label,
        description=description,
        metrics={"accuracy": acc, "n_support": n_sv, "hinge_loss": hinge, "iteration": iteration},
        surface=class_surface(
            labels, n_classes=data.n_classes, confidence=margin_confidence(decision)
        ),
        extras={
            # Indices into the caller's point list, so the frontend can ring them directly.
            "support_indices": model.support_.tolist(),
            "margin_lines": _margin_lines(model, scaler, grid),
            "class_values": data.class_values,
        },
    )
