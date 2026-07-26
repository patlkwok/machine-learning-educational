"""Gaussian naive Bayes, fed the data in chunks so the fit builds up on screen."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB

from ..grid import Grid, class_surface, confidence_from_scores
from .base import (
    METRIC_LABELS,
    AlgorithmSpec,
    FitResult,
    Param,
    Step,
    make_split,
    prepare_labelled,
    split_notes,
)

SPEC = AlgorithmSpec(
    id="naive_bayes",
    name="Gaussian Naive Bayes",
    task="classification",
    tagline="Model each class as a bell curve, then apply Bayes' rule.",
    description=[
        "Naive Bayes fits one Gaussian per class per feature and combines them with Bayes' rule: "
        "<code>P(class | x) ∝ P(class) · P(x₁ | class) · P(x₂ | class)</code>. The "
        "<strong>naive</strong> part is that product — it assumes the features are independent given "
        "the class, which is almost never true and works surprisingly often anyway.",
        "Because the features are assumed independent, the fitted Gaussians are always "
        "<em>axis-aligned</em>. The dashed ellipses are the two-standard-deviation contours; you will "
        "never see one tilt, no matter how diagonal the data is. That is the model's blind spot drawn "
        "on screen.",
        "There is no iterative training to watch, so this animation feeds the data in a few points at "
        "a time using <code>partial_fit</code>. It shows how quickly the estimates settle: naive Bayes "
        "needs remarkably little data to get roughly the right answer.",
    ],
    watch_for=[
        "The ellipses stay axis-aligned even on the stretched-blobs dataset — that is the independence assumption failing visibly.",
        "The boundary is almost final after two or three chunks. Naive Bayes is very data-efficient.",
        "Between two classes with equal variance the boundary is straight; with unequal variances it curves.",
        "On concentric circles it fails badly: one class surrounds the other, which no single bell curve can express.",
    ],
    step_unit="chunk",
    step_hint="Each frame adds another slice of the training data to the running estimates.",
    params=[
        Param(
            name="chunks",
            label="Data chunks",
            type="int",
            default=10,
            min=2,
            max=30,
            step=1,
            help="How many pieces the training data is fed in.",
        ),
        Param(
            name="var_smoothing",
            label="Variance smoothing",
            type="float",
            default=1e-3,
            min=1e-6,
            max=1.0,
            step=1e-4,
            help="A floor added to every variance, which widens the bells and stops degenerate fits.",
        ),
        Param(
            name="equal_priors",
            label="Equal class priors",
            type="bool",
            default=False,
            help="Ignore how common each class is and treat them as equally likely.",
        ),
    ],
)

SIGMA = 2.0


def _ellipses(model: GaussianNB, class_values: list[int]) -> list[dict]:
    out = []
    for index, (mean, var) in enumerate(zip(model.theta_, model.var_)):
        out.append(
            {
                "class_index": index,
                "class_value": class_values[index],
                "cx": float(mean[0]),
                "cy": float(mean[1]),
                "rx": float(SIGMA * np.sqrt(max(var[0], 1e-12))),
                "ry": float(SIGMA * np.sqrt(max(var[1], 1e-12))),
            }
        )
    return out


def fit(points, params, grid: Grid, validation: float = 0.0) -> FitResult:
    data = prepare_labelled(points, min_per_class=2)
    n_chunks = int(params["chunks"])
    var_smoothing = float(params["var_smoothing"])
    equal_priors = bool(params["equal_priors"])

    split = make_split(len(data.y), validation, data.y)
    X_train, y_train = data.X[split.train], data.y[split.train]
    X_val, y_val = data.X[split.val], data.y[split.val]

    classes = np.arange(data.n_classes)
    priors = np.full(data.n_classes, 1.0 / data.n_classes) if equal_priors else None
    model = GaussianNB(var_smoothing=var_smoothing, priors=priors)

    # Shuffle the training points, then make sure the first chunk contains every
    # class so the very first partial_fit is well defined.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(y_train))
    first = [int(np.flatnonzero(y_train[order] == c)[0]) for c in classes]
    rest = [i for i in range(len(order)) if i not in set(first)]
    order = order[np.array(first + rest)]

    X, y = X_train[order], y_train[order]
    n_chunks = int(min(n_chunks, max(1, len(y) - len(first) + 1)))
    bounds = np.linspace(len(first), len(y), n_chunks + 1).round().astype(int)
    bounds[0] = max(len(first), 1)

    steps: list[Step] = []
    seen = 0
    for chunk in range(n_chunks):
        start = 0 if chunk == 0 else bounds[chunk]
        end = bounds[chunk + 1]
        if end <= start and chunk > 0:
            continue
        model.partial_fit(X[start:end], y[start:end], classes=classes if chunk == 0 else None)
        seen = end

        proba_train = model.predict_proba(X_train)
        acc = float((np.argmax(proba_train, axis=1) == y_train).mean())
        loss = float(log_loss(y_train, proba_train, labels=classes))

        val_acc = val_loss = None
        if split.active:
            proba_val = model.predict_proba(X_val)
            val_acc = float((np.argmax(proba_val, axis=1) == y_val).mean())
            val_loss = float(log_loss(y_val, proba_val, labels=classes))

        proba_grid = model.predict_proba(grid.points)
        labels = np.argmax(proba_grid, axis=1)

        steps.append(
            Step(
                label=f"{seen}/{len(y)} points",
                description=(
                    f"Estimates updated from {seen} of {len(y)} training points. Training accuracy "
                    f"{acc * 100:.1f}%"
                    + (f", validation {val_acc * 100:.1f}%." if val_acc is not None else ".")
                    + " The dashed ellipses are each class's fitted 2σ contour."
                ),
                metrics={
                    "train_accuracy": acc,
                    "val_accuracy": val_acc,
                    "train_log_loss": loss,
                    "val_log_loss": val_loss,
                    "points_seen": seen,
                },
                surface=class_surface(
                    labels, n_classes=data.n_classes, confidence=confidence_from_scores(proba_grid)
                ),
                extras={
                    "ellipses": _ellipses(model, data.class_values),
                    "class_values": data.class_values,
                    "priors": model.class_prior_.tolist(),
                },
            )
        )

    notes = [
        "Every ellipse is axis-aligned because the model assumes x and y are independent within a class.",
    ]
    if equal_priors:
        notes.append("Priors are forced equal, so class sizes do not influence the boundary.")
    notes.extend(split_notes(split, steps[-1].metrics))

    final = steps[-1]
    return FitResult(
        task="classification",
        steps=steps,
        metric_labels={**METRIC_LABELS, "points_seen": "Points seen"},
        chart_metrics=["train_accuracy", "val_accuracy", "train_log_loss", "val_log_loss"],
        summary={
            "Training accuracy": final.metrics["train_accuracy"],
            "Validation accuracy": final.metrics["val_accuracy"],
            "Log loss": final.metrics["train_log_loss"],
            "Classes": data.n_classes,
        },
        notes=notes,
        split=split,
        extras={"class_values": data.class_values},
    )
