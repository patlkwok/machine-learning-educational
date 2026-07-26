"""k-nearest neighbours, sweeping k from 1 upwards."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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
    thin,
)

SPEC = AlgorithmSpec(
    id="knn",
    name="k-Nearest Neighbours",
    task="classification",
    tagline="No training at all — just ask the k closest points what they are.",
    description=[
        "k-NN is the laziest algorithm here: it stores the training set and does nothing else. To "
        "classify a new point it finds the <code>k</code> closest stored points and takes a vote.",
        "Because there is no training loop, this animation sweeps <code>k</code> instead. "
        "<code>k = 1</code> carves the plane into a Voronoi diagram where every training point owns its "
        "own territory — training accuracy is a perfect 100%, and it means nothing. As <code>k</code> "
        "grows the boundary smooths out and small islands get absorbed.",
        "The chart puts training accuracy next to cross-validated accuracy. The gap between them is "
        "exactly how much the model is memorising rather than generalising.",
    ],
    watch_for=[
        "At k = 1, training accuracy is 100% while cross-validated accuracy is much lower. That gap is overfitting.",
        "Isolated points create little colour islands at small k and vanish as k grows.",
        "Push k towards the dataset size and the model degenerates into 'always predict the majority class'.",
        "Switch to distance weighting: nearer neighbours count more, so islands survive to larger k.",
    ],
    step_unit="k",
    step_hint="Each frame re-classifies the plane using one more neighbour.",
    params=[
        Param(
            name="max_k",
            label="Maximum k",
            type="int",
            default=25,
            min=1,
            max=99,
            step=1,
            help="The animation sweeps k = 1 up to this value.",
        ),
        Param(
            name="weights",
            label="Vote weighting",
            type="select",
            default="uniform",
            options=[
                {"value": "uniform", "label": "Uniform (one vote each)"},
                {"value": "distance", "label": "Distance (closer counts more)"},
            ],
            help="Whether all k neighbours count equally.",
        ),
        Param(
            name="metric",
            label="Distance metric",
            type="select",
            default="euclidean",
            options=[
                {"value": "euclidean", "label": "Euclidean (straight line)"},
                {"value": "manhattan", "label": "Manhattan (city block)"},
                {"value": "chebyshev", "label": "Chebyshev (max axis)"},
            ],
            help="How 'close' is measured. Manhattan gives blockier boundaries.",
        ),
    ],
)


def fit(points, params, grid: Grid, validation: float = 0.0) -> FitResult:
    data = prepare_labelled(points)
    weights = params["weights"]
    metric = params["metric"]

    split = make_split(len(data.y), validation, data.y)
    X_train, y_train = data.X[split.train], data.y[split.train]
    X_val, y_val = data.X[split.val], data.y[split.val]

    # k can never exceed the number of points the model actually stores.
    max_k = int(min(int(params["max_k"]), len(y_train)))
    ks = thin(list(range(1, max_k + 1)))

    steps: list[Step] = []
    best = (-1.0, 1)

    for k in ks:
        model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
        model.fit(X_train, y_train)
        train_acc = float((model.predict(X_train) == y_train).mean())

        val_acc = None
        if split.active:
            val_acc = float((model.predict(X_val) == y_val).mean())
            if val_acc > best[0]:
                best = (val_acc, k)

        proba = model.predict_proba(grid.points)
        labels = np.argmax(proba, axis=1)

        if k == 1:
            note = " With k = 1 every training point owns a Voronoi cell, so training accuracy is trivially perfect."
        elif val_acc is not None and train_acc - val_acc > 0.15:
            note = " Training accuracy is well above validation accuracy — still memorising."
        else:
            note = ""

        steps.append(
            Step(
                label=f"k = {k}",
                description=(
                    f"Each point on the plane takes a vote of its {k} nearest neighbour"
                    f"{'s' if k > 1 else ''}. Training accuracy {train_acc * 100:.1f}%"
                    + (f", validation {val_acc * 100:.1f}%." if val_acc is not None else ".")
                    + note
                ),
                metrics={"train_accuracy": train_acc, "val_accuracy": val_acc, "k": k},
                surface=class_surface(
                    labels, n_classes=data.n_classes, confidence=confidence_from_scores(proba)
                ),
                extras={"k": k, "class_values": data.class_values},
            )
        )

    notes = ["k-NN never trains: every frame is the same stored data, queried differently."]
    notes.extend(split_notes(split, steps[-1].metrics))
    if split.active:
        notes.append(f"Best validation accuracy is at k = {best[1]} ({best[0] * 100:.1f}%).")

    final = steps[-1]
    return FitResult(
        task="classification",
        steps=steps,
        metric_labels={**METRIC_LABELS, "k": "k"},
        chart_metrics=["train_accuracy", "val_accuracy"],
        summary={
            "Final k": final.metrics["k"],
            "Training accuracy": final.metrics["train_accuracy"],
            "Validation accuracy": final.metrics["val_accuracy"],
            "Best k": best[1] if split.active else None,
        },
        notes=notes,
        split=split,
        extras={"class_values": data.class_values, "best_k": best[1] if split.active else None},
    )
