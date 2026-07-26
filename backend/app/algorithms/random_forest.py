"""Random forest, adding one tree per frame so the ensemble effect is visible."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
    id="random_forest",
    name="Random Forest",
    task="classification",
    tagline="Average hundreds of deliberately mediocre trees into one good model.",
    description=[
        "A random forest grows many decision trees and lets them vote. Two sources of randomness keep "
        "the trees from being identical: each tree is trained on a <strong>bootstrap sample</strong> "
        "(a random resample of the data), and at each split it may only consider a random subset of the "
        "features.",
        "Individually the trees are worse than a single carefully tuned tree — they are noisy and "
        "overfit. Averaged, their errors partly cancel and the ensemble is far more stable. This is "
        "<strong>bagging</strong>, and the animation shows it directly: turn on the single-tree overlay "
        "to compare the newest tree's jagged boundary with the smooth ensemble behind it.",
        "The <strong>out-of-bag score</strong> is a free validation estimate. Each tree only saw about "
        "63% of the data, so the remaining third can be used to score it without a separate holdout set.",
    ],
    watch_for=[
        "The boundary goes from jagged to smooth within the first 10–20 trees, then barely changes.",
        "Toggle the single-tree overlay: every individual tree is a mess, yet the average is clean.",
        "Out-of-bag accuracy climbs fast, then plateaus. More trees never hurt accuracy — only runtime.",
        "Unlike a single tree, a deep forest does not visibly overfit as you add trees.",
    ],
    step_unit="tree",
    step_hint="Each frame adds more trees to the ensemble and re-votes.",
    params=[
        Param(
            name="n_estimators",
            label="Number of trees",
            type="int",
            default=40,
            min=1,
            max=200,
            step=1,
            help="Trees are added one at a time up to this total.",
        ),
        Param(
            name="max_depth",
            label="Maximum tree depth",
            type="int",
            default=6,
            min=1,
            max=14,
            step=1,
            help="Depth limit for every tree in the forest.",
        ),
        Param(
            name="max_features",
            label="Features per split",
            type="select",
            default="sqrt",
            options=[
                {"value": "sqrt", "label": "1 of 2 (random subset)"},
                {"value": "all", "label": "Both features"},
            ],
            help="Restricting features makes trees more different from each other.",
        ),
        Param(
            name="min_samples_leaf",
            label="Minimum samples per leaf",
            type="int",
            default=1,
            min=1,
            max=30,
            step=1,
            help="Higher values give simpler, smoother trees.",
        ),
        Param(
            name="show_single_tree",
            label="Single-tree overlay",
            type="bool",
            default=True,
            help="Also send the newest individual tree's boundary so you can compare it to the ensemble.",
        ),
    ],
)


def fit(points, params, grid: Grid, validation: float = 0.0) -> FitResult:
    data = prepare_labelled(points, min_per_class=2)
    split = make_split(len(data.y), validation, data.y)
    X_train, y_train = data.X[split.train], data.y[split.train]
    X_val, y_val = data.X[split.val], data.y[split.val]
    n_estimators = int(params["n_estimators"])
    max_depth = int(params["max_depth"])
    max_features = None if params["max_features"] == "all" else "sqrt"
    min_samples_leaf = int(params["min_samples_leaf"])
    show_single = bool(params["show_single_tree"])

    model = RandomForestClassifier(
        n_estimators=1,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        oob_score=True,
        warm_start=True,
        random_state=0,
        n_jobs=1,
    )

    schedule = thin(list(range(1, n_estimators + 1)))
    steps: list[Step] = []

    with warnings.catch_warnings():
        # Small forests legitimately leave some points out-of-bag for no tree.
        warnings.simplefilter("ignore", category=UserWarning)
        for count in schedule:
            model.set_params(n_estimators=count)
            model.fit(X_train, y_train)

            train_acc = float((model.predict(X_train) == y_train).mean())
            val_acc = (
                float((model.predict(X_val) == y_val).mean()) if split.active else None
            )
            oob = getattr(model, "oob_score_", None)
            oob = float(oob) if oob is not None and np.isfinite(oob) else None

            proba = model.predict_proba(grid.points)
            labels = np.argmax(proba, axis=1)

            extras = {"n_trees": count, "class_values": data.class_values}
            if show_single:
                single = model.estimators_[-1].predict(grid.points).astype(int)
                extras["single_tree_surface"] = class_surface(single, n_classes=data.n_classes)

            if count == 1:
                note = " One tree on its own: jagged, and it has only seen a bootstrap sample of the data."
            elif count <= 5:
                note = " The vote is already smoothing the worst of the jaggedness."
            else:
                note = ""

            steps.append(
                Step(
                    label=f"{count} tree{'s' if count > 1 else ''}",
                    description=(
                        f"{count} tree{'s' if count > 1 else ''} voting. Training accuracy "
                        f"{train_acc * 100:.1f}%"
                        + (f", validation {val_acc * 100:.1f}%" if val_acc is not None else "")
                        + (f", out-of-bag {oob * 100:.1f}%." if oob is not None else ".")
                        + note
                    ),
                    metrics={
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "oob_accuracy": oob,
                        "n_trees": count,
                    },
                    surface=class_surface(
                        labels,
                        n_classes=data.n_classes,
                        confidence=confidence_from_scores(proba),
                    ),
                    extras=extras,
                )
            )

    importances = model.feature_importances_
    notes = [
        f"Feature importance: x {importances[0] * 100:.0f}%, y {importances[1] * 100:.0f}% — "
        f"how much each axis contributed to reducing impurity across the forest.",
        "Out-of-bag accuracy is computed on the points each tree did not see, so it needs no holdout set.",
    ]
    if show_single:
        notes.append("The faint dashed regions are the newest single tree, for comparison with the ensemble.")
    notes.extend(split_notes(split, steps[-1].metrics))

    final = steps[-1]
    return FitResult(
        task="classification",
        steps=steps,
        metric_labels={
            **METRIC_LABELS,
            "oob_accuracy": "Out-of-bag accuracy",
            "n_trees": "Trees",
        },
        chart_metrics=["train_accuracy", "val_accuracy", "oob_accuracy"],
        summary={
            "Trees": final.metrics["n_trees"],
            "Training accuracy": final.metrics["train_accuracy"],
            "Validation accuracy": final.metrics["val_accuracy"],
            "Out-of-bag accuracy": final.metrics["oob_accuracy"],
        },
        notes=notes,
        split=split,
        extras={
            "class_values": data.class_values,
            "feature_importances": importances.tolist(),
            "has_single_tree": show_single,
        },
    )
