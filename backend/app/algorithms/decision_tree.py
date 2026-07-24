"""Decision tree, grown one level of depth per frame."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from ..grid import Grid, class_surface, confidence_from_scores
from .base import AlgorithmSpec, FitResult, Param, Step, prepare_labelled

FEATURE_NAMES = ("x", "y")

SPEC = AlgorithmSpec(
    id="decision_tree",
    name="Decision Tree",
    task="classification",
    tagline="Ask one yes/no question at a time, splitting the plane into rectangles.",
    description=[
        "A decision tree repeatedly asks questions of the form <code>is x &lt; 1.4?</code> — always about "
        "a single feature, always a threshold. Each answer sends the point down one branch. Because "
        "every question involves one axis, the regions are always <strong>axis-aligned rectangles</strong>. "
        "A diagonal boundary has to be approximated as a staircase.",
        "At every node the tree tries all possible splits and keeps the one that most reduces impurity — "
        "<em>Gini</em> or <em>entropy</em>, both measures of how mixed the labels are in a region. "
        "A perfectly pure region has impurity 0 and stops splitting.",
        "Each frame here allows one more level of depth. Depth 1 is a single line across the plot; "
        "by depth 6 or 7 the tree is usually carving out individual noisy points, which is overfitting "
        "you can literally see.",
    ],
    watch_for=[
        "Depth 1 is a single split — the best question you could ask if you only got one.",
        "Watch thin slivers appear at high depth: each one is the tree memorising a single noisy point.",
        "Cross-validated accuracy peaks and then declines while training accuracy marches to 100%.",
        "On the diagonal-ish stretched blobs the boundary is a staircase — trees cannot draw a slanted line.",
    ],
    step_unit="depth",
    step_hint="Each frame lets the tree grow one level deeper.",
    params=[
        Param(
            name="max_depth",
            label="Maximum depth",
            type="int",
            default=8,
            min=1,
            max=14,
            step=1,
            help="The animation grows the tree from depth 1 to this value.",
        ),
        Param(
            name="criterion",
            label="Split criterion",
            type="select",
            default="gini",
            options=[
                {"value": "gini", "label": "Gini impurity"},
                {"value": "entropy", "label": "Entropy (information gain)"},
            ],
            help="How the mixedness of a region is measured.",
        ),
        Param(
            name="min_samples_leaf",
            label="Minimum samples per leaf",
            type="int",
            default=1,
            min=1,
            max=30,
            step=1,
            help="Raising this is the simplest way to stop the tree memorising noise.",
        ),
    ],
)


def _serialize(tree, node: int, depth: int, class_values: list[int]) -> dict:
    counts = tree.value[node][0]
    total = float(counts.sum()) or 1.0
    predicted = int(np.argmax(counts))
    payload = {
        "depth": depth,
        "samples": int(tree.n_node_samples[node]),
        "impurity": round(float(tree.impurity[node]), 4),
        "counts": [round(float(c * tree.weighted_n_node_samples[node] / total), 2) for c in counts],
        "predicted_index": predicted,
        "predicted_class": class_values[predicted],
        "is_leaf": tree.children_left[node] == -1,
    }
    if payload["is_leaf"]:
        return payload
    payload["feature"] = FEATURE_NAMES[int(tree.feature[node])]
    payload["threshold"] = round(float(tree.threshold[node]), 3)
    payload["left"] = _serialize(tree, int(tree.children_left[node]), depth + 1, class_values)
    payload["right"] = _serialize(tree, int(tree.children_right[node]), depth + 1, class_values)
    return payload


def _split_lines(tree, grid: Grid) -> list[dict]:
    """Axis-aligned segments showing where each internal node cuts the plane."""
    vp = grid.viewport
    lines: list[dict] = []

    def walk(node: int, box: tuple[float, float, float, float], depth: int) -> None:
        if tree.children_left[node] == -1:
            return
        x0, x1, y0, y1 = box
        feature = int(tree.feature[node])
        threshold = float(tree.threshold[node])
        if feature == 0:
            threshold = float(np.clip(threshold, x0, x1))
            lines.append({"depth": depth, "points": [[threshold, y0], [threshold, y1]]})
            left_box = (x0, threshold, y0, y1)
            right_box = (threshold, x1, y0, y1)
        else:
            threshold = float(np.clip(threshold, y0, y1))
            lines.append({"depth": depth, "points": [[x0, threshold], [x1, threshold]]})
            left_box = (x0, x1, y0, threshold)
            right_box = (x0, x1, threshold, y1)
        walk(int(tree.children_left[node]), left_box, depth + 1)
        walk(int(tree.children_right[node]), right_box, depth + 1)

    walk(0, (vp.x_min, vp.x_max, vp.y_min, vp.y_max), 0)
    return lines


def fit(points, params, grid: Grid) -> FitResult:
    data = prepare_labelled(points)
    max_depth = int(params["max_depth"])
    criterion = params["criterion"]
    min_samples_leaf = int(params["min_samples_leaf"])

    counts = np.bincount(data.y, minlength=data.n_classes)
    folds = int(min(5, counts.min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0) if folds >= 2 else None

    steps: list[Step] = []
    best = (-1.0, 1)
    previous_leaves = 0

    for depth in range(1, max_depth + 1):
        model = DecisionTreeClassifier(
            max_depth=depth,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
        ).fit(data.X, data.y)

        train_acc = float((model.predict(data.X) == data.y).mean())
        cv_acc = None
        if cv is not None:
            cv_acc = float(
                np.mean(
                    cross_val_score(
                        DecisionTreeClassifier(
                            max_depth=depth,
                            criterion=criterion,
                            min_samples_leaf=min_samples_leaf,
                            random_state=0,
                        ),
                        data.X,
                        data.y,
                        cv=cv,
                    )
                )
            )
            if cv_acc > best[0]:
                best = (cv_acc, depth)

        proba = model.predict_proba(grid.points)
        labels = np.argmax(proba, axis=1)
        n_leaves = int(model.get_n_leaves())
        actual_depth = int(model.get_depth())

        if depth == 1:
            note = " A single question splits the whole plane in two."
        elif n_leaves == previous_leaves:
            note = " The tree stopped growing — every remaining region is already pure or too small to split."
        else:
            note = ""
        previous_leaves = n_leaves

        steps.append(
            Step(
                label=f"Depth {depth}",
                description=(
                    f"Depth {actual_depth}, {n_leaves} leaves. Training accuracy {train_acc * 100:.1f}%"
                    + (f", cross-validated {cv_acc * 100:.1f}%." if cv_acc is not None else ".")
                    + note
                ),
                metrics={
                    "train_accuracy": train_acc,
                    "cv_accuracy": cv_acc,
                    "leaves": n_leaves,
                    "depth": actual_depth,
                },
                surface=class_surface(
                    labels, n_classes=data.n_classes, confidence=confidence_from_scores(proba)
                ),
                extras={
                    "tree": _serialize(model.tree_, 0, 0, data.class_values),
                    "split_lines": _split_lines(model.tree_, grid),
                    "class_values": data.class_values,
                },
            )
        )

    notes = ["Every boundary is a horizontal or vertical cut — a tree can only split on one feature at a time."]
    if cv is not None:
        notes.append(f"Best cross-validated depth: {best[1]} ({best[0] * 100:.1f}%).")
    else:
        notes.append("Cross-validation needs at least 2 points in the smallest class.")

    final = steps[-1]
    return FitResult(
        task="classification",
        steps=steps,
        metric_labels={
            "train_accuracy": "Training accuracy",
            "cv_accuracy": "Cross-validated accuracy",
            "leaves": "Leaf count",
            "depth": "Depth",
        },
        chart_metrics=["train_accuracy", "cv_accuracy"],
        summary={
            "Depth": final.metrics["depth"],
            "Leaves": final.metrics["leaves"],
            "Training accuracy": final.metrics["train_accuracy"],
            "Cross-validated accuracy": final.metrics["cv_accuracy"],
        },
        notes=notes,
        extras={"class_values": data.class_values, "best_depth": best[1] if cv is not None else None},
    )
