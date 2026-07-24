"""Logistic regression trained epoch by epoch with SGD."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from ..grid import Grid, class_surface, confidence_from_scores
from .base import AlgorithmSpec, FitResult, Param, Step, prepare_labelled, thin

SPEC = AlgorithmSpec(
    id="logistic_regression",
    name="Logistic Regression",
    task="classification",
    tagline="A straight decision boundary, fitted by maximising the likelihood of the labels.",
    description=[
        "Logistic regression scores each point with a linear function <code>z = w·x + b</code> and "
        "squashes it through the sigmoid <code>σ(z) = 1 / (1 + e^(−z))</code> to get a probability. "
        "The decision boundary is where the probability is exactly 0.5, which is always a straight "
        "line in 2D (a hyperplane in general).",
        "Training minimises <strong>log loss</strong> (cross-entropy), which punishes confident wrong "
        "answers far more than hesitant ones. Each frame is one epoch of stochastic gradient descent.",
        "The shading shows the model's confidence: strong colour where it is sure, washed out near the "
        "boundary where it is guessing. With three or more classes scikit-learn fits one line per class "
        "and takes the highest score, so the regions meet at straight seams.",
    ],
    watch_for=[
        "On the two-moons or XOR data the boundary can never fit properly — that is the whole limitation of a linear model.",
        "Watch the shaded band around the boundary narrow as training sharpens the model's confidence.",
        "Crank up L2 regularisation: the boundary stays put but the confidence band widens, because the weights are kept small.",
        "Log loss keeps improving after accuracy has stopped changing — the model is still getting more certain.",
    ],
    step_unit="epoch",
    step_hint="Each frame is one pass of stochastic gradient descent over the data.",
    params=[
        Param(
            name="learning_rate",
            label="Learning rate",
            type="float",
            default=0.05,
            min=0.001,
            max=1.0,
            step=0.001,
            help="Step size for each gradient update.",
        ),
        Param(
            name="epochs",
            label="Epochs",
            type="int",
            default=40,
            min=1,
            max=300,
            step=1,
            help="Number of passes over the training data.",
        ),
        Param(
            name="regularization",
            label="Regularisation",
            type="select",
            default="l2",
            options=[
                {"value": "none", "label": "None"},
                {"value": "l2", "label": "L2 (ridge)"},
                {"value": "l1", "label": "L1 (lasso)"},
            ],
            help="Penalty on large weights, which keeps the boundary from becoming over-confident.",
        ),
        Param(
            name="alpha",
            label="Regularisation strength",
            type="float",
            default=0.001,
            min=0.0,
            max=1.0,
            step=0.0005,
            help="Higher values shrink the weights harder.",
        ),
    ],
)


def _equation(model, scaler: StandardScaler, class_values: list[int]) -> str:
    if len(class_values) != 2:
        return f"{len(class_values)} one-vs-rest linear scores"
    # Undo standardisation so the equation reads in plot coordinates.
    w = model.coef_[0] / scaler.scale_
    b = float(model.intercept_[0] - np.dot(model.coef_[0], scaler.mean_ / scaler.scale_))
    return f"P(class {class_values[1]}) = σ({w[0]:.2f}·x + {w[1]:.2f}·y + {b:.2f})"


def fit(points, params, grid: Grid) -> FitResult:
    data = prepare_labelled(points)
    lr = float(params["learning_rate"])
    epochs = int(params["epochs"])
    penalty = params["regularization"]
    alpha = float(params["alpha"])

    scaler = StandardScaler().fit(data.X)
    Xs = scaler.transform(data.X)
    grid_s = scaler.transform(grid.points)
    classes = np.arange(data.n_classes)

    model = SGDClassifier(
        loss="log_loss",
        penalty=None if penalty == "none" else penalty,
        alpha=alpha if penalty != "none" else 0.0,
        learning_rate="constant",
        eta0=lr,
        shuffle=True,
        random_state=0,
    )

    frames = []
    for epoch in range(1, epochs + 1):
        model.partial_fit(Xs, data.y, classes=classes)
        if not np.isfinite(model.coef_).all():
            break
        proba = model.predict_proba(Xs)
        frames.append(
            (
                epoch,
                float(log_loss(data.y, proba, labels=classes)),
                float((model.predict(Xs) == data.y).mean()),
                model.coef_.copy(),
                model.intercept_.copy(),
                _equation(model, scaler, data.class_values),
            )
        )

    if not frames:
        raise ValueError("Training produced no usable frames; try a smaller learning rate.")

    steps: list[Step] = []
    for epoch, loss, acc, coef, intercept, equation in thin(frames):
        model.coef_, model.intercept_ = coef, intercept
        proba_grid = model.predict_proba(grid_s)
        labels = np.argmax(proba_grid, axis=1)
        steps.append(
            Step(
                label=f"Epoch {epoch}",
                description=(
                    f"After epoch {epoch}: log loss {loss:.4f}, training accuracy {acc * 100:.1f}%. "
                    f"{equation}"
                ),
                metrics={"log_loss": loss, "accuracy": acc},
                surface=class_surface(
                    labels,
                    n_classes=data.n_classes,
                    confidence=confidence_from_scores(proba_grid),
                ),
                extras={"equation": equation, "class_values": data.class_values},
            )
        )

    notes = [
        "The boundary is a straight line by construction — no amount of training makes it bend."
    ]
    final = steps[-1]
    if final.metrics["accuracy"] < 0.9:
        notes.append(
            "Accuracy is stuck well below 100%, which usually means the classes are not linearly "
            "separable. Try an SVM with an RBF kernel, a decision tree, or the neural network."
        )

    return FitResult(
        task="classification",
        steps=steps,
        metric_labels={"log_loss": "Log loss", "accuracy": "Training accuracy"},
        chart_metrics=["log_loss", "accuracy"],
        summary={
            "Accuracy": final.metrics["accuracy"],
            "Log loss": final.metrics["log_loss"],
            "Model": final.extras["equation"],
        },
        notes=notes,
        extras={"class_values": data.class_values},
    )
