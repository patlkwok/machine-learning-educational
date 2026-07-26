"""A small multi-layer perceptron, trained one epoch per frame.

One or two hidden layers of up to 100 neurons each, over two input features.
Even the largest option trains and renders every animation frame in about a
second on a laptop CPU, which is the only reason a neural network earns a place
in this playground.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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

MAX_NEURONS = 100

MAX_DIAGRAM_WEIGHTS = 900

SPEC = AlgorithmSpec(
    id="mlp",
    name="Neural Network (MLP)",
    task="classification",
    tagline="Stack a few layers of neurons and let backpropagation bend the boundary anywhere.",
    description=[
        "A multi-layer perceptron feeds the two coordinates through one or more hidden layers of "
        "neurons. Each neuron computes a weighted sum and passes it through a non-linear activation; "
        "stacking them lets the network compose simple pieces into an arbitrarily shaped boundary.",
        "Training is <strong>backpropagation</strong>: run the data forward, measure the loss, then "
        "push the error gradients backwards to nudge every weight. Each frame here is one epoch.",
        "You choose the shape: one or two hidden layers of up to 100 neurons each. Even the widest "
        "option is tiny by modern standards and trains in about a second, but the lesson is the same "
        "as for a large network: the boundary starts near-linear and progressively acquires curvature "
        "as the hidden units specialise. Set the second layer to 0 for a single-layer network.",
    ],
    watch_for=[
        "Early epochs look almost like logistic regression. The curvature appears later, as hidden units differentiate.",
        "On spirals, a single narrow hidden layer stalls; widen it or set layer 2 above 0 and it suddenly untangles.",
        "Compare 2 neurons against 100 in layer 1: too few and the network physically cannot bend the boundary enough.",
        "Switch activation from ReLU to tanh: ReLU gives piecewise-linear, faceted boundaries, tanh gives smooth ones.",
        "Too high a learning rate makes the loss curve bounce instead of descend.",
        "The diagram shades each connection by weight — red for negative, blue for positive — so you can see weights grow apart from zero.",
    ],
    step_unit="epoch",
    step_hint="Each frame is one epoch of backpropagation over the whole dataset.",
    params=[
        Param(
            name="layer1",
            label="Hidden layer 1 neurons",
            type="int",
            default=16,
            min=1,
            max=MAX_NEURONS,
            step=1,
            help="Width of the first hidden layer. Wider layers can carve more pieces out of the plane.",
        ),
        Param(
            name="layer2",
            label="Hidden layer 2 neurons",
            type="int",
            default=16,
            min=0,
            max=MAX_NEURONS,
            step=1,
            help="Width of the second hidden layer. Set to 0 for a single-hidden-layer network.",
        ),
        Param(
            name="activation",
            label="Activation",
            type="select",
            default="relu",
            options=[
                {"value": "relu", "label": "ReLU (piecewise linear)"},
                {"value": "tanh", "label": "tanh (smooth)"},
                {"value": "logistic", "label": "Sigmoid"},
            ],
            help="The non-linearity applied at each hidden neuron.",
        ),
        Param(
            name="epochs",
            label="Epochs",
            type="int",
            default=60,
            min=1,
            max=100,
            step=1,
            help="Passes over the training data.",
        ),
        Param(
            name="learning_rate",
            label="Learning rate",
            type="float",
            default=0.02,
            min=1e-5,
            max=10.0,
            scale="log",
            help="Adam's step size. Too high and the loss bounces around.",
        ),
        Param(
            name="alpha",
            label="L2 penalty (α)",
            type="float",
            default=0.0001,
            min=0.0,
            max=1.0,
            step=0.0001,
            help="Weight decay. Higher values give smoother, simpler boundaries.",
        ),
    ],
)


def _network_diagram(model: MLPClassifier) -> dict | None:
    total = sum(w.size for w in model.coefs_)
    if total > MAX_DIAGRAM_WEIGHTS:
        return None
    layers = [int(model.coefs_[0].shape[0])] + [int(w.shape[1]) for w in model.coefs_]
    scale = max(float(max(np.abs(w).max() for w in model.coefs_)), 1e-9)
    return {
        "layers": layers,
        "scale": round(scale, 4),
        "weights": [np.round(w / scale, 3).tolist() for w in model.coefs_],
    }


def fit(points, params, grid: Grid, validation: float = 0.0) -> FitResult:
    data = prepare_labelled(points)
    # layer2 = 0 means "one hidden layer"; scikit-learn rejects a zero-width layer.
    layer1 = int(np.clip(params["layer1"], 1, MAX_NEURONS))
    layer2 = int(np.clip(params["layer2"], 0, MAX_NEURONS))
    hidden = (layer1, layer2) if layer2 > 0 else (layer1,)
    activation = params["activation"]
    epochs = int(params["epochs"])
    lr = float(params["learning_rate"])
    alpha = float(params["alpha"])

    split = make_split(len(data.y), validation, data.y)
    X_train, y_train = data.X[split.train], data.y[split.train]
    X_val, y_val = data.X[split.val], data.y[split.val]

    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    Xs_val = scaler.transform(X_val) if split.active else None
    grid_s = scaler.transform(grid.points)

    model = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation=activation,
        solver="adam",
        learning_rate_init=lr,
        alpha=alpha,
        max_iter=1,
        warm_start=True,
        random_state=0,
        batch_size=min(32, max(2, len(y_train))),
    )

    snapshots = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for epoch in range(1, epochs + 1):
            model.fit(Xs, y_train)
            snapshots.append(
                (
                    epoch,
                    float(model.loss_),
                    float((model.predict(Xs) == y_train).mean()),
                    float((model.predict(Xs_val) == y_val).mean()) if split.active else None,
                    [w.copy() for w in model.coefs_],
                    [b.copy() for b in model.intercepts_],
                )
            )

        # scikit-learn uses a single logistic output unit for two classes and
        # one unit per class beyond that; report what it actually built.
        n_outputs = int(model.coefs_[-1].shape[1])
        architecture = f"2 → {' → '.join(str(h) for h in hidden)} → {n_outputs}"

        steps: list[Step] = []
        for epoch, loss, acc, val_acc, coefs, intercepts in thin(snapshots):
            model.coefs_, model.intercepts_ = coefs, intercepts
            proba = model.predict_proba(grid_s)
            labels = np.argmax(proba, axis=1)
            steps.append(
                Step(
                    label=f"Epoch {epoch}",
                    description=(
                        f"Epoch {epoch}: loss {loss:.4f}, training accuracy {acc * 100:.1f}%"
                        + (f", validation {val_acc * 100:.1f}%. " if val_acc is not None else ". ")
                        + f"Architecture {architecture}."
                    ),
                    metrics={
                        "train_loss": loss,
                        "train_accuracy": acc,
                        "val_accuracy": val_acc,
                        "epoch": epoch,
                    },
                    surface=class_surface(
                        labels,
                        n_classes=data.n_classes,
                        confidence=confidence_from_scores(proba),
                    ),
                    extras={
                        "network": _network_diagram(model),
                        "class_values": data.class_values,
                    },
                )
            )

    n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    notes = [
        f"{n_params} trainable parameters — small enough to train instantly, large enough to bend the boundary.",
        "Inputs are standardised before training; neural networks converge much faster on scaled features.",
    ]
    final = steps[-1]
    if len(steps) >= 3 and final.metrics["train_loss"] > steps[-3].metrics["train_loss"]:
        notes.append("The loss went up over the last few epochs — try a lower learning rate.")
    elif final.metrics["train_accuracy"] < 0.85:
        notes.append("Still underfitting: give it more epochs, a bigger hidden layer, or a higher learning rate.")
    notes.extend(split_notes(split, final.metrics))

    return FitResult(
        task="classification",
        steps=steps,
        metric_labels={**METRIC_LABELS, "epoch": "Epoch"},
        chart_metrics=["train_loss", "train_accuracy", "val_accuracy"],
        summary={
            "Architecture": architecture,
            "Parameters": n_params,
            "Final loss": final.metrics["train_loss"],
            "Training accuracy": final.metrics["train_accuracy"],
            "Validation accuracy": final.metrics["val_accuracy"],
        },
        notes=notes,
        split=split,
        extras={"class_values": data.class_values, "hidden": list(hidden)},
    )
