"""A small multi-layer perceptron, trained one epoch per frame.

Deliberately tiny: a couple of hidden layers over two input features trains in
well under a second on a laptop CPU, which is the only reason a neural network
earns a place in this playground.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..grid import Grid, class_surface, confidence_from_scores
from .base import AlgorithmSpec, FitResult, Param, Step, prepare_labelled, thin

ARCHITECTURES = {
    "4": (4,),
    "8": (8,),
    "16": (16,),
    "8,8": (8, 8),
    "16,16": (16, 16),
    "32,16": (32, 16),
}

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
        "This network is deliberately small — a few dozen neurons over two input features — so it "
        "trains in a fraction of a second on any laptop. The lesson is the same as for a large one: "
        "the boundary starts near-linear and progressively acquires curvature as the hidden units "
        "specialise.",
    ],
    watch_for=[
        "Early epochs look almost like logistic regression. The curvature appears later, as hidden units differentiate.",
        "On spirals, a single small hidden layer stalls; add a second layer and it suddenly untangles.",
        "Switch activation from ReLU to tanh: ReLU gives piecewise-linear, faceted boundaries, tanh gives smooth ones.",
        "Too high a learning rate makes the loss curve bounce instead of descend.",
        "The diagram shades each connection by weight — red for negative, blue for positive — so you can see weights grow apart from zero.",
    ],
    step_unit="epoch",
    step_hint="Each frame is one epoch of backpropagation over the whole dataset.",
    params=[
        Param(
            name="hidden",
            label="Hidden layers",
            type="select",
            default="16,16",
            options=[
                {"value": "4", "label": "1 layer x 4 neurons"},
                {"value": "8", "label": "1 layer x 8 neurons"},
                {"value": "16", "label": "1 layer x 16 neurons"},
                {"value": "8,8", "label": "2 layers x 8 neurons"},
                {"value": "16,16", "label": "2 layers x 16 neurons"},
                {"value": "32,16", "label": "2 layers: 32 then 16"},
            ],
            help="More and wider layers can express more complex boundaries.",
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
            max=400,
            step=1,
            help="Passes over the training data.",
        ),
        Param(
            name="learning_rate",
            label="Learning rate",
            type="float",
            default=0.02,
            min=0.0005,
            max=0.5,
            step=0.0005,
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


def fit(points, params, grid: Grid) -> FitResult:
    data = prepare_labelled(points)
    hidden = ARCHITECTURES.get(params["hidden"], (16, 16))
    activation = params["activation"]
    epochs = int(params["epochs"])
    lr = float(params["learning_rate"])
    alpha = float(params["alpha"])

    scaler = StandardScaler().fit(data.X)
    Xs = scaler.transform(data.X)
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
        batch_size=min(32, max(2, len(data.y))),
    )

    snapshots = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for epoch in range(1, epochs + 1):
            model.fit(Xs, data.y)
            snapshots.append(
                (
                    epoch,
                    float(model.loss_),
                    float((model.predict(Xs) == data.y).mean()),
                    [w.copy() for w in model.coefs_],
                    [b.copy() for b in model.intercepts_],
                )
            )

        # scikit-learn uses a single logistic output unit for two classes and
        # one unit per class beyond that; report what it actually built.
        n_outputs = int(model.coefs_[-1].shape[1])
        architecture = f"2 → {' → '.join(str(h) for h in hidden)} → {n_outputs}"

        steps: list[Step] = []
        for epoch, loss, acc, coefs, intercepts in thin(snapshots):
            model.coefs_, model.intercepts_ = coefs, intercepts
            proba = model.predict_proba(grid_s)
            labels = np.argmax(proba, axis=1)
            steps.append(
                Step(
                    label=f"Epoch {epoch}",
                    description=(
                        f"Epoch {epoch}: loss {loss:.4f}, training accuracy {acc * 100:.1f}%. "
                        f"Architecture {architecture}."
                    ),
                    metrics={"loss": loss, "accuracy": acc, "epoch": epoch},
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
    if len(steps) >= 3 and final.metrics["loss"] > steps[-3].metrics["loss"]:
        notes.append("The loss went up over the last few epochs — try a lower learning rate.")
    elif final.metrics["accuracy"] < 0.85:
        notes.append("Still underfitting: give it more epochs, a bigger hidden layer, or a higher learning rate.")

    return FitResult(
        task="classification",
        steps=steps,
        metric_labels={"loss": "Training loss", "accuracy": "Training accuracy", "epoch": "Epoch"},
        chart_metrics=["loss", "accuracy"],
        summary={
            "Architecture": architecture,
            "Parameters": n_params,
            "Final loss": final.metrics["loss"],
            "Accuracy": final.metrics["accuracy"],
        },
        notes=notes,
        extras={"class_values": data.class_values, "hidden": list(hidden)},
    )
