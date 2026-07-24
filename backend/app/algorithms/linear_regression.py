"""Linear regression trained by gradient descent, one epoch per frame."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

from ..grid import Grid
from .base import (
    AlgorithmSpec,
    FitResult,
    Param,
    Step,
    prepare_regression,
    thin,
)

SPEC = AlgorithmSpec(
    id="linear_regression",
    name="Linear Regression",
    task="regression",
    tagline="Fit the straight line that minimises squared error.",
    description=[
        "Linear regression assumes the target is a straight-line function of the input: "
        "<code>ŷ = w·x + b</code>. Training means choosing <code>w</code> and <code>b</code> so the "
        "mean squared error — the average of the squared vertical gaps between the points and the "
        "line — is as small as possible.",
        "This demo trains with <strong>gradient descent</strong>: it starts from a flat line at zero "
        "and repeatedly nudges the slope and intercept downhill along the error surface. Each frame is "
        "one epoch, i.e. one pass over the whole dataset.",
        "A straight line also has an exact closed-form solution (ordinary least squares). The dashed "
        "grey line shows it, so you can watch gradient descent walk towards the answer algebra gives "
        "instantly.",
    ],
    watch_for=[
        "Raise the learning rate and the line converges in fewer epochs — until it overshoots and the error chart explodes.",
        "Drag the learning rate very low and the line crawls: it is heading the right way, just slowly.",
        "Add one far-away outlier and watch the whole line tilt. Squared error punishes big misses hard.",
        "Turn on L2 regularisation and the slope is pulled gently towards zero.",
    ],
    step_unit="epoch",
    step_hint="Each frame is one full pass of gradient descent over the data.",
    params=[
        Param(
            name="learning_rate",
            label="Learning rate",
            type="float",
            default=0.08,
            min=0.001,
            max=1.2,
            step=0.001,
            help="How big a step each epoch takes downhill. Too large and it overshoots.",
        ),
        Param(
            name="epochs",
            label="Epochs",
            type="int",
            default=30,
            min=1,
            max=300,
            step=1,
            help="How many passes over the data to run.",
        ),
        Param(
            name="regularization",
            label="Regularisation",
            type="select",
            default="none",
            options=[
                {"value": "none", "label": "None"},
                {"value": "l2", "label": "L2 (ridge)"},
                {"value": "l1", "label": "L1 (lasso)"},
            ],
            help="A penalty on large coefficients, which shrinks the slope towards zero.",
        ),
        Param(
            name="alpha",
            label="Regularisation strength",
            type="float",
            default=0.01,
            min=0.0,
            max=1.0,
            step=0.001,
            help="Only used when regularisation is on.",
        ),
    ],
)


def _line_curve(grid: Grid, slope: float, intercept: float) -> list[list[float]]:
    xs = grid.curve_x()
    ys = slope * xs + intercept
    return [[float(a), float(b)] for a, b in zip(xs, ys)]


def fit(points, params, grid: Grid) -> FitResult:
    X, y = prepare_regression(points)
    lr = float(params["learning_rate"])
    epochs = int(params["epochs"])
    penalty = params["regularization"]
    alpha = float(params["alpha"])

    # Gradient descent is scale sensitive, so standardise and map the learned
    # coefficients back into plot coordinates for display.
    x_mean, x_std = float(X.mean()), float(X.std()) or 1.0
    y_mean, y_std = float(y.mean()), float(y.std()) or 1.0
    Xs = (X - x_mean) / x_std
    ys = (y - y_mean) / y_std

    def to_plot(w: float, b: float) -> tuple[float, float]:
        slope = y_std * w / x_std
        intercept = y_mean + y_std * b - slope * x_mean
        return slope, intercept

    ols = LinearRegression().fit(X, y)
    ols_slope, ols_intercept = float(ols.coef_[0]), float(ols.intercept_)
    ols_mse = float(mean_squared_error(y, ols.predict(X)))

    model = SGDRegressor(
        loss="squared_error",
        penalty=None if penalty == "none" else penalty,
        alpha=alpha if penalty != "none" else 0.0,
        learning_rate="constant",
        eta0=lr,
        shuffle=True,
        random_state=0,
    )

    def snapshot(slope: float, intercept: float) -> tuple[float, float]:
        pred = slope * X[:, 0] + intercept
        return float(mean_squared_error(y, pred)), float(r2_score(y, pred))

    frames: list[tuple[int, float, float, float, float]] = []
    mse, r2 = snapshot(0.0, 0.0)
    frames.append((0, 0.0, 0.0, mse, r2))

    diverged = False
    for epoch in range(1, epochs + 1):
        model.partial_fit(Xs, ys)
        slope, intercept = to_plot(float(model.coef_[0]), float(model.intercept_[0]))
        if not (np.isfinite(slope) and np.isfinite(intercept)) or abs(slope) > 1e6:
            diverged = True
            break
        mse, r2 = snapshot(slope, intercept)
        frames.append((epoch, slope, intercept, mse, r2))
        if not np.isfinite(mse) or mse > 1e12:
            diverged = True
            break

    steps: list[Step] = []
    for epoch, slope, intercept, mse, r2 in thin(frames):
        if epoch == 0:
            label = "Start"
            description = (
                "Both parameters start at zero, so the model predicts a flat line through "
                "the origin. Every vertical grey stub is one residual — the error gradient "
                "descent is about to shrink."
            )
        else:
            description = (
                f"After epoch {epoch} the line is <code>ŷ = {slope:.3f}·x + {intercept:.3f}</code>, "
                f"with mean squared error {mse:.3f} (R² = {r2:.3f})."
            )
            label = f"Epoch {epoch}"
        steps.append(
            Step(
                label=label,
                description=description,
                metrics={"mse": mse, "r2": r2, "slope": slope, "intercept": intercept},
                curve=_line_curve(grid, slope, intercept),
                extras={
                    "slope": slope,
                    "intercept": intercept,
                    "show_residuals": True,
                    "equation": f"ŷ = {slope:.3f}·x + {intercept:.3f}",
                },
            )
        )

    notes = [
        f"Closed-form least squares optimum: ŷ = {ols_slope:.3f}·x + {ols_intercept:.3f} "
        f"with MSE {ols_mse:.3f} (shown dashed).",
    ]
    if diverged:
        notes.append(
            "Training diverged — the learning rate is too large for this data, so each step "
            "overshoots the minimum and the error grows. Lower it and run again."
        )

    final = steps[-1]
    return FitResult(
        task="regression",
        steps=steps,
        metric_labels={
            "mse": "Mean squared error",
            "r2": "R²",
            "slope": "Slope (w)",
            "intercept": "Intercept (b)",
        },
        chart_metrics=["mse", "r2"],
        summary={
            "Equation": final.extras["equation"],
            "MSE": final.metrics["mse"],
            "R²": final.metrics["r2"],
            "Optimal MSE": ols_mse,
        },
        notes=notes,
        extras={
            "reference_curve": _line_curve(grid, ols_slope, ols_intercept),
            "reference_label": "Least-squares optimum",
        },
    )
