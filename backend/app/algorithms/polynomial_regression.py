"""Polynomial regression, sweeping the degree to show under- and over-fitting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

from ..grid import Grid
from .base import (
    METRIC_LABELS,
    AlgorithmSpec,
    FitResult,
    Param,
    Step,
    make_split,
    prepare_regression,
    split_notes,
)

SPEC = AlgorithmSpec(
    id="polynomial_regression",
    name="Polynomial Regression",
    task="regression",
    tagline="Add curvature by fitting powers of x — and watch overfitting appear.",
    description=[
        "Polynomial regression is still linear regression; the trick is that the features are "
        "<code>x, x², x³, …, x^d</code>. The model stays linear <em>in its coefficients</em>, so it "
        "still has a closed-form solution, but the curve it draws can bend.",
        "Each frame raises the degree by one. Low degrees <strong>underfit</strong>: the curve is too "
        "stiff to follow the data. High degrees <strong>overfit</strong>: the curve wriggles through "
        "individual noise points and predicts nonsense between them.",
        "The chart plots training error against 5-fold cross-validated error. Training error only ever "
        "falls as degree grows; cross-validated error falls, bottoms out at the right complexity, and "
        "then climbs. That U-shape is the bias–variance trade-off made visible.",
    ],
    watch_for=[
        "On the sine-wave dataset, watch degree 1–3 underfit badly, then degree 5–7 lock on.",
        "Push the degree to 12+ on a small dataset: the curve whips off the top of the plot between points.",
        "Raise the ridge penalty and high-degree curves calm down — regularisation buys back stability.",
        "The lowest point of the cross-validation curve is the degree you would actually ship.",
    ],
    step_unit="degree",
    step_hint="Each frame fits a fresh polynomial one degree higher than the last.",
    params=[
        Param(
            name="max_degree",
            label="Maximum degree",
            type="int",
            default=9,
            min=1,
            max=15,
            step=1,
            help="The animation sweeps from degree 1 up to this value.",
        ),
        Param(
            name="alpha",
            label="Ridge penalty (α)",
            type="float",
            default=0.0,
            min=0.0,
            max=5.0,
            step=0.01,
            help="L2 shrinkage on the coefficients. 0 is plain least squares.",
        ),
    ],
)

Y_CLAMP = 1e6


def _pipeline(degree: int, alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("ridge", Ridge(alpha=max(alpha, 1e-10))),
        ]
    )


def fit(points, params, grid: Grid, validation: float = 0.0) -> FitResult:
    X, y = prepare_regression(points)
    max_degree = int(params["max_degree"])
    alpha = float(params["alpha"])

    split = make_split(len(y), validation)
    X_train, y_train = X[split.train], y[split.train]
    X_val, y_val = X[split.val], y[split.val]

    curve_x = grid.curve_x()
    steps: list[Step] = []
    best = (float("inf"), 1)

    for degree in range(1, max_degree + 1):
        model = _pipeline(degree, alpha)
        model.fit(X_train, y_train)

        predicted = model.predict(X_train)
        train_mse = float(mean_squared_error(y_train, predicted))
        train_r2 = float(r2_score(y_train, predicted))

        val_mse = val_r2 = None
        if split.active:
            predicted_val = model.predict(X_val)
            val_mse = float(mean_squared_error(y_val, predicted_val))
            val_r2 = float(r2_score(y_val, predicted_val))
            if np.isfinite(val_mse) and val_mse < best[0]:
                best = (val_mse, degree)

        curve_y = np.clip(model.predict(curve_x.reshape(-1, 1)), -Y_CLAMP, Y_CLAMP)
        coefs = model.named_steps["ridge"].coef_

        verdict = ""
        if degree == 1:
            verdict = " Degree 1 is just a straight line — the baseline to beat."
        elif val_mse is not None and val_mse > train_mse * 3 and degree > 3:
            verdict = " Validation error is now far above training error: this is overfitting."

        description = (
            f"Degree {degree} polynomial: training MSE {train_mse:.3f}, R² {train_r2:.3f}"
            + (f", validation MSE {val_mse:.3f}." if val_mse is not None else ".")
            + verdict
        )

        steps.append(
            Step(
                label=f"Degree {degree}",
                description=description,
                metrics={
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "train_r2": train_r2,
                    "val_r2": val_r2,
                    "degree": degree,
                    "coef_norm": float(np.linalg.norm(coefs)),
                },
                curve=[[float(a), float(b)] for a, b in zip(curve_x, curve_y)],
                extras={"degree": degree, "show_residuals": True},
            )
        )

    notes = list(split_notes(split, steps[-1].metrics))
    if split.active:
        notes.append(
            f"Validation error is lowest at degree {best[1]} — the best complexity for this data, "
            f"and the degree you would actually ship."
        )
    if alpha == 0 and max_degree >= 10:
        notes.append(
            "With no ridge penalty, high-degree fits are numerically delicate; nudge α up to see them steady."
        )

    final = steps[-1]
    return FitResult(
        task="regression",
        steps=steps,
        metric_labels={
            **METRIC_LABELS,
            "coef_norm": "Coefficient size ‖w‖",
            "degree": "Degree",
        },
        chart_metrics=["train_mse", "val_mse"],
        summary={
            "Final degree": final.metrics["degree"],
            "Training MSE": final.metrics["train_mse"],
            "Validation MSE": final.metrics["val_mse"],
            "Best degree": best[1] if split.active else None,
        },
        notes=notes,
        split=split,
        extras={"best_degree": best[1] if split.active else None},
    )
