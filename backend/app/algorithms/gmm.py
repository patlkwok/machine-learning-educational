"""Gaussian mixture models, one EM iteration per frame.

scikit-learn's GaussianMixture reports only the converged fit, so the loop is
driven here with ``warm_start`` and ``max_iter=1`` — the same approach used for
the MLP, where each call advances the fit by exactly one step.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from ..grid import Grid, class_surface, confidence_from_scores
from .base import AlgorithmSpec, FitResult, Param, Step, prepare_unlabelled, thin

# Contour drawn for each component, in standard deviations.
SIGMA = 2.0

# A floor on the covariance diagonal. Clicked points are often duplicated or
# collinear, which makes a component collapse to zero width without this.
REG_COVAR = 1e-4

SPEC = AlgorithmSpec(
    id="gmm",
    name="Gaussian Mixture Model",
    task="clustering",
    tagline="Fit overlapping bell curves, and let every point belong partly to each.",
    description=[
        "A Gaussian mixture assumes the data was produced by <em>k</em> Gaussian blobs mixed "
        "together, and works backwards to find them. Unlike k-means it does not force each point "
        "into one cluster: every point gets a <strong>responsibility</strong> for each component — "
        "the probability that this component produced it. Points deep inside a blob are drawn solid, "
        "points caught between two are drawn faded, because the model genuinely is not sure.",
        "Training is <strong>expectation-maximisation</strong>. The <em>E-step</em> computes each "
        "point's responsibilities given the current blobs; the <em>M-step</em> moves each blob to the "
        "weighted mean and covariance of the points that claimed it. Each frame is one such round, "
        "and the log-likelihood is guaranteed never to fall — that is a theorem about EM, not a "
        "property of this data.",
        "<strong>Covariance type</strong> is the interesting dial, because it turns this one algorithm "
        "into three familiar ones. <em>Spherical</em> forces round blobs, which is essentially a soft "
        "k-means. <em>Diagonal</em> allows axis-aligned ellipses, exactly the shape naive Bayes is "
        "restricted to. <em>Full</em> lets them stretch and tilt in any direction, which is what "
        "neither of the others can do.",
    ],
    watch_for=[
        "Set covariance to full on the stretched-blobs dataset, then compare with k-means: the ellipses tilt to follow the data, where k-means cuts straight across it.",
        "Switch between spherical, diagonal and full on the same data. Spherical behaves like k-means; diagonal gives the axis-aligned ellipses of naive Bayes; only full can tilt.",
        "Watch points between two components stay faded for the whole run. They are not misclassified — the model is reporting genuine uncertainty.",
        "The log-likelihood curve rises and then flattens. It can never fall: if it appears to, that is a bug rather than a bad run.",
        "Lower BIC is better, and it charges you for every extra component. Step k up on well-separated blobs and BIC bottoms out at the right number.",
        "On overlapping blobs BIC often prefers fewer components than actually generated the data — it reports what the data supports, not what you know is there. Raise the noise until two blobs merge and watch the chosen k drop.",
    ],
    step_unit="iteration",
    step_hint="Each frame is one expectation-maximisation round: responsibilities, then new blobs.",
    reseed={
        "param": "seed",
        "label": "New starting blobs",
        "help": (
            "Restart EM from a different initial guess. Like k-means, it finds a local "
            "optimum, so a different start can find a different answer."
        ),
    },
    params=[
        Param(
            name="n_components",
            label="Components (k)",
            type="int",
            default=3,
            min=1,
            max=10,
            step=1,
            help="How many Gaussian blobs to fit.",
        ),
        Param(
            name="covariance_type",
            label="Covariance type",
            type="select",
            default="full",
            options=[
                {"value": "full", "label": "Full (any shape, can tilt)"},
                {"value": "tied", "label": "Tied (one shape, shared)"},
                {"value": "diag", "label": "Diagonal (axis-aligned)"},
                {"value": "spherical", "label": "Spherical (round, like k-means)"},
            ],
            help="What shapes the blobs are allowed to take. This changes the answer more than k does.",
        ),
        Param(
            name="max_iter",
            label="Maximum iterations",
            type="int",
            default=25,
            min=1,
            max=80,
            step=1,
            help="EM stops early once the log-likelihood settles.",
        ),
        Param(
            name="init_params",
            label="Initialisation",
            type="select",
            default="kmeans",
            options=[
                {"value": "kmeans", "label": "k-means (spread out)"},
                {"value": "random", "label": "Random"},
            ],
            help="Where the blobs start before the first E-step.",
        ),
        Param(
            name="seed",
            label="Starting blobs",
            type="int",
            default=0,
            min=0,
            max=999999,
            step=1,
            hidden=True,
            help="Which starting configuration to use.",
        ),
    ],
)


def _covariance_matrices(model: GaussianMixture, k: int) -> np.ndarray:
    """Every covariance_type expanded to k full 2x2 matrices."""
    cov = model.covariances_
    kind = model.covariance_type
    if kind == "full":
        return np.asarray(cov)
    if kind == "tied":
        return np.repeat(np.asarray(cov)[None, :, :], k, axis=0)
    if kind == "diag":
        return np.array([np.diag(c) for c in cov])
    return np.array([np.eye(2) * c for c in cov])  # spherical


def _ellipses(model: GaussianMixture, k: int) -> list[dict]:
    """2-sigma contours, as centre, radii and a rotation in radians.

    The eigenvectors of the covariance give the ellipse axes; this is what lets
    a full covariance tilt where a diagonal one cannot.
    """
    out = []
    for index, (mean, cov) in enumerate(zip(model.means_, _covariance_matrices(model, k))):
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        out.append(
            {
                "class_index": index,
                "cx": float(mean[0]),
                "cy": float(mean[1]),
                "rx": float(SIGMA * np.sqrt(max(values[0], 1e-12))),
                "ry": float(SIGMA * np.sqrt(max(values[1], 1e-12))),
                "angle": float(np.arctan2(vectors[1, 0], vectors[0, 0])),
                "weight": float(model.weights_[index]),
            }
        )
    return out


def fit(points, params, grid: Grid, validation: float = 0.0) -> FitResult:
    # validation is unused: clustering has no labels to score a holdout against.
    del validation

    k = int(params["n_components"])
    covariance_type = params["covariance_type"]
    max_iter = int(params["max_iter"])
    init_params = params["init_params"]
    seed = int(params["seed"])

    X = prepare_unlabelled(points, min_points=max(3, k))
    k = min(k, len(X))

    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        max_iter=1,
        warm_start=True,
        init_params=init_params,
        random_state=seed,
        reg_covar=REG_COVAR,
    )

    frames: list[tuple[int, float, list[dict], np.ndarray, np.ndarray, np.ndarray]] = []
    converged_at = None

    with warnings.catch_warnings():
        # max_iter=1 always "fails to converge"; convergence is detected below.
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        previous = None
        for iteration in range(1, max_iter + 1):
            model.fit(X)
            log_likelihood = float(model.lower_bound_)

            responsibilities = model.predict_proba(X)
            grid_proba = model.predict_proba(grid.points)
            frames.append(
                (
                    iteration,
                    log_likelihood,
                    _ellipses(model, k),
                    responsibilities,
                    grid_proba,
                    np.array([model.bic(X), model.aic(X)]),
                )
            )

            if previous is not None and abs(log_likelihood - previous) < 1e-7:
                converged_at = iteration
                break
            previous = log_likelihood

    steps: list[Step] = []
    for iteration, log_likelihood, ellipses, responsibilities, grid_proba, criteria in thin(frames):
        assignments = np.argmax(responsibilities, axis=1)
        confidence = responsibilities.max(axis=1)
        uncertain = int((confidence < 0.9).sum())

        steps.append(
            Step(
                label=f"Iteration {iteration}",
                description=(
                    f"E-step assigned responsibilities, M-step moved the blobs. Log-likelihood "
                    f"{log_likelihood:.4f}, BIC {criteria[0]:.1f}. "
                    + (
                        f"{uncertain} point{'s' if uncertain != 1 else ''} are still split between "
                        f"components — drawn faded."
                        if uncertain
                        else "Every point now belongs clearly to one component."
                    )
                ),
                metrics={
                    "log_likelihood": log_likelihood,
                    "bic": float(criteria[0]),
                    "aic": float(criteria[1]),
                    "uncertain": uncertain,
                },
                surface=class_surface(
                    np.argmax(grid_proba, axis=1),
                    n_classes=k,
                    confidence=confidence_from_scores(grid_proba),
                ),
                extras={
                    "ellipses": ellipses,
                    "assignments": assignments.tolist(),
                    # Max responsibility per point: how sure the model is, drawn
                    # as opacity so uncertainty is visible rather than hidden.
                    "responsibility": np.round(confidence, 3).tolist(),
                    "n_clusters": k,
                },
            )
        )

    final_labels = np.argmax(frames[-1][3], axis=1)
    silhouette = None
    if 2 <= k < len(X) and len(set(final_labels.tolist())) > 1:
        silhouette = float(silhouette_score(X, final_labels))

    notes = []
    if converged_at is not None:
        notes.append(f"EM converged after {converged_at} iterations: the log-likelihood stopped moving.")
    else:
        notes.append(
            f"Stopped at the {max_iter}-iteration limit while the log-likelihood was still rising."
        )
    notes.append(
        {
            "full": "Full covariance: each blob can stretch and tilt independently. The most flexible "
            "option, and the only one that can follow diagonal structure.",
            "tied": "Tied covariance: every blob shares one shape, so they can tilt but not differ "
            "from each other.",
            "diag": "Diagonal covariance: ellipses stay axis-aligned, exactly the restriction that "
            "makes Gaussian naive Bayes naive.",
            "spherical": "Spherical covariance: blobs are forced round, which makes this close to a "
            "soft-assignment k-means.",
        }[covariance_type]
    )
    notes.append(
        f"BIC {frames[-1][5][0]:.1f} — lower is better, and it charges you for every extra "
        f"component. Comparing BIC across k is the principled way to choose it."
    )
    if silhouette is not None:
        notes.append(f"Silhouette score {silhouette:.3f}, computed on the hard assignment.")
    notes.append(
        "Colour shows the most likely component; opacity shows how sure the model is. Faded points "
        "are genuinely shared between components rather than misassigned."
    )

    final = steps[-1]
    return FitResult(
        task="clustering",
        steps=steps,
        metric_labels={
            "log_likelihood": "Log-likelihood",
            "bic": "BIC (lower is better)",
            "aic": "AIC",
            "uncertain": "Uncertain points",
        },
        chart_metrics=["log_likelihood", "bic"],
        summary={
            "Components": k,
            "Covariance": covariance_type,
            "Log-likelihood": final.metrics["log_likelihood"],
            "BIC": final.metrics["bic"],
            "Silhouette": silhouette,
        },
        notes=notes,
        extras={"n_clusters": k, "covariance_type": covariance_type},
    )
