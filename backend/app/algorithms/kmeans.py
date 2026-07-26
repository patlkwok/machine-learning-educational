"""k-means clustering, showing the assign step and the update step separately."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import pairwise_distances_argmin, silhouette_score

from ..grid import Grid, class_surface
from .base import AlgorithmSpec, DataError, FitResult, Param, Step, prepare_unlabelled, thin

SPEC = AlgorithmSpec(
    id="kmeans",
    name="k-Means Clustering",
    task="clustering",
    tagline="Alternate between assigning points to centroids and moving centroids to their points.",
    description=[
        "k-means is the classic unsupervised algorithm: no labels, just geometry. You pick "
        "<code>k</code>, and it looks for <code>k</code> centres such that the total squared distance "
        "from every point to its nearest centre — the <strong>inertia</strong> — is as small as possible.",
        "Lloyd's algorithm alternates two steps, and this animation shows them as separate frames. "
        "<strong>Assign:</strong> colour each point by whichever centroid is nearest. "
        "<strong>Update:</strong> move each centroid to the mean of the points that chose it. Repeat "
        "until nothing moves. Inertia can only ever go down, so it always converges — though not "
        "necessarily to the best answer.",
        "Initialisation matters enormously. <em>k-means++</em> spreads the starting centroids out on "
        "purpose; pure random starts can strand two centroids inside one cluster and leave another "
        "cluster unclaimed, and the algorithm has no way to recover.",
    ],
    watch_for=[
        "Inertia drops steeply on the first couple of iterations, then barely moves. Most of the work happens early.",
        "To see initialisation matter, you need a hard enough problem: on three well-separated blobs both options find the same answer every single time.",
        "Try two moons with k = 6, then press \"New starting centroids\" repeatedly. Random starts land on a visibly worse arrangement about twice as often as k-means++, and when they do, they are roughly three times further from the best solution.",
        "Ask for the wrong k — say 4 on 3 blobs — and it will happily split a real cluster in half. k-means always finds exactly k clusters, whether or not they exist.",
        "On the stretched-blobs dataset the round Voronoi cells cut straight across the elongated clusters. k-means assumes clusters are round and equally sized.",
    ],
    step_unit="phase",
    step_hint="Frames alternate between the assign step and the update step.",
    reseed={
        "param": "seed",
        "label": "New starting centroids",
        "help": (
            "Restart from a different initial guess. Whether the answer changes depends "
            "on the data: see the suggestions under Things to try."
        ),
    },
    params=[
        Param(
            name="k",
            label="Clusters (k)",
            type="int",
            default=3,
            min=1,
            max=10,
            step=1,
            help="How many centroids to fit. k-means will always find exactly this many.",
        ),
        Param(
            name="init",
            label="Initialisation",
            type="select",
            default="kmeans++",
            options=[
                {"value": "kmeans++", "label": "k-means++ (spread out)"},
                {"value": "random", "label": "Random points"},
            ],
            help="How the starting centroids are chosen.",
        ),
        Param(
            name="seed",
            label="Starting centroids",
            type="int",
            default=0,
            min=0,
            max=999999,
            step=1,
            # Driven by the "New starting centroids" button below: the number
            # itself teaches nothing, but changing it teaches a great deal.
            hidden=True,
            help="Which starting configuration to use.",
        ),
        Param(
            name="max_iter",
            label="Maximum iterations",
            type="int",
            default=15,
            min=1,
            max=60,
            step=1,
            help="Stops early as soon as the centroids stop moving.",
        ),
    ],
)


def _voronoi(centroids: np.ndarray, grid: Grid) -> dict:
    labels = pairwise_distances_argmin(grid.points, centroids)
    return class_surface(labels, n_classes=len(centroids))


def _inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    return float(((X - centroids[labels]) ** 2).sum())


def fit(points, params, grid: Grid, validation: float = 0.0) -> FitResult:
    # validation is unused: clustering has no labels to score a holdout against.
    del validation
    k = int(params["k"])
    seed = int(params["seed"])
    max_iter = int(params["max_iter"])

    X = prepare_unlabelled(points, min_points=max(3, k))
    if len(X) < k:
        raise DataError(f"k = {k} needs at least {k} points; there are only {len(X)}.")

    if params["init"] == "kmeans++":
        centroids, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed)
        init_note = "k-means++ picked starting centroids that are deliberately far apart."
    else:
        rng = np.random.default_rng(seed)
        centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
        init_note = "Starting centroids were picked uniformly at random from the data points."
    centroids = np.asarray(centroids, dtype=float)

    frames: list[dict] = []
    labels = pairwise_distances_argmin(X, centroids)
    frames.append(
        {
            "phase": "init",
            "centroids": centroids.copy(),
            "previous": None,
            "labels": labels.copy(),
            "inertia": _inertia(X, centroids, labels),
            "shift": None,
            "iteration": 0,
        }
    )

    converged_at = None
    for iteration in range(1, max_iter + 1):
        labels = pairwise_distances_argmin(X, centroids)
        frames.append(
            {
                "phase": "assign",
                "centroids": centroids.copy(),
                "previous": None,
                "labels": labels.copy(),
                "inertia": _inertia(X, centroids, labels),
                "shift": None,
                "iteration": iteration,
            }
        )

        new_centroids = centroids.copy()
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members):
                new_centroids[cluster] = members.mean(axis=0)
        shift = float(np.linalg.norm(new_centroids - centroids, axis=1).max())

        frames.append(
            {
                "phase": "update",
                "centroids": new_centroids.copy(),
                "previous": centroids.copy(),
                "labels": labels.copy(),
                "inertia": _inertia(X, new_centroids, labels),
                "shift": shift,
                "iteration": iteration,
            }
        )
        centroids = new_centroids
        if shift < 1e-9:
            converged_at = iteration
            break

    final_labels = pairwise_distances_argmin(X, centroids)
    empty = [c for c in range(k) if not np.any(final_labels == c)]

    steps: list[Step] = []
    for frame in thin(frames):
        centroid_list = frame["centroids"].tolist()
        if frame["phase"] == "init":
            label = "Initialise"
            description = f"{init_note} Nothing has been fitted yet — this is just the starting guess."
        elif frame["phase"] == "assign":
            label = f"Iter {frame['iteration']} · assign"
            description = (
                f"Each point is recoloured to match its nearest centroid. Inertia "
                f"{frame['inertia']:.2f}. Centroids have not moved in this frame."
            )
        else:
            label = f"Iter {frame['iteration']} · update"
            description = (
                f"Each centroid jumps to the mean of its assigned points (largest move "
                f"{frame['shift']:.3f}). Inertia falls to {frame['inertia']:.2f}."
            )
        steps.append(
            Step(
                label=label,
                description=description,
                metrics={
                    "inertia": frame["inertia"],
                    "shift": frame["shift"],
                    "iteration": frame["iteration"],
                },
                surface=_voronoi(frame["centroids"], grid),
                extras={
                    "centroids": centroid_list,
                    "previous_centroids": frame["previous"].tolist() if frame["previous"] is not None else None,
                    "assignments": frame["labels"].tolist(),
                    "phase": frame["phase"],
                    "n_clusters": k,
                },
            )
        )

    silhouette = None
    if 2 <= k < len(X) and len(set(final_labels.tolist())) > 1:
        silhouette = float(silhouette_score(X, final_labels))

    notes = []
    if converged_at is not None:
        notes.append(f"Converged after {converged_at} iteration(s): no centroid moved any further.")
    else:
        notes.append(
            f"Stopped at the {max_iter}-iteration limit without fully converging — raise the limit to continue."
        )
    if empty:
        notes.append(
            f"Cluster(s) {', '.join(str(c + 1) for c in empty)} ended up with no points. "
            "That is a classic random-initialisation failure."
        )
    if silhouette is not None:
        notes.append(
            f"Silhouette score {silhouette:.3f} (1 = tight and well separated, 0 = overlapping, "
            "negative = points are closer to another cluster than their own)."
        )

    final = steps[-1]
    return FitResult(
        task="clustering",
        steps=steps,
        metric_labels={
            "inertia": "Inertia (within-cluster sum of squares)",
            "shift": "Largest centroid move",
            "iteration": "Iteration",
        },
        chart_metrics=["inertia", "shift"],
        summary={
            "Clusters": k,
            "Inertia": final.metrics["inertia"],
            "Silhouette": silhouette,
            "Iterations": converged_at if converged_at is not None else max_iter,
        },
        notes=notes,
        extras={"n_clusters": k},
    )
