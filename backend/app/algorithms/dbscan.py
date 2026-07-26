"""DBSCAN, animated as the flood fill it actually is.

scikit-learn's DBSCAN returns only the final labelling, so the expansion is
driven here explicitly on top of ``NearestNeighbors`` radius queries — the same
approach used for k-means, where the interesting part is the loop rather than
the answer.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from ..grid import Grid, class_surface
from .base import AlgorithmSpec, FitResult, Param, Step, prepare_unlabelled, thin

# Point roles, mirrored in plot.js.
ROLE_PENDING = 0
ROLE_CORE = 1
ROLE_BORDER = 2
ROLE_NOISE = 3

SPEC = AlgorithmSpec(
    id="dbscan",
    name="DBSCAN",
    task="clustering",
    tagline="Grow clusters through dense regions, and call the leftovers noise.",
    description=[
        "DBSCAN clusters by density rather than by distance to a centre. Two numbers define it: "
        "<code>eps</code>, a radius, and <code>min_samples</code>, a population. Any point with at "
        "least <code>min_samples</code> neighbours inside <code>eps</code> is a <strong>core "
        "point</strong>. Clusters are grown by flood fill: start at an unvisited core point, absorb "
        "everything within <code>eps</code>, and keep expanding from every new core point you reach.",
        "Points that get absorbed but are not themselves dense enough are <strong>border points</strong> "
        "— drawn as hollow rings. Points that no cluster ever reaches are <strong>noise</strong>, drawn "
        "as grey crosses. No other algorithm here is allowed to say 'this point belongs to nothing', "
        "and that is DBSCAN's most useful property.",
        "It also never asks you for <code>k</code>. The number of clusters falls out of the density "
        "structure, which means DBSCAN handles the crescents, rings and spirals that defeat k-means — "
        "and it means a badly chosen <code>eps</code> can silently return one giant cluster, or nothing "
        "but noise.",
    ],
    watch_for=[
        "Run it on two moons or concentric circles, then compare with k-means. This is the case k-means cannot do.",
        "The circle drawn on the expanding point is the eps radius — the actual reachability test, frame by frame.",
        "Nudge eps down and clusters shatter into noise; nudge it up and separate clusters fuse into one.",
        "Raise min_samples to thin out the core points: cluster edges retreat and the noise count climbs.",
        "On the uniform-noise dataset there is no density structure, so almost everything is correctly labelled noise.",
    ],
    step_unit="expansion",
    step_hint="Each frame absorbs more points into the cluster currently being grown.",
    params=[
        Param(
            name="eps",
            label="eps (neighbourhood radius)",
            type="float",
            default=0.45,
            min=0.02,
            max=5.0,
            scale="log",
            help="How close counts as neighbouring. The single most important setting.",
        ),
        Param(
            name="min_samples",
            label="min_samples (density threshold)",
            type="int",
            default=5,
            min=2,
            max=30,
            step=1,
            help="Neighbours required inside eps for a point to be a core point.",
        ),
    ],
)


def _roles(labels: np.ndarray, core: np.ndarray, final: bool) -> np.ndarray:
    """Per-point role for the current state of the flood fill."""
    assigned = labels >= 0
    roles = np.full(len(labels), ROLE_PENDING)
    roles[assigned & core] = ROLE_CORE
    roles[assigned & ~core] = ROLE_BORDER
    if final:
        # Anything still unclaimed once the fill is finished really is noise.
        roles[~assigned] = ROLE_NOISE
    return roles


def _surface(X: np.ndarray, labels: np.ndarray, core: np.ndarray, eps: float, grid: Grid, n_clusters: int) -> dict:
    """Shade each cell by the nearest *assigned* core point within eps.

    DBSCAN has no predict(): this is an extrapolation, flagged as such in the
    notes. It hugs the density the algorithm actually found, which is the point.
    """
    assigned_core = np.flatnonzero(core & (labels >= 0))
    if len(assigned_core) == 0:
        return class_surface(np.full(len(grid.points), -1), n_classes=max(n_clusters, 1))

    nn = NearestNeighbors(n_neighbors=1).fit(X[assigned_core])
    distance, index = nn.kneighbors(grid.points)
    nearest = labels[assigned_core][index.ravel()]
    cells = np.where(distance.ravel() <= eps, nearest, -1)
    return class_surface(cells, n_classes=max(n_clusters, 1))


def fit(points, params, grid: Grid, validation: float = 0.0) -> FitResult:
    # validation is unused: clustering has no labels to score a holdout against.
    del validation
    X = prepare_unlabelled(points, min_points=3)
    eps = float(params["eps"])
    min_samples = int(params["min_samples"])

    neighbourhoods = NearestNeighbors(radius=eps).fit(X).radius_neighbors(X, return_distance=False)
    core = np.array([len(nb) >= min_samples for nb in neighbourhoods])

    # Flood fill, logging one event per point claimed. The log is compact, so a
    # thousand events cost nothing; frames are replayed from it afterwards.
    labels = np.full(len(X), -1, dtype=int)
    events: list[tuple[int, int, int]] = []  # (point, cluster, seed being expanded)
    cluster = -1
    for seed in np.flatnonzero(core):
        if labels[seed] >= 0:
            continue
        cluster += 1
        stack = [seed]
        while stack:
            current = stack.pop()
            if labels[current] >= 0:
                continue
            labels[current] = cluster
            events.append((int(current), cluster, int(seed)))
            if core[current]:
                stack.extend(j for j in neighbourhoods[current] if labels[j] < 0)

    n_clusters = cluster + 1
    final_labels = labels.copy()
    n_noise = int((final_labels < 0).sum())

    # Replay the log up to a handful of checkpoints to build the animation.
    checkpoints = thin(range(1, len(events) + 1), max_items=44) if events else []
    states: list[tuple[int, np.ndarray, int | None, bool]] = [
        (0, np.full(len(X), -1, dtype=int), None, False)
    ]
    for count in checkpoints:
        replay = np.full(len(X), -1, dtype=int)
        for point, cluster_id, _ in events[:count]:
            replay[point] = cluster_id
        states.append((count, replay, events[count - 1][0], False))
    states.append((len(events), final_labels, None, True))

    steps: list[Step] = []
    for claimed, state, active, final in states:
        found = int(state.max()) + 1 if state.max() >= 0 else 0
        roles = _roles(state, core, final)
        pending = int((state < 0).sum())

        if claimed == 0:
            label = "Core points"
            description = (
                f"{int(core.sum())} of {len(X)} points have at least {min_samples} neighbours "
                f"within eps = {eps:g}, making them core points. Nothing has been clustered yet — "
                f"the fill is about to start from the first of them."
            )
        elif final:
            label = "Complete"
            description = (
                f"{n_clusters} cluster{'s' if n_clusters != 1 else ''} found, "
                f"{n_noise} point{'s' if n_noise != 1 else ''} left as noise. "
                f"DBSCAN was never told how many clusters to look for."
            )
        else:
            label = f"{claimed} claimed"
            description = (
                f"Growing cluster {found} — {claimed} of {len(X)} points claimed, {pending} still "
                f"unvisited. The circle marks the eps radius around the point being expanded."
            )

        steps.append(
            Step(
                label=label,
                description=description,
                metrics={
                    "clusters": found,
                    "claimed": claimed,
                    "unassigned": pending,
                    "core_points": int(core.sum()),
                },
                surface=_surface(X, state, core, eps, grid, max(found, n_clusters)),
                extras={
                    "assignments": state.tolist(),
                    "roles": roles.tolist(),
                    "eps": eps,
                    "active_index": active,
                    "n_clusters": found,
                },
            )
        )

    silhouette = None
    clustered = final_labels >= 0
    if n_clusters >= 2 and clustered.sum() > n_clusters:
        silhouette = float(silhouette_score(X[clustered], final_labels[clustered]))

    notes = [
        "The shaded regions are an extrapolation, not DBSCAN's own output: the algorithm labels only "
        "the points it was given and has no predict() for new locations. Each cell is coloured by the "
        "nearest core point within eps.",
    ]
    if n_clusters == 0:
        notes.append(
            "No core point exists at these settings, so everything is noise. Increase eps or lower "
            "min_samples."
        )
    elif n_clusters == 1 and n_noise == 0:
        notes.append(
            "Everything landed in a single cluster — eps is large enough to bridge every gap. Reduce it."
        )
    if n_noise > 0.5 * len(X):
        notes.append(
            f"{n_noise / len(X) * 100:.0f}% of points are noise, which usually means eps is too small "
            f"for this data."
        )
    if silhouette is not None:
        notes.append(f"Silhouette score {silhouette:.3f}, computed over the non-noise points only.")

    return FitResult(
        task="clustering",
        steps=steps,
        metric_labels={
            "clusters": "Clusters found",
            "claimed": "Points claimed",
            "unassigned": "Unassigned",
            "core_points": "Core points",
        },
        chart_metrics=["claimed", "clusters"],
        summary={
            "Clusters": n_clusters,
            "Noise points": n_noise,
            "Core points": int(core.sum()),
            "Silhouette": silhouette,
        },
        notes=notes,
        extras={"eps": eps, "n_clusters": n_clusters},
    )
