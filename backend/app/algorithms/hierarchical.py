"""Agglomerative hierarchical clustering, animated by lowering the cut.

The linkage tree is computed once with scipy; each frame is that same tree cut
at a different height, so the animation shows clusters fusing pairwise while a
cut line slides down the dendrogram.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from ..grid import Grid, class_surface
from .base import AlgorithmSpec, FitResult, Param, Step, prepare_unlabelled

# How many of the topmost merges the drawn dendrogram expands before collapsing
# subtrees into a single "n points" leaf.
MAX_DRAWN_MERGES = 31

# Where the animation starts.
#
# This is also a correctness bound, not just a taste one. Stabilised labels are
# inherited from the first frame, so they never exceed this value, and the
# frontend palette cycles: if this were larger than the number of colours in
# CLASS_COLORS (12), two live clusters would be painted the same colour and the
# plot would appear to show fewer clusters than it found.
START_CLUSTERS = 12

SPEC = AlgorithmSpec(
    id="hierarchical",
    name="Hierarchical Clustering",
    task="clustering",
    tagline="Merge the two closest clusters, over and over, and read the tree.",
    description=[
        "Agglomerative clustering starts with every point as its own cluster and repeatedly merges "
        "the closest pair. It never needs <code>k</code> up front: it builds the entire hierarchy, "
        "from <em>n</em> clusters down to one, and you choose where to cut afterwards.",
        "That hierarchy is the <strong>dendrogram</strong> in the panel on the right. Height is the "
        "distance at which two groups merged, so a tall vertical run means those clusters stayed "
        "distinct for a long time — the sign of a real gap in the data. The dashed line is the "
        "current cut; everything below it is a cluster.",
        "<strong>Linkage</strong> defines the distance between two clusters, and changes the answer "
        "completely. <em>Ward</em> minimises within-cluster variance and produces compact, k-means-like "
        "blobs. <em>Single</em> uses the closest pair of points, so clusters chain along thin bridges — "
        "which is why it is the one linkage that can follow the two-moons shape.",
    ],
    watch_for=[
        "Switch linkage to single on the two-moons dataset: it traces the crescents. Then switch to ward and watch it cut them in half.",
        "A tall jump in the merge-distance chart is the natural number of clusters — the cut just below it.",
        "Single linkage on noisy data often chains everything into one blob plus a few singletons. That failure is why ward is the usual default.",
        "The dendrogram is computed once. Changing the cluster count just moves the cut line — no refitting.",
    ],
    step_unit="cut",
    step_hint="Each frame lowers the cut by one, merging exactly two clusters.",
    params=[
        Param(
            name="linkage",
            label="Linkage",
            type="select",
            default="ward",
            options=[
                {"value": "ward", "label": "Ward (compact, variance-based)"},
                {"value": "average", "label": "Average (mean pairwise distance)"},
                {"value": "complete", "label": "Complete (furthest pair)"},
                {"value": "single", "label": "Single (nearest pair, chains)"},
            ],
            help="How the distance between two clusters is measured. This changes the result more than anything else.",
        ),
        Param(
            name="n_clusters",
            label="Clusters to keep",
            type="int",
            default=3,
            min=1,
            max=10,
            step=1,
            help="Where to cut the tree. The animation merges down to this number.",
        ),
    ],
)


def _serialise(Z: np.ndarray, node: int, n_points: int, min_row: int) -> dict:
    """Nested dendrogram, collapsing everything below the top merges."""
    if node < n_points:
        return {"kind": "leaf", "count": 1, "height": 0.0}
    row = node - n_points
    if row < min_row:
        return {"kind": "collapsed", "count": int(Z[row][3]), "height": float(Z[row][2])}
    return {
        "kind": "node",
        "height": float(Z[row][2]),
        "count": int(Z[row][3]),
        "children": [
            _serialise(Z, int(Z[row][0]), n_points, min_row),
            _serialise(Z, int(Z[row][1]), n_points, min_row),
        ],
    }


def _stabilise(previous: np.ndarray | None, current: np.ndarray) -> np.ndarray:
    """Keep cluster colours steady as clusters merge.

    fcluster renumbers freely between cut levels, which would make the whole
    plot change colour on every frame. Each cluster instead inherits the colour
    of the largest group it came from.
    """
    if previous is None:
        return current
    out = np.empty_like(current)
    taken: set[int] = set()
    order = sorted(np.unique(current), key=lambda c: -int((current == c).sum()))
    unresolved = []
    for cluster in order:
        members = current == cluster
        values, counts = np.unique(previous[members], return_counts=True)
        for candidate in values[np.argsort(-counts)]:
            if int(candidate) not in taken:
                out[members] = candidate
                taken.add(int(candidate))
                break
        else:
            unresolved.append(cluster)
    # Bounded well below NOISE_CLASS (254), which a label must never collide
    # with. Merging alone cannot reach here, but a fallback should stay safe.
    spare = (c for c in range(200) if c not in taken)
    for cluster in unresolved:
        out[current == cluster] = next(spare)
    return out


def _centroids(X: np.ndarray, labels: np.ndarray) -> dict[int, list[float]]:
    return {
        int(c): X[labels == c].mean(axis=0).tolist() for c in np.unique(labels)
    }


def fit(points, params, grid: Grid, validation: float = 0.0) -> FitResult:
    # validation is unused: clustering has no labels to score a holdout against.
    del validation
    X = prepare_unlabelled(points, min_points=3)
    method = params["linkage"]
    target = int(params["n_clusters"])
    n_points = len(X)
    target = min(target, n_points)

    Z = linkage(X, method=method)

    # Cells inherit the cluster of the nearest labelled point. Like DBSCAN, this
    # is an extrapolation: agglomerative clustering has no predict().
    _, nearest = NearestNeighbors(n_neighbors=1).fit(X).kneighbors(grid.points)
    nearest = nearest.ravel()

    start = min(START_CLUSTERS, n_points)
    levels = list(range(start, target - 1, -1)) if start >= target else [target]

    steps: list[Step] = []
    previous_labels: np.ndarray | None = None
    previous_centroids: dict[int, list[float]] = {}

    for k in levels:
        labels = _stabilise(previous_labels, fcluster(Z, t=k, criterion="maxclust") - 1)
        found = len(np.unique(labels))
        # The merge that produced k clusters is the (n-k)th, i.e. row n-k-1.
        row = n_points - k - 1
        height = float(Z[row][2]) if 0 <= row < len(Z) else 0.0

        merge_link = None
        if previous_labels is not None:
            # Exactly two of the previous clusters now share a label.
            for cluster in np.unique(labels):
                came_from = np.unique(previous_labels[labels == cluster])
                if len(came_from) == 2:
                    a, b = (previous_centroids.get(int(c)) for c in came_from)
                    if a and b:
                        merge_link = [a, b]
                    break

        steps.append(
            Step(
                label=f"{found} cluster{'s' if found != 1 else ''}",
                description=(
                    f"Cut at height {height:.3f} leaves {found} cluster"
                    f"{'s' if found != 1 else ''}."
                    + (
                        " The dashed line joins the two clusters that just merged."
                        if merge_link
                        else " This is the starting cut, before any merging is shown."
                    )
                ),
                metrics={"clusters": found, "merge_distance": height},
                # Colour stability means labels keep their original values as
                # clusters merge, so they are not contiguous: the surface bound
                # is the largest label, not the number of clusters.
                surface=class_surface(labels[nearest], n_classes=int(labels.max()) + 1),
                extras={
                    "assignments": labels.tolist(),
                    # Deliberately no centroids: this algorithm has none, and
                    # drawing them would imply a k-means-like model.
                    "cut_height": height,
                    "merge_link": merge_link,
                    "n_clusters": found,
                },
            )
        )
        previous_labels = labels
        previous_centroids = _centroids(X, labels)

    final_labels = previous_labels
    silhouette = None
    if final_labels is not None and 2 <= len(np.unique(final_labels)) < n_points:
        silhouette = float(silhouette_score(X, final_labels))

    heights = [s.metrics["merge_distance"] for s in steps]
    biggest_jump = None
    if len(heights) > 2:
        jumps = [(heights[i] - heights[i + 1], steps[i].metrics["clusters"]) for i in range(len(heights) - 1)]
        biggest_jump = max(jumps)[1]

    notes = [
        "The shaded regions are an extrapolation, not the algorithm's own output: hierarchical "
        "clustering labels only the points it was given and has no predict() for new locations. "
        "Each cell takes the cluster of its nearest labelled point.",
        "The tree is built once. Moving the cut costs nothing, which is why you can pick k after "
        "looking at the dendrogram rather than before.",
    ]
    if biggest_jump is not None:
        notes.append(
            f"The largest jump in merge distance happens at {biggest_jump} clusters — usually the "
            f"most defensible place to cut."
        )
    if method == "single":
        notes.append(
            "Single linkage merges on the closest pair of points, so it follows thin shapes but is "
            "easily fooled into chaining separate clusters through a bridge of noise."
        )
    if silhouette is not None:
        notes.append(f"Silhouette score {silhouette:.3f} at the final cut.")

    return FitResult(
        task="clustering",
        steps=steps,
        metric_labels={
            "clusters": "Clusters",
            "merge_distance": "Merge distance (cut height)",
        },
        chart_metrics=["merge_distance", "clusters"],
        summary={
            "Clusters": int(len(np.unique(final_labels))) if final_labels is not None else 0,
            "Linkage": method,
            "Final cut height": heights[-1] if heights else 0.0,
            "Silhouette": silhouette,
        },
        notes=notes,
        extras={
            "dendrogram": _serialise(Z, 2 * n_points - 2, n_points, max(0, len(Z) - MAX_DRAWN_MERGES)),
            "max_height": float(Z[-1][2]),
            "linkage": method,
        },
    )
