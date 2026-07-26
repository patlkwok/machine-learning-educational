"""End-to-end tests over the HTTP API."""

from __future__ import annotations

import base64
import math

import pytest
from fastapi.testclient import TestClient

from app import datasets
from app.main import app

client = TestClient(app)

VIEWPORT = {"x_min": -5.0, "x_max": 5.0, "y_min": -5.0, "y_max": 5.0}
RESOLUTION = 32


@pytest.fixture(scope="module")
def catalogue() -> dict:
    response = client.get("/api/algorithms")
    assert response.status_code == 200
    return response.json()


def points_for(task: str) -> list[dict]:
    generator = {
        "regression": "wave_regression",
        "classification": "moons",
        "clustering": "blobs",
    }[task]
    points = datasets.generate(generator, n_samples=90, noise=0.2, seed=3, classes=3)
    if task == "clustering":
        points = [{**point, "label": None} for point in points]
    return points


def fit(algorithm: str, points: list[dict], params: dict | None = None) -> dict:
    response = client.post(
        "/api/fit",
        json={
            "algorithm": algorithm,
            "params": params or {},
            "points": points,
            "viewport": VIEWPORT,
            "grid_resolution": RESOLUTION,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------- basics --


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_catalogue_shape(catalogue):
    assert len(catalogue["algorithms"]) == 13
    assert {spec["task"] for spec in catalogue["algorithms"]} == {
        "regression",
        "classification",
        "clustering",
    }
    for spec in catalogue["algorithms"]:
        assert spec["id"] and spec["name"] and spec["tagline"]
        assert spec["description"] and spec["watch_for"]
        for param in spec["params"]:
            assert param["type"] in {"int", "float", "select", "bool"}
            assert param["scale"] in {"linear", "log"}
            if param["type"] == "select":
                assert param["options"]
                assert param["default"] in [o["value"] for o in param["options"]]
            if param["type"] in {"int", "float"}:
                # The frontend needs both bounds to build a slider and to clamp
                # whatever the user types into the box beside it.
                assert param["min"] is not None and param["max"] is not None
                assert param["min"] <= param["default"] <= param["max"]


def test_epochs_are_capped_at_100():
    for spec in client.get("/api/algorithms").json()["algorithms"]:
        for param in spec["params"]:
            if param["name"] == "epochs":
                assert param["max"] == 100, f"{spec['id']} allows {param['max']} epochs"


def test_learning_rates_are_log_scaled():
    found = 0
    for spec in client.get("/api/algorithms").json()["algorithms"]:
        for param in spec["params"]:
            if param["name"] != "learning_rate":
                continue
            found += 1
            assert param["scale"] == "log"
            # A log slider divides by zero on a zero lower bound.
            assert param["min"] > 0
            assert param["min"] == pytest.approx(1e-5)
            assert param["max"] == pytest.approx(10.0)
    assert found == 3, f"expected 3 learning-rate params, found {found}"


@pytest.mark.parametrize(
    "algorithm,epochs,expected",
    [
        ("linear_regression", 500, 100),  # clamped down to the cap
        ("linear_regression", 0, 1),
        ("logistic_regression", 500, 100),
        ("mlp", 500, 100),
    ],
)
def test_epoch_values_are_clamped(algorithm, epochs, expected):
    task = ALGORITHM_TASKS[algorithm]
    payload = fit(algorithm, points_for(task), {"epochs": epochs})
    assert payload["params"]["epochs"] == expected


def test_extreme_learning_rate_is_accepted_not_crashed():
    """The top of the log range diverges; that is a lesson, not an error."""
    payload = fit("linear_regression", points_for("regression"), {"learning_rate": 10.0})
    assert payload["params"]["learning_rate"] == pytest.approx(10.0)
    assert any("diverged" in note for note in payload["notes"])

    tiny = fit("linear_regression", points_for("regression"), {"learning_rate": 1e-5})
    assert tiny["params"]["learning_rate"] == pytest.approx(1e-5)


def test_index_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "ML Playground" in response.text


# ------------------------------------------------------------ generators --


@pytest.mark.parametrize("generator", list(datasets.GENERATORS))
def test_every_generator(generator):
    spec = datasets.GENERATORS[generator]
    response = client.post(
        "/api/generate",
        json={
            "generator": generator,
            "n_samples": 60,
            "noise": 0.3,
            "seed": 5,
            "classes": spec.max_classes,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    points = payload["points"]

    assert 40 <= len(points) <= 80
    assert all(-5 <= p["x"] <= 5 and -5 <= p["y"] <= 5 for p in points)

    if spec.kind == "regression":
        assert all(p["label"] is None for p in points)
    else:
        labels = {p["label"] for p in points}
        assert None not in labels
        assert len(labels) >= 2


def test_generator_is_deterministic():
    body = {"generator": "moons", "n_samples": 40, "noise": 0.2, "seed": 11, "classes": 2}
    first = client.post("/api/generate", json=body).json()
    second = client.post("/api/generate", json=body).json()
    assert first == second


def test_unknown_generator_404():
    response = client.post("/api/generate", json={"generator": "nope"})
    assert response.status_code == 404


# ------------------------------------------------------------ every fit --


ALGORITHM_TASKS = {
    "linear_regression": "regression",
    "polynomial_regression": "regression",
    "logistic_regression": "classification",
    "knn": "classification",
    "naive_bayes": "classification",
    "svm": "classification",
    "decision_tree": "classification",
    "random_forest": "classification",
    "mlp": "classification",
    "kmeans": "clustering",
    "dbscan": "clustering",
    "hierarchical": "clustering",
    "gmm": "clustering",
}

# Reserved surface byte for "no cluster here"; see NOISE_CLASS in grid.py.
NOISE_CLASS = 254

# CLASS_COLORS.length in frontend/js/palette.js. classColor() cycles that list,
# so two clusters whose labels are congruent modulo this are drawn identically.
PALETTE_SIZE = 12


@pytest.mark.parametrize("algorithm,task", sorted(ALGORITHM_TASKS.items()))
def test_fit_returns_a_usable_animation(algorithm, task):
    payload = fit(algorithm, points_for(task))

    assert payload["task"] == task
    assert payload["grid"]["resolution"] == RESOLUTION
    assert len(payload["steps"]) >= 2
    assert payload["elapsed_ms"] >= 0

    for index, step in enumerate(payload["steps"]):
        assert step["index"] == index
        assert step["label"] and step["description"]
        assert step["metrics"]

        if task == "regression":
            assert step["curve"], "regression steps must carry a curve to draw"
            assert all(len(pair) == 2 for pair in step["curve"])
        else:
            surface = step["surface"]
            assert surface, "classification/clustering steps must carry a decision surface"
            decoded = base64.b64decode(surface["classes"])
            assert len(decoded) == RESOLUTION**2
            limit = max(surface["n_classes"], 1)
            assert all(cell < limit or cell == NOISE_CLASS for cell in decoded)
            if surface["confidence"]:
                assert len(base64.b64decode(surface["confidence"])) == RESOLUTION**2

    # Every charted series must line up with the step list.
    for series in payload["metric_series"]:
        assert len(series["values"]) == len(payload["steps"])


@pytest.mark.parametrize("algorithm", sorted(ALGORITHM_TASKS))
def test_fit_is_deterministic(algorithm):
    points = points_for(ALGORITHM_TASKS[algorithm])
    assert fit(algorithm, points)["steps"] == fit(algorithm, points)["steps"]


def test_params_are_echoed_and_clamped():
    payload = fit("knn", points_for("classification"), {"max_k": 10_000, "weights": "bogus"})
    assert payload["params"]["max_k"] == 99  # clamped to the spec maximum
    assert payload["params"]["weights"] == "uniform"  # unknown option falls back to default


# ------------------------------------------------------ algorithm detail --


def test_gradient_descent_reduces_error():
    payload = fit("linear_regression", points_for("regression"), {"learning_rate": 0.1, "epochs": 40})
    mse = [step["metrics"]["train_mse"] for step in payload["steps"]]
    assert mse[-1] < mse[0]


# --------------------------------------------------- validation split --

SUPERVISED = {a: t for a, t in ALGORITHM_TASKS.items() if t != "clustering"}


def fit_split(algorithm, points, split, params=None):
    response = client.post(
        "/api/fit",
        json={
            "algorithm": algorithm,
            "params": params or {},
            "points": points,
            "viewport": VIEWPORT,
            "grid_resolution": RESOLUTION,
            "validation_split": split,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("algorithm,task", sorted(SUPERVISED.items()))
def test_every_supervised_algorithm_reports_validation_metrics(algorithm, task):
    payload = fit_split(algorithm, points_for(task), 0.25)
    keys = {series["key"] for series in payload["metric_series"]}
    assert any(key.startswith("val_") for key in keys), f"no validation series: {sorted(keys)}"
    assert any(key.startswith("train_") for key in keys), f"no training series: {sorted(keys)}"

    split = payload["split"]
    assert split["n_validation"] > 0
    assert split["n_train"] + split["n_validation"] == len(points_for(task))
    # Every step must carry both, so the chart never has a ragged series.
    for step in payload["steps"]:
        assert any(k.startswith("val_") and v is not None for k, v in step["metrics"].items())


@pytest.mark.parametrize("algorithm,task", sorted(SUPERVISED.items()))
def test_zero_split_reports_training_metrics_only(algorithm, task):
    payload = fit_split(algorithm, points_for(task), 0.0)
    keys = {series["key"] for series in payload["metric_series"]}
    assert not any(key.startswith("val_") for key in keys), sorted(keys)
    assert payload["split"]["n_validation"] == 0
    assert any("No points are held back" in note for note in payload["notes"])


def test_validation_points_are_excluded_from_training():
    """The held-out indices must be a genuine partition of the point list."""
    points = points_for("classification")
    payload = fit_split("decision_tree", points, 0.3)
    held = payload["split"]["validation_indices"]

    assert len(held) == len(set(held)), "indices must be unique"
    assert all(0 <= i < len(points) for i in held)
    assert len(held) == payload["split"]["n_validation"]
    assert payload["split"]["n_train"] == len(points) - len(held)


def test_split_is_stratified_for_classification():
    points = points_for("classification")
    payload = fit_split("knn", points, 0.4)
    held = set(payload["split"]["validation_indices"])
    labels = [p["label"] for i, p in enumerate(points) if i in held]
    # Both classes must appear on the held-out side, or validation accuracy is
    # measuring something other than the model's ability to separate them.
    assert len(set(labels)) == 2, f"held-out labels: {set(labels)}"


def test_split_is_deterministic():
    points = points_for("classification")
    first = fit_split("svm", points, 0.3)["split"]["validation_indices"]
    second = fit_split("svm", points, 0.3)["split"]["validation_indices"]
    assert first == second


def fit_reseed(algorithm, points, split, seed):
    response = client.post(
        "/api/fit",
        json={
            "algorithm": algorithm,
            "points": points,
            "viewport": VIEWPORT,
            "grid_resolution": RESOLUTION,
            "validation_split": split,
            "validation_seed": seed,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_resampling_redraws_the_split_without_touching_the_data():
    """The whole point: a different partition of exactly the same points."""
    points = points_for("classification")
    first = fit_reseed("knn", points, 0.3, 0)
    second = fit_reseed("knn", points, 0.3, 12345)

    a = set(first["split"]["validation_indices"])
    b = set(second["split"]["validation_indices"])
    assert a != b, "a new seed must give a new partition"
    assert len(a) == len(b) == first["split"]["n_validation"]
    # Same data, same sizes: only which points were held back has changed.
    assert first["split"]["n_train"] == second["split"]["n_train"]
    assert first["params"] == second["params"]


def test_resampling_is_itself_reproducible():
    points = points_for("classification")
    first = fit_reseed("decision_tree", points, 0.3, 777)["split"]["validation_indices"]
    second = fit_reseed("decision_tree", points, 0.3, 777)["split"]["validation_indices"]
    assert first == second


def test_every_resample_stays_stratified():
    points = points_for("classification")
    for seed in (0, 1, 99, 4242, 999999):
        payload = fit_reseed("knn", points, 0.4, seed)
        held = set(payload["split"]["validation_indices"])
        labels = {p["label"] for i, p in enumerate(points) if i in held}
        assert len(labels) == 2, f"seed {seed} held out only class {labels}"


def test_validation_seed_is_rejected_outside_its_range():
    for bad in (-1, 1_000_000):
        response = client.post(
            "/api/fit",
            json={
                "algorithm": "knn",
                "points": points_for("classification"),
                "viewport": VIEWPORT,
                "validation_split": 0.2,
                "validation_seed": bad,
            },
        )
        assert response.status_code == 422, bad


MIN_TRAIN_POINTS = 4  # mirrors base.MIN_TRAIN_POINTS


@pytest.mark.parametrize("n_points", [4, 5, 6, 10, 40])
def test_split_never_starves_the_training_set(n_points):
    """Validation is given up before the training set drops below the minimum."""
    points = [{"x": i * 0.4 - 2, "y": (i % 2) * 1.5, "label": i % 2} for i in range(n_points)]
    payload = fit_split("knn", points, 0.5)  # the largest split on offer
    assert payload["split"]["n_train"] >= min(MIN_TRAIN_POINTS, n_points)
    assert payload["split"]["n_train"] + payload["split"]["n_validation"] == n_points


def test_dataset_too_small_to_split_says_so():
    """At 4 points there is nothing to spare, so the split is refused outright."""
    points = [{"x": i * 0.9 - 1.5, "y": (i % 2) * 1.5, "label": i % 2} for i in range(4)]
    payload = fit_split("knn", points, 0.5)
    assert payload["split"]["n_validation"] == 0
    assert payload["split"]["n_train"] == 4
    assert any("too few points" in note.lower() for note in payload["notes"])


def test_overfitting_gap_is_called_out():
    """A deep tree on noisy data should trip the overfitting note."""
    points = [
        {**p, "label": p["label"]}
        for p in datasets.generate("uniform", n_samples=120, noise=0.5, seed=1, classes=2)
    ]
    payload = fit_split("decision_tree", points, 0.3, {"max_depth": 12})
    final = payload["steps"][-1]["metrics"]
    assert final["train_accuracy"] > final["val_accuracy"]
    assert any("overfitting" in note.lower() for note in payload["notes"])


def test_split_is_rejected_outside_its_range():
    for bad in (-0.1, 0.75):
        response = client.post(
            "/api/fit",
            json={
                "algorithm": "knn",
                "points": points_for("classification"),
                "viewport": VIEWPORT,
                "validation_split": bad,
            },
        )
        assert response.status_code == 422, bad


def test_clustering_ignores_the_split():
    payload = fit_split("kmeans", points_for("clustering"), 0.4)
    assert payload["split"] is None


def test_kmeans_offers_a_reseed_button_instead_of_a_seed_slider():
    catalogue = client.get("/api/algorithms").json()
    kmeans = next(s for s in catalogue["algorithms"] if s["id"] == "kmeans")

    seed = next(p for p in kmeans["params"] if p["name"] == "seed")
    assert seed["hidden"] is True, "the seed number should never be drawn"
    assert kmeans["reseed"] == {
        "param": "seed",
        "label": kmeans["reseed"]["label"],
        "help": kmeans["reseed"]["help"],
    }
    assert kmeans["reseed"]["param"] == "seed"
    assert kmeans["reseed"]["label"]

    # No other algorithm shows a visible seed control either.
    for spec in catalogue["algorithms"]:
        visible = [p["name"] for p in spec["params"] if not p["hidden"]]
        assert not any("seed" in name for name in visible), f"{spec['id']}: {visible}"


def test_kmeans_reseeding_changes_the_starting_centroids():
    points = points_for("clustering")
    first = fit("kmeans", points, {"init": "random", "seed": 1})
    second = fit("kmeans", points, {"init": "random", "seed": 98765})
    assert first["steps"][0]["extras"]["centroids"] != second["steps"][0]["extras"]["centroids"]
    # Same seed must still reproduce, so a run can be reasoned about.
    assert fit("kmeans", points, {"init": "random", "seed": 1})["steps"] == first["steps"]


def _blobs(n=240, classes=3, generator="anisotropic", noise=0.2, seed=2):
    return [
        {**p, "label": None}
        for p in datasets.generate(generator, n_samples=n, noise=noise, seed=seed, classes=classes)
    ]


@pytest.mark.parametrize("covariance", ["full", "tied", "diag", "spherical"])
def test_gmm_log_likelihood_never_falls(covariance):
    """EM cannot decrease the log-likelihood. That is a theorem, not a hope."""
    payload = fit("gmm", _blobs(), {"covariance_type": covariance, "max_iter": 30})
    values = [step["metrics"]["log_likelihood"] for step in payload["steps"]]
    assert all(b >= a - 1e-6 for a, b in zip(values, values[1:])), values


def test_gmm_reports_soft_responsibilities():
    payload = fit("gmm", _blobs(), {"n_components": 3})
    for step in payload["steps"]:
        responsibility = step["extras"]["responsibility"]
        assert len(responsibility) == 240
        # A max-probability can never be below 1/k or above 1.
        assert all(1 / 3 - 1e-6 <= r <= 1.0 for r in responsibility)


def test_only_full_and_tied_covariance_can_tilt():
    """The covariance type is the whole lesson: it controls ellipse shape."""
    points = _blobs()

    def angles(covariance):
        payload = fit("gmm", points, {"covariance_type": covariance})
        return [e["angle"] for e in payload["steps"][-1]["extras"]["ellipses"]]

    # Axis-aligned means every angle is a multiple of a right angle: the ellipse
    # axes line up with x and y even when the major axis is the vertical one.
    for covariance in ("diag", "spherical"):
        for angle in angles(covariance):
            assert min(abs(angle % (math.pi / 2)), math.pi / 2 - abs(angle % (math.pi / 2))) < 1e-6, (
                f"{covariance} produced a tilted ellipse: {angle}"
            )

    # Full covariance on deliberately rotated blobs must tilt off the axes.
    tilted = [
        a for a in angles("full")
        if min(abs(a % (math.pi / 2)), math.pi / 2 - abs(a % (math.pi / 2))) > 0.05
    ]
    assert tilted, "full covariance should tilt on the stretched-blobs dataset"


def test_tied_covariance_shares_one_shape():
    payload = fit("gmm", _blobs(), {"covariance_type": "tied", "n_components": 3})
    ellipses = payload["steps"][-1]["extras"]["ellipses"]
    radii = {(round(e["rx"], 6), round(e["ry"], 6), round(e["angle"], 6)) for e in ellipses}
    assert len(radii) == 1, f"tied components should share a shape, got {radii}"


def test_gmm_bic_finds_the_right_k_on_separated_blobs():
    """Seed 9 puts the three centres far apart; BIC should then recover k = 3.

    It does not do so universally. When two centres fall close together BIC
    prefers fewer components, which is BIC being right about what the data
    supports rather than BIC failing.
    """
    points = _blobs(generator="blobs", classes=3, noise=0.08, seed=9)
    bic = {
        k: fit("gmm", points, {"n_components": k, "covariance_type": "full"})["summary"]["BIC"]
        for k in range(1, 7)
    }
    assert min(bic, key=bic.get) == 3, bic


def test_gmm_bic_charges_for_unused_components():
    points = _blobs(generator="blobs", classes=3, noise=0.08, seed=9)
    bic = {
        k: fit("gmm", points, {"n_components": k, "covariance_type": "full"})["summary"]["BIC"]
        for k in (3, 10)
    }
    assert bic[10] > bic[3], f"ten components should cost more than three: {bic}"


def test_gmm_offers_a_reseed_button():
    catalogue = client.get("/api/algorithms").json()
    spec = next(s for s in catalogue["algorithms"] if s["id"] == "gmm")
    assert spec["task"] == "clustering"
    assert spec["reseed"]["param"] == "seed"
    assert next(p for p in spec["params"] if p["name"] == "seed")["hidden"] is True


def test_kmeans_inertia_never_increases():
    payload = fit("kmeans", points_for("clustering"), {"k": 3, "max_iter": 20})
    inertia = [step["metrics"]["inertia"] for step in payload["steps"]]
    assert all(b <= a + 1e-6 for a, b in zip(inertia, inertia[1:]))
    assert all(step["extras"]["centroids"] for step in payload["steps"])


def _moons(n=200, noise=0.1):
    return [
        {**p, "label": None}
        for p in datasets.generate("moons", n_samples=n, noise=noise, seed=2)
    ]


def test_dbscan_separates_the_moons_that_kmeans_cannot():
    payload = fit("dbscan", _moons(), {"eps": 0.45, "min_samples": 5})
    assert payload["summary"]["Clusters"] == 2
    final = payload["steps"][-1]
    assert final["metrics"]["unassigned"] == payload["summary"]["Noise points"]


def test_dbscan_flood_fill_only_ever_claims_more_points():
    payload = fit("dbscan", _moons(), {"eps": 0.45, "min_samples": 5})
    claimed = [step["metrics"]["claimed"] for step in payload["steps"]]
    assert claimed[0] == 0
    assert all(b >= a for a, b in zip(claimed, claimed[1:]))
    assert claimed[-1] == max(claimed)


def test_dbscan_reports_point_roles_and_eps():
    payload = fit("dbscan", _moons(), {"eps": 0.45, "min_samples": 5})
    for step in payload["steps"]:
        roles = step["extras"]["roles"]
        assert len(roles) == 200
        assert set(roles) <= {0, 1, 2, 3}
        assert step["extras"]["eps"] == pytest.approx(0.45)
    # Only the finished frame may declare a point to be noise.
    assert 3 not in set(payload["steps"][0]["extras"]["roles"])


def test_dbscan_tiny_eps_makes_everything_noise():
    payload = fit("dbscan", _moons(), {"eps": 0.02, "min_samples": 5})
    assert payload["summary"]["Clusters"] == 0
    assert payload["summary"]["Noise points"] == 200
    assert any("noise" in note.lower() for note in payload["notes"])
    # Every grid cell must fall back to the reserved noise byte.
    cells = set(base64.b64decode(payload["steps"][-1]["surface"]["classes"]))
    assert cells == {NOISE_CLASS}


def test_dbscan_surface_is_flagged_as_extrapolated():
    payload = fit("dbscan", _moons())
    assert any("extrapolation" in note for note in payload["notes"])


def test_hierarchical_merges_down_to_the_requested_cut():
    payload = fit("hierarchical", points_for("clustering"), {"n_clusters": 4})
    counts = [step["metrics"]["clusters"] for step in payload["steps"]]
    assert counts[-1] == 4
    assert counts[0] > counts[-1]
    assert all(b <= a for a, b in zip(counts, counts[1:])), "cluster count must only fall"


def test_hierarchical_merge_distance_only_grows():
    payload = fit("hierarchical", points_for("clustering"), {"n_clusters": 2})
    heights = [step["metrics"]["merge_distance"] for step in payload["steps"]]
    assert all(b >= a - 1e-9 for a, b in zip(heights, heights[1:]))


@pytest.mark.parametrize("method", ["ward", "average", "complete", "single"])
@pytest.mark.parametrize("k", [2, 3, 4, 6, 10])
def test_hierarchical_clusters_never_share_a_colour(method, k):
    """Distinct clusters must land on distinct palette slots.

    Colour stability keeps label values fixed as clusters merge, so labels are
    not contiguous. If any two exceed the palette length apart, they cycle onto
    the same colour and the plot appears to show fewer clusters than it found.
    """
    payload = fit("hierarchical", points_for("clustering"), {"linkage": method, "n_clusters": k})
    for step in payload["steps"]:
        labels = set(step["extras"]["assignments"])
        slots = {label % PALETTE_SIZE for label in labels}
        assert len(slots) == len(labels), (
            f"{method} at {step['label']}: labels {sorted(labels)} collapse to "
            f"{len(slots)} colours"
        )
    assert payload["steps"][-1]["metrics"]["clusters"] == k


def test_kmeans_clusters_never_share_a_colour():
    payload = fit("kmeans", points_for("clustering"), {"k": 10})
    for step in payload["steps"]:
        labels = set(step["extras"]["assignments"])
        assert len({label % PALETTE_SIZE for label in labels}) == len(labels)


def test_hierarchical_emits_a_dendrogram():
    payload = fit("hierarchical", points_for("clustering"), {"n_clusters": 3})
    tree = payload["extras"]["dendrogram"]
    assert tree["kind"] == "node"
    assert payload["extras"]["max_height"] > 0

    def leaves(node):
        return 1 if node["kind"] != "node" else sum(leaves(c) for c in node["children"])

    # Truncated for legibility rather than one leaf per point.
    assert 2 <= leaves(tree) <= 40
    assert all(step["extras"]["cut_height"] >= 0 for step in payload["steps"])


@pytest.mark.parametrize("method", ["ward", "average", "complete", "single"])
def test_every_linkage_runs(method):
    payload = fit("hierarchical", points_for("clustering"), {"linkage": method, "n_clusters": 3})
    assert payload["summary"]["Linkage"] == method
    assert payload["steps"][-1]["metrics"]["clusters"] == 3


def test_single_linkage_differs_from_ward():
    """Linkage is the parameter that actually changes the answer."""
    points = _moons(noise=0.06)
    ward = fit("hierarchical", points, {"linkage": "ward", "n_clusters": 2})
    single = fit("hierarchical", points, {"linkage": "single", "n_clusters": 2})
    assert ward["steps"][-1]["extras"]["assignments"] != single["steps"][-1]["extras"]["assignments"]


def test_knn_k1_memorises_the_training_set():
    payload = fit("knn", points_for("classification"), {"max_k": 5})
    assert payload["steps"][0]["metrics"]["train_accuracy"] == pytest.approx(1.0)


def test_decision_tree_grows_monotonically():
    payload = fit("decision_tree", points_for("classification"), {"max_depth": 6})
    leaves = [step["metrics"]["leaves"] for step in payload["steps"]]
    assert all(b >= a for a, b in zip(leaves, leaves[1:]))
    assert payload["steps"][-1]["extras"]["tree"]["is_leaf"] is False


def test_random_forest_adds_trees():
    payload = fit("random_forest", points_for("classification"), {"n_estimators": 12})
    counts = [step["metrics"]["n_trees"] for step in payload["steps"]]
    assert counts[0] == 1 and counts[-1] == 12
    assert all(b > a for a, b in zip(counts, counts[1:]))


def test_svm_reports_support_vectors():
    payload = fit("svm", points_for("classification"), {"kernel": "linear", "frames": 6})
    final = payload["steps"][-1]
    assert final["metrics"]["n_support"] >= 2
    assert final["extras"]["support_indices"]
    assert len(final["extras"]["margin_lines"]) == 3  # boundary plus both margins


def test_svm_frame_budget_only_trims_on_large_datasets():
    """Each SVM frame refits the solver, so the budget scales with dataset size."""
    small = [
        {**p} for p in datasets.generate("moons", n_samples=200, noise=0.2, seed=5)
    ]
    large = [
        {**p} for p in datasets.generate("moons", n_samples=1000, noise=0.2, seed=5)
    ]

    asked = 20
    small_payload = fit("svm", small, {"frames": asked})
    large_payload = fit("svm", large, {"frames": asked})

    # Small datasets get every frame they asked for, and say nothing about it.
    assert not any("requested frames" in note for note in small_payload["notes"])
    # Large ones are trimmed, and say so rather than silently disagreeing.
    assert len(large_payload["steps"]) < len(small_payload["steps"])
    assert any("requested frames" in note for note in large_payload["notes"])
    # Never trimmed so far that the animation stops being an animation.
    assert len(large_payload["steps"]) >= 6


def test_generator_supports_the_full_point_range():
    """The frontend slider tops out at 1000; the API must accept that."""
    response = client.post(
        "/api/generate",
        json={"generator": "moons", "n_samples": 1000, "noise": 0.2, "seed": 1, "classes": 2},
    )
    assert response.status_code == 200
    assert len(response.json()["points"]) == 1000


def test_mlp_emits_a_network_diagram():
    payload = fit("mlp", points_for("classification"), {"layer1": 8, "layer2": 8, "epochs": 5})
    network = payload["steps"][-1]["extras"]["network"]
    # Two features in; two hidden layers; a single logistic output for 2 classes.
    assert network["layers"] == [2, 8, 8, 1]
    assert len(network["weights"]) == 3


def test_mlp_layer2_zero_gives_one_hidden_layer():
    payload = fit("mlp", points_for("classification"), {"layer1": 12, "layer2": 0, "epochs": 5})
    assert payload["steps"][-1]["extras"]["network"]["layers"] == [2, 12, 1]
    assert payload["summary"]["Architecture"] == "2 → 12 → 1"


def test_mlp_neurons_are_capped_at_100():
    catalogue = client.get("/api/algorithms").json()
    mlp = next(s for s in catalogue["algorithms"] if s["id"] == "mlp")
    widths = {p["name"]: p for p in mlp["params"] if p["name"] in {"layer1", "layer2"}}
    assert widths["layer1"]["min"] == 1 and widths["layer1"]["max"] == 100
    assert widths["layer2"]["min"] == 0 and widths["layer2"]["max"] == 100

    payload = fit("mlp", points_for("classification"), {"layer1": 5000, "layer2": 5000, "epochs": 2})
    assert payload["params"]["layer1"] == 100 and payload["params"]["layer2"] == 100
    assert payload["summary"]["Architecture"] == "2 → 100 → 100 → 1"


def test_widest_mlp_stays_responsive():
    """2-100-100-1 is the heaviest configuration the UI can request."""
    payload = fit("mlp", points_for("classification"), {"layer1": 100, "layer2": 100, "epochs": 100})
    assert payload["elapsed_ms"] < 8000, f"took {payload['elapsed_ms']} ms"
    # Too many weights to draw legibly, so the diagram is omitted by design.
    assert payload["steps"][-1]["extras"]["network"] is None


def test_naive_bayes_ellipses_are_axis_aligned():
    payload = fit("naive_bayes", points_for("classification"), {"chunks": 4})
    ellipses = payload["steps"][-1]["extras"]["ellipses"]
    assert len(ellipses) == 2
    assert all(e["rx"] > 0 and e["ry"] > 0 for e in ellipses)


def test_polynomial_degree_sweep_lowers_training_error():
    payload = fit("polynomial_regression", points_for("regression"), {"max_degree": 6})
    mse = [step["metrics"]["train_mse"] for step in payload["steps"]]
    assert mse[-1] < mse[0]


def test_accuracy_summaries_are_percentages():
    payload = fit("decision_tree", points_for("classification"))
    assert payload["summary"]["Training accuracy"].endswith("%")
    assert payload["metric_formats"]["train_accuracy"] == "percent"


# ------------------------------------------------------------- failures --


def test_classifier_rejects_unlabelled_points():
    response = client.post(
        "/api/fit",
        json={
            "algorithm": "logistic_regression",
            "points": points_for("regression"),
            "viewport": VIEWPORT,
            "grid_resolution": RESOLUTION,
        },
    )
    assert response.status_code == 400
    assert "class labels" in response.json()["detail"]


def test_classifier_rejects_a_single_class():
    points = [{"x": i * 0.4 - 2, "y": 0.3 * i, "label": 0} for i in range(10)]
    response = client.post(
        "/api/fit",
        json={"algorithm": "svm", "points": points, "viewport": VIEWPORT, "grid_resolution": 16},
    )
    assert response.status_code == 400
    assert "one class" in response.json()["detail"].lower()


def test_too_few_points_is_a_clear_error():
    response = client.post(
        "/api/fit",
        json={
            "algorithm": "linear_regression",
            "points": [{"x": 0.0, "y": 0.0, "label": None}],
            "viewport": VIEWPORT,
        },
    )
    assert response.status_code == 400
    assert "at least" in response.json()["detail"]


def test_degenerate_regression_input_is_rejected():
    points = [{"x": 1.0, "y": float(i), "label": None} for i in range(6)]
    response = client.post(
        "/api/fit",
        json={"algorithm": "linear_regression", "points": points, "viewport": VIEWPORT},
    )
    assert response.status_code == 400
    assert "same x value" in response.json()["detail"]


def test_unknown_algorithm_404():
    response = client.post(
        "/api/fit",
        json={"algorithm": "quantum_regression", "points": points_for("regression")},
    )
    assert response.status_code == 404


def test_too_many_points_rejected():
    points = [{"x": 0.0, "y": 0.0, "label": 0} for _ in range(2001)]
    response = client.post("/api/fit", json={"algorithm": "knn", "points": points})
    assert response.status_code == 422


def test_bad_viewport_rejected():
    response = client.post(
        "/api/fit",
        json={
            "algorithm": "knn",
            "points": points_for("classification"),
            "viewport": {"x_min": 5, "x_max": -5, "y_min": -5, "y_max": 5},
        },
    )
    assert response.status_code == 422
