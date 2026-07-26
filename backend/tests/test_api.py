"""End-to-end tests over the HTTP API."""

from __future__ import annotations

import base64

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
    assert len(catalogue["algorithms"]) == 12
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
    mse = [step["metrics"]["mse"] for step in payload["steps"]]
    assert mse[-1] < mse[0]


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
