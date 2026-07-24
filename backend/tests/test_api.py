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
    assert len(catalogue["algorithms"]) == 10
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
            if param["type"] == "select":
                assert param["options"]
                assert param["default"] in [o["value"] for o in param["options"]]


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
}


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
            assert max(decoded) < max(surface["n_classes"], 1)
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
    payload = fit("mlp", points_for("classification"), {"hidden": "8,8", "epochs": 5})
    network = payload["steps"][-1]["extras"]["network"]
    # Two features in; two hidden layers; a single logistic output for 2 classes.
    assert network["layers"] == [2, 8, 8, 1]
    assert len(network["weights"]) == 3


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
