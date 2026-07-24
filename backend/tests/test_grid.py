"""Unit tests for the grid/encoding helpers and the param coercion rules."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from app.algorithms.base import DataError, Param, prepare_labelled, prepare_regression, thin
from app.grid import (
    Grid,
    Viewport,
    class_surface,
    confidence_from_scores,
    encode_confidence,
    margin_confidence,
)


def test_grid_is_row_major_and_bottom_up():
    resolution = 16
    grid = Grid.build(Viewport(-1, 1, -2, 2), resolution=resolution)
    assert grid.points.shape == (resolution**2, 2)
    # First row sits at y_min and runs left to right; the last row is at y_max.
    assert grid.points[0].tolist() == [-1.0, -2.0]
    assert grid.points[resolution - 1].tolist() == [1.0, -2.0]
    assert grid.points[-1].tolist() == [1.0, 2.0]
    assert grid.points[resolution].tolist()[0] == -1.0  # second row, first column


def test_resolution_is_clamped():
    assert Grid.build(Viewport(-1, 1, -1, 1), resolution=4).resolution == 16
    assert Grid.build(Viewport(-1, 1, -1, 1), resolution=500).resolution == 96


def test_class_surface_round_trips():
    labels = np.array([0, 1, 2, 1])
    surface = class_surface(labels, n_classes=3, confidence=np.array([0.0, 0.5, 1.0, 0.25]))
    assert list(base64.b64decode(surface["classes"])) == [0, 1, 2, 1]
    decoded = list(base64.b64decode(surface["confidence"]))
    assert decoded[0] == 0 and decoded[2] == 255


def test_confidence_is_zero_on_a_two_class_boundary():
    scores = np.array([[0.5, 0.5], [0.99, 0.01]])
    confidence = confidence_from_scores(scores)
    assert confidence[0] == pytest.approx(0.0)
    assert confidence[1] > 0.9


def test_confidence_clamps_out_of_range_values():
    assert list(base64.b64decode(encode_confidence(np.array([-3.0, 7.0])))) == [0, 255]


def test_margin_confidence_saturates():
    confidence = margin_confidence(np.array([0.0, 1.0, 50.0]), scale=1.0)
    assert confidence[0] == 0.0
    assert confidence[-1] == 1.0


def test_thin_keeps_the_ends():
    thinned = thin(list(range(100)), max_items=10)
    assert len(thinned) <= 10
    assert thinned[0] == 0 and thinned[-1] == 99


@pytest.mark.parametrize(
    "param,raw,expected",
    [
        (Param("k", "k", "int", 3, min=1, max=10), "7.6", 8),
        (Param("k", "k", "int", 3, min=1, max=10), 99, 10),
        (Param("k", "k", "int", 3, min=1, max=10), "nonsense", 3),
        (Param("a", "a", "float", 0.5, min=0.0, max=1.0), 2.0, 1.0),
        (Param("f", "f", "bool", False), "true", True),
        (Param("f", "f", "bool", False), None, False),
        (
            Param("m", "m", "select", "x", options=[{"value": "x"}, {"value": "y"}]),
            "y",
            "y",
        ),
        (
            Param("m", "m", "select", "x", options=[{"value": "x"}, {"value": "y"}]),
            "z",
            "x",
        ),
    ],
)
def test_param_coercion(param, raw, expected):
    assert param.coerce(raw) == expected


def test_prepare_labelled_remaps_sparse_labels():
    points = [{"x": float(i), "y": 0.0, "label": 3 if i % 2 else 7} for i in range(8)]
    data = prepare_labelled(points)
    assert data.class_values == [3, 7]
    assert set(data.y.tolist()) == {0, 1}


def test_prepare_labelled_enforces_min_per_class():
    points = [{"x": float(i), "y": 0.0, "label": 0} for i in range(6)]
    points.append({"x": 1.0, "y": 1.0, "label": 1})
    with pytest.raises(DataError, match="at least 2 per class"):
        prepare_labelled(points, min_per_class=2)


def test_prepare_regression_splits_feature_and_target():
    points = [{"x": 1.0, "y": 5.0, "label": None}, {"x": 2.0, "y": 6.0, "label": None},
              {"x": 3.0, "y": 7.0, "label": None}]
    X, y = prepare_regression(points)
    assert X.shape == (3, 1)
    assert y.tolist() == [5.0, 6.0, 7.0]
