"""Algorithm implementations, one module per algorithm."""

from . import (  # noqa: F401
    decision_tree,
    kmeans,
    knn,
    linear_regression,
    logistic_regression,
    mlp,
    naive_bayes,
    polynomial_regression,
    random_forest,
    svm,
)

MODULES = [
    linear_regression,
    polynomial_regression,
    logistic_regression,
    knn,
    naive_bayes,
    svm,
    decision_tree,
    random_forest,
    mlp,
    kmeans,
]

__all__ = ["MODULES"]
