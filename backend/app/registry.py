"""Lookup table mapping algorithm ids to their spec and fit function."""

from __future__ import annotations

from typing import Callable

from .algorithms import MODULES
from .algorithms.base import AlgorithmSpec, FitResult, SplitSpec
from .grid import Grid

_REGISTRY: dict[str, tuple[AlgorithmSpec, Callable[..., FitResult]]] = {
    module.SPEC.id: (module.SPEC, module.fit) for module in MODULES
}

# Order matters: this is the order the frontend lists them in.
ORDER = [module.SPEC.id for module in MODULES]

TASK_LABELS = {
    "regression": "Regression",
    "classification": "Classification",
    "clustering": "Clustering",
}


def algorithm_specs() -> list[dict]:
    return [_REGISTRY[key][0].as_dict() for key in ORDER]


def get_spec(algorithm_id: str) -> AlgorithmSpec:
    entry = _REGISTRY.get(algorithm_id)
    if entry is None:
        raise KeyError(algorithm_id)
    return entry[0]


def run(
    algorithm_id: str,
    params: dict,
    points: list[dict],
    grid: Grid,
    validation: float = 0.0,
    validation_seed: int = 0,
) -> dict:
    entry = _REGISTRY.get(algorithm_id)
    if entry is None:
        raise KeyError(algorithm_id)
    spec, fit = entry
    resolved = spec.resolve(params)
    split_spec = SplitSpec(fraction=validation, seed=validation_seed)
    result = fit(points, resolved, grid, split_spec)
    payload = result.as_dict(algorithm_id, grid)
    payload["params"] = resolved
    payload["validation_split"] = validation
    payload["validation_seed"] = validation_seed
    return payload
