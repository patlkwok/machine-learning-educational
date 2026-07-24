"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .grid import DEFAULT_RESOLUTION, MAX_RESOLUTION, MIN_RESOLUTION

MAX_POINTS = 2000


class Point(BaseModel):
    x: float
    y: float
    label: int | None = None

    @field_validator("x", "y")
    @classmethod
    def _finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("coordinates must be finite numbers")
        return value

    @field_validator("label")
    @classmethod
    def _sane_label(cls, value: int | None) -> int | None:
        if value is not None and not 0 <= value <= 15:
            raise ValueError("label must be between 0 and 15")
        return value


class ViewportModel(BaseModel):
    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -5.0
    y_max: float = 5.0

    @field_validator("x_max")
    @classmethod
    def _x_ordered(cls, value: float, info) -> float:
        x_min = info.data.get("x_min")
        if x_min is not None and value <= x_min:
            raise ValueError("x_max must be greater than x_min")
        return value

    @field_validator("y_max")
    @classmethod
    def _y_ordered(cls, value: float, info) -> float:
        y_min = info.data.get("y_min")
        if y_min is not None and value <= y_min:
            raise ValueError("y_max must be greater than y_min")
        return value


class FitRequest(BaseModel):
    algorithm: str
    params: dict[str, Any] = Field(default_factory=dict)
    points: list[Point]
    viewport: ViewportModel = Field(default_factory=ViewportModel)
    grid_resolution: int = Field(default=DEFAULT_RESOLUTION, ge=MIN_RESOLUTION, le=MAX_RESOLUTION)

    @field_validator("points")
    @classmethod
    def _not_too_many(cls, value: list[Point]) -> list[Point]:
        if len(value) > MAX_POINTS:
            raise ValueError(f"at most {MAX_POINTS} points are supported")
        return value


class GenerateRequest(BaseModel):
    generator: str
    n_samples: int = Field(default=200, ge=4, le=1200)
    noise: float = Field(default=0.2, ge=0.0, le=1.0)
    seed: int = Field(default=0, ge=0, le=999_999)
    classes: int = Field(default=2, ge=2, le=6)
