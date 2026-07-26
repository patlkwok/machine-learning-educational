"""FastAPI application: JSON API plus the static frontend."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import datasets, registry
from .algorithms.base import DataError
from .grid import Grid, Viewport
from .schemas import FitRequest, GenerateRequest

LOGGER = logging.getLogger("ml_playground")

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(
    title="ML Playground",
    version="1.0.0",
    description="Interactive visualisations of classical machine learning algorithms.",
)

# Animation payloads are mostly base64 label grids, which compress very well.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/algorithms")
def list_algorithms() -> dict:
    return {
        "algorithms": registry.algorithm_specs(),
        "generators": datasets.generator_specs(),
        "task_labels": registry.TASK_LABELS,
    }


@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict:
    try:
        points = datasets.generate(
            generator=request.generator,
            n_samples=request.n_samples,
            noise=request.noise,
            seed=request.seed,
            classes=request.classes,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{request.generator}'")

    spec = datasets.GENERATORS[request.generator]
    return {"generator": request.generator, "kind": spec.kind, "points": points}


@app.post("/api/fit")
def fit(request: FitRequest) -> dict:
    viewport = Viewport(
        x_min=request.viewport.x_min,
        x_max=request.viewport.x_max,
        y_min=request.viewport.y_min,
        y_max=request.viewport.y_max,
    )
    grid = Grid.build(viewport, request.grid_resolution)
    points = [p.model_dump() for p in request.points]

    started = time.perf_counter()
    try:
        payload = registry.run(
            request.algorithm,
            request.params,
            points,
            grid,
            request.validation_split,
            request.validation_seed,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown algorithm '{request.algorithm}'")
    except DataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:  # numpy's LinAlgError lands here too
        LOGGER.exception("fit failed for %s", request.algorithm)
        raise HTTPException(status_code=400, detail=f"Could not fit this model: {exc}")

    payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return payload


if FRONTEND_DIR.is_dir():

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
