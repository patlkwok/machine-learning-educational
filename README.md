# ML Playground

An interactive, browser-based playground for classical machine learning. Click on a 2D plane to
place your own data points (or generate a dataset), pick an algorithm, and watch it train — one
real step at a time, with the decision boundary, the fitted curve and the error metrics updating
as it goes.

Every frame of every animation is a genuine intermediate state of the algorithm: an epoch of
gradient descent, one Lloyd iteration of k-means, one more level of tree depth, one more tree in
the forest. Nothing is faked for the sake of the visuals.

Everything runs on a laptop CPU. The heaviest configuration — a two-layer neural network of 100
neurons each over 1000 points — trains and renders every frame in about a second.

```
./run.sh          # then open http://127.0.0.1:8000
```

---

## What is in it

| Algorithm | Task | What the animation steps through |
|---|---|---|
| **Linear regression** | Regression | Epochs of gradient descent, walking towards the closed-form least-squares line (drawn dashed for comparison) |
| **Polynomial regression** | Regression | Degree 1 → *d*, with training and validation error side by side so the overfitting U-curve appears |
| **Logistic regression** | Classification | Epochs of SGD on the log loss, with the confidence band around the boundary narrowing |
| **k-nearest neighbours** | Classification | k = 1 upwards: the Voronoi partition smoothing out as neighbours are added |
| **Gaussian naive Bayes** | Classification | Data fed in chunks via `partial_fit`, with each class's fitted 2σ ellipse drawn |
| **Support vector machine** | Classification | The solver's own iterations, with support vectors circled and the ±1 margins drawn for a linear kernel |
| **Decision tree** | Classification | One more level of depth per frame, plus a rendered tree diagram and the split lines on the plot |
| **Random forest** | Classification | One more tree per frame, with an optional overlay of the newest single tree to compare against the ensemble |
| **Neural network (MLP)** | Classification | Epochs of backpropagation, with a weight-shaded network diagram |
| **k-means** | Clustering | Assign and update shown as *separate* frames, with centroid movement trails |
| **Gaussian mixture** | Clustering | EM iterations, with covariance ellipses that tilt and points faded by how strongly they belong |
| **DBSCAN** | Clustering | The flood fill itself, with the eps radius drawn on the expanding point and core / border / noise points drawn differently |
| **Hierarchical** | Clustering | The cut sliding down a dendrogram, merging two clusters per frame, with selectable linkage |

Synthetic datasets: Gaussian blobs, stretched blobs, two moons, concentric circles, spirals, XOR
quadrants, uniform noise, and four regression shapes (noisy line, sine wave, cubic, step). All have
adjustable sample count and noise. **Shuffle** draws a new dataset and **Resample split** redraws the holdout without touching the data; no seed is ever shown.

## Using it

* **Click** the plot to add a point. In classification mode the coloured chips above the plot pick
  which class you are placing; press **+ class** for a third, fourth, … class.
* **Drag** a point to move it and watch the model follow in real time.
* **Shift-click** or **right-click** a point to delete it.
* **Space** plays/pauses; **←** and **→** step through frames.
* *Auto-run* re-fits on every change. Turn it off and changes are staged until you press **Generate** for data or **Train** for the model.
* **Validation split** holds back 0–50% of the points, scored separately, so training and validation metrics can be compared. Held-out points are ringed on the plot.
* Pressing **Train** or **Generate** rewinds and plays the animation; tweaking a slider re-fits in
  place so you can compare two settings at the same frame.

## Running it

The only requirements are Python 3.10 or newer and a shell — Bash on macOS and Linux, PowerShell
on Windows. No editor, IDE or Node toolchain is needed: the frontend is plain ES modules served
straight from disk, with no build step. Working in a virtual environment is recommended, and both
run scripts pick `.venv` up automatically if it exists.

**macOS / Linux**

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
./run.sh
```

```bash
./run.sh --dev          # auto-reload while editing
PORT=9000 ./run.sh      # pick a port
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\run.ps1
```

```powershell
.\run.ps1 -Dev            # auto-reload while editing
.\run.ps1 -Port 9000      # pick a port
```

Then open <http://127.0.0.1:8000>.

Or drive uvicorn yourself, on any platform:

```bash
python -m uvicorn backend.app.main:app --port 8000
```

### Tests

```bash
python -m pytest
```

170 tests cover every algorithm endpoint, every dataset generator, the grid encoding, parameter
coercion and clamping, and the error paths (unlabelled data, a single class, too few points,
degenerate input).

## How it is built

```
backend/
  app/
    main.py          FastAPI app: JSON API + serves the frontend
    registry.py      algorithm id -> spec + fit function
    schemas.py       request validation
    datasets.py      synthetic data generators
    grid.py          decision-surface sampling and byte encoding
    algorithms/
      base.py        Step / FitResult / AlgorithmSpec, data preparation, shared helpers
      *.py           one module per algorithm
  tests/
frontend/
  index.html
  css/style.css
  js/
    main.js          app wiring: catalogue -> controls -> fit -> playback -> insights
    plot.js          the interactive canvas: surfaces, points, curves, drag-to-edit
    chart.js         per-step metric chart
    diagrams.js      SVG tree and neural-network diagrams
    controls.js      builds controls from the parameter specs the API returns
    api.js, palette.js
```

The models are scikit-learn. Nothing is reimplemented by hand — the animations come from asking
scikit-learn for genuine intermediate states:

* `partial_fit` for SGD regressors/classifiers and naive Bayes (one epoch or one chunk per frame)
* `warm_start` for the random forest (one more tree per frame), the MLP (one epoch per frame) and the Gaussian mixture (one EM iteration per frame)
* `SVC(max_iter=n)` to cap the libsvm solver part-way through
* refitting at each depth / k / degree for trees, k-NN and polynomials
* an explicit Lloyd loop over `kmeans_plusplus` + `pairwise_distances_argmin` for k-means, so the
  assign and update halves can be shown as separate frames
* an explicit flood fill over `NearestNeighbors` radius queries for DBSCAN, for the same reason
* one `scipy` linkage tree for hierarchical clustering, re-cut with `fcluster` at each frame

DBSCAN and hierarchical clustering are **transductive**: they label the points they were given and
have no `predict()` for anywhere else. Their shaded regions are therefore an extrapolation — nearest
core point within `eps`, and nearest labelled point respectively — and each run says so in its notes.

### The API

Three endpoints, all JSON:

* `GET /api/algorithms` — the catalogue. Each algorithm ships its own parameter specs, prose
  description and "things to try" list; **the frontend builds its entire control panel from this**,
  so adding a hyperparameter needs no frontend change.
* `POST /api/generate` — `{generator, n_samples, noise, seed, classes}` → points.
* `POST /api/fit` — `{algorithm, params, points, viewport, grid_resolution, validation_split,
  validation_seed}` → one `step` per animation frame, each with metrics, a decision surface and
  any algorithm-specific extras.

Decision surfaces are the bulky part of the payload, so each frame's grid is sent as a base64
`uint8` array — one byte of class index per cell, plus an optional confidence byte that the
frontend turns into per-pixel alpha. Responses are gzipped; a 48-frame animation is a few
hundred kilobytes.

### Adding an algorithm

1. Drop a module in `backend/app/algorithms/` exposing `SPEC` (an `AlgorithmSpec`) and
   `fit(points, params, grid) -> FitResult`.
2. Add it to `MODULES` in `backend/app/algorithms/__init__.py`.

That is all — the sidebar entry, the hyperparameter sliders, the explanation card, the metric chart
and the playback bar are all driven by the spec and the returned steps.
