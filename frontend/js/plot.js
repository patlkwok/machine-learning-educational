/**
 * The interactive 2D plot.
 *
 * Draw order matters: decision surface, then grid, then model geometry
 * (split lines, margins, ellipses, curves), then the data points on top so
 * they are never obscured.
 */

import { decodeBytes } from './api.js';
import { CURVE_COLOR, NEUTRAL, REFERENCE_COLOR, classColor, classRgb, rgba } from './palette.js';

const HIT_RADIUS = 11;
const POINT_RADIUS = 5;

export class Plot {
  constructor(canvas, { viewport, onPointsChange }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.viewport = viewport;
    this.onPointsChange = onPointsChange;

    this.points = [];
    this.task = 'classification';
    this.activeClass = 0;
    this.result = null;
    this.stepIndex = 0;
    this.overlays = {};
    this.hoverIndex = -1;
    this.dragIndex = -1;
    this.surfaceCache = new Map();

    this._bindEvents();
    this._resize();
    window.addEventListener('resize', () => {
      this._resize();
      this.render();
    });
  }

  /* ----------------------------------------------------------- state */

  setPoints(points) {
    this.points = points;
    this.render();
  }

  setTask(task) {
    this.task = task;
    this.render();
  }

  setActiveClass(index) {
    this.activeClass = index;
  }

  setResult(result) {
    this.result = result;
    this.stepIndex = result ? result.steps.length - 1 : 0;
    this.surfaceCache.clear();
    this.render();
  }

  setStep(index) {
    this.stepIndex = index;
    this.render();
  }

  setOverlays(overlays) {
    this.overlays = overlays;
    this.render();
  }

  get step() {
    if (!this.result || !this.result.steps.length) return null;
    const index = Math.min(Math.max(this.stepIndex, 0), this.result.steps.length - 1);
    return this.result.steps[index];
  }

  /* ------------------------------------------------------ geometry */

  _resize() {
    const ratio = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(rect.width, 1);
    this.height = Math.max(rect.height, 1);
    this.canvas.width = Math.round(this.width * ratio);
    this.canvas.height = Math.round(this.height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  toScreenX(x) {
    const { x_min, x_max } = this.viewport;
    return ((x - x_min) / (x_max - x_min)) * this.width;
  }

  toScreenY(y) {
    const { y_min, y_max } = this.viewport;
    return this.height - ((y - y_min) / (y_max - y_min)) * this.height;
  }

  toWorld(px, py) {
    const { x_min, x_max, y_min, y_max } = this.viewport;
    return {
      x: x_min + (px / this.width) * (x_max - x_min),
      y: y_min + ((this.height - py) / this.height) * (y_max - y_min),
    };
  }

  _eventPos(event) {
    const rect = this.canvas.getBoundingClientRect();
    return { px: event.clientX - rect.left, py: event.clientY - rect.top };
  }

  _hitTest(px, py) {
    let best = -1;
    let bestDistance = HIT_RADIUS;
    this.points.forEach((point, index) => {
      const dx = this.toScreenX(point.x) - px;
      const dy = this.toScreenY(point.y) - py;
      const distance = Math.hypot(dx, dy);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    });
    return best;
  }

  /* -------------------------------------------------- interaction */

  _bindEvents() {
    const canvas = this.canvas;

    canvas.addEventListener('contextmenu', (event) => {
      event.preventDefault();
      const { px, py } = this._eventPos(event);
      const index = this._hitTest(px, py);
      if (index >= 0) this._removePoint(index);
    });

    canvas.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      const { px, py } = this._eventPos(event);
      const index = this._hitTest(px, py);

      if (index >= 0 && (event.shiftKey || event.altKey)) {
        this._removePoint(index);
        return;
      }
      if (index >= 0) {
        this.dragIndex = index;
        this.dragMoved = false;
        canvas.setPointerCapture(event.pointerId);
        return;
      }
      this._addPoint(this.toWorld(px, py));
    });

    canvas.addEventListener('pointermove', (event) => {
      const { px, py } = this._eventPos(event);
      if (this.dragIndex >= 0) {
        const world = this.toWorld(px, py);
        const point = this.points[this.dragIndex];
        point.x = this._clampX(world.x);
        point.y = this._clampY(world.y);
        this.dragMoved = true;
        this.render();
        this.onPointsChange({ reason: 'drag', settled: false });
        return;
      }
      const hover = this._hitTest(px, py);
      if (hover !== this.hoverIndex) {
        this.hoverIndex = hover;
        canvas.style.cursor = hover >= 0 ? 'grab' : 'crosshair';
        this.render();
      }
    });

    const endDrag = (event) => {
      if (this.dragIndex < 0) return;
      this.dragIndex = -1;
      if (canvas.hasPointerCapture?.(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      if (this.dragMoved) this.onPointsChange({ reason: 'drag', settled: true });
    };
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);

    canvas.addEventListener('pointerleave', () => {
      if (this.hoverIndex !== -1) {
        this.hoverIndex = -1;
        this.render();
      }
    });
  }

  _clampX(x) {
    const { x_min, x_max } = this.viewport;
    return Math.min(Math.max(x, x_min), x_max);
  }

  _clampY(y) {
    const { y_min, y_max } = this.viewport;
    return Math.min(Math.max(y, y_min), y_max);
  }

  _addPoint({ x, y }) {
    const label = this.task === 'classification' ? this.activeClass : null;
    this.points.push({ x: this._clampX(x), y: this._clampY(y), label });
    this.render();
    this.onPointsChange({ reason: 'add', settled: true });
  }

  _removePoint(index) {
    this.points.splice(index, 1);
    this.hoverIndex = -1;
    this.render();
    this.onPointsChange({ reason: 'remove', settled: true });
  }

  /* -------------------------------------------------------- render */

  render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = '#0b0f18';
    ctx.fillRect(0, 0, this.width, this.height);

    const step = this.step;
    if (step) {
      if (this.overlays.singleTree && step.extras?.single_tree_surface) {
        // Drawn crisp so one tree's blocky rectangles stand out against the
        // smoothly blended ensemble underneath.
        this._drawSurface(step.extras.single_tree_surface, `single-${this.stepIndex}`, 0.34, false);
      }
      if (step.surface && this.overlays.surface !== false) {
        this._drawSurface(step.surface, `main-${this.stepIndex}`, 0.62);
      }
    }

    this._drawGrid();

    if (step) {
      this._drawSplitLines(step);
      this._drawMarginLines(step);
      this._drawEllipses(step);
      this._drawCentroidTrails(step);
      if (this.result?.extras?.reference_curve && this.overlays.reference !== false) {
        this._drawCurve(this.result.extras.reference_curve, REFERENCE_COLOR, 1.6, [6, 5]);
      }
      if (step.curve) {
        if (step.extras?.show_residuals && this.overlays.residuals !== false) {
          this._drawResiduals(step.curve);
        }
        this._drawCurve(step.curve, CURVE_COLOR, 2.4);
      }
    }

    this._drawPoints(step);
    if (step) this._drawCentroids(step);
  }

  _drawSurface(surface, cacheKey, alpha, smooth = true) {
    let bitmap = this.surfaceCache.get(cacheKey);
    if (!bitmap) {
      bitmap = this._buildSurface(surface);
      this.surfaceCache.set(cacheKey, bitmap);
    }
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.imageSmoothingEnabled = smooth;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, 0, 0, this.width, this.height);
    ctx.restore();
  }

  _buildSurface(surface) {
    const classes = decodeBytes(surface.classes);
    const confidence = surface.confidence ? decodeBytes(surface.confidence) : null;
    const resolution = this.result.grid.resolution;

    const offscreen = document.createElement('canvas');
    offscreen.width = resolution;
    offscreen.height = resolution;
    const image = offscreen.getContext('2d').createImageData(resolution, resolution);
    const data = image.data;

    for (let row = 0; row < resolution; row += 1) {
      // Backend rows run bottom-up; canvas rows run top-down.
      const flipped = resolution - 1 - row;
      for (let col = 0; col < resolution; col += 1) {
        const source = flipped * resolution + col;
        const target = (row * resolution + col) * 4;
        const [r, g, b] = classRgb(classes[source]);
        const sureness = confidence ? confidence[source] / 255 : 1;
        data[target] = r;
        data[target + 1] = g;
        data[target + 2] = b;
        data[target + 3] = Math.round(255 * (0.3 + 0.7 * sureness));
      }
    }
    offscreen.getContext('2d').putImageData(image, 0, 0);
    return offscreen;
  }

  _drawGrid() {
    const ctx = this.ctx;
    const { x_min, x_max, y_min, y_max } = this.viewport;

    ctx.save();
    ctx.lineWidth = 1;
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = 'rgba(147, 161, 187, 0.55)';

    for (let x = Math.ceil(x_min); x <= x_max; x += 1) {
      const sx = Math.round(this.toScreenX(x)) + 0.5;
      const axis = x === 0;
      ctx.strokeStyle = axis ? 'rgba(147, 161, 187, 0.34)' : 'rgba(147, 161, 187, 0.11)';
      ctx.beginPath();
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, this.height);
      ctx.stroke();
      if (!axis && x % 2 === 0) ctx.fillText(String(x), sx + 3, this.height - 5);
    }

    for (let y = Math.ceil(y_min); y <= y_max; y += 1) {
      const sy = Math.round(this.toScreenY(y)) + 0.5;
      const axis = y === 0;
      ctx.strokeStyle = axis ? 'rgba(147, 161, 187, 0.34)' : 'rgba(147, 161, 187, 0.11)';
      ctx.beginPath();
      ctx.moveTo(0, sy);
      ctx.lineTo(this.width, sy);
      ctx.stroke();
      if (!axis && y % 2 === 0) ctx.fillText(String(y), 4, sy - 4);
    }

    const labels =
      this.task === 'regression'
        ? { x: 'feature x →', y: '↑ target y' }
        : { x: 'feature x₁ →', y: '↑ feature x₂' };
    ctx.fillStyle = 'rgba(147, 161, 187, 0.75)';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(labels.x, this.width - 82, this.height - 20);
    ctx.fillText(labels.y, 10, 18);
    ctx.restore();
  }

  _drawCurve(curve, color, width, dash) {
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, this.width, this.height);
    ctx.clip();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    curve.forEach(([x, y], index) => {
      const sx = this.toScreenX(x);
      const sy = this.toScreenY(y);
      if (index === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    });
    ctx.stroke();
    ctx.restore();
  }

  _curveValueAt(curve, x) {
    if (!curve.length) return null;
    if (x <= curve[0][0]) return curve[0][1];
    if (x >= curve[curve.length - 1][0]) return curve[curve.length - 1][1];
    const span = curve[curve.length - 1][0] - curve[0][0];
    const approx = Math.floor(((x - curve[0][0]) / span) * (curve.length - 1));
    const index = Math.min(Math.max(approx, 0), curve.length - 2);
    const [x0, y0] = curve[index];
    const [x1, y1] = curve[index + 1];
    if (x1 === x0) return y0;
    return y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
  }

  _drawResiduals(curve) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = 'rgba(249, 115, 98, 0.55)';
    ctx.lineWidth = 1.2;
    this.points.forEach((point) => {
      const predicted = this._curveValueAt(curve, point.x);
      if (predicted === null) return;
      ctx.beginPath();
      ctx.moveTo(this.toScreenX(point.x), this.toScreenY(point.y));
      ctx.lineTo(this.toScreenX(point.x), this.toScreenY(predicted));
      ctx.stroke();
    });
    ctx.restore();
  }

  _drawSplitLines(step) {
    const lines = step.extras?.split_lines;
    if (!lines || this.overlays.splits === false) return;
    const ctx = this.ctx;
    ctx.save();
    lines.forEach(({ depth, points }) => {
      ctx.strokeStyle = `rgba(232, 237, 247, ${Math.max(0.18, 0.75 - depth * 0.11)})`;
      ctx.lineWidth = Math.max(0.7, 2.1 - depth * 0.22);
      ctx.beginPath();
      ctx.moveTo(this.toScreenX(points[0][0]), this.toScreenY(points[0][1]));
      ctx.lineTo(this.toScreenX(points[1][0]), this.toScreenY(points[1][1]));
      ctx.stroke();
    });
    ctx.restore();
  }

  _drawMarginLines(step) {
    const lines = step.extras?.margin_lines;
    if (!lines || !lines.length) return;
    const ctx = this.ctx;
    ctx.save();
    lines.forEach(({ kind, points }) => {
      ctx.strokeStyle = kind === 'boundary' ? 'rgba(232, 237, 247, 0.9)' : 'rgba(232, 237, 247, 0.45)';
      ctx.lineWidth = kind === 'boundary' ? 2 : 1.2;
      ctx.setLineDash(kind === 'boundary' ? [] : [7, 5]);
      ctx.beginPath();
      ctx.moveTo(this.toScreenX(points[0][0]), this.toScreenY(points[0][1]));
      ctx.lineTo(this.toScreenX(points[1][0]), this.toScreenY(points[1][1]));
      ctx.stroke();
    });
    ctx.restore();
  }

  _drawEllipses(step) {
    const ellipses = step.extras?.ellipses;
    if (!ellipses || this.overlays.ellipses === false) return;
    const ctx = this.ctx;
    const scaleX = this.width / (this.viewport.x_max - this.viewport.x_min);
    const scaleY = this.height / (this.viewport.y_max - this.viewport.y_min);
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.6;
    ellipses.forEach((ellipse) => {
      ctx.strokeStyle = rgba(classColor(ellipse.class_index), 0.9);
      ctx.beginPath();
      ctx.ellipse(
        this.toScreenX(ellipse.cx),
        this.toScreenY(ellipse.cy),
        Math.max(ellipse.rx * scaleX, 1),
        Math.max(ellipse.ry * scaleY, 1),
        0,
        0,
        Math.PI * 2
      );
      ctx.stroke();
    });
    ctx.restore();
  }

  _drawCentroidTrails(step) {
    const previous = step.extras?.previous_centroids;
    const current = step.extras?.centroids;
    if (!previous || !current) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    previous.forEach((from, index) => {
      const to = current[index];
      if (!to) return;
      ctx.strokeStyle = rgba(classColor(index), 0.85);
      ctx.beginPath();
      ctx.moveTo(this.toScreenX(from[0]), this.toScreenY(from[1]));
      ctx.lineTo(this.toScreenX(to[0]), this.toScreenY(to[1]));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(this.toScreenX(from[0]), this.toScreenY(from[1]), 3, 0, Math.PI * 2);
      ctx.fillStyle = rgba(classColor(index), 0.5);
      ctx.fill();
      ctx.setLineDash([4, 4]);
    });
    ctx.restore();
  }

  _drawPoints(step) {
    const ctx = this.ctx;
    const assignments = step?.extras?.assignments || null;
    const supportSet = new Set(step?.extras?.support_indices || []);

    this.points.forEach((point, index) => {
      const sx = this.toScreenX(point.x);
      const sy = this.toScreenY(point.y);

      let color = NEUTRAL;
      if (assignments && assignments[index] !== undefined) color = classColor(assignments[index]);
      else if (point.label !== null && point.label !== undefined) color = classColor(point.label);
      else if (this.task === 'regression') color = '#cfd9ea';

      if (supportSet.has(index)) {
        ctx.beginPath();
        ctx.arc(sx, sy, POINT_RADIUS + 4.5, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(sx, sy, index === this.hoverIndex ? POINT_RADIUS + 1.5 : POINT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = 'rgba(11, 15, 24, 0.9)';
      ctx.stroke();
    });
  }

  _drawCentroids(step) {
    const centroids = step.extras?.centroids;
    if (!centroids) return;
    const ctx = this.ctx;
    centroids.forEach(([x, y], index) => {
      const sx = this.toScreenX(x);
      const sy = this.toScreenY(y);
      ctx.save();
      ctx.translate(sx, sy);
      ctx.beginPath();
      ctx.arc(0, 0, 9, 0, Math.PI * 2);
      ctx.fillStyle = rgba(classColor(index), 0.95);
      ctx.fill();
      ctx.lineWidth = 2.2;
      ctx.strokeStyle = '#0b0f18';
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(-4, 0);
      ctx.lineTo(4, 0);
      ctx.moveTo(0, -4);
      ctx.lineTo(0, 4);
      ctx.strokeStyle = '#0b0f18';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
    });
  }
}
