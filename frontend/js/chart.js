/**
 * Small multi-series line chart for the per-step metrics.
 *
 * Metrics rarely share a scale (log loss vs accuracy vs support-vector count),
 * so each series is normalised to its own range and the legend carries the
 * real numbers. The vertical rule marks the frame currently on screen.
 */

const SERIES_COLORS = ['#5b9cf8', '#f5c451', '#4dd6a8', '#f28ec8'];
const PADDING = { top: 10, right: 8, bottom: 16, left: 8 };

export class MetricChart {
  constructor(canvas, legendElement) {
    this.canvas = canvas;
    this.legend = legendElement;
    this.ctx = canvas.getContext('2d');
    this.series = [];
    this.stepIndex = 0;
    window.addEventListener('resize', () => this.render());
  }

  setResult(result) {
    this.series = result ? result.metric_series : [];
    this.formats = result ? result.metric_formats || {} : {};
    this.render();
  }

  setStep(index) {
    this.stepIndex = index;
    this.render();
  }

  _resize() {
    const ratio = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(rect.width, 1);
    this.height = Math.max(rect.height, 1);
    this.canvas.width = Math.round(this.width * ratio);
    this.canvas.height = Math.round(this.height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  render() {
    this._resize();
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    const usable = this.series.filter((s) => s.values.some((v) => v !== null && v !== undefined));
    if (!usable.length) {
      this.legend.innerHTML = '';
      ctx.fillStyle = 'rgba(100, 116, 143, 0.9)';
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('Train a model to see its metrics.', 8, this.height / 2);
      return;
    }

    const plotWidth = this.width - PADDING.left - PADDING.right;
    const plotHeight = this.height - PADDING.top - PADDING.bottom;
    const count = Math.max(usable[0].values.length, 2);
    const xAt = (index) => PADDING.left + (index / (count - 1)) * plotWidth;

    // Baseline + frame marker.
    ctx.strokeStyle = 'rgba(147, 161, 187, 0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PADDING.left, PADDING.top + plotHeight + 0.5);
    ctx.lineTo(PADDING.left + plotWidth, PADDING.top + plotHeight + 0.5);
    ctx.stroke();

    const markerX = xAt(Math.min(this.stepIndex, count - 1));
    ctx.strokeStyle = 'rgba(232, 237, 247, 0.35)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(markerX, PADDING.top);
    ctx.lineTo(markerX, PADDING.top + plotHeight);
    ctx.stroke();
    ctx.setLineDash([]);

    const legendRows = [];

    usable.forEach((serie, serieIndex) => {
      const color = SERIES_COLORS[serieIndex % SERIES_COLORS.length];
      const numbers = serie.values.filter((v) => v !== null && v !== undefined);
      const min = Math.min(...numbers);
      const max = Math.max(...numbers);
      const span = max - min || 1;
      const yAt = (value) => PADDING.top + plotHeight - ((value - min) / span) * plotHeight;

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.9;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      let started = false;
      serie.values.forEach((value, index) => {
        if (value === null || value === undefined) {
          started = false;
          return;
        }
        const px = xAt(index);
        const py = yAt(value);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      });
      ctx.stroke();

      const current = serie.values[Math.min(this.stepIndex, serie.values.length - 1)];
      if (current !== null && current !== undefined) {
        ctx.beginPath();
        ctx.arc(markerX, yAt(current), 3.4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#161d2c';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      legendRows.push({
        label: serie.label,
        color,
        value: current,
        format: (this.formats || {})[serie.key],
      });
    });

    this.legend.innerHTML = legendRows
      .map(
        (row) => `
        <div class="legend-row">
          <span class="legend-swatch" style="background:${row.color}"></span>
          <span>${escapeHtml(row.label)}</span>
          <span class="legend-value">${formatValue(row.value, row.format)}</span>
        </div>`
      )
      .join('');
  }
}

export function formatValue(value, format) {
  if (value === null || value === undefined) return '—';
  if (typeof value !== 'number') return String(value);
  if (format === 'percent') return `${(value * 100).toFixed(1)}%`;
  if (Number.isInteger(value)) return String(value);
  if (Math.abs(value) >= 1000 || (Math.abs(value) < 0.001 && value !== 0)) {
    return value.toExponential(2);
  }
  return value.toFixed(Math.abs(value) < 1 ? 4 : 3);
}

export function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[character]);
}
