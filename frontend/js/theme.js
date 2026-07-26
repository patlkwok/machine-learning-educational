/**
 * Theme handling.
 *
 * Canvas and SVG cannot use CSS variables directly, so the colours are declared
 * once in style.css and read back from the document here. That keeps a single
 * source of truth: adding a theme means editing CSS, not JavaScript.
 */

const STORAGE_KEY = 'ml-playground-theme';
const VARIABLES = [
  'plot-bg',
  'plot-grid',
  'plot-axis',
  'plot-label',
  'plot-ink',
  'plot-halo',
  'diagram-node-bg',
  'chart-marker-halo',
  'text',
  'text-muted',
  'text-faint',
  'accent',
  'border',
];

let cache = null;

/** Current theme colours, keyed by camelCase name (plotBg, plotGrid, ...). */
export function colors() {
  if (cache) return cache;
  const style = getComputedStyle(document.documentElement);
  cache = {};
  VARIABLES.forEach((name) => {
    const key = name.replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    cache[key] = style.getPropertyValue(`--${name}`).trim();
  });
  return cache;
}

/** Same colour with an explicit alpha. Accepts #rgb, #rrggbb and rgb/rgba(). */
export function fade(color, alpha) {
  const value = String(color).trim();
  if (value.startsWith('#')) {
    const hex = value.slice(1);
    const full = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex;
    const n = parseInt(full, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
  }
  const parts = value.match(/[\d.]+/g);
  if (!parts || parts.length < 3) return value;
  return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
}

export function current() {
  return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

/** Apply a theme, remember it, and invalidate the colour cache. */
export function apply(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  cache = null;
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* private mode: the choice simply does not persist */
  }
  return theme;
}

/** Stored choice, else the OS preference, else dark. */
export function initial() {
  let stored = null;
  try {
    stored = localStorage.getItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
  if (stored === 'light' || stored === 'dark') return stored;
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}
