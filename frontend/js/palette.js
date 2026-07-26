/** Class colours, shared by the plot, the chart and every diagram. */

// At least as many entries as the largest number of clusters any algorithm can
// show at once (hierarchical starts at 12, k-means allows k = 10). classColor()
// cycles this list, so a shorter palette paints distinct clusters identically.
export const CLASS_COLORS = [
  '#4f9cf9', // blue
  '#f97362', // coral
  '#4dd6a8', // spring green
  '#f5c451', // amber
  '#b48bf2', // violet
  '#f28ec8', // pink
  '#5ad1e6', // cyan
  '#a8d95b', // lime
  '#ff9d4d', // orange
  '#8896f5', // indigo
  '#6fcf6f', // green
  '#e57ae5', // magenta
];

export const NEUTRAL = '#8ba0c0';

// Reserved surface byte for "no cluster here". Must match NOISE_CLASS in grid.py.
export const NOISE_CLASS = 254;
export const NOISE_COLOR = '#5c6a83';
export const CURVE_COLOR = '#ffd166';
export const REFERENCE_COLOR = '#7e8ca6';

/** Colour for class index `i`, cycling if there are more classes than colours.
 *  Negative indices and the reserved noise byte render as neutral grey. */
export function classColor(index) {
  if (index < 0 || index === NOISE_CLASS) return NOISE_COLOR;
  return CLASS_COLORS[((index % CLASS_COLORS.length) + CLASS_COLORS.length) % CLASS_COLORS.length];
}

const HEX = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i;

export function toRgb(hex) {
  const match = HEX.exec(hex);
  if (!match) return [255, 255, 255];
  return [parseInt(match[1], 16), parseInt(match[2], 16), parseInt(match[3], 16)];
}

export function rgba(hex, alpha) {
  const [r, g, b] = toRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Pre-computed RGB triples, so the surface renderer never parses hex per pixel. */
export const CLASS_RGB = CLASS_COLORS.map(toRgb);
export const NOISE_RGB = toRgb(NOISE_COLOR);

export function classRgb(index) {
  if (index === NOISE_CLASS) return NOISE_RGB;
  return CLASS_RGB[((index % CLASS_RGB.length) + CLASS_RGB.length) % CLASS_RGB.length];
}
