/** Class colours, shared by the plot, the chart and every diagram. */

export const CLASS_COLORS = [
  '#4f9cf9',
  '#f97362',
  '#4dd6a8',
  '#f5c451',
  '#b48bf2',
  '#f28ec8',
];

export const NEUTRAL = '#8ba0c0';
export const CURVE_COLOR = '#ffd166';
export const REFERENCE_COLOR = '#7e8ca6';

/** Colour for class index `i`, cycling if there are more classes than colours. */
export function classColor(index) {
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

export function classRgb(index) {
  return CLASS_RGB[((index % CLASS_RGB.length) + CLASS_RGB.length) % CLASS_RGB.length];
}
