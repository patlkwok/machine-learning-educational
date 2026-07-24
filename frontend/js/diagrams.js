/** SVG diagrams of the fitted model: decision trees and neural networks. */

import { classColor } from './palette.js';
import { escapeHtml } from './chart.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

function el(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

/* ----------------------------------------------------------- tree ----- */

const NODE_WIDTH = 108;
const NODE_HEIGHT = 40;
const LEVEL_GAP = 62;
const SIBLING_GAP = 14;
const MAX_DRAWN_DEPTH = 5;

/** Lay the tree out left-to-right, giving each subtree the width it needs. */
function layout(node, depth, cursor) {
  const truncated = depth >= MAX_DRAWN_DEPTH && !node.is_leaf;
  if (node.is_leaf || truncated) {
    const box = {
      node,
      depth,
      truncated,
      x: cursor.x,
      width: NODE_WIDTH,
      children: [],
    };
    cursor.x += NODE_WIDTH + SIBLING_GAP;
    return box;
  }
  const left = layout(node.left, depth + 1, cursor);
  const right = layout(node.right, depth + 1, cursor);
  const centre = (left.x + left.width / 2 + right.x + right.width / 2) / 2 - NODE_WIDTH / 2;
  return {
    node,
    depth,
    truncated: false,
    x: centre,
    width: NODE_WIDTH,
    children: [left, right],
  };
}

function collect(box, out = []) {
  out.push(box);
  box.children.forEach((child) => collect(child, out));
  return out;
}

export function renderTree(container, tree, classValues) {
  container.innerHTML = '';
  if (!tree) return;

  const cursor = { x: 0 };
  const root = layout(tree, 0, cursor);
  const boxes = collect(root);
  const maxDepth = Math.max(...boxes.map((box) => box.depth));
  const width = Math.max(cursor.x, NODE_WIDTH) + 8;
  const height = maxDepth * LEVEL_GAP + NODE_HEIGHT + 8;

  const svg = el('svg', { width, height, viewBox: `0 0 ${width} ${height}` });

  const yOf = (depth) => depth * LEVEL_GAP + 4;

  // Edges first so nodes sit on top of them.
  boxes.forEach((box) => {
    box.children.forEach((child, index) => {
      svg.appendChild(
        el('path', {
          d: `M ${box.x + NODE_WIDTH / 2} ${yOf(box.depth) + NODE_HEIGHT}
              C ${box.x + NODE_WIDTH / 2} ${yOf(box.depth) + NODE_HEIGHT + 22},
                ${child.x + NODE_WIDTH / 2} ${yOf(child.depth) - 22},
                ${child.x + NODE_WIDTH / 2} ${yOf(child.depth)}`,
          fill: 'none',
          stroke: 'rgba(147, 161, 187, 0.45)',
          'stroke-width': 1.2,
        })
      );
      const midX = (box.x + child.x) / 2 + NODE_WIDTH / 2;
      const edgeLabel = el('text', {
        x: midX,
        y: yOf(box.depth) + NODE_HEIGHT + 14,
        'text-anchor': 'middle',
        fill: 'rgba(147, 161, 187, 0.9)',
        'font-size': 9,
        'font-family': 'ui-monospace, monospace',
      });
      edgeLabel.textContent = index === 0 ? 'yes' : 'no';
      svg.appendChild(edgeLabel);
    });
  });

  boxes.forEach((box) => {
    const { node } = box;
    const y = yOf(box.depth);
    const isLeaf = node.is_leaf || box.truncated;
    const fill = isLeaf
      ? hexToRgba(classColor(node.predicted_index), 0.28)
      : 'rgba(22, 29, 44, 0.95)';

    svg.appendChild(
      el('rect', {
        x: box.x,
        y,
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
        rx: 6,
        fill,
        stroke: isLeaf ? classColor(node.predicted_index) : 'rgba(147, 161, 187, 0.4)',
        'stroke-width': 1.2,
      })
    );

    const title = el('text', {
      x: box.x + NODE_WIDTH / 2,
      y: y + 16,
      'text-anchor': 'middle',
      fill: '#e8edf7',
      'font-size': 10.5,
      'font-family': 'ui-monospace, monospace',
    });
    if (box.truncated) title.textContent = '⋯ pruned view';
    else if (node.is_leaf) title.textContent = `class ${classValues?.[node.predicted_index] ?? node.predicted_class}`;
    else title.textContent = `${node.feature} < ${node.threshold}`;
    svg.appendChild(title);

    const subtitle = el('text', {
      x: box.x + NODE_WIDTH / 2,
      y: y + 30,
      'text-anchor': 'middle',
      fill: 'rgba(147, 161, 187, 0.95)',
      'font-size': 9.5,
      'font-family': 'ui-monospace, monospace',
    });
    subtitle.textContent = `n=${node.samples} · imp ${node.impurity}`;
    svg.appendChild(subtitle);
  });

  container.appendChild(svg);

  const caption = document.createElement('p');
  caption.className = 'field-help';
  caption.style.marginTop = '8px';
  caption.textContent =
    maxDepth >= MAX_DRAWN_DEPTH
      ? `Showing the top ${MAX_DRAWN_DEPTH} levels; deeper branches are collapsed. "imp" is the node's impurity.`
      : 'Each box is one question; leaves are coloured by the class they predict.';
  container.appendChild(caption);
}

function hexToRgba(hex, alpha) {
  const value = hex.replace('#', '');
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* -------------------------------------------------------- network ----- */

export function renderNetwork(container, network) {
  container.innerHTML = '';
  if (!network) {
    container.innerHTML =
      '<p class="field-help">This network has too many weights to draw legibly.</p>';
    return;
  }

  const { layers, weights } = network;
  const nodeRadius = 7;
  const columnGap = 82;
  const rowGap = 20;
  const width = (layers.length - 1) * columnGap + 60;
  const tallest = Math.max(...layers);
  const height = Math.max(tallest * rowGap + 30, 120);

  const svg = el('svg', { width, height, viewBox: `0 0 ${width} ${height}` });

  const positions = layers.map((size, layerIndex) => {
    const x = 30 + layerIndex * columnGap;
    const offset = (height - (size - 1) * rowGap) / 2;
    return Array.from({ length: size }, (_, i) => ({ x, y: offset + i * rowGap }));
  });

  weights.forEach((matrix, layerIndex) => {
    matrix.forEach((row, from) => {
      row.forEach((weight, to) => {
        const magnitude = Math.abs(weight);
        if (magnitude < 0.06) return;
        const start = positions[layerIndex][from];
        const end = positions[layerIndex + 1][to];
        svg.appendChild(
          el('line', {
            x1: start.x,
            y1: start.y,
            x2: end.x,
            y2: end.y,
            stroke: weight >= 0 ? 'rgba(91, 156, 248, 0.85)' : 'rgba(242, 104, 92, 0.85)',
            'stroke-width': Math.min(2.6, 0.3 + magnitude * 2.4),
            'stroke-opacity': Math.min(0.9, 0.16 + magnitude),
          })
        );
      });
    });
  });

  positions.forEach((column, layerIndex) => {
    column.forEach((position) => {
      svg.appendChild(
        el('circle', {
          cx: position.x,
          cy: position.y,
          r: nodeRadius,
          fill:
            layerIndex === 0
              ? '#4dd6a8'
              : layerIndex === positions.length - 1
                ? '#f5c451'
                : '#8ba0c0',
          stroke: 'rgba(11, 15, 24, 0.9)',
          'stroke-width': 1.4,
        })
      );
    });
    const label = el('text', {
      x: column[0].x,
      y: height - 6,
      'text-anchor': 'middle',
      fill: 'rgba(147, 161, 187, 0.9)',
      'font-size': 9.5,
      'font-family': 'ui-monospace, monospace',
    });
    label.textContent =
      layerIndex === 0 ? 'in' : layerIndex === positions.length - 1 ? 'out' : `h${layerIndex}`;
    svg.appendChild(label);
  });

  container.appendChild(svg);

  const caption = document.createElement('p');
  caption.className = 'field-help';
  caption.style.marginTop = '8px';
  caption.innerHTML =
    'Blue connections are positive weights, red negative; thickness is magnitude. ' +
    `Largest weight this frame: <code>${escapeHtml(String(network.scale))}</code>.`;
  container.appendChild(caption);
}
