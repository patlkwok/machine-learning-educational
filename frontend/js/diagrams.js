/** SVG diagrams of the fitted model: decision trees and neural networks. */

import { classColor } from './palette.js';
import { escapeHtml } from './chart.js';
import * as theme from './theme.js';

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
          stroke: theme.fade(theme.colors().textMuted, 0.45),
          'stroke-width': 1.2,
        })
      );
      const midX = (box.x + child.x) / 2 + NODE_WIDTH / 2;
      const edgeLabel = el('text', {
        x: midX,
        y: yOf(box.depth) + NODE_HEIGHT + 14,
        'text-anchor': 'middle',
        fill: theme.fade(theme.colors().textMuted, 0.9),
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
      : theme.colors().diagramNodeBg;

    svg.appendChild(
      el('rect', {
        x: box.x,
        y,
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
        rx: 6,
        fill,
        stroke: isLeaf ? classColor(node.predicted_index) : theme.fade(theme.colors().textMuted, 0.4),
        'stroke-width': 1.2,
      })
    );

    const title = el('text', {
      x: box.x + NODE_WIDTH / 2,
      y: y + 16,
      'text-anchor': 'middle',
      fill: theme.colors().text,
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
      fill: theme.fade(theme.colors().textMuted, 0.95),
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

/* ----------------------------------------------------- dendrogram ----- */

const LEAF_GAP = 19;
const DENDRO_HEIGHT = 210;
const DENDRO_PAD = { top: 12, bottom: 24, left: 34, right: 12 };

/**
 * Classic bracket dendrogram with a movable cut line.
 *
 * Links that merged below the cut are inside a cluster and drawn bright; links
 * above it were severed by the cut and are drawn muted.
 */
export function renderDendrogram(container, tree, cutHeight, maxHeight) {
  container.innerHTML = '';
  if (!tree) return;

  // Left-to-right leaf order fixes every x position; parents sit above the
  // midpoint of their two children.
  const xs = new Map();
  let leaves = 0;
  const place = (node) => {
    if (node.kind !== 'node') {
      const x = leaves;
      leaves += 1;
      xs.set(node, x);
      return x;
    }
    const a = place(node.children[0]);
    const b = place(node.children[1]);
    const x = (a + b) / 2;
    xs.set(node, x);
    return x;
  };
  place(tree);

  const top = maxHeight || tree.height || 1;
  const width = DENDRO_PAD.left + leaves * LEAF_GAP + DENDRO_PAD.right;
  const height = DENDRO_PAD.top + DENDRO_HEIGHT + DENDRO_PAD.bottom;
  const xOf = (x) => DENDRO_PAD.left + x * LEAF_GAP + LEAF_GAP / 2;
  const yOf = (h) => DENDRO_PAD.top + DENDRO_HEIGHT - (Math.min(h, top) / top) * DENDRO_HEIGHT;

  const svg = el('svg', { width, height, viewBox: `0 0 ${width} ${height}` });

  // Height axis.
  [0, top / 2, top].forEach((h) => {
    svg.appendChild(
      el('line', {
        x1: DENDRO_PAD.left - 4, y1: yOf(h), x2: width - DENDRO_PAD.right, y2: yOf(h),
        stroke: theme.fade(theme.colors().textMuted, 0.12), 'stroke-width': 1,
      })
    );
    const tick = el('text', {
      x: DENDRO_PAD.left - 7, y: yOf(h) + 3, 'text-anchor': 'end',
      fill: theme.fade(theme.colors().textMuted, 0.8), 'font-size': 9,
      'font-family': 'ui-monospace, monospace',
    });
    tick.textContent = h >= 100 ? h.toFixed(0) : h.toFixed(h < 1 ? 2 : 1);
    svg.appendChild(tick);
  });

  const draw = (node) => {
    if (node.kind !== 'node') return;
    const [left, right] = node.children;
    const x1 = xOf(xs.get(left));
    const x2 = xOf(xs.get(right));
    const yTop = yOf(node.height);
    const within = node.height < cutHeight;
    const stroke = within ? theme.colors().accent : theme.fade(theme.colors().textMuted, 0.5);

    svg.appendChild(
      el('path', {
        d: `M ${x1} ${yOf(left.height || 0)} L ${x1} ${yTop} L ${x2} ${yTop} L ${x2} ${yOf(right.height || 0)}`,
        fill: 'none',
        stroke,
        'stroke-width': within ? 1.7 : 1.2,
      })
    );
    draw(left);
    draw(right);
  };
  draw(tree);

  // Collapsed subtrees get a marker and their point count.
  xs.forEach((x, node) => {
    if (node.kind !== 'collapsed') return;
    const cx = xOf(x);
    const cy = yOf(node.height);
    svg.appendChild(
      el('path', {
        d: `M ${cx} ${cy} L ${cx - 6} ${yOf(0)} L ${cx + 6} ${yOf(0)} Z`,
        fill: theme.fade(theme.colors().textMuted, 0.28),
        stroke: theme.fade(theme.colors().textMuted, 0.55),
        'stroke-width': 1,
      })
    );
    const label = el('text', {
      x: cx, y: yOf(0) + 12, 'text-anchor': 'middle',
      fill: theme.fade(theme.colors().textMuted, 0.9), 'font-size': 8.5,
      'font-family': 'ui-monospace, monospace',
    });
    label.textContent = String(node.count);
    svg.appendChild(label);
  });

  // The cut.
  svg.appendChild(
    el('line', {
      x1: DENDRO_PAD.left - 4, y1: yOf(cutHeight), x2: width - DENDRO_PAD.right, y2: yOf(cutHeight),
      stroke: '#ffd166', 'stroke-width': 1.6, 'stroke-dasharray': '6 4',
    })
  );

  container.appendChild(svg);

  const caption = document.createElement('p');
  caption.className = 'field-help';
  caption.style.marginTop = '8px';
  caption.innerHTML =
    'Height is the distance at which two groups merged. The dashed line is the current cut; ' +
    'blue links merged below it, so they sit inside a cluster. Triangles are collapsed subtrees ' +
    'labelled with their point count. Link colour is not matched to the plot colours.';
  container.appendChild(caption);
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
          stroke: theme.fade(theme.colors().plotHalo, 0.9),
          'stroke-width': 1.4,
        })
      );
    });
    const label = el('text', {
      x: column[0].x,
      y: height - 6,
      'text-anchor': 'middle',
      fill: theme.fade(theme.colors().textMuted, 0.9),
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
