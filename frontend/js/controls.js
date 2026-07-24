/** Builds hyperparameter and dataset controls from the specs the API returns. */

import { escapeHtml } from './chart.js';

/**
 * Render one control per parameter spec into `container`.
 * Returns a `read()` function giving the current values as an object.
 */
export function buildParamControls(container, params, { values = {}, onChange }) {
  container.innerHTML = '';
  const readers = {};

  params.forEach((param) => {
    const current = values[param.name] !== undefined ? values[param.name] : param.default;

    if (param.type === 'bool') {
      const label = document.createElement('label');
      label.className = 'checkbox';
      label.title = param.help || '';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = Boolean(current);
      const span = document.createElement('span');
      span.textContent = param.label;
      label.append(input, span);
      container.appendChild(label);
      input.addEventListener('change', () => onChange(param.name, input.checked));
      readers[param.name] = () => input.checked;
      return;
    }

    const field = document.createElement('label');
    field.className = 'field';

    const head = document.createElement('span');
    head.className = 'field-label';
    head.innerHTML = `<span>${escapeHtml(param.label)}</span>`;
    field.appendChild(head);

    if (param.type === 'select') {
      const select = document.createElement('select');
      param.options.forEach((option) => {
        const node = document.createElement('option');
        node.value = option.value;
        node.textContent = option.label;
        node.selected = option.value === current;
        select.appendChild(node);
      });
      field.appendChild(select);
      select.addEventListener('change', () => onChange(param.name, select.value));
      readers[param.name] = () => select.value;
    } else {
      const readout = document.createElement('span');
      readout.className = 'field-value';
      readout.textContent = formatNumber(current, param);
      head.appendChild(readout);

      const input = document.createElement('input');
      input.type = 'range';
      input.min = param.min ?? 0;
      input.max = param.max ?? 100;
      input.step = param.step ?? (param.type === 'int' ? 1 : 0.01);
      input.value = current;
      field.appendChild(input);

      const parse = (raw) => (param.type === 'int' ? Math.round(Number(raw)) : Number(raw));
      input.addEventListener('input', () => {
        readout.textContent = formatNumber(parse(input.value), param);
      });
      input.addEventListener('change', () => onChange(param.name, parse(input.value)));
      readers[param.name] = () => parse(input.value);
    }

    if (param.help) {
      field.title = param.help;
      const help = document.createElement('p');
      help.className = 'field-help';
      help.textContent = param.help;
      field.appendChild(help);
    }

    container.appendChild(field);
  });

  return () => {
    const out = {};
    Object.entries(readers).forEach(([name, read]) => {
      out[name] = read();
    });
    return out;
  };
}

function formatNumber(value, param) {
  if (param.type === 'int') return String(Math.round(value));
  const step = param.step ?? 0.01;
  const decimals = Math.min(6, Math.max(0, Math.ceil(-Math.log10(step))));
  return Number(value).toFixed(decimals);
}

/** Class-picker chips shown above the plot in classification mode. */
export function buildClassPicker(container, { classes, active, onSelect, onAdd, enabled }) {
  container.innerHTML = '';
  if (!enabled) {
    container.innerHTML =
      '<span class="picker-label">Click the plot to add points — this mode has no class labels.</span>';
    return;
  }

  const label = document.createElement('span');
  label.className = 'picker-label';
  label.textContent = 'Placing:';
  container.appendChild(label);

  classes.forEach((entry) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = `class-chip${entry.index === active ? ' active' : ''}`;
    chip.style.color = entry.index === active ? entry.color : '';
    chip.innerHTML = `<span class="dot" style="background:${entry.color}"></span>Class ${entry.index}`;
    chip.addEventListener('click', () => onSelect(entry.index));
    container.appendChild(chip);
  });

  if (classes.length < 6) {
    const add = document.createElement('button');
    add.type = 'button';
    add.className = 'class-chip add-chip';
    add.textContent = '+ class';
    add.title = 'Add another class to place';
    add.addEventListener('click', onAdd);
    container.appendChild(add);
  }
}

/** Checkbox row under the plot for the per-algorithm visual overlays. */
export function buildOverlayToggles(container, toggles, onChange) {
  container.innerHTML = '';
  toggles.forEach((toggle) => {
    const label = document.createElement('label');
    label.className = 'checkbox';
    label.title = toggle.help || '';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = toggle.checked;
    input.addEventListener('change', () => onChange(toggle.key, input.checked));
    const span = document.createElement('span');
    span.textContent = toggle.label;
    label.append(input, span);
    container.appendChild(label);
  });
}
