/** Builds hyperparameter and dataset controls from the specs the API returns. */

import { escapeHtml } from './chart.js';

// Resolution of a log-scale slider, in positions across its whole travel.
const LOG_STEPS = 1000;

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

    // A div rather than a label: numeric fields hold two controls (slider and
    // text box), so there is no single input for a label to point at.
    const field = document.createElement('div');
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
      const min = param.min ?? 0;
      const max = param.max ?? 100;
      // Log scaling needs a strictly positive range; fall back to linear if not.
      const isLog = param.scale === 'log' && min > 0 && max > min;

      const clamp = (v) => Math.min(Math.max(v, min), max);
      const toInt = (v) => (param.type === 'int' ? Math.round(v) : v);
      // Slider positions carry no meaningful precision beyond ~3 significant
      // figures, so snap them; typed values are kept exactly as entered.
      const snap = (v) => (param.type === 'int' ? Math.round(v) : isLog ? Number(v.toPrecision(3)) : v);

      const toPosition = (v) =>
        isLog ? (LOG_STEPS * Math.log(clamp(v) / min)) / Math.log(max / min) : clamp(v);
      const fromPosition = (p) =>
        isLog ? clamp(min * (max / min) ** (Number(p) / LOG_STEPS)) : clamp(Number(p));

      const box = document.createElement('input');
      box.type = 'text';
      box.inputMode = 'decimal';
      box.className = 'field-value-input';
      box.setAttribute('aria-label', `${param.label}, type a value`);
      box.title = `Type a value between ${formatNumber(min, param)} and ${formatNumber(max, param)}`;
      head.appendChild(box);

      const slider = document.createElement('input');
      slider.type = 'range';
      slider.setAttribute('aria-label', param.label);
      if (isLog) {
        slider.min = 0;
        slider.max = LOG_STEPS;
        slider.step = 1;
      } else {
        slider.min = min;
        slider.max = max;
        slider.step = param.step ?? (param.type === 'int' ? 1 : 0.01);
      }
      field.appendChild(slider);

      let value = toInt(clamp(Number(current)));
      const sync = () => {
        slider.value = String(toPosition(value));
        box.value = formatNumber(value, param);
      };
      sync();

      slider.addEventListener('input', () => {
        value = snap(fromPosition(slider.value));
        box.value = formatNumber(value, param);
      });
      slider.addEventListener('change', () => {
        value = snap(fromPosition(slider.value));
        sync();
        onChange(param.name, value);
      });
      // Fires on Enter and on blur. Anything unparseable or out of range is
      // silently corrected by sync(), so the box can never show a bad value.
      box.addEventListener('change', () => {
        const typed = Number(box.value.trim());
        if (Number.isFinite(typed)) value = toInt(clamp(typed));
        sync();
        onChange(param.name, value);
      });

      readers[param.name] = () => value;
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

  if (param.scale === 'log') {
    const v = Number(value);
    if (v === 0) return '0';
    // Decade extremes read far better in exponential form than as 0.00001.
    if (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) return v.toExponential(2).replace('e+', 'e');
    return String(Number(v.toPrecision(6)));
  }

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
