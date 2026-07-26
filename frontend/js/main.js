/** Application wiring: catalogue -> controls -> fit -> playback -> insights. */

import { api } from './api.js';
import { MetricChart, formatValue, escapeHtml } from './chart.js';
import { buildClassPicker, buildOverlayToggles, buildParamControls } from './controls.js';
import { renderDendrogram, renderNetwork, renderTree } from './diagrams.js';
import { Plot } from './plot.js';
import { classColor } from './palette.js';
import * as theme from './theme.js';

const VIEWPORT = { x_min: -5, x_max: 5, y_min: -5, y_max: 5 };

// Seeds are internal: the buttons randomise them and the API validates them,
// but no number is ever shown. Must match the API's limit in schemas.py.
const SEED_MAX = 999999;

const DEFAULT_GENERATOR = {
  regression: 'linear_regression',
  classification: 'moons',
  clustering: 'blobs',
};

const dom = {
  algorithmList: document.getElementById('algorithm-list'),
  generatorSelect: document.getElementById('generator-select'),
  generatorHelp: document.getElementById('generator-help'),
  generatorParams: document.getElementById('generator-params'),
  validationControl: document.getElementById('validation-control'),
  generateBtn: document.getElementById('generate-btn'),
  reseedBtn: document.getElementById('reseed-btn'),
  paramControls: document.getElementById('param-controls'),
  trainBtn: document.getElementById('train-btn'),
  autoTrain: document.getElementById('auto-train'),
  classPicker: document.getElementById('class-picker'),
  resolutionSelect: document.getElementById('resolution-select'),
  clearBtn: document.getElementById('clear-btn'),
  plotEmpty: document.getElementById('plot-empty'),
  plotError: document.getElementById('plot-error'),
  playBtn: document.getElementById('play-btn'),
  prevBtn: document.getElementById('prev-btn'),
  nextBtn: document.getElementById('next-btn'),
  slider: document.getElementById('step-slider'),
  stepCounter: document.getElementById('step-counter'),
  speedSelect: document.getElementById('speed-select'),
  stepLabel: document.getElementById('step-label'),
  stepDescription: document.getElementById('step-description'),
  overlayToggles: document.getElementById('overlay-toggles'),
  summary: document.getElementById('summary'),
  notes: document.getElementById('notes'),
  explain: document.getElementById('explain'),
  explainTitle: document.getElementById('explain-title'),
  watchFor: document.getElementById('watch-for'),
  diagramBlock: document.getElementById('diagram-block'),
  diagramTitle: document.getElementById('diagram-title'),
  diagram: document.getElementById('diagram'),
  statusPill: document.getElementById('status-pill'),
  helpToggle: document.getElementById('help-toggle'),
  themeToggle: document.getElementById('theme-toggle'),
  helpDialog: document.getElementById('help-dialog'),
  metricChart: document.getElementById('metric-chart'),
  metricLegend: document.getElementById('metric-legend'),
};

const state = {
  catalogue: null,
  algorithm: null,
  params: {},
  readParams: () => ({}),
  generator: null,
  generatorOptions: { n_samples: 200, noise: 0.2, seed: 1, classes: 3 },
  // Fraction of points held out of training, shared by every supervised algorithm.
  validationSplit: 0.2,
  // Which partition to draw; changed only by the Resample split button, so the
  // split stays fixed while you tune anything else.
  validationSeed: 0,
  activeClass: 0,
  classCount: 2,
  result: null,
  stepIndex: 0,
  playing: false,
  timer: null,
  inFlight: false,
  queued: false,
  overlays: {},
  // 'data' | 'model' | false - a change staged while auto-run is off.
  stale: false,
};

const plot = new Plot(document.getElementById('plot'), {
  viewport: VIEWPORT,
  onPointsChange: handlePointsChanged,
});
const chart = new MetricChart(dom.metricChart, dom.metricLegend);

/* ------------------------------------------------------------- boot --- */

async function boot() {
  applyTheme(theme.initial());
  setStatus('busy', 'Loading…');
  try {
    state.catalogue = await api.catalogue();
  } catch (error) {
    setStatus('error', 'Backend unreachable');
    showError(`Could not reach the backend: ${error.message}`);
    return;
  }
  renderAlgorithmList();
  bindGlobalControls();
  await selectAlgorithm(state.catalogue.algorithms[0].id, { generate: true });
}

function bindGlobalControls() {
  dom.generateBtn.addEventListener('click', () => generateData({ mode: 'play' }));
  dom.reseedBtn.addEventListener('click', () => {
    state.generatorOptions.seed = Math.floor(Math.random() * (SEED_MAX + 1));
    renderGeneratorControls();
    generateData({ mode: 'play' });
  });
  dom.trainBtn.addEventListener('click', () => train({ mode: 'play' }));
  dom.clearBtn.addEventListener('click', () => {
    plot.setPoints([]);
    clearResult();
    updateEmptyState();
  });
  dom.generatorSelect.addEventListener('change', () => {
    state.generator = dom.generatorSelect.value;
    renderGeneratorControls();
    if (autoRun()) generateData({ mode: 'play' });
    else markStale('data');
  });
  dom.resolutionSelect.addEventListener('change', () => {
    if (autoRun()) train({ mode: 'keep' });
    else markStale('model');
  });
  dom.autoTrain.addEventListener('change', () => {
    // Turning it back on should apply whatever was staged while it was off.
    if (!autoRun() || !state.stale) return;
    const pending = state.stale;
    state.stale = false;
    if (pending === 'data') generateData({ mode: 'final' });
    else train({ mode: 'keep' });
  });

  dom.playBtn.addEventListener('click', togglePlay);
  dom.prevBtn.addEventListener('click', () => {
    stopPlayback();
    goToStep(state.stepIndex - 1);
  });
  dom.nextBtn.addEventListener('click', () => {
    stopPlayback();
    goToStep(state.stepIndex + 1);
  });
  dom.slider.addEventListener('input', () => {
    stopPlayback();
    goToStep(Number(dom.slider.value));
  });
  dom.speedSelect.addEventListener('change', () => {
    if (state.playing) {
      stopPlayback();
      startPlayback();
    }
  });

  dom.helpToggle.addEventListener('click', () => dom.helpDialog.showModal());

  dom.themeToggle.addEventListener('click', () => {
    applyTheme(theme.current() === 'light' ? 'dark' : 'light');
  });

  document.addEventListener('keydown', (event) => {
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (event.key === ' ') {
      event.preventDefault();
      togglePlay();
    } else if (event.key === 'ArrowLeft') {
      stopPlayback();
      goToStep(state.stepIndex - 1);
    } else if (event.key === 'ArrowRight') {
      stopPlayback();
      goToStep(state.stepIndex + 1);
    }
  });
}

/* -------------------------------------------------------- algorithms --- */

function renderAlgorithmList() {
  const groups = new Map();
  state.catalogue.algorithms.forEach((spec) => {
    if (!groups.has(spec.task)) groups.set(spec.task, []);
    groups.get(spec.task).push(spec);
  });

  dom.algorithmList.innerHTML = '';
  ['regression', 'classification', 'clustering'].forEach((task) => {
    const specs = groups.get(task);
    if (!specs) return;
    const heading = document.createElement('div');
    heading.className = 'algo-group-label';
    heading.textContent = state.catalogue.task_labels[task] || task;
    dom.algorithmList.appendChild(heading);

    specs.forEach((spec) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'algo-btn';
      button.dataset.algorithm = spec.id;
      button.innerHTML = `<strong>${escapeHtml(spec.name)}</strong><span>${escapeHtml(spec.tagline)}</span>`;
      button.addEventListener('click', () => selectAlgorithm(spec.id, { generate: 'if-needed' }));
      dom.algorithmList.appendChild(button);
    });
  });
}

function currentSpec() {
  return state.catalogue.algorithms.find((spec) => spec.id === state.algorithm);
}

async function selectAlgorithm(algorithmId, { generate } = {}) {
  state.algorithm = algorithmId;
  const spec = currentSpec();

  document.querySelectorAll('.algo-btn').forEach((button) => {
    button.classList.toggle('active', button.dataset.algorithm === algorithmId);
  });

  state.params = {};
  state.readParams = buildParamControls(dom.paramControls, spec.params, {
    values: state.params,
    onChange: (name, value) => {
      state.params[name] = value;
      if (autoRun()) scheduleTrain(120);
      else markStale('model');
    },
  });

  // Some algorithms expose a randomisable value as a button rather than a
  // number, because the number means nothing but changing it means a lot.
  if (spec.reseed) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'btn btn-subtle';
    button.id = 'reseed-param';
    button.textContent = spec.reseed.label;
    button.title = spec.reseed.help || '';
    button.addEventListener('click', () => {
      state.params[spec.reseed.param] = Math.floor(Math.random() * (SEED_MAX + 1));
      train({ mode: 'play' });
    });
    dom.paramControls.appendChild(button);
  }

  plot.setTask(spec.task);
  renderClassPicker();
  renderExplanation(spec);
  clearResult();

  const wantsRegression = spec.task === 'regression';
  const generatorKind = wantsRegression ? 'regression' : 'labelled';
  const needsNewGenerator =
    !state.generator ||
    state.catalogue.generators.find((g) => g.id === state.generator)?.kind !== generatorKind;

  if (needsNewGenerator) state.generator = DEFAULT_GENERATOR[spec.task];
  renderGeneratorOptions(generatorKind);
  renderGeneratorControls();

  const incompatible = !pointsSuitFor(spec.task);
  if (generate === true || (generate === 'if-needed' && (incompatible || !plot.points.length))) {
    await generateData({ mode: 'final' });
  } else if (plot.points.length) {
    plot.render();
    await train({ mode: 'final' });
  }
  updateEmptyState();
}

/** Classification needs labelled points; the other tasks accept anything. */
function pointsSuitFor(task) {
  if (!plot.points.length) return false;
  if (task !== 'classification') return true;
  const labels = new Set(plot.points.map((point) => point.label));
  return !labels.has(null) && !labels.has(undefined) && labels.size >= 2;
}

/* ----------------------------------------------------------- datasets --- */

function renderGeneratorOptions(kind) {
  const options = state.catalogue.generators.filter((generator) => generator.kind === kind);
  dom.generatorSelect.innerHTML = '';
  options.forEach((generator) => {
    const node = document.createElement('option');
    node.value = generator.id;
    node.textContent = generator.name;
    node.selected = generator.id === state.generator;
    dom.generatorSelect.appendChild(node);
  });
  if (!options.some((generator) => generator.id === state.generator) && options.length) {
    state.generator = options[0].id;
    dom.generatorSelect.value = state.generator;
  }
}

function generatorSpec() {
  return state.catalogue.generators.find((generator) => generator.id === state.generator);
}

function renderGeneratorControls() {
  const spec = generatorSpec();
  dom.generatorHelp.textContent = spec ? spec.description : '';
  dom.generatorSelect.value = state.generator;

  const params = [
    {
      name: 'n_samples',
      label: 'Points',
      type: 'int',
      default: state.generatorOptions.n_samples,
      min: 10,
      max: 1000,
      step: 10,
      help: '',
    },
    {
      name: 'noise',
      label: 'Noise',
      type: 'float',
      default: state.generatorOptions.noise,
      min: 0,
      max: 1,
      step: 0.01,
      help: '',
    },
    // No seed control. Everything here is seeded and reproducible either way,
    // so showing the number buys nothing a learner can use: "Shuffle" and
    // "Resample split" say what they do, where "seed 731469" does not.
  ];

  if (spec && spec.supports_classes) {
    params.push({
      name: 'classes',
      label: currentSpec().task === 'clustering' ? 'True clusters' : 'Classes',
      type: 'int',
      default: Math.min(Math.max(state.generatorOptions.classes, spec.min_classes), spec.max_classes),
      min: spec.min_classes,
      max: spec.max_classes,
      step: 1,
      help: '',
    });
  }

  renderValidationControl();

  state.readGeneratorOptions = buildParamControls(dom.generatorParams, params, {
    values: state.generatorOptions,
    onChange: (name, value) => {
      state.generatorOptions[name] = value;
      if (autoRun()) generateData({ mode: 'final' });
      else markStale('data');
    },
  });
}

/**
 * The validation-split control, shown only where it means something.
 *
 * Clustering has no labels, so there is nothing a held-out point could be
 * scored against; offering the control there would imply otherwise.
 */
function renderValidationControl() {
  const supervised = currentSpec().task !== 'clustering';
  dom.validationControl.innerHTML = '';
  if (!supervised) return;

  const resample = document.createElement('button');
  resample.type = 'button';
  resample.className = 'btn btn-subtle';
  resample.id = 'resample-split';
  resample.textContent = 'Resample split';
  resample.title =
    'Draw a different set of held-out points from the same data. ' +
    'If the metrics move a lot, the split was doing the work, not the model.';
  resample.addEventListener('click', () => {
    state.validationSeed = Math.floor(Math.random() * (SEED_MAX + 1));
    train({ mode: 'keep' });
  });

  state.readValidation = buildParamControls(
    dom.validationControl,
    [
      {
        name: 'validation_split',
        label: 'Validation split',
        type: 'float',
        default: state.validationSplit,
        min: 0,
        max: 0.5,
        step: 0.05,
        help:
          'Fraction of points held out of training and scored separately. ' +
          'The gap between training and validation is overfitting. 0 disables it.',
      },
    ],
    {
      values: { validation_split: state.validationSplit },
      onChange: (_name, value) => {
        state.validationSplit = value;
        resample.disabled = value <= 0;
        if (autoRun()) scheduleTrain(120);
        else markStale('model');
      },
    }
  );

  // Nothing to redraw when no points are being held back.
  resample.disabled = state.validationSplit <= 0;
  dom.validationControl.appendChild(resample);
}

async function generateData({ mode = 'final' } = {}) {
  const spec = currentSpec();
  const options = state.readGeneratorOptions ? state.readGeneratorOptions() : state.generatorOptions;
  Object.assign(state.generatorOptions, options);

  setStatus('busy', 'Generating…');
  try {
    const response = await api.generate({
      generator: state.generator,
      n_samples: state.generatorOptions.n_samples,
      noise: state.generatorOptions.noise,
      seed: state.generatorOptions.seed,
      classes: state.generatorOptions.classes,
    });

    // Clustering is unsupervised: hide the ground-truth labels.
    const points = response.points.map((point) => ({
      ...point,
      label: spec.task === 'clustering' ? null : point.label,
    }));

    state.classCount = Math.max(
      2,
      ...points.map((point) => (point.label === null ? 1 : point.label + 1))
    );
    state.activeClass = Math.min(state.activeClass, state.classCount - 1);
    renderClassPicker();

    plot.setPoints(points);
    clearError();
    updateEmptyState();
    await train({ mode });
  } catch (error) {
    setStatus('error', 'Generate failed');
    showError(error.message);
  }
}

/* --------------------------------------------------------------- fit --- */

let trainTimer = null;

function scheduleTrain(delay = 300) {
  clearTimeout(trainTimer);
  trainTimer = setTimeout(() => train({ mode: 'keep' }), delay);
}

function handlePointsChanged({ settled }) {
  updateEmptyState();
  const labels = plot.points.map((point) => point.label).filter((label) => label !== null);
  if (labels.length) state.classCount = Math.max(state.classCount, Math.max(...labels) + 1);
  // Mid-drag changes are not worth reacting to either way.
  if (!settled) return;
  if (!autoRun()) {
    markStale('model');
    return;
  }
  scheduleTrain(280);
}

/**
 * `mode` decides where playback lands once the new result arrives:
 *   'play'  - explicit Train/Generate: rewind and animate from the start
 *   'keep'  - a tweak while exploring: stay on the same frame to compare
 *   'final' - a fresh algorithm: show the trained result straight away
 */
async function train({ mode = 'keep' } = {}) {
  if (!plot.points.length) {
    clearResult();
    updateEmptyState();
    return;
  }
  if (state.inFlight) {
    state.queued = { mode };
    return;
  }

  state.inFlight = true;
  dom.trainBtn.disabled = true;
  setStatus('busy', 'Training…');

  const previousStep = state.stepIndex;
  // If the last frame was already on screen, follow the end of the new
  // animation rather than freezing on an early, half-trained frame.
  const wasAtEnd = !state.result || state.stepIndex >= state.result.steps.length - 1;

  try {
    const params = state.readParams();
    Object.assign(state.params, params);
    const result = await api.fit({
      algorithm: state.algorithm,
      params,
      points: plot.points,
      viewport: VIEWPORT,
      grid_resolution: Number(dom.resolutionSelect.value),
      validation_split: currentSpec().task === 'clustering' ? 0 : state.validationSplit,
      validation_seed: state.validationSeed,
    });
    state.result = result;
    state.stale = false;
    clearError();
    applyResult(result, { mode, previousStep, wasAtEnd });
    setStatus('ok', `Trained in ${Math.round(result.elapsed_ms)} ms`);
  } catch (error) {
    clearResult();
    setStatus('error', 'Could not train');
    showError(error.message);
  } finally {
    state.inFlight = false;
    dom.trainBtn.disabled = false;
    if (state.queued) {
      const queued = state.queued;
      state.queued = false;
      train(queued);
    }
  }
}

function applyResult(result, { mode, previousStep, wasAtEnd }) {
  plot.setResult(result);
  chart.setResult(result);

  const last = Math.max(result.steps.length - 1, 0);
  dom.slider.max = String(last);

  let target = last;
  if (mode === 'play') target = 0;
  else if (mode === 'keep' && !wasAtEnd) target = Math.min(previousStep, last);
  goToStep(target);

  renderOverlayToggles(result);
  renderSummary(result);

  if (mode === 'play' && result.steps.length > 1) startPlayback();
}

function clearResult() {
  stopPlayback();
  state.result = null;
  state.stepIndex = 0;
  plot.setResult(null);
  chart.setResult(null);
  dom.slider.max = '0';
  dom.slider.value = '0';
  dom.stepCounter.textContent = '—';
  dom.stepLabel.textContent = 'No model trained yet';
  dom.stepDescription.textContent = 'Pick an algorithm, add some data, and press Train.';
  dom.summary.innerHTML = '';
  dom.notes.innerHTML = '';
  dom.overlayToggles.innerHTML = '';
  dom.diagramBlock.hidden = true;
}

/* ---------------------------------------------------------- playback --- */

function goToStep(index) {
  if (!state.result) return;
  const count = state.result.steps.length;
  const clamped = Math.min(Math.max(index, 0), count - 1);
  state.stepIndex = clamped;
  dom.slider.value = String(clamped);
  dom.stepCounter.textContent = `${clamped + 1} / ${count}`;

  plot.setStep(clamped);
  chart.setStep(clamped);
  renderStepReadout(state.result.steps[clamped]);
  renderDiagram(state.result.steps[clamped]);
}

function togglePlay() {
  if (!state.result || state.result.steps.length < 2) return;
  if (state.playing) stopPlayback();
  else startPlayback();
}

function startPlayback() {
  // Always clear first: starting over a live timer would orphan the old
  // interval, which would then keep ticking against a stale result.
  stopPlayback();
  if (!state.result || state.result.steps.length < 2) return;
  if (state.stepIndex >= state.result.steps.length - 1) goToStep(0);
  state.playing = true;
  dom.playBtn.textContent = '❚❚';
  state.timer = setInterval(() => {
    if (!state.result || state.stepIndex >= state.result.steps.length - 1) {
      stopPlayback();
      return;
    }
    goToStep(state.stepIndex + 1);
  }, Number(dom.speedSelect.value));
}

function stopPlayback() {
  state.playing = false;
  dom.playBtn.textContent = '▶';
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

/* ---------------------------------------------------------- insights --- */

function renderStepReadout(step) {
  const spec = currentSpec();
  dom.stepLabel.textContent = `${spec.step_unit} · ${step.label}`;
  dom.stepLabel.title = spec.step_hint;
  dom.stepDescription.innerHTML = step.description;
}

function renderSummary(result) {
  dom.summary.innerHTML = Object.entries(result.summary)
    .filter(([, value]) => value !== null && value !== undefined)
    .map(
      ([key, value]) =>
        `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(
          typeof value === 'number' ? formatValue(value) : String(value)
        )}</dd>`
    )
    .join('');

  dom.notes.innerHTML = result.notes.map((note) => `<li>${note}</li>`).join('');
}

function renderExplanation(spec) {
  dom.explainTitle.textContent = `How ${spec.name} works`;
  dom.explain.innerHTML = spec.description.map((paragraph) => `<p>${paragraph}</p>`).join('');
  dom.watchFor.innerHTML = spec.watch_for.map((item) => `<li>${escapeHtml(item)}</li>`).join('');
}

function renderDiagram(step) {
  if (step.extras?.tree) {
    dom.diagramBlock.hidden = false;
    dom.diagramTitle.textContent = 'Tree structure';
    renderTree(dom.diagram, step.extras.tree, step.extras.class_values);
    return;
  }
  // The dendrogram is built once per fit; only the cut line moves per frame.
  if (state.result?.extras?.dendrogram) {
    dom.diagramBlock.hidden = false;
    dom.diagramTitle.textContent = 'Dendrogram';
    renderDendrogram(
      dom.diagram,
      state.result.extras.dendrogram,
      step.extras?.cut_height ?? 0,
      state.result.extras.max_height
    );
    return;
  }
  if (state.algorithm === 'mlp') {
    dom.diagramBlock.hidden = false;
    dom.diagramTitle.textContent = 'Network weights';
    renderNetwork(dom.diagram, step.extras?.network);
    return;
  }
  dom.diagramBlock.hidden = true;
  dom.diagram.innerHTML = '';
}

function renderOverlayToggles(result) {
  const toggles = [];
  const step = result.steps[0] || {};
  const task = result.task;

  if (task !== 'regression') {
    toggles.push({
      key: 'surface',
      label: 'Decision regions',
      checked: state.overlays.surface !== false,
      help: 'Shade the plane by what the model predicts there.',
    });
  }
  if (step.extras?.split_lines) {
    toggles.push({
      key: 'splits',
      label: 'Split lines',
      checked: state.overlays.splits !== false,
      help: 'Show where each node of the tree cuts the plane.',
    });
  }
  if (step.extras?.eps) {
    toggles.push({
      key: 'eps',
      label: 'eps radius',
      checked: state.overlays.eps !== false,
      help: "Circle the neighbourhood radius around the point being expanded.",
    });
  }
  if (step.extras?.ellipses) {
    toggles.push({
      key: 'ellipses',
      label: 'Class ellipses (2σ)',
      checked: state.overlays.ellipses !== false,
      help: 'The Gaussian each class was fitted with.',
    });
  }
  if (result.extras?.has_single_tree) {
    toggles.push({
      key: 'singleTree',
      label: 'Newest single tree',
      checked: Boolean(state.overlays.singleTree),
      help: 'Overlay one tree of the forest to compare it with the ensemble.',
    });
  }
  if (step.extras?.show_residuals) {
    toggles.push({
      key: 'residuals',
      label: 'Residuals',
      checked: state.overlays.residuals !== false,
      help: 'The vertical error between each point and the fitted curve.',
    });
  }
  if (result.extras?.reference_curve) {
    toggles.push({
      key: 'reference',
      label: 'Optimal fit',
      checked: state.overlays.reference !== false,
      help: 'The exact least-squares solution, for comparison.',
    });
  }

  toggles.forEach((toggle) => {
    if (state.overlays[toggle.key] === undefined) state.overlays[toggle.key] = toggle.checked;
  });

  buildOverlayToggles(dom.overlayToggles, toggles, (key, value) => {
    state.overlays[key] = value;
    plot.setOverlays({ ...state.overlays });
  });
  plot.setOverlays({ ...state.overlays });
}

function renderClassPicker() {
  const spec = currentSpec();
  const enabled = spec.task === 'classification';
  const classes = Array.from({ length: state.classCount }, (_, index) => ({
    index,
    color: classColor(index),
  }));

  buildClassPicker(dom.classPicker, {
    classes,
    active: state.activeClass,
    enabled,
    onSelect: (index) => {
      state.activeClass = index;
      plot.setActiveClass(index);
      renderClassPicker();
    },
    onAdd: () => {
      state.classCount = Math.min(state.classCount + 1, 6);
      state.activeClass = state.classCount - 1;
      plot.setActiveClass(state.activeClass);
      renderClassPicker();
    },
  });
  plot.setActiveClass(state.activeClass);
}

/* ------------------------------------------------------------ status --- */

function setStatus(kind, text) {
  dom.statusPill.className = `pill pill-${kind}`;
  dom.statusPill.textContent = text;
}

/**
 * Switch theme and repaint everything that draws its own colours.
 *
 * The surface cache holds bitmaps built from the palette, and canvas/SVG can
 * neither inherit CSS variables nor be restyled after the fact, so all three
 * renderers have to redraw rather than merely re-render.
 */
function applyTheme(next) {
  theme.apply(next);
  dom.themeToggle.textContent = next === 'light' ? 'Dark mode' : 'Light mode';
  plot.surfaceCache.clear();
  plot.render();
  chart.render();
  if (state.result) renderDiagram(state.result.steps[state.stepIndex]);
}

/** Whether changes should recompute immediately, or wait for a button. */
function autoRun() {
  return dom.autoTrain.checked;
}

/** Remember that a change is staged, and say which button applies it. */
function markStale(what) {
  state.stale = what;
  setStatus('stale', what === 'data' ? 'Press Generate to apply' : 'Press Train to apply');
}

function showError(message) {
  dom.plotError.hidden = false;
  dom.plotError.textContent = message;
}

function clearError() {
  dom.plotError.hidden = true;
  dom.plotError.textContent = '';
}

function updateEmptyState() {
  dom.plotEmpty.hidden = plot.points.length > 0;
}

boot();
