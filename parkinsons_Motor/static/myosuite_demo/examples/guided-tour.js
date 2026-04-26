// Judge-facing guided tour. Four popups walk through what each panel does.
// Auto-shows on every fresh load *after* all 3D assets have finished
// downloading (otherwise the spotlight anchors would be missing on slow
// Hugging Face cold starts). A help bubble bottom-right relaunches it.

const STEPS = [
  {
    title: 'Brain — live signal simulation',
    selector: '.brain-panel',
    fallbackSelector: 'canvas',
    placement: 'right',
    body: (
      'The 3D brain shows resting cortical tissue (skin tone). Red hotspots ' +
      'and nerve fibers light up only when neural pathways fire. The dual ' +
      'EEG below shows the pathological STN beta rhythm (top, red) and the ' +
      'motor cortex output (bottom, cyan). Watch the STN ↔ M1 loop diagram ' +
      'turn cool when DBS suppresses pathology.'
    ),
  },
  {
    title: 'OpenEnv Controls - drive the agent',
    selector: '.openenv-dock',
    placement: 'right',
    body: (
      'Pick a task from the Task dropdown - each task swaps the 3D body ' +
      'model (finger, hand, leg, elbow) to match the clinical motion. ' +
      'Click Connect to open a session. Then tweak DBS params (Amp mA, ' +
      'PW ms, Freq Hz) and press Step. Each Step prints reward + signal ' +
      'deltas (beta, tremor, DBS entrainment) above the JSON output.'
    ),
  },
  {
    title: 'Body model - UPDRS-style motion',
    selector: 'canvas',
    placement: 'top',
    body: (
      'The musculoskeletal model performs the motor task assigned by the ' +
      'env. High beta = bradykinesia (smaller, slower movements). High ' +
      'tremor = visible 4–6 Hz shake. Effective DBS shrinks the tremor ' +
      'overlay and restores voluntary command amplitude.'
    ),
  },
  {
    title: 'Agent Panel - autonomous control',
    selector: '.agent-panel',
    placement: 'left',
    body: (
      'Pick an agent (Auto / HF Qwen / Local Heuristic) and click Start ' +
      'Agent to stream a full episode. Live signal bars track Beta, ' +
      'Tremor, DBS, Force, Side FX, Track. Event chips fire mid-episode ' +
      '(L-DOPA OFF, Dyskinesia, Tachyphylaxis) - the agent must react.'
    ),
  },
];

function clearChildren(node) {
  while (node.firstChild) { node.removeChild(node.firstChild); }
}

class GuidedTour {
  constructor() {
    this.idx = 0;
    this.overlay = null;
    this.popup = null;
    this.spotlight = null;
    this._reposition = this._reposition.bind(this);
    this._buildHelpButton();
    // Auto-start on every load, but only once the 3D assets and UI are in
    // the DOM. The previous fixed 1.8s delay raced the slow Hugging Face
    // cold-start: on first visits the GLTF brain (~5MB) and MuJoCo WASM
    // were still streaming when the tour fired, so anchors were missing
    // and users never saw the steps. We now wait for explicit ready
    // signals (`motorassist:loader-hidden`, `motorassist:brain-ready`)
    // and DOM presence of the openenv-dock + agent-panel before opening.
    this._autoStartWhenReady();
  }

  // Resolves only when:
  //   - the bootstrap loader has hidden itself (`motorassist:loader-hidden`),
  //   - the brain overlay has finished loading meshes (`motorassist:brain-ready`),
  //   - the MuJoCo body viewer global is live (`window.myoDemo`),
  //   - the viewer-controller has appended `.agent-panel` and `.openenv-dock`,
  //   - and a real `<canvas>` is in the DOM.
  // Falls back after a hard ceiling so a partial failure (e.g. brain GLTF
  // 404) doesn't permanently block the tour.
  _waitForReady() {
    const HARD_CEILING_MS = 60000;
    const POLL_MS = 200;
    const start = Date.now();
    const loaderEl = () => document.getElementById('viewer-loader');
    const loaderGone = () => {
      if (window.motorAssistLoaderHidden) { return true; }
      const el = loaderEl();
      return !el || el.classList.contains('is-hidden');
    };
    const domReady = () =>
      !!document.querySelector('.agent-panel') &&
      !!document.querySelector('.openenv-dock') &&
      !!document.querySelector('canvas');
    const brainReady = () => !!(window.motorAssistBrain && (window.motorAssistBrain.ready || window.motorAssistBrain.brainRoot));
    const bodyReady = () => !!window.myoDemo;
    return new Promise((resolve) => {
      const tick = () => {
        if (loaderGone() && domReady() && brainReady() && bodyReady()) {
          resolve(true);
          return;
        }
        if (Date.now() - start > HARD_CEILING_MS) {
          resolve(false);
          return;
        }
        window.setTimeout(tick, POLL_MS);
      };
      tick();
    });
  }

  async _autoStartWhenReady() {
    await this._waitForReady();
    // Small grace delay so layout has a frame to settle after the loader
    // fades out, otherwise the first popup positions against a transient
    // rect (e.g. .brain-panel still has its loading text width).
    window.setTimeout(() => this.start(), 350);
  }

  _buildHelpButton() {
    const btn = document.createElement('button');
    btn.className = 'tour-help-btn';
    btn.textContent = '? Tour';
    btn.title = 'Replay the guided tour';
    btn.addEventListener('click', () => this.start());
    document.body.appendChild(btn);
  }

  start() {
    this.idx = 0;
    if (this.overlay) { this.stop(); }
    this._buildOverlay();
    this._render();
  }

  stop() {
    if (this.overlay) {
      this.overlay.remove();
      this.overlay = null;
      this.popup = null;
      this.spotlight = null;
    }
    window.removeEventListener('resize', this._reposition);
  }

  _buildOverlay() {
    this.overlay = document.createElement('div');
    this.overlay.className = 'tour-overlay';

    this.spotlight = document.createElement('div');
    this.spotlight.className = 'tour-spotlight';
    this.overlay.appendChild(this.spotlight);

    this.popup = document.createElement('div');
    this.popup.className = 'tour-popup';
    this.overlay.appendChild(this.popup);

    document.body.appendChild(this.overlay);
    window.addEventListener('resize', this._reposition);
  }

  _resolveTarget(step) {
    let target = document.querySelector(step.selector);
    if (!target && step.fallbackSelector) {
      target = document.querySelector(step.fallbackSelector);
    }
    return target;
  }

  _render() {
    const step = STEPS[this.idx];
    const target = this._resolveTarget(step);
    const rect = target ? target.getBoundingClientRect() : null;

    if (rect && rect.width > 0 && rect.height > 0) {
      const pad = 10;
      this.spotlight.style.display = 'block';
      this.spotlight.style.left = `${rect.left - pad}px`;
      this.spotlight.style.top = `${rect.top - pad}px`;
      this.spotlight.style.width = `${rect.width + pad * 2}px`;
      this.spotlight.style.height = `${rect.height + pad * 2}px`;
    } else {
      this.spotlight.style.display = 'none';
    }

    clearChildren(this.popup);
    const kicker = document.createElement('div');
    kicker.className = 'tour-kicker';
    kicker.textContent = `Step ${this.idx + 1} of ${STEPS.length}`;
    const title = document.createElement('div');
    title.className = 'tour-title';
    title.textContent = step.title;
    const body = document.createElement('div');
    body.className = 'tour-body';
    body.textContent = step.body;

    const actions = document.createElement('div');
    actions.className = 'tour-actions';

    const skip = document.createElement('button');
    skip.className = 'tour-btn tour-btn-ghost';
    skip.textContent = 'Skip';
    skip.addEventListener('click', () => this.stop());

    const back = document.createElement('button');
    back.className = 'tour-btn tour-btn-ghost';
    back.textContent = 'Back';
    back.disabled = this.idx === 0;
    back.addEventListener('click', () => { if (this.idx > 0) { this.idx--; this._render(); } });

    const next = document.createElement('button');
    next.className = 'tour-btn tour-btn-primary';
    const last = this.idx === STEPS.length - 1;
    next.textContent = last ? 'Got it' : 'Next';
    next.addEventListener('click', () => {
      if (last) { this.stop(); }
      else { this.idx++; this._render(); }
    });

    actions.append(skip, back, next);
    this.popup.append(kicker, title, body, actions);

    this._placePopup(rect, step.placement || 'right');
  }

  _placePopup(rect, placement) {
    const popup = this.popup;
    popup.style.visibility = 'hidden';
    popup.style.left = '0px';
    popup.style.top = '0px';
    const pw = popup.offsetWidth;
    const ph = popup.offsetHeight;
    const margin = 18;
    let x = window.innerWidth / 2 - pw / 2;
    let y = window.innerHeight / 2 - ph / 2;

    if (rect) {
      switch (placement) {
        case 'right':
          x = rect.right + margin;
          y = rect.top;
          if (x + pw > window.innerWidth - 12) { x = rect.left - pw - margin; }
          break;
        case 'left':
          x = rect.left - pw - margin;
          y = rect.top;
          if (x < 12) { x = rect.right + margin; }
          break;
        case 'top':
          x = rect.left + rect.width / 2 - pw / 2;
          y = rect.top - ph - margin;
          if (y < 12) { y = rect.bottom + margin; }
          break;
        case 'bottom':
        default:
          x = rect.left + rect.width / 2 - pw / 2;
          y = rect.bottom + margin;
          break;
      }
    }
    x = Math.max(12, Math.min(window.innerWidth - pw - 12, x));
    y = Math.max(12, Math.min(window.innerHeight - ph - 12, y));
    popup.style.left = `${x}px`;
    popup.style.top = `${y}px`;
    popup.style.visibility = 'visible';
  }

  _reposition() {
    if (this.overlay) { this._render(); }
  }
}

window.motorAssistTour = new GuidedTour();
