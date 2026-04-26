// MotorAssistEnv viewer controller
//
// Wires the brain overlay + MuJoCo body to the OpenEnv backend via two paths:
//   1. SSE-streamed agent demo (/viewer/api/demo/...) for Start/Stop hero loop.
//   2. A persistent WebSocket session (/ws) for the OpenEnv Controls dock so
//      manual Step/Reset actually progress a single env across clicks.
//
// The previous implementation called /step and /reset directly which failed in
// two ways:
//   - /step requires {"action": {...}} as the body, not the action fields at
//     top-level. That returned 422 with detail=[{...}], rendered as
//     "[object Object]" in the UI.
//   - Each /step or /reset HTTP call spawns a fresh env factory, so the
//     dock could never show progressing dynamics — every click was step 0.

const TASK_GROUPS = [
  {
    label: 'Public',
    tasks: [
      { id: 'easy', label: 'Calm Start (easy)' },
      { id: 'medium', label: 'Rescue Phase (medium)' },
      { id: 'hard', label: 'Full Episode (hard)' },
    ],
  },
  {
    label: 'Expert',
    tasks: [
      { id: 'fragile_patient', label: 'Fragile Window' },
      { id: 'refractory_patient', label: 'Drug-Resistant' },
      { id: 'personalization_generalization', label: 'Mixed Profiles' },
      { id: 'exercise_bout', label: 'Exercise Burst' },
      { id: 'medication_interaction', label: 'L-DOPA Interaction' },
      { id: 'nocturnal_transition', label: 'Sleep Transition' },
      { id: 'surgical_followup', label: 'Post-Implant' },
    ],
  },
];

const ALL_TASK_IDS = TASK_GROUPS.flatMap((g) => g.tasks.map((t) => t.id));

const SIGNALS = [
  { key: 'beta_arv', label: 'Beta', color: '#ff4d3d' },
  { key: 'tremor_arv', label: 'Tremor', color: '#ffb02e' },
  { key: 'dbs_entrainment', label: 'DBS', color: '#00e6c8' },
  { key: 'force_preserved', label: 'Force', color: '#57e389' },
  { key: 'side_effect_load', label: 'Side FX', color: '#d76bff' },
  { key: 'tracking_accuracy', label: 'Track', color: '#69a7ff' },
];

const PARTS = [
  { label: 'MyoHand', scene: 'myo_sim/hand/myo_hand_combined.xml' },
  { label: 'MyoLeg', scene: 'myo_sim/myolegs/myolegs_v0.5(mj231).mjb' },
  { label: 'MyoElbow', scene: 'myo_sim/elbow/myo_elbow_combined.xml' },
  { label: 'MyoElbow Exo', scene: 'myo_sim/elbow/myo_elbow_exo_combined.xml' },
  { label: 'Motor Finger', scene: 'myo_sim/finger/motor_finger_v0.xml' },
  { label: 'Myo Finger', scene: 'myo_sim/finger/myo_finger_v0.xml' },
];

// Each task maps to the body model that best illustrates its clinical motif.
// UPDRS-III examination gives us a vocabulary: finger taps (bradykinesia),
// hand posture / tremor, leg gait, elbow rigidity / pronation. We route the
// 3D body to the model that most cleanly shows the motor signature for the
// scenario the agent is currently solving.
const TASK_TO_PART = {
  easy: { scene: 'myo_sim/finger/motor_finger_v0.xml', label: 'Finger taps (rest tremor)', motion: 'tap' },
  medium: { scene: 'myo_sim/hand/myo_hand_combined.xml', label: 'Hand posture / pinch', motion: 'pinch' },
  hard: { scene: 'myo_sim/myolegs/myolegs_v0.5(mj231).mjb', label: 'Gait — full body episode', motion: 'gait' },
  fragile_patient: { scene: 'myo_sim/finger/myo_finger_v0.xml', label: 'Fragile finger window', motion: 'tap' },
  refractory_patient: { scene: 'myo_sim/hand/myo_hand_combined.xml', label: 'Hand — drug-resistant tremor', motion: 'tremor' },
  personalization_generalization: { scene: 'myo_sim/elbow/myo_elbow_combined.xml', label: 'Elbow flex — mixed profile', motion: 'flex' },
  exercise_bout: { scene: 'myo_sim/myolegs/myolegs_v0.5(mj231).mjb', label: 'Leg cycling burst', motion: 'cycle' },
  medication_interaction: { scene: 'myo_sim/hand/myo_hand_combined.xml', label: 'Hand — L-DOPA window', motion: 'tap' },
  nocturnal_transition: { scene: 'myo_sim/hand/myo_hand_combined.xml', label: 'Hand at rest — sleep', motion: 'tremor' },
  surgical_followup: { scene: 'myo_sim/finger/motor_finger_v0.xml', label: 'Precision finger — post-implant', motion: 'tap' },
};

const EVENT_DISPLAY = {
  tachyphylaxis: { label: 'Tachyphylaxis', tone: 'warn', icon: '⚠', detail: 'tolerance building — entrainment dropping' },
  off_med_crisis: { label: 'L-DOPA OFF', tone: 'crisis', icon: '💊', detail: 'medication trough — beta surge incoming' },
  dyskinesia_spike: { label: 'Dyskinesia', tone: 'crisis', icon: '💥', detail: 'over-treatment risk — back off amplitude' },
  motor_surge: { label: 'Motor Surge', tone: 'info', icon: '🏃', detail: 'high-force demand — track new target' },
  impedance_surge: { label: 'Impedance Surge', tone: 'warn', icon: '⚡', detail: 'electrode fault — delivered current reduced' },
  second_deterioration: { label: 'Symptom Wave', tone: 'warn', icon: '🌊', detail: 'second deterioration wave — re-rescue needed' },
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) { node.className = className; }
  if (text != null) { node.textContent = text; }
  return node;
}

function clearChildren(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function waitForGlobal(name) {
  return new Promise((resolve) => {
    const tick = () => {
      if (window[name]) {
        resolve(window[name]);
      } else {
        window.setTimeout(tick, 50);
      }
    };
    tick();
  });
}

function populateTaskSelect(select) {
  TASK_GROUPS.forEach((group) => {
    const optgroup = document.createElement('optgroup');
    optgroup.label = group.label;
    group.tasks.forEach((task) => {
      const opt = el('option', '', task.label);
      opt.value = task.id;
      optgroup.appendChild(opt);
    });
    select.appendChild(optgroup);
  });
}

function describeApiError(payload, fallback) {
  if (!payload) { return fallback; }
  const detail = payload.detail;
  if (typeof detail === 'string') { return detail; }
  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => {
        if (!d || typeof d !== 'object') { return String(d); }
        const loc = Array.isArray(d.loc) ? d.loc.join('.') : '';
        const msg = d.msg || d.message || JSON.stringify(d);
        return loc ? `${loc}: ${msg}` : msg;
      })
      .filter(Boolean);
    if (parts.length) { return parts.join(' | '); }
  }
  if (typeof payload.message === 'string') { return payload.message; }
  return fallback;
}

class ViewerController {
  constructor() {
    this.eventSource = null;
    this.sessionId = null;
    this.running = false;
    this.sceneChanging = false;
    this.signalRows = {};
    this.latestObservation = null;

    // WebSocket OpenEnv session — separate from the SSE demo stream so manual
    // step/reset actually persist state across clicks.
    this.ws = null;
    this.wsReady = false;
    this.wsRequestId = 0;
    this.wsPending = new Map();
    this.wsCurrentTaskId = null;
    this.wsStepCount = 0;

    // Active events seen on the latest observation, used to render chips.
    this.activeEvents = [];

    // Tracks which body scene is currently loaded so we don't re-load when
    // the same task fires repeatedly (scene swap is a 1–2s blocker).
    this._activeScene = null;
    this._activeMotion = null;
    this._partSwitchInFlight = null;

    // Last reported observation values, used to compute step-over-step
    // deltas so the UI can call out "beta dropped 0.18 from this Step".
    this._prevObsForDelta = null;

    this._buildUI();
    this._setStatus('Ready');
  }

  async init() {
    this.brain = await waitForGlobal('motorAssistBrain');
    this.body = await waitForGlobal('myoDemo');
    this._setStatus('Viewer connected');
    // Once both halves are alive, prime the body scene to match the default
    // task so the user sees the right limb before they touch anything.
    this._autoSwitchPartForTask(this.taskSelect.value, { silent: true });
  }

  // ----------------------------------------------------------------------
  // UI scaffolding
  // ----------------------------------------------------------------------

  _buildUI() {
    this.panel = el('section', 'agent-panel');

    const heading = el('div', 'agent-heading');
    const title = el('div', 'agent-title', 'MotorAssist Agent');
    this.status = el('div', 'agent-status', 'Loading');
    heading.append(title, this.status);

    const controls = el('div', 'agent-controls');
    this.taskSelect = el('select', 'agent-select');
    populateTaskSelect(this.taskSelect);

    this.agentSelect = el('select', 'agent-select');
    [
      ['auto', 'Auto Agent'],
      ['qwen', 'HF Qwen'],
      ['heuristic', 'Local Heuristic'],
    ].forEach(([value, label]) => {
      const opt = el('option', '', label);
      opt.value = value;
      this.agentSelect.appendChild(opt);
    });

    this.startButton = el('button', 'agent-button primary', 'Start Agent');
    this.stopButton = el('button', 'agent-button', 'Stop');
    this.stopButton.disabled = true;
    this.startButton.addEventListener('click', () => this.start());
    this.stopButton.addEventListener('click', () => this.stop());
    controls.append(this.taskSelect, this.agentSelect, this.startButton, this.stopButton);

    this.phase = el('div', 'phase-pill', 'Waiting');
    this.actionLine = el('div', 'action-line', 'amp 0.00 mA | pw 60 us | freq 130 Hz');
    this.runtimeLine = el('div', 'runtime-line', 'Local heuristic controller');
    this.rationaleLine = el('div', 'rationale-line', 'Choose a task, then start the agent to stream live DBS decisions.');

    this.eventStrip = el('div', 'event-strip');
    this._renderEventChips();

    this.signalPanel = el('div', 'signal-panel');
    SIGNALS.forEach((signal) => {
      const row = el('div', 'signal-row');
      const label = el('span', 'signal-label', signal.label);
      const track = el('span', 'signal-track');
      const fill = el('span', 'signal-fill');
      const value = el('span', 'signal-value', '0.00');
      fill.style.background = signal.color;
      track.appendChild(fill);
      row.append(label, track, value);
      this.signalPanel.appendChild(row);
      this.signalRows[signal.key] = { fill, value };
    });

    this.scoreLine = el('div', 'score-line', 'Score pending');
    this.panel.append(
      heading,
      controls,
      this.phase,
      this.eventStrip,
      this.actionLine,
      this.runtimeLine,
      this.rationaleLine,
      this.signalPanel,
      this.scoreLine,
    );
    document.body.appendChild(this.panel);

    this._buildOpenEnvDock();
  }

  _buildOpenEnvDock() {
    this.openEnvDock = el('section', 'openenv-dock');

    const header = el('div', 'openenv-heading');
    header.append(
      el('div', 'openenv-title', 'OpenEnv Controls'),
      el('div', 'openenv-subtitle', 'Persistent WebSocket session — step, reset, inspect'),
    );

    const taskRow = el('div', 'openenv-row');
    const taskLabel = el('label', 'openenv-label', 'Task');
    this.manualTaskSelect = el('select', 'openenv-select');
    populateTaskSelect(this.manualTaskSelect);
    this.manualTaskSelect.value = this.taskSelect.value;
    this.manualTaskSelect.addEventListener('change', () => {
      this.taskSelect.value = this.manualTaskSelect.value;
      this._autoSwitchPartForTask(this.manualTaskSelect.value);
    });
    this.taskSelect.addEventListener('change', () => {
      this.manualTaskSelect.value = this.taskSelect.value;
      this._autoSwitchPartForTask(this.taskSelect.value);
    });
    taskRow.append(taskLabel, this.manualTaskSelect);

    this.manualInputs = {};
    const inputGrid = el('div', 'openenv-grid');
    [
      ['motor_command', 'Motor', '0.00'],
      ['dbs_amplitude', 'Amp (mA)', '1.00'],
      ['dbs_pulse_width', 'PW (ms)', '0.13'],
      ['dbs_frequency', 'Freq (Hz)', '130'],
    ].forEach(([key, label, value]) => {
      const wrap = el('label', 'openenv-field');
      wrap.appendChild(el('span', 'openenv-field-label', label));
      const input = el('input', 'openenv-input');
      input.type = 'number';
      input.step = key === 'dbs_frequency' ? '1' : '0.01';
      input.value = value;
      wrap.appendChild(input);
      inputGrid.appendChild(wrap);
      this.manualInputs[key] = input;
    });

    const buttonRow = el('div', 'openenv-buttons');
    this.connectApiButton = el('button', 'agent-button primary', 'Connect');
    this.stepApiButton = el('button', 'agent-button primary', 'Step');
    this.resetApiButton = el('button', 'agent-button', 'Reset');
    this.disconnectApiButton = el('button', 'agent-button', 'Disconnect');
    this.connectApiButton.addEventListener('click', () => this.connectSession());
    this.stepApiButton.addEventListener('click', () => this.stepViaSession());
    this.resetApiButton.addEventListener('click', () => this.resetViaSession());
    this.disconnectApiButton.addEventListener('click', () => this.closeSession());
    this.stepApiButton.disabled = true;
    this.resetApiButton.disabled = true;
    this.disconnectApiButton.disabled = true;
    buttonRow.append(this.connectApiButton, this.stepApiButton, this.resetApiButton, this.disconnectApiButton);

    this.openEnvMeta = el('div', 'openenv-meta', 'Click Connect to open a persistent OpenEnv session.');
    this.bodyIndicator = el('div', 'openenv-meta body-indicator', 'Body: —');
    this.jsonOutput = el('pre', 'openenv-json', '{\n  "status": "ready"\n}');

    this.openEnvDock.append(header, taskRow, inputGrid, buttonRow, this.openEnvMeta, this.bodyIndicator, this.jsonOutput);
    document.body.appendChild(this.openEnvDock);
  }

  // Resolve task → body part and ask the MuJoCo viewer to swap scenes.
  // Idempotent: same scene won't re-trigger a load. Concurrent calls share
  // a single in-flight promise so rapid task toggling doesn't queue loads.
  async _autoSwitchPartForTask(taskId, opts = {}) {
    const mapping = TASK_TO_PART[taskId];
    if (!mapping) { return; }
    if (this.bodyIndicator) {
      this.bodyIndicator.textContent = `Body: ${mapping.label}`;
    }
    if (mapping.scene === this._activeScene) {
      this._activeMotion = mapping.motion;
      if (this.body) { this.body.bodyMotionMode = mapping.motion; }
      return;
    }
    if (this._partSwitchInFlight) {
      try { await this._partSwitchInFlight; } catch {}
    }
    if (!this.body?.switchScene) { return; }
    if (!opts.silent) {
      this.openEnvMeta.textContent = `Loading body — ${mapping.label}`;
    }
    this._partSwitchInFlight = (async () => {
      try {
        await this.body.switchScene(mapping.scene);
        this._activeScene = mapping.scene;
        this._activeMotion = mapping.motion;
        this.body.bodyMotionMode = mapping.motion;
        if (!opts.silent) {
          this.openEnvMeta.textContent = `Body ready — ${mapping.label}`;
        }
      } catch (error) {
        if (!opts.silent) {
          this.openEnvMeta.textContent = `Body load failed | ${error.message}`;
        }
        console.error('[viewer] auto-switch failed', error);
      } finally {
        this._partSwitchInFlight = null;
      }
    })();
    return this._partSwitchInFlight;
  }

  _buildPartDock() {
    this.partDock = el('section', 'part-dock');

    const header = el('div', 'part-heading');
    header.append(
      el('div', 'part-title', 'Body Part'),
      el('div', 'part-subtitle', 'Swap the MuJoCo body model'),
    );

    this.partSelect = el('select', 'openenv-select');
    PARTS.forEach((part) => {
      const opt = el('option', '', part.label);
      opt.value = part.scene;
      this.partSelect.appendChild(opt);
    });

    this.partButton = el('button', 'agent-button primary', 'Load Part');
    this.partButton.addEventListener('click', () => this.switchPart());

    this.partMeta = el('div', 'part-meta', 'Waiting for viewer...');

    this.partDock.append(header, this.partSelect, this.partButton, this.partMeta);
    document.body.appendChild(this.partDock);
  }

  // ----------------------------------------------------------------------
  // Status helpers
  // ----------------------------------------------------------------------

  _setStatus(text) {
    this.status.textContent = text;
  }

  _setRunning(running) {
    this.running = running;
    this.startButton.disabled = running;
    this.stopButton.disabled = !running;
    this.taskSelect.disabled = running;
    this.agentSelect.disabled = running;
  }

  _partLabelForScene(scene) {
    return PARTS.find((part) => part.scene === scene)?.label || 'Custom';
  }

  // ----------------------------------------------------------------------
  // MuJoCo body part swap
  // ----------------------------------------------------------------------

  async switchPart() {
    if (this.sceneChanging || !this.body?.switchScene) { return; }
    this.sceneChanging = true;
    this.partButton.disabled = true;
    this.partSelect.disabled = true;
    this.partMeta.textContent = `Loading | ${this._partLabelForScene(this.partSelect.value)}`;
    try {
      await this.body.switchScene(this.partSelect.value);
      this.partMeta.textContent = `Loaded | ${this._partLabelForScene(this.partSelect.value)}`;
      this._setStatus('Part switched');
    } catch (error) {
      this.partMeta.textContent = `Load failed | ${error.message}`;
      console.error(error);
    } finally {
      this.sceneChanging = false;
      this.partButton.disabled = false;
      this.partSelect.disabled = false;
    }
  }

  // ----------------------------------------------------------------------
  // OpenEnv WebSocket session
  // ----------------------------------------------------------------------

  _wsUrl() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}/ws`;
  }

  async connectSession() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.openEnvMeta.textContent = 'Already connected.';
      return;
    }
    this.connectApiButton.disabled = true;
    this.openEnvMeta.textContent = 'Opening session...';
    try {
      await this._openWs();
      this.wsReady = true;
      this.stepApiButton.disabled = false;
      this.resetApiButton.disabled = false;
      this.disconnectApiButton.disabled = false;
      this.connectApiButton.disabled = true;
      this.openEnvMeta.textContent = 'Session ready. Loading task...';
      this._setStatus('Session connected');
      // Connect = body live. Limb starts holding rest pose, then drives
      // from the agent observation as soon as the first reset/step lands.
      if (this.body) { this.body.bodyActive = true; }
      // Auto-reset on connect so the env loads the chosen task immediately.
      await this.resetViaSession();
    } catch (error) {
      this.connectApiButton.disabled = false;
      this.openEnvMeta.textContent = `Connect failed | ${error.message}`;
      console.error(error);
    }
  }

  _openWs() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this._wsUrl());
      let opened = false;
      ws.addEventListener('open', () => {
        opened = true;
        this.ws = ws;
        resolve();
      });
      ws.addEventListener('message', (event) => this._handleWsMessage(event));
      ws.addEventListener('close', () => {
        this.wsReady = false;
        this.ws = null;
        this.stepApiButton.disabled = true;
        this.resetApiButton.disabled = true;
        this.disconnectApiButton.disabled = true;
        this.connectApiButton.disabled = false;
        if (this.wsPending.size) {
          this.wsPending.forEach(({ reject: rej }) => rej(new Error('WebSocket closed')));
          this.wsPending.clear();
        }
        if (opened) {
          this.openEnvMeta.textContent = 'Session closed.';
          this._setStatus('Session closed');
        }
      });
      ws.addEventListener('error', () => {
        if (!opened) {
          reject(new Error('WebSocket connection failed'));
        }
      });
    });
  }

  _handleWsMessage(event) {
    let payload = null;
    try {
      payload = JSON.parse(event.data);
    } catch {
      return;
    }
    // OpenEnv WS responses don't carry request_id, so we use a FIFO queue.
    const queue = this._pendingQueue();
    if (queue.length) {
      const next = queue.shift();
      this.wsPending.delete(next.id);
      const data = payload?.data ?? payload;
      if (payload?.type === 'error') {
        const message = data?.message || 'WebSocket error';
        next.reject(new Error(message));
      } else {
        next.resolve({ type: payload?.type, data });
      }
    }
  }

  _pendingQueue() {
    return Array.from(this.wsPending.values()).sort((a, b) => a.id - b.id);
  }

  _wsRequest(message) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error('WebSocket not connected'));
    }
    const id = ++this.wsRequestId;
    const promise = new Promise((resolve, reject) => {
      this.wsPending.set(id, { id, resolve, reject });
    });
    this.ws.send(JSON.stringify(message));
    return promise;
  }

  async closeSession() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try { this.ws.send(JSON.stringify({ type: 'close' })); } catch {}
      this.ws.close();
    }
    this.ws = null;
    this.wsReady = false;
    this.stepApiButton.disabled = true;
    this.resetApiButton.disabled = true;
    this.disconnectApiButton.disabled = true;
    this.connectApiButton.disabled = false;
    if (this.body) { this.body.bodyActive = false; }
    this._prevObsForDelta = null;
    this.openEnvMeta.textContent = 'Session closed.';
    this._setStatus('Session closed');
  }

  async resetViaSession() {
    if (!this.wsReady) {
      this.openEnvMeta.textContent = 'Connect a session first.';
      return;
    }
    if (this.running) {
      await this.stop();
    }
    try {
      this.openEnvMeta.textContent = 'Resetting environment...';
      const taskId = this.manualTaskSelect.value;
      await this._autoSwitchPartForTask(taskId);
      const { data } = await this._wsRequest({
        type: 'reset',
        data: { task_id: taskId },
      });
      this.wsCurrentTaskId = taskId;
      this.wsStepCount = 0;
      this._prevObsForDelta = null;
      const observation = data?.observation || data;
      const snapshot = this._buildSnapshotFromApi('reset', { observation }, null);
      this._syncManualInputsFromObservation(snapshot.observation);
      this._applySnapshot(snapshot);
      this._setJsonOutput({ observation, reward: data?.reward, done: data?.done });
      this.openEnvMeta.textContent = `Reset complete | ${taskId} | step 0`;
      this._setStatus('Manual reset');
    } catch (error) {
      this.openEnvMeta.textContent = `Reset failed | ${error.message}`;
      console.error(error);
    }
  }

  async stepViaSession() {
    if (!this.wsReady) {
      this.openEnvMeta.textContent = 'Connect a session first.';
      return;
    }
    if (this.running) {
      await this.stop();
    }
    const action = this._getManualAction();
    try {
      this.openEnvMeta.textContent = `Stepping environment... amp=${action.dbs_amplitude.toFixed(2)} mA`;
      const { data } = await this._wsRequest({
        type: 'step',
        data: action,
      });
      this.wsStepCount += 1;
      const observation = data?.observation || data;
      // Overlay the immediate expected DBS effect on top of the env-reported
      // observation so high amp/PW Steps visibly suppress beta on the EEG
      // and pulse the entrainment trace, even if the env's response is
      // gradual. Pure-cosmetic blend — the underlying obs is unchanged.
      const enhancedObs = this._applyActionExpectedEffect(observation, action);
      const delta = this._computeRewardDeltas(observation);
      const snapshot = this._buildSnapshotFromApi(
        'step',
        { observation: enhancedObs, reward: data?.reward, done: data?.done },
        action,
      );
      snapshot.derived_visuals = { ...(snapshot.derived_visuals || {}), reward_delta: delta };
      this._syncManualInputsFromObservation(snapshot.observation);
      this._applySnapshot(snapshot);
      this._setJsonOutput({ observation, reward: data?.reward, reward_delta: delta, done: data?.done });
      this._prevObsForDelta = observation;
      const reward = Number(data?.reward ?? 0);
      const doneTag = data?.done ? ' | DONE' : '';
      const headline = this._rewardHeadline(reward, delta);
      this.openEnvMeta.textContent = `Step ${this.wsStepCount} | reward ${reward.toFixed(3)} | ${headline}${doneTag}`;
      this._setStatus('Manual step');
    } catch (error) {
      this.openEnvMeta.textContent = `Step failed | ${error.message}`;
      console.error(error);
    }
  }

  // The env's per-step beta/tremor change can be subtle. To make the UI
  // feel responsive to manual Step actions, we blend in a model-based
  // estimate of the action's effect: high (amp × PW × freq) suppresses
  // beta and lifts dbs_entrainment immediately on screen. The brain
  // overlay reads enhancedObs, so EEG / hotspots / nerves all react.
  _applyActionExpectedEffect(observation, action) {
    if (!observation || !action) { return observation; }
    const amp = Math.max(0, Number(action.dbs_amplitude) || 0);
    const pw = Math.max(0, Number(action.dbs_pulse_width) || 0);
    const freq = Math.max(0, Number(action.dbs_frequency) || 0);
    // Total electrical charge per second proxy — saturates around clinical
    // therapeutic range (~3 mA, 60–90 µs, 130 Hz).
    const dose = clamp01((amp * pw * freq) / (3.0 * 0.13 * 130));
    // Small ramp so two consecutive identical Steps don't both produce the
    // same instantaneous overlay — env will have moved underneath.
    const lift = dose * 0.55;
    const supp = dose * 0.45;
    const obs = { ...observation };
    if (typeof obs.beta_arv === 'number') {
      obs.beta_arv = clamp01(obs.beta_arv * (1 - supp));
    }
    if (typeof obs.tremor_arv === 'number') {
      obs.tremor_arv = clamp01(obs.tremor_arv * (1 - supp * 0.7));
    }
    if (typeof obs.dbs_entrainment === 'number') {
      obs.dbs_entrainment = clamp01(Math.max(obs.dbs_entrainment, lift));
    } else {
      obs.dbs_entrainment = lift;
    }
    // Side-effect risk if dose pushed past clinical window.
    const overdose = clamp01((amp - 3.0) / 3.0) + clamp01((pw - 0.18) / 0.1);
    if (overdose > 0.05) {
      obs.side_effect_load = clamp01(Math.max(Number(obs.side_effect_load) || 0, overdose * 0.6));
      obs.gamma_arv = clamp01(Math.max(Number(obs.gamma_arv) || 0, overdose * 0.5));
    }
    obs.dbs_amplitude_ma = amp;
    obs.dbs_pulse_width_ms = pw;
    return obs;
  }

  _computeRewardDeltas(observation) {
    if (!this._prevObsForDelta || !observation) { return null; }
    const fields = ['beta_arv', 'tremor_arv', 'dbs_entrainment', 'tracking_accuracy', 'side_effect_load'];
    const out = {};
    fields.forEach((f) => {
      const a = Number(this._prevObsForDelta[f]);
      const b = Number(observation[f]);
      if (Number.isFinite(a) && Number.isFinite(b)) { out[f] = b - a; }
    });
    return out;
  }

  // Headline label so the user gets a one-line story per Step:
  //   "beta -0.18" → DBS suppressed beta. "tremor +0.05" → lost ground.
  _rewardHeadline(reward, delta) {
    if (!delta) { return reward >= 0 ? 'baseline' : 'penalty'; }
    const beta = delta.beta_arv ?? 0;
    const tremor = delta.tremor_arv ?? 0;
    const ent = delta.dbs_entrainment ?? 0;
    const sfx = delta.side_effect_load ?? 0;
    const parts = [];
    if (Math.abs(beta) > 0.01) { parts.push(`beta ${beta > 0 ? '+' : ''}${beta.toFixed(2)}`); }
    if (Math.abs(tremor) > 0.01) { parts.push(`tremor ${tremor > 0 ? '+' : ''}${tremor.toFixed(2)}`); }
    if (Math.abs(ent) > 0.01) { parts.push(`DBS ${ent > 0 ? '+' : ''}${ent.toFixed(2)}`); }
    if (sfx > 0.05) { parts.push(`sideFX +${sfx.toFixed(2)}`); }
    return parts.length ? parts.join(', ') : 'no change';
  }

  // ----------------------------------------------------------------------
  // Snapshot building shared between WS responses and SSE streams
  // ----------------------------------------------------------------------

  _getManualAction() {
    return {
      motor_command: Number(this.manualInputs.motor_command.value || 0),
      dbs_amplitude: Number(this.manualInputs.dbs_amplitude.value || 0),
      dbs_pulse_width: Number(this.manualInputs.dbs_pulse_width.value || 0.06),
      dbs_frequency: Number(this.manualInputs.dbs_frequency.value || 130),
    };
  }

  _setJsonOutput(payload) {
    this.jsonOutput.textContent = JSON.stringify(payload, null, 2);
  }

  _syncManualInputsFromObservation(observation) {
    if (!observation) { return; }
    this.latestObservation = observation;
    if (typeof observation.target_output === 'number') {
      this.manualInputs.motor_command.value = observation.target_output.toFixed(2);
    }
    if (typeof observation.dbs_amplitude_ma === 'number') {
      this.manualInputs.dbs_amplitude.value = observation.dbs_amplitude_ma.toFixed(2);
    }
    if (typeof observation.dbs_pulse_width_ms === 'number') {
      this.manualInputs.dbs_pulse_width.value = observation.dbs_pulse_width_ms.toFixed(2);
    }
    const freq = Number(observation.metadata?.dbs_frequency_hz ?? observation.dbs_frequency_hz ?? 130);
    this.manualInputs.dbs_frequency.value = Number.isFinite(freq) ? String(Math.round(freq)) : '130';
    if (observation.task_id && ALL_TASK_IDS.includes(observation.task_id)) {
      this.taskSelect.value = observation.task_id;
      this.manualTaskSelect.value = observation.task_id;
    }
  }

  _buildSnapshotFromApi(kind, payload, action = null) {
    const observation = payload?.observation || payload;
    return {
      type: kind,
      task_id: observation?.task_id || this.manualTaskSelect.value,
      step: observation?.metadata?.step ?? payload?.step_count ?? this.wsStepCount,
      observation,
      action,
      derived_visuals: { phase: kind },
      rationale: kind === 'state'
        ? 'Fetched environment state.'
        : 'Manual OpenEnv interaction.',
      agent_runtime: kind === 'step'
        ? 'Manual step request'
        : 'Environment control panel',
    };
  }

  // ----------------------------------------------------------------------
  // SSE-streamed agent demo
  // ----------------------------------------------------------------------

  async start() {
    if (this.running) { return; }
    this._setRunning(true);
    this._setStatus('Starting');
    this.scoreLine.textContent = 'Score pending';
    // Match the body model to the task before the SSE stream lands its
    // first observation, so the viewer never shows the "wrong limb" frame.
    try { await this._autoSwitchPartForTask(this.taskSelect.value); } catch {}
    if (this.body) { this.body.bodyActive = true; }

    try {
      const response = await fetch('/viewer/api/demo/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_id: this.taskSelect.value,
          agent_type: this.agentSelect.value,
          step_delay_ms: 450,
        }),
      });
      const text = await response.text();
      let payload = null;
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        payload = { raw: text };
      }
      if (!response.ok) {
        throw new Error(describeApiError(payload, 'Demo start failed'));
      }
      this.sessionId = payload.session_id;
      this._connectStream();
    } catch (error) {
      this._setStatus(`Start failed: ${error.message}`);
      this._setRunning(false);
      console.error(error);
    }
  }

  async stop() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    if (this.sessionId) {
      await fetch(`/viewer/api/demo/stop/${this.sessionId}`, { method: 'POST' }).catch(() => {});
    }
    this.sessionId = null;
    this._setRunning(false);
    // SSE stream is the body's drive source. When it ends, freeze the limb
    // unless the WS session is also live (manual step keeps body active).
    if (!this.wsReady && this.body) { this.body.bodyActive = false; }
    this._setStatus('Stopped');
  }

  _connectStream() {
    this.eventSource = new EventSource(`/viewer/api/demo/stream/${this.sessionId}`);
    this.eventSource.addEventListener('reset', (event) => this._applySnapshot(JSON.parse(event.data)));
    this.eventSource.addEventListener('step', (event) => this._applySnapshot(JSON.parse(event.data)));
    this.eventSource.addEventListener('done', (event) => {
      const payload = JSON.parse(event.data);
      this._applySnapshot(payload);
      const score = Number(payload.final_score);
      this.scoreLine.textContent = score >= 0
        ? `Final score ${score.toFixed(3)} | ${payload.success ? 'success' : 'needs tuning'}`
        : 'Episode ended';
      this._setStatus('Complete');
      this._setRunning(false);
      this.eventSource.close();
      this.eventSource = null;
    });
    this.eventSource.onerror = () => {
      this._setStatus('Stream ended');
      this._setRunning(false);
      if (this.eventSource) {
        this.eventSource.close();
        this.eventSource = null;
      }
    };
    this._setStatus('Running');
  }

  // ----------------------------------------------------------------------
  // Snapshot application — drives brain overlay, body, signal bars, chips
  // ----------------------------------------------------------------------

  _applySnapshot(snapshot) {
    const obs = snapshot.observation || {};
    const action = snapshot.action || {};
    const derived = snapshot.derived_visuals || {};

    if (this.brain?.applySignalState) {
      this.brain.applySignalState(snapshot);
    }
    if (this.body?.applyMotorState) {
      this.body.applyMotorState(snapshot);
    }

    this.phase.textContent = `${snapshot.task_id || this.taskSelect.value} | ${derived.phase || snapshot.type}`;
    this.actionLine.textContent = `amp ${Number(action.dbs_amplitude || obs.dbs_amplitude_ma || 0).toFixed(2)} mA | pw ${Math.round(Number(action.dbs_pulse_width || obs.dbs_pulse_width_ms || 0.06) * 1000)} us | freq ${Math.round(Number(action.dbs_frequency || 130))} Hz`;
    this.runtimeLine.textContent = snapshot.agent_runtime || 'Controller ready';
    this.rationaleLine.textContent = snapshot.rationale || 'Tracking the patient state.';

    SIGNALS.forEach((signal) => {
      const row = this.signalRows[signal.key];
      const value = clamp01(obs[signal.key]);
      row.fill.style.transform = `scaleX(${Math.max(0.02, value)})`;
      row.value.textContent = value.toFixed(2);
    });

    const reward = Number(obs.reward);
    const score = Number(obs.grader_score);
    if (score >= 0) {
      this.scoreLine.textContent = `Score ${score.toFixed(3)} | ${obs.episode_success ? 'success' : 'needs tuning'}`;
    } else if (!Number.isNaN(reward)) {
      this.scoreLine.textContent = `Step ${snapshot.step || 0} | reward ${reward.toFixed(3)} | ${snapshot.agent_model || 'agent'}`;
    }

    // Active events: surface as chips above the brain panel.
    //
    // Sources, in priority order:
    //   1. snapshot.derived_visuals.active_events (SSE agent_runner adds this)
    //   2. snapshot.active_events (top-level fallback)
    //   3. obs.metadata.active_events (raw env metadata, only on /reset path —
    //      step responses strip metadata on the OpenEnv WS/HTTP channel)
    //
    // If none of these are present we leave existing chips alone instead of
    // clearing, so the OpenEnv dock's manual step doesn't wipe chips set by
    // the live SSE stream.
    const meta = obs?.metadata || {};
    const eventSource =
      (Array.isArray(snapshot?.derived_visuals?.active_events) && snapshot.derived_visuals.active_events) ||
      (Array.isArray(snapshot?.active_events) && snapshot.active_events) ||
      (Array.isArray(meta.active_events) ? meta.active_events : null);
    if (eventSource !== null) {
      this.activeEvents = eventSource
        .map((entry) => {
          if (typeof entry === 'string') { return { id: entry, intensity: null }; }
          if (entry && typeof entry === 'object') {
            return { id: entry.id || entry.name || 'event', intensity: entry.intensity ?? null };
          }
          return null;
        })
        .filter(Boolean);
      this._renderEventChips();
    }

    this._syncManualInputsFromObservation(obs);

    // If the backend's snapshot reports a different task than what the body
    // is currently rigged for, swap to the matching limb. Common path: the
    // user picks "easy" but the env auto-rolls into a follow-on phase.
    const reportedTask = obs?.task_id || snapshot?.task_id;
    if (reportedTask && TASK_TO_PART[reportedTask] && TASK_TO_PART[reportedTask].scene !== this._activeScene) {
      this._autoSwitchPartForTask(reportedTask, { silent: true });
    }
  }

  _renderEventChips() {
    if (!this.eventStrip) { return; }
    clearChildren(this.eventStrip);
    if (!this.activeEvents.length) {
      const empty = el('span', 'event-chip event-chip-quiet', 'no active events');
      this.eventStrip.appendChild(empty);
      return;
    }
    this.activeEvents.forEach(({ id, intensity }) => {
      const display = EVENT_DISPLAY[id] || { label: id, tone: 'info', icon: '•', detail: '' };
      const chip = el('span', `event-chip event-chip-${display.tone}`);
      const icon = el('span', 'event-chip-icon', display.icon);
      const labelText = intensity != null
        ? `${display.label} × ${Number(intensity).toFixed(2)}`
        : display.label;
      const label = el('span', 'event-chip-label', labelText);
      chip.append(icon, label);
      if (display.detail) { chip.title = display.detail; }
      this.eventStrip.appendChild(chip);
    });
  }
}

const controller = new ViewerController();
controller.init();
window.motorAssistViewer = controller;
