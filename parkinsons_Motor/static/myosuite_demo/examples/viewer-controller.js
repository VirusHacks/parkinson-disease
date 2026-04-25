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

    this._buildUI();
    this._setStatus('Ready');
  }

  async init() {
    this.brain = await waitForGlobal('motorAssistBrain');
    this.body = await waitForGlobal('myoDemo');
    this.partSelect.value = this.body?.params?.scene || PARTS[0].scene;
    this.partMeta.textContent = `Loaded | ${this._partLabelForScene(this.partSelect.value)}`;
    this._setStatus('Viewer connected');
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
    this._buildPartDock();
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
    });
    this.taskSelect.addEventListener('change', () => {
      this.manualTaskSelect.value = this.taskSelect.value;
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
    this.jsonOutput = el('pre', 'openenv-json', '{\n  "status": "ready"\n}');

    this.openEnvDock.append(header, taskRow, inputGrid, buttonRow, this.openEnvMeta, this.jsonOutput);
    document.body.appendChild(this.openEnvDock);
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
      const { data } = await this._wsRequest({
        type: 'reset',
        data: { task_id: taskId },
      });
      this.wsCurrentTaskId = taskId;
      this.wsStepCount = 0;
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
      const snapshot = this._buildSnapshotFromApi('step', { observation, reward: data?.reward, done: data?.done }, action);
      this._syncManualInputsFromObservation(snapshot.observation);
      this._applySnapshot(snapshot);
      this._setJsonOutput({ observation, reward: data?.reward, done: data?.done });
      const reward = Number(data?.reward ?? 0);
      const doneTag = data?.done ? ' | DONE' : '';
      this.openEnvMeta.textContent = `Step ${this.wsStepCount} | reward ${reward.toFixed(3)}${doneTag}`;
      this._setStatus('Manual step');
    } catch (error) {
      this.openEnvMeta.textContent = `Step failed | ${error.message}`;
      console.error(error);
    }
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
