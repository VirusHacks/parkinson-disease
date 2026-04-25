const TASKS = [
  { id: 'beta_suppression', label: 'Calm Start' },
  { id: 'tremor_correction', label: 'Rescue Phase' },
  { id: 'full_episode', label: 'Full Episode' },
];

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

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) { node.className = className; }
  if (text) { node.textContent = text; }
  return node;
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

class ViewerController {
  constructor() {
    this.eventSource = null;
    this.sessionId = null;
    this.running = false;
    this.sceneChanging = false;
    this.signalRows = {};
    this.latestObservation = null;
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

  _buildUI() {
    this.panel = el('section', 'agent-panel');

    const heading = el('div', 'agent-heading');
    const title = el('div', 'agent-title', 'MotorAssist Agent');
    this.status = el('div', 'agent-status', 'Loading');
    heading.append(title, this.status);

    const controls = el('div', 'agent-controls');
    this.taskSelect = el('select', 'agent-select');
    TASKS.forEach((task) => {
      const opt = el('option', '', task.label);
      opt.value = task.id;
      this.taskSelect.appendChild(opt);
    });

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
      el('div', 'openenv-subtitle', 'Direct reset, step, and state access'),
    );

    const taskRow = el('div', 'openenv-row');
    const taskLabel = el('label', 'openenv-label', 'Task');
    this.manualTaskSelect = el('select', 'openenv-select');
    TASKS.forEach((task) => {
      const opt = el('option', '', task.label);
      opt.value = task.id;
      this.manualTaskSelect.appendChild(opt);
    });
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
      ['dbs_amplitude', 'Amp', '0.00'],
      ['dbs_pulse_width', 'PW', '0.06'],
      ['dbs_frequency', 'Freq', '130'],
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
    this.stepApiButton = el('button', 'agent-button primary', 'Step');
    this.resetApiButton = el('button', 'agent-button', 'Reset');
    this.getStateButton = el('button', 'agent-button', 'Get state');
    this.stepApiButton.addEventListener('click', () => this.stepViaApi());
    this.resetApiButton.addEventListener('click', () => this.resetViaApi());
    this.getStateButton.addEventListener('click', () => this.getStateViaApi());
    buttonRow.append(this.stepApiButton, this.resetApiButton, this.getStateButton);

    this.openEnvMeta = el('div', 'openenv-meta', 'Episode state will appear here.');
    this.jsonOutput = el('pre', 'openenv-json', '{\n  "status": "ready"\n}');

    this.openEnvDock.append(header, taskRow, inputGrid, buttonRow, this.openEnvMeta, this.jsonOutput);
    document.body.appendChild(this.openEnvDock);
  }

  _buildPartDock() {
    this.partDock = el('section', 'part-dock');

    const header = el('div', 'part-heading');
    header.append(
      el('div', 'part-title', 'Part Switcher'),
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

  _getManualAction() {
    return {
      motor_command: Number(this.manualInputs.motor_command.value || 0),
      dbs_amplitude: Number(this.manualInputs.dbs_amplitude.value || 0),
      dbs_pulse_width: Number(this.manualInputs.dbs_pulse_width.value || 0.06),
      dbs_frequency: Number(this.manualInputs.dbs_frequency.value || 130),
      task_id: '',
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
    if (observation.task_id) {
      this.taskSelect.value = observation.task_id;
      this.manualTaskSelect.value = observation.task_id;
    }
  }

  _buildSnapshotFromApi(kind, payload, action = null) {
    const observation = payload?.observation || payload;
    return {
      type: kind,
      task_id: observation?.task_id || this.manualTaskSelect.value,
      step: observation?.metadata?.step ?? payload?.step_count ?? 0,
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

  async _requestJson(path, options, fallbackError) {
    const response = await fetch(path, options);
    const text = await response.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch {
      payload = { raw: text };
    }
    if (!response.ok) {
      throw new Error(payload?.detail || payload?.message || fallbackError);
    }
    return payload;
  }

  async resetViaApi() {
    if (this.running) {
      await this.stop();
    }
    try {
      this.openEnvMeta.textContent = 'Resetting environment...';
      const payload = await this._requestJson(
        '/reset',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task_id: this.manualTaskSelect.value }),
        },
        'Reset failed',
      );
      const snapshot = this._buildSnapshotFromApi('reset', payload, null);
      this._syncManualInputsFromObservation(snapshot.observation);
      this._applySnapshot(snapshot);
      this._setJsonOutput(payload);
      this.openEnvMeta.textContent = `Reset complete | ${snapshot.task_id}`;
      this._setStatus('Manual reset');
    } catch (error) {
      this.openEnvMeta.textContent = `Reset failed | ${error.message}`;
      console.error(error);
    }
  }

  async stepViaApi() {
    if (this.running) {
      await this.stop();
    }
    const action = this._getManualAction();
    try {
      this.openEnvMeta.textContent = 'Stepping environment...';
      const payload = await this._requestJson(
        '/step',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(action),
        },
        'Step failed',
      );
      const snapshot = this._buildSnapshotFromApi('step', payload, action);
      this._syncManualInputsFromObservation(snapshot.observation);
      this._applySnapshot(snapshot);
      this._setJsonOutput(payload);
      this.openEnvMeta.textContent = `Step complete | reward ${Number(payload.reward ?? 0).toFixed(3)}`;
      this._setStatus('Manual step');
    } catch (error) {
      this.openEnvMeta.textContent = `Step failed | ${error.message}`;
      console.error(error);
    }
  }

  async getStateViaApi() {
    try {
      this.openEnvMeta.textContent = 'Fetching state...';
      const payload = await this._requestJson('/state', { method: 'GET' }, 'State fetch failed');
      this._setJsonOutput(payload);
      this.openEnvMeta.textContent = `Episode ${payload.episode_id || 'n/a'} | step ${payload.step_count ?? 0}`;
      this._setStatus('State fetched');
    } catch (error) {
      this.openEnvMeta.textContent = `Get state failed | ${error.message}`;
      console.error(error);
    }
  }

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
      const payload = await response.json();
      this.sessionId = payload.session_id;
      this._connectStream();
    } catch (error) {
      this._setStatus('Start failed');
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

    this._syncManualInputsFromObservation(obs);
  }
}

const controller = new ViewerController();
controller.init();
window.motorAssistViewer = controller;
