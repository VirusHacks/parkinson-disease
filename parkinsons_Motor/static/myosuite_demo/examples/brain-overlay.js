import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Brain overlay panel.
//
// Three vertically stacked sections:
//   1. 3D brain (GLTF) with bilateral STN leads, glowing electrodes, animated
//      pulse shells, and a screen-space SVG label layer that pins anatomical
//      labels (Motor Cortex, Thalamus, STN-L, STN-R) to projected 3D points.
//   2. Dual-trace EEG strip showing:
//        - Pathological STN beta (~20 Hz) in red, scaled by beta_arv
//        - Effective motor cortex output in cyan/green, scaled by
//          dbs_entrainment * (1 - beta_arv)
//      The visual contrast between the two traces is the "before vs. after
//      DBS" story in one glance.
//   3. Compact pathway diagram (M1 -> Thalamus -> STN with a DBS lead arrow)
//      that lights up cooler when DBS is suppressing pathology.

const PANEL_W = 340;
const BRAIN_W = 320;
const BRAIN_H = 280;
const EEG_W = 320;
const EEG_H = 96;
const PATHWAY_H = 64;

function makeStat(label, value) {
  const span = document.createElement('span');
  span.appendChild(document.createTextNode(label + ' '));
  const val = document.createElement('b');
  val.style.color = '#00ffc8';
  val.textContent = value;
  span.appendChild(val);
  return { span, val };
}

function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

class BrainOverlay {
  constructor() {
    this.time = 0;
    this.tipMeshes = [];
    this.tipPositions = [];
    this.pulseGroups = [];
    this._stnMeshes = [];
    // Dual EEG ring buffers — pathological beta and DBS-corrected motor signal.
    this.betaBuffer = new Float32Array(EEG_W).fill(EEG_H * 0.30);
    this.motorBuffer = new Float32Array(EEG_W).fill(EEG_H * 0.72);
    this.brainRoot = null;
    this.signalState = {
      beta_arv: 0.55,
      tremor_arv: 0.35,
      dbs_entrainment: 0.0,
      side_effect_load: 0.0,
      gamma_arv: 0.0,
      force_preserved: 0.0,
      tracking_accuracy: 0.0,
      dbs_amplitude: 0.0,
      dbs_pulse_width: 0.06,
      dbs_frequency: 130.0,
      phase: 'ready',
    };

    // Anchor points (in pivot-local 3D space) used for screen-space labels.
    // These are set once we know where electrodes / STN are.
    this.labelAnchors = [];

    this._setupPanel();
    this._setupRenderer();
    this._setupScene();
    this._buildLights();
    this._buildEEG();
    this._buildPathwayDiagram();
    this._loadBrain();

    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  // -------------------------------------------------------------------------
  // Panel & DOM
  // -------------------------------------------------------------------------

  _setupPanel() {
    this.panel = document.createElement('div');
    Object.assign(this.panel.style, {
      position: 'fixed',
      top: '10px',
      left: '10px',
      width: `${PANEL_W}px`,
      background: 'rgba(3, 10, 22, 0.92)',
      border: '1px solid rgba(0, 210, 255, 0.30)',
      borderRadius: '14px',
      padding: '10px 10px 10px',
      zIndex: '100',
      backdropFilter: 'blur(8px)',
      boxShadow: '0 0 28px rgba(0,140,255,0.18), inset 0 0 40px rgba(0,0,0,0.4)',
      fontFamily: 'monospace',
      color: '#b8d8f0',
      userSelect: 'none',
    });
    document.body.appendChild(this.panel);

    const headerRow = document.createElement('div');
    Object.assign(headerRow.style, {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: '7px',
    });

    const title = document.createElement('div');
    Object.assign(title.style, {
      fontSize: '10px',
      letterSpacing: '2.5px',
      textTransform: 'uppercase',
      color: '#00d8ff',
      textShadow: '0 0 8px rgba(0,200,255,0.6)',
    });
    title.textContent = '⚡ Deep Brain Stimulation';

    this.statusPill = document.createElement('span');
    Object.assign(this.statusPill.style, {
      fontSize: '9px',
      padding: '2px 7px',
      borderRadius: '999px',
      letterSpacing: '0.8px',
      textTransform: 'uppercase',
      border: '1px solid rgba(255, 90, 70, 0.5)',
      color: '#ff8d6f',
      background: 'rgba(255, 90, 70, 0.08)',
    });
    this.statusPill.textContent = 'STN over-active';

    headerRow.append(title, this.statusPill);
    this.panel.appendChild(headerRow);

    // Brain canvas + label overlay live in a positioned wrapper so SVG labels
    // can sit on top of the WebGL canvas.
    const brainWrap = document.createElement('div');
    Object.assign(brainWrap.style, {
      position: 'relative',
      width: `${BRAIN_W}px`,
      height: `${BRAIN_H}px`,
    });
    this.panel.appendChild(brainWrap);

    this.canvas = document.createElement('canvas');
    this.canvas.width = BRAIN_W;
    this.canvas.height = BRAIN_H;
    Object.assign(this.canvas.style, {
      display: 'block',
      width: `${BRAIN_W}px`,
      height: `${BRAIN_H}px`,
      borderRadius: '8px',
      border: '1px solid rgba(0,160,200,0.15)',
    });
    brainWrap.appendChild(this.canvas);

    this.labelSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this.labelSvg.setAttribute('width', String(BRAIN_W));
    this.labelSvg.setAttribute('height', String(BRAIN_H));
    Object.assign(this.labelSvg.style, {
      position: 'absolute',
      top: '0',
      left: '0',
      width: `${BRAIN_W}px`,
      height: `${BRAIN_H}px`,
      pointerEvents: 'none',
    });
    brainWrap.appendChild(this.labelSvg);

    this.loadingDiv = document.createElement('div');
    Object.assign(this.loadingDiv.style, {
      position: 'absolute',
      top: '0',
      left: '0',
      width: `${BRAIN_W}px`,
      height: `${BRAIN_H}px`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#00d8ff',
      fontSize: '11px',
      letterSpacing: '1px',
      pointerEvents: 'none',
    });
    this.loadingDiv.textContent = 'Loading brain model…';
    brainWrap.appendChild(this.loadingDiv);

    const stats = document.createElement('div');
    Object.assign(stats.style, {
      display: 'flex',
      justifyContent: 'space-between',
      fontSize: '9.5px',
      marginTop: '7px',
      color: '#80b8cc',
      letterSpacing: '0.5px',
    });
    this.statFields = {
      target: makeStat('Target', 'STN bilateral'),
      freq: makeStat('Freq', '130 Hz'),
      amp: makeStat('Amp', '0.0 mA'),
      pw: makeStat('PW', '60 µs'),
    };
    stats.appendChild(this.statFields.target.span);
    stats.appendChild(this.statFields.freq.span);
    stats.appendChild(this.statFields.amp.span);
    stats.appendChild(this.statFields.pw.span);
    this.panel.appendChild(stats);

    this.pulseBar = document.createElement('div');
    Object.assign(this.pulseBar.style, {
      height: '3px',
      borderRadius: '2px',
      marginTop: '7px',
      background: 'linear-gradient(90deg, #003eff, #00ffc8)',
      transformOrigin: 'left center',
      transform: 'scaleX(0)',
      boxShadow: '0 0 6px rgba(0,255,200,0.5)',
    });
    this.panel.appendChild(this.pulseBar);

    this.phaseLine = document.createElement('div');
    Object.assign(this.phaseLine.style, {
      marginTop: '6px',
      color: '#e5f8ff',
      fontSize: '10px',
      textAlign: 'center',
      letterSpacing: '0.6px',
    });
    this.phaseLine.textContent = 'Waiting for agent';
    this.panel.appendChild(this.phaseLine);
  }

  applySignalState(snapshot) {
    const obs = snapshot?.observation || {};
    const action = snapshot?.action || {};
    const derived = snapshot?.derived_visuals || {};
    this.signalState = {
      beta_arv: clamp01(obs.beta_arv),
      tremor_arv: clamp01(obs.tremor_arv),
      dbs_entrainment: clamp01(obs.dbs_entrainment),
      side_effect_load: clamp01(obs.side_effect_load),
      gamma_arv: clamp01(obs.gamma_arv),
      force_preserved: clamp01(obs.force_preserved),
      tracking_accuracy: clamp01(obs.tracking_accuracy),
      dbs_amplitude: Number(action.dbs_amplitude ?? obs.dbs_amplitude_ma ?? 0),
      dbs_pulse_width: Number(action.dbs_pulse_width ?? obs.dbs_pulse_width_ms ?? 0.06),
      dbs_frequency: Number(action.dbs_frequency ?? 130),
      phase: derived.phase || snapshot?.type || 'running',
    };
    this.statFields.freq.val.textContent = `${Math.round(this.signalState.dbs_frequency)} Hz`;
    this.statFields.amp.val.textContent = `${this.signalState.dbs_amplitude.toFixed(2)} mA`;
    this.statFields.pw.val.textContent = `${Math.round(this.signalState.dbs_pulse_width * 1000)} µs`;
    this.phaseLine.textContent = this.signalState.phase.toUpperCase();

    const pathology = Math.max(this.signalState.beta_arv, this.signalState.tremor_arv);
    const dbsControl = this.signalState.dbs_entrainment;
    const warn = this.signalState.side_effect_load > 0.5 || this.signalState.gamma_arv > 0.5;
    if (warn) {
      this._setStatusPill('side-effect risk', '#c13cff');
    } else if (dbsControl > pathology * 0.85 && pathology < 0.45) {
      this._setStatusPill('STN suppressed', '#00e6c8');
    } else if (dbsControl > pathology * 0.6) {
      this._setStatusPill('DBS engaging', '#69a7ff');
    } else {
      this._setStatusPill('STN over-active', '#ff5a46');
    }
    this._updatePathwayDiagram();
  }

  _setStatusPill(text, color) {
    this.statusPill.textContent = text;
    this.statusPill.style.color = color;
    this.statusPill.style.borderColor = color + '99';
    this.statusPill.style.background = color + '14';
  }

  // -------------------------------------------------------------------------
  // WebGL setup
  // -------------------------------------------------------------------------

  _setupRenderer() {
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(BRAIN_W, BRAIN_H);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.renderer.shadowMap.enabled = true;
  }

  _setupScene() {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, BRAIN_W / BRAIN_H, 0.01, 60);
    this.camera.position.set(0.05, 0.18, 3.6);
    this.camera.lookAt(0, 0, 0);

    this.pivot = new THREE.Group();
    this.scene.add(this.pivot);
  }

  // -------------------------------------------------------------------------
  // Brain GLTF + fallback
  // -------------------------------------------------------------------------

  _loadBrain() {
    const loader = new GLTFLoader();
    const modelPath = './models/brain/scene.gltf';

    loader.load(
      modelPath,
      (gltf) => {
        const model = gltf.scene;

        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2.0 / maxDim;

        model.scale.setScalar(scale);
        model.position.sub(center.multiplyScalar(scale));
        model.position.y += 0.1;

        model.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            if (child.material) {
              const mats = Array.isArray(child.material) ? child.material : [child.material];
              mats.forEach((mat) => {
                if (mat.isMeshStandardMaterial || mat.isMeshPhongMaterial) {
                  mat.roughness = Math.max(mat.roughness ?? 0.8, 0.72);
                  mat.metalness = Math.min(mat.metalness ?? 0.0, 0.05);
                  mat.envMapIntensity = 0.4;
                }
              });
            }
          }
        });

        this.brainRoot = model;
        this.pivot.add(model);
        this.loadingDiv.style.display = 'none';
        this._buildElectrodes();
        this._buildPulses();
        this._buildSTN();
        this._buildAnatomicalLabels();
      },
      (xhr) => {
        const pct = Math.round((xhr.loaded / (xhr.total || 1)) * 100);
        this.loadingDiv.textContent = 'Loading… ' + pct + '%';
      },
      (err) => {
        console.warn('[BrainOverlay] GLTF load failed:', err);
        this.loadingDiv.textContent = '⚠ Place brain/scene.gltf in models/brain/';
        this._buildFallbackBrain();
        this._buildElectrodes();
        this._buildPulses();
        this._buildSTN();
        this._buildAnatomicalLabels();
      },
    );
  }

  _buildFallbackBrain() {
    const geo = new THREE.IcosahedronGeometry(1.0, 5);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      const d =
        Math.sin(x * 8.4 + z * 2.0) * Math.cos(y * 7.1) * 0.095 +
        Math.sin(x * 14.1 + y * 3.6) * Math.cos(z * 11.5) * 0.048 +
        Math.cos(x * 5.2 + y * 4.8 + z * 3.1) * 0.068 +
        Math.sin(y * 17.8 + z * 6.8) * 0.022;
      const len = Math.sqrt(x * x + y * y + z * z);
      pos.setXYZ(i, (x * (1 + d)) / len, (y * (1 + d)) / len, (z * (1 + d)) / len);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();

    const mesh = new THREE.Mesh(
      geo,
      new THREE.MeshStandardMaterial({ color: 0xbe7882, roughness: 0.84 }),
    );
    mesh.scale.set(1.08, 0.87, 0.96);
    this.pivot.add(mesh);

    const cbl = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.42, 3),
      new THREE.MeshStandardMaterial({ color: 0xbe7882, roughness: 0.84 }),
    );
    cbl.position.set(0, -0.72, -0.58);
    cbl.scale.set(1.3, 0.7, 0.9);
    this.pivot.add(cbl);

    const stem = new THREE.Mesh(
      new THREE.CylinderGeometry(0.13, 0.10, 0.58, 10),
      new THREE.MeshStandardMaterial({ color: 0xbe7882, roughness: 0.84 }),
    );
    stem.position.set(0, -1.02, -0.18);
    this.pivot.add(stem);
  }

  _buildElectrodes() {
    const shaftMat = new THREE.MeshStandardMaterial({
      color: 0xd8d8d8, metalness: 0.96, roughness: 0.12,
    });
    const contactMat = new THREE.MeshStandardMaterial({
      color: 0x999999, metalness: 1.0, roughness: 0.08,
    });

    const leads = [
      { entry: new THREE.Vector3(-0.30, 1.28, 0.10), target: new THREE.Vector3(-0.17, -0.08, 0.04) },
      { entry: new THREE.Vector3(0.30, 1.28, 0.10), target: new THREE.Vector3(0.17, -0.08, 0.04) },
    ];

    leads.forEach(({ entry, target }) => {
      const top = entry.clone().add(entry.clone().sub(target).normalize().multiplyScalar(0.28));
      const dir = target.clone().sub(top);
      const len = dir.length();
      const mid = top.clone().add(target).multiplyScalar(0.5);

      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.018, len, 8), shaftMat);
      shaft.position.copy(mid);
      shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
      this.pivot.add(shaft);

      for (let c = 0; c < 4; c++) {
        const cp = target.clone().lerp(top, (c + 0.5) / 6);
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.027, 0.007, 6, 22), contactMat);
        ring.position.copy(cp);
        ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
        this.pivot.add(ring);
      }

      const tipMat = new THREE.MeshStandardMaterial({
        color: 0x00ffee, emissive: 0x00ffee, emissiveIntensity: 3.0,
        roughness: 0.0, metalness: 0.2, transparent: true, opacity: 0.95,
      });
      const tip = new THREE.Mesh(new THREE.SphereGeometry(0.038, 16, 16), tipMat);
      tip.position.copy(target);
      this.pivot.add(tip);
      this.tipMeshes.push(tip);
      this.tipPositions.push(target.clone());
    });
  }

  _buildSTN() {
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffcc44, emissive: 0xffaa00, emissiveIntensity: 1.8, roughness: 0.2,
    });
    for (const sx of [-0.17, 0.17]) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.05, 12, 12), mat.clone());
      m.position.set(sx, -0.08, 0.04);
      this.pivot.add(m);
      this._stnMeshes.push(m);
    }
  }

  _buildPulses() {
    this.tipPositions.forEach((pos, i) => {
      const group = [];
      for (let s = 0; s < 4; s++) {
        const mat = new THREE.MeshBasicMaterial({
          color: 0x00ffdd, transparent: true, opacity: 0,
          wireframe: true, depthWrite: false,
        });
        const mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), mat);
        mesh.position.copy(pos);
        mesh.userData.phase = (s / 4) * Math.PI * 2 + i * Math.PI;
        mesh.userData.speed = 1.8 + i * 0.3;
        this.pivot.add(mesh);
        group.push(mesh);
      }
      this.pulseGroups.push(group);
    });
  }

  // Anchor anatomical labels to specific 3D points. We project each anchor
  // every frame in screen space and keep the SVG <text> positioned over it.
  _buildAnatomicalLabels() {
    this.labelAnchors = [
      { id: 'cortex', text: 'Motor Cortex', position: new THREE.Vector3(-0.55, 0.95, 0.35), color: '#9adfff', offset: { x: -10, y: -8 } },
      { id: 'thalamus', text: 'Thalamus', position: new THREE.Vector3(0.0, 0.05, 0.35), color: '#ffcf6e', offset: { x: 6, y: 4 } },
      { id: 'stnL', text: 'STN-L', position: new THREE.Vector3(-0.17, -0.08, 0.04), color: '#ff7f7f', offset: { x: -38, y: 12 } },
      { id: 'stnR', text: 'STN-R', position: new THREE.Vector3(0.17, -0.08, 0.04), color: '#ff7f7f', offset: { x: 14, y: 12 } },
      { id: 'lead', text: 'DBS Lead', position: new THREE.Vector3(-0.27, 0.85, 0.10), color: '#00ffe1', offset: { x: -68, y: 0 } },
    ];

    this.labelNodes = this.labelAnchors.map((anchor) => {
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', '0');
      text.setAttribute('y', '0');
      text.setAttribute('fill', anchor.color);
      text.setAttribute('font-family', 'monospace');
      text.setAttribute('font-size', '9');
      text.setAttribute('font-weight', '600');
      text.setAttribute('letter-spacing', '0.6');
      text.style.textShadow = '0 0 6px rgba(0,0,0,0.9)';
      text.style.paintOrder = 'stroke';
      text.style.stroke = 'rgba(0,0,0,0.7)';
      text.style.strokeWidth = '2.5px';
      text.textContent = anchor.text;
      this.labelSvg.appendChild(text);

      const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tick.setAttribute('stroke', anchor.color);
      tick.setAttribute('stroke-width', '0.9');
      tick.setAttribute('opacity', '0.6');
      this.labelSvg.appendChild(tick);

      return { anchor, text, tick };
    });
  }

  _updateAnatomicalLabels() {
    if (!this.labelNodes || !this.labelNodes.length) { return; }
    const proj = new THREE.Vector3();
    this.labelNodes.forEach(({ anchor, text, tick }) => {
      proj.copy(anchor.position).applyMatrix4(this.pivot.matrixWorld).project(this.camera);
      const sx = (proj.x * 0.5 + 0.5) * BRAIN_W;
      const sy = (1 - (proj.y * 0.5 + 0.5)) * BRAIN_H;
      const visible = proj.z < 1 && proj.z > -1;
      const tx = sx + anchor.offset.x;
      const ty = sy + anchor.offset.y;
      text.setAttribute('x', String(tx));
      text.setAttribute('y', String(ty));
      text.setAttribute('opacity', visible ? '0.95' : '0.0');
      tick.setAttribute('x1', String(sx));
      tick.setAttribute('y1', String(sy));
      tick.setAttribute('x2', String(tx + (anchor.offset.x < 0 ? 6 : -6)));
      tick.setAttribute('y2', String(ty - 3));
      tick.setAttribute('opacity', visible ? '0.55' : '0.0');
    });
  }

  // -------------------------------------------------------------------------
  // Lights
  // -------------------------------------------------------------------------

  _buildLights() {
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.45));

    const key = new THREE.DirectionalLight(0xfff3e8, 1.6);
    key.position.set(1.5, 2.5, 2.8);
    key.castShadow = true;
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0xd0e8ff, 0.6);
    fill.position.set(-2.5, 0.5, 1.0);
    this.scene.add(fill);

    const rim = new THREE.PointLight(0x0055ff, 1.0, 15);
    rim.position.set(0, -0.5, -3.5);
    this.scene.add(rim);

    const top = new THREE.DirectionalLight(0xffddaa, 0.4);
    top.position.set(0, 4, 0);
    this.scene.add(top);

    this.innerGlow = new THREE.PointLight(0x00ffcc, 0.0, 3);
    this.innerGlow.position.set(0, -0.08, 0.04);
    this.pivot.add(this.innerGlow);
  }

  // -------------------------------------------------------------------------
  // Dual-trace EEG strip
  // -------------------------------------------------------------------------

  _buildEEG() {
    const eegCanvas = document.createElement('canvas');
    eegCanvas.width = EEG_W;
    eegCanvas.height = EEG_H;
    Object.assign(eegCanvas.style, {
      display: 'block', width: `${EEG_W}px`, height: `${EEG_H}px`,
      marginTop: '8px', borderRadius: '6px',
      border: '1px solid rgba(0,160,200,0.18)',
    });
    this.panel.appendChild(eegCanvas);
    this.eegCtx = eegCanvas.getContext('2d');
  }

  _tickEEG(t) {
    const ctx = this.eegCtx;
    const beta = this.betaBuffer;
    const motor = this.motorBuffer;
    beta.copyWithin(0, 1);
    motor.copyWithin(0, 1);

    const betaArv = this.signalState.beta_arv;
    const tremor = this.signalState.tremor_arv;
    const dbs = this.signalState.dbs_entrainment;
    const sideFx = this.signalState.side_effect_load;
    const tracking = this.signalState.tracking_accuracy || 0;

    // Pathological STN beta: ~20 Hz oscillation, amplitude grows with beta_arv
    // and tremor_arv. Sits in upper half of canvas around y=EEG_H*0.30.
    const betaCenter = EEG_H * 0.30;
    const betaAmp = 4 + 18 * betaArv + 8 * tremor;
    const betaWave =
      Math.sin(t * 125) * 0.55 +
      Math.sin(t * 207) * 0.30 +
      Math.sin(t * 47) * 0.15 +
      Math.sin(t * 313) * (0.10 + tremor * 0.20);
    beta[EEG_W - 1] = betaCenter + betaWave * betaAmp;

    // Effective motor cortex output: clean smooth low-freq sinusoid that
    // grows as DBS engages and tracking improves; lower noise floor as
    // side-effect load drops.
    const motorCenter = EEG_H * 0.72;
    const motorAmp = 4 + 14 * dbs + 6 * tracking;
    const noise = (1 - dbs) * 4 + sideFx * 6;
    const motorWave =
      Math.sin(t * 6.5) * 0.7 +
      Math.sin(t * 11.2) * 0.25 +
      (Math.random() - 0.5) * noise * 0.18;
    motor[EEG_W - 1] = motorCenter + motorWave * motorAmp;

    ctx.clearRect(0, 0, EEG_W, EEG_H);
    ctx.fillStyle = 'rgba(0,0,0,0.78)';
    ctx.fillRect(0, 0, EEG_W, EEG_H);

    // Mid divider so the two traces have visual separation.
    ctx.strokeStyle = 'rgba(0,180,140,0.18)';
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    ctx.moveTo(0, EEG_H * 0.51);
    ctx.lineTo(EEG_W, EEG_H * 0.51);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(0,180,140,0.10)';
    for (let x = 0; x < EEG_W; x += 48) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, EEG_H);
      ctx.stroke();
    }

    ctx.strokeStyle = '#ff5a46';
    ctx.lineWidth = 1.2;
    ctx.shadowBlur = 4;
    ctx.shadowColor = '#ff5a46';
    ctx.beginPath();
    for (let i = 0; i < EEG_W; i++) {
      i === 0 ? ctx.moveTo(i, beta[i]) : ctx.lineTo(i, beta[i]);
    }
    ctx.stroke();

    ctx.strokeStyle = dbs > 0.4 ? '#00ffcc' : '#69a7ff';
    ctx.shadowColor = '#00ffcc';
    ctx.beginPath();
    for (let i = 0; i < EEG_W; i++) {
      i === 0 ? ctx.moveTo(i, motor[i]) : ctx.lineTo(i, motor[i]);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = 'rgba(255,90,70,0.85)';
    ctx.font = '8.5px monospace';
    ctx.fillText(`STN β  ${(betaArv * 100).toFixed(0)}%`, 6, 11);
    ctx.fillStyle = dbs > 0.4 ? 'rgba(0,255,200,0.88)' : 'rgba(105,167,255,0.85)';
    ctx.fillText(`Motor  ${(tracking * 100).toFixed(0)}% track`, 6, EEG_H - 5);
    ctx.fillStyle = 'rgba(120,200,255,0.45)';
    ctx.fillText(`${Math.round(this.signalState.dbs_frequency)} Hz`, EEG_W - 56, 11);
  }

  // -------------------------------------------------------------------------
  // Pathway diagram (bottom strip)
  // -------------------------------------------------------------------------

  _buildPathwayDiagram() {
    const wrap = document.createElement('div');
    Object.assign(wrap.style, {
      marginTop: '8px',
      padding: '6px 4px 4px',
      borderRadius: '6px',
      background: 'rgba(0,40,70,0.35)',
      border: '1px solid rgba(0,160,200,0.18)',
      display: 'flex',
      flexDirection: 'column',
      gap: '4px',
    });

    const heading = document.createElement('div');
    Object.assign(heading.style, {
      fontSize: '8.5px',
      letterSpacing: '1.6px',
      textTransform: 'uppercase',
      color: 'rgba(155, 220, 255, 0.6)',
      textAlign: 'center',
    });
    heading.textContent = 'basal ganglia – thalamo-cortical loop';
    wrap.appendChild(heading);

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', String(BRAIN_W));
    svg.setAttribute('height', String(PATHWAY_H));
    svg.setAttribute('viewBox', `0 0 ${BRAIN_W} ${PATHWAY_H}`);
    svg.style.display = 'block';
    wrap.appendChild(svg);
    this.panel.appendChild(wrap);
    this.pathwaySvg = svg;

    const SVG_NS = 'http://www.w3.org/2000/svg';
    const nodes = [
      { id: 'm1', label: 'M1', x: 36, y: 32, color: '#9adfff' },
      { id: 'th', label: 'Thal', x: 130, y: 32, color: '#ffcf6e' },
      { id: 'stn', label: 'STN', x: 224, y: 32, color: '#ff7f7f' },
    ];
    this.pathwayNodes = {};
    nodes.forEach((node) => {
      const circle = document.createElementNS(SVG_NS, 'circle');
      circle.setAttribute('cx', String(node.x));
      circle.setAttribute('cy', String(node.y));
      circle.setAttribute('r', '14');
      circle.setAttribute('fill', node.color + '33');
      circle.setAttribute('stroke', node.color);
      circle.setAttribute('stroke-width', '1.4');
      svg.appendChild(circle);

      const label = document.createElementNS(SVG_NS, 'text');
      label.setAttribute('x', String(node.x));
      label.setAttribute('y', String(node.y + 3.5));
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('font-family', 'monospace');
      label.setAttribute('font-size', '9');
      label.setAttribute('font-weight', '700');
      label.setAttribute('fill', node.color);
      label.textContent = node.label;
      svg.appendChild(label);

      this.pathwayNodes[node.id] = { circle, label, color: node.color };
    });

    // Arrows between nodes — direction: STN → Thal (inhibitory in PD), Thal → M1.
    const arrows = [
      { id: 'stn_th', from: { x: 210, y: 32 }, to: { x: 144, y: 32 } },
      { id: 'th_m1', from: { x: 116, y: 32 }, to: { x: 50, y: 32 } },
    ];
    this.pathwayArrows = {};
    arrows.forEach((arr) => {
      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', String(arr.from.x));
      line.setAttribute('y1', String(arr.from.y));
      line.setAttribute('x2', String(arr.to.x));
      line.setAttribute('y2', String(arr.to.y));
      line.setAttribute('stroke', '#ff7f7f');
      line.setAttribute('stroke-width', '1.6');
      line.setAttribute('marker-end', 'url(#arrowhead)');
      svg.appendChild(line);
      this.pathwayArrows[arr.id] = line;
    });

    const defs = document.createElementNS(SVG_NS, 'defs');
    const marker = document.createElementNS(SVG_NS, 'marker');
    marker.setAttribute('id', 'arrowhead');
    marker.setAttribute('viewBox', '0 0 10 10');
    marker.setAttribute('refX', '8');
    marker.setAttribute('refY', '5');
    marker.setAttribute('markerWidth', '6');
    marker.setAttribute('markerHeight', '6');
    marker.setAttribute('orient', 'auto-start-reverse');
    const arrowPath = document.createElementNS(SVG_NS, 'path');
    arrowPath.setAttribute('d', 'M0,0 L10,5 L0,10 Z');
    arrowPath.setAttribute('fill', 'currentColor');
    marker.appendChild(arrowPath);
    defs.appendChild(marker);
    svg.appendChild(defs);

    // DBS lead arrow pointing into STN from above.
    const dbsLine = document.createElementNS(SVG_NS, 'line');
    dbsLine.setAttribute('x1', String(224));
    dbsLine.setAttribute('y1', '4');
    dbsLine.setAttribute('x2', String(224));
    dbsLine.setAttribute('y2', String(20));
    dbsLine.setAttribute('stroke', '#00ffcc');
    dbsLine.setAttribute('stroke-width', '2');
    dbsLine.setAttribute('marker-end', 'url(#arrowhead)');
    svg.appendChild(dbsLine);
    this.dbsArrow = dbsLine;

    const dbsLabel = document.createElementNS(SVG_NS, 'text');
    dbsLabel.setAttribute('x', String(244));
    dbsLabel.setAttribute('y', '12');
    dbsLabel.setAttribute('font-family', 'monospace');
    dbsLabel.setAttribute('font-size', '8');
    dbsLabel.setAttribute('fill', '#00ffcc');
    dbsLabel.textContent = 'DBS';
    svg.appendChild(dbsLabel);
    this.dbsArrowLabel = dbsLabel;

    this._updatePathwayDiagram();
  }

  _updatePathwayDiagram() {
    if (!this.pathwayNodes) { return; }
    const beta = this.signalState.beta_arv;
    const dbs = this.signalState.dbs_entrainment;
    const sideFx = this.signalState.side_effect_load;
    const gamma = this.signalState.gamma_arv;
    const warning = Math.max(sideFx, gamma);

    const stnHot = warning > 0.55 ? '#c13cff' : (dbs > beta * 0.85 ? '#00ffc8' : '#ff7f7f');
    const stn = this.pathwayNodes.stn;
    stn.circle.setAttribute('stroke', stnHot);
    stn.circle.setAttribute('fill', stnHot + '33');
    stn.label.setAttribute('fill', stnHot);

    const arrowColor = dbs > beta * 0.85 ? '#69a7ff' : '#ff7f7f';
    Object.values(this.pathwayArrows).forEach((line) => {
      line.setAttribute('stroke', arrowColor);
      line.style.color = arrowColor;
    });

    const dbsActive = this.signalState.dbs_amplitude > 0.05;
    const dbsColor = warning > 0.55 ? '#c13cff' : '#00ffcc';
    this.dbsArrow.setAttribute('stroke', dbsColor);
    this.dbsArrow.setAttribute('opacity', dbsActive ? '1' : '0.25');
    this.dbsArrow.style.color = dbsColor;
    this.dbsArrowLabel.setAttribute('fill', dbsColor);
    this.dbsArrowLabel.setAttribute('opacity', dbsActive ? '1' : '0.4');
  }

  // -------------------------------------------------------------------------
  // Animation loop
  // -------------------------------------------------------------------------

  _loop(timeMS) {
    const t = timeMS * 0.001;
    this.time = t;

    this.pivot.rotation.y = t * 0.22;
    this.pivot.rotation.x = Math.sin(t * 0.08) * 0.06;

    const beta = this.signalState.beta_arv;
    const tremor = this.signalState.tremor_arv;
    const entrainment = this.signalState.dbs_entrainment;
    const side = this.signalState.side_effect_load;
    const warning = Math.max(side, this.signalState.gamma_arv);
    const pathology = Math.max(beta, tremor);
    const pulsePower = clamp01(0.2 + entrainment * 0.9 + this.signalState.dbs_amplitude / 2.4);

    this.tipMeshes.forEach((tip, i) => {
      tip.material.emissiveIntensity = 1.1 + pulsePower * 4.0 + Math.sin(t * 7.0 + i * Math.PI) * 1.2;
      tip.material.color.set(entrainment > pathology ? 0x00ffee : 0xffd447);
      tip.material.emissive.set(entrainment > pathology ? 0x00ffee : 0xff8a00);
    });
    this._stnMeshes.forEach((m, i) => {
      const controlled = entrainment > pathology * 0.75;
      const color = warning > 0.55 ? 0xc13cff : (controlled ? 0x00ffc8 : 0xff3d2e);
      m.material.color.set(color);
      m.material.emissive.set(color);
      m.material.emissiveIntensity = 0.9 + pathology * 3.4 + Math.sin(t * 6.0 + i * Math.PI) * 0.7;
    });
    this.innerGlow.color.set(warning > 0.55 ? 0xc13cff : 0x00ffcc);
    this.innerGlow.intensity = 0.2 + entrainment * 1.8 + pathology * 0.7 + warning * 0.8;

    this.pulseGroups.forEach((group) => {
      group.forEach((shell) => {
        const ph = ((t * shell.userData.speed + shell.userData.phase) % (Math.PI * 2)) / (Math.PI * 2);
        shell.scale.setScalar(0.04 + ph * (0.22 + pulsePower * 0.55));
        shell.material.color.set(warning > 0.55 ? 0xc13cff : 0x00ffdd);
        shell.material.opacity = Math.max(0, (0.25 + pulsePower * 0.48) * (1.0 - ph * 1.4));
      });
    });

    this.pulseBar.style.transform = `scaleX(${Math.max(0.04, entrainment).toFixed(3)})`;
    this.pulseBar.style.background = warning > 0.55
      ? 'linear-gradient(90deg, #ff3ac8, #8b5cff)'
      : 'linear-gradient(90deg, #003eff, #00ffc8)';
    this._tickEEG(t);
    this.renderer.render(this.scene, this.camera);
    this._updateAnatomicalLabels();
    requestAnimationFrame(this._loop);
  }
}

window.motorAssistBrain = new BrainOverlay();
