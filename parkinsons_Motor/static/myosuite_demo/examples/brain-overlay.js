import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

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
    this.time         = 0;
    this.tipMeshes    = [];
    this.tipPositions = [];
    this.pulseGroups  = [];
    this._stnMeshes   = [];
    this.eegBuffer    = new Float32Array(240).fill(19);
    this.brainRoot    = null;
    this.signalState   = {
      beta_arv: 0.55,
      tremor_arv: 0.35,
      dbs_entrainment: 0.0,
      side_effect_load: 0.0,
      gamma_arv: 0.0,
      dbs_amplitude: 0.0,
      dbs_pulse_width: 0.06,
      dbs_frequency: 130.0,
      phase: 'ready',
    };

    this._setupPanel();
    this._setupRenderer();
    this._setupScene();
    this._buildLights();
    this._buildEEG();
    this._loadBrain();

    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  // -------------------------------------------------------------------------
  _setupPanel() {
    this.panel = document.createElement('div');
    Object.assign(this.panel.style, {
      position:       'fixed',
      top:            '10px',
      left:           '10px',
      width:          '260px',
      background:     'rgba(3, 10, 22, 0.90)',
      border:         '1px solid rgba(0, 210, 255, 0.30)',
      borderRadius:   '14px',
      padding:        '10px 10px 8px',
      zIndex:         '100',
      backdropFilter: 'blur(8px)',
      boxShadow:      '0 0 28px rgba(0,140,255,0.18), inset 0 0 40px rgba(0,0,0,0.4)',
      fontFamily:     'monospace',
      color:          '#b8d8f0',
      userSelect:     'none',
    });
    document.body.appendChild(this.panel);

    const title = document.createElement('div');
    Object.assign(title.style, {
      fontSize:      '10px',
      letterSpacing: '2.5px',
      textTransform: 'uppercase',
      color:         '#00d8ff',
      marginBottom:  '7px',
      textAlign:     'center',
      textShadow:    '0 0 8px rgba(0,200,255,0.6)',
    });
    title.textContent = '⚡  Deep Brain Stimulation';
    this.panel.appendChild(title);

    this.canvas = document.createElement('canvas');
    this.canvas.width  = 240;
    this.canvas.height = 230;
    Object.assign(this.canvas.style, {
      display:      'block',
      width:        '240px',
      height:       '230px',
      borderRadius: '8px',
      border:       '1px solid rgba(0,160,200,0.15)',
    });
    this.panel.appendChild(this.canvas);

    // Loading overlay (shown while GLTF loads)
    this.loadingDiv = document.createElement('div');
    Object.assign(this.loadingDiv.style, {
      position:   'absolute',
      top:        '40px',
      left:       '10px',
      width:      '240px',
      height:     '230px',
      display:    'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color:      '#00d8ff',
      fontSize:   '11px',
      letterSpacing: '1px',
      pointerEvents: 'none',
    });
    this.loadingDiv.textContent = 'Loading brain model…';
    this.panel.appendChild(this.loadingDiv);

    const stats = document.createElement('div');
    Object.assign(stats.style, {
      display:        'flex',
      justifyContent: 'space-between',
      fontSize:       '9.5px',
      marginTop:      '7px',
      color:          '#80b8cc',
      letterSpacing:  '0.5px',
    });
    this.statFields = {
      target: makeStat('Target', 'STN'),
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
      height:          '3px',
      borderRadius:    '2px',
      marginTop:       '7px',
      background:      'linear-gradient(90deg, #003eff, #00ffc8)',
      transformOrigin: 'left center',
      transform:       'scaleX(0)',
      boxShadow:       '0 0 6px rgba(0,255,200,0.5)',
    });
    this.panel.appendChild(this.pulseBar);

    this.phaseLine = document.createElement('div');
    Object.assign(this.phaseLine.style, {
      marginTop: '7px',
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
  }

  // -------------------------------------------------------------------------
  _setupRenderer() {
    this.renderer = new THREE.WebGLRenderer({
      canvas:    this.canvas,
      antialias: true,
      alpha:     true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(240, 230);
    this.renderer.toneMapping         = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.renderer.shadowMap.enabled   = true;
  }

  // -------------------------------------------------------------------------
  _setupScene() {
    this.scene  = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 240 / 230, 0.01, 60);
    this.camera.position.set(0.05, 0.15, 3.8);
    this.camera.lookAt(0, 0, 0);

    this.pivot = new THREE.Group();
    this.scene.add(this.pivot);
  }

  // -------------------------------------------------------------------------
  _loadBrain() {
    const loader = new GLTFLoader();

    // Paths to try — user may place file as scene.gltf or scene.glb
    const modelPath = './models/brain/scene.gltf';

    loader.load(
      modelPath,
      (gltf) => {
        const model = gltf.scene;

        // Auto-fit the loaded model to a ~2-unit bounding box
        const box    = new THREE.Box3().setFromObject(model);
        const size   = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale  = 2.0 / maxDim;

        model.scale.setScalar(scale);
        model.position.sub(center.multiplyScalar(scale));
        // Shift slightly upward so brain stem is below center
        model.position.y += 0.1;

        // Enhance materials: boost roughness, keep original colors
        model.traverse((child) => {
          if (child.isMesh) {
            child.castShadow    = true;
            child.receiveShadow = true;
            if (child.material) {
              const mats = Array.isArray(child.material) ? child.material : [child.material];
              mats.forEach((mat) => {
                if (mat.isMeshStandardMaterial || mat.isMeshPhongMaterial) {
                  mat.roughness  = Math.max(mat.roughness  ?? 0.8, 0.72);
                  mat.metalness  = Math.min(mat.metalness  ?? 0.0, 0.05);
                  mat.envMapIntensity = 0.4;
                }
              });
            }
          }
        });

        this.brainRoot = model;
        this.pivot.add(model);

        // Hide loading indicator
        this.loadingDiv.style.display = 'none';

        // Now that brain is loaded, add electrodes targeting its center
        this._buildElectrodes();
        this._buildPulses();
        this._buildSTN();
      },
      (xhr) => {
        const pct = Math.round((xhr.loaded / (xhr.total || 1)) * 100);
        this.loadingDiv.textContent = 'Loading… ' + pct + '%';
      },
      (err) => {
        console.warn('[BrainOverlay] GLTF load failed:', err);
        this.loadingDiv.textContent = '⚠ Place brain/scene.gltf in models/brain/';
        // Fallback to procedural brain
        this._buildFallbackBrain();
        this._buildElectrodes();
        this._buildPulses();
        this._buildSTN();
      }
    );
  }

  // -------------------------------------------------------------------------
  _buildFallbackBrain() {
    // Procedural fallback if model file not found
    const geo = new THREE.IcosahedronGeometry(1.0, 5);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      const d =
        Math.sin(x * 8.4 + z * 2.0)  * Math.cos(y * 7.1)          * 0.095 +
        Math.sin(x * 14.1 + y * 3.6) * Math.cos(z * 11.5)          * 0.048 +
        Math.cos(x * 5.2 + y * 4.8 + z * 3.1)                      * 0.068 +
        Math.sin(y * 17.8 + z * 6.8)                                * 0.022;
      const len = Math.sqrt(x*x + y*y + z*z);
      pos.setXYZ(i, x*(1+d)/len, y*(1+d)/len, z*(1+d)/len);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();

    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: 0xBE7882, roughness: 0.84,
    }));
    mesh.scale.set(1.08, 0.87, 0.96);
    this.pivot.add(mesh);

    // Cerebellum
    const cbl = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.42, 3),
      new THREE.MeshStandardMaterial({ color: 0xBE7882, roughness: 0.84 })
    );
    cbl.position.set(0, -0.72, -0.58);
    cbl.scale.set(1.3, 0.7, 0.9);
    this.pivot.add(cbl);

    this.pivot.add(Object.assign(new THREE.Mesh(
      new THREE.CylinderGeometry(0.13, 0.10, 0.58, 10),
      new THREE.MeshStandardMaterial({ color: 0xBE7882, roughness: 0.84 })
    ), { position: new THREE.Vector3(0, -1.02, -0.18) }));
  }

  // -------------------------------------------------------------------------
  _buildElectrodes() {
    const shaftMat = new THREE.MeshStandardMaterial({
      color: 0xd8d8d8, metalness: 0.96, roughness: 0.12,
    });
    const contactMat = new THREE.MeshStandardMaterial({
      color: 0x999999, metalness: 1.0, roughness: 0.08,
    });

    // Bilateral STN leads — entry at skull top, target at STN depth
    const leads = [
      { entry: new THREE.Vector3(-0.30, 1.28, 0.10), target: new THREE.Vector3(-0.17, -0.08, 0.04) },
      { entry: new THREE.Vector3( 0.30, 1.28, 0.10), target: new THREE.Vector3( 0.17, -0.08, 0.04) },
    ];

    leads.forEach(({ entry, target }) => {
      const top  = entry.clone().add(entry.clone().sub(target).normalize().multiplyScalar(0.28));
      const dir  = target.clone().sub(top);
      const len  = dir.length();
      const mid  = top.clone().add(target).multiplyScalar(0.5);

      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.018, len, 8), shaftMat);
      shaft.position.copy(mid);
      shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
      this.pivot.add(shaft);

      // 4 platinum contact rings — actual DBS lead anatomy
      for (let c = 0; c < 4; c++) {
        const cp   = target.clone().lerp(top, (c + 0.5) / 6);
        const ring = new THREE.Mesh(new THREE.TorusGeometry(0.027, 0.007, 6, 22), contactMat);
        ring.position.copy(cp);
        ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
        this.pivot.add(ring);
      }

      // Glowing active tip
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

  // -------------------------------------------------------------------------
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

  // -------------------------------------------------------------------------
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
  _buildEEG() {
    const eegCanvas = document.createElement('canvas');
    eegCanvas.width  = 240;
    eegCanvas.height = 38;
    Object.assign(eegCanvas.style, {
      display: 'block', width: '240px', height: '38px',
      marginTop: '6px', borderRadius: '4px',
      border: '1px solid rgba(0,160,200,0.12)',
    });
    this.panel.appendChild(eegCanvas);
    this.eegCtx = eegCanvas.getContext('2d');
  }

  _tickEEG(t) {
    const ctx = this.eegCtx;
    const buf = this.eegBuffer;
    buf.copyWithin(0, 1);

    const freq = Math.max(60, Math.min(185, this.signalState.dbs_frequency || 130));
    const dbs = clamp01(this.signalState.dbs_entrainment);
    const pathology = Math.max(this.signalState.beta_arv, this.signalState.tremor_arv);
    const phase = (t % (1 / freq)) / (1 / freq);
    let s = 19;
    const amp = 5 + 11 * dbs + 8 * pathology;
    if      (phase < 0.06) { s = 19 - Math.sin((phase / 0.06) * Math.PI) * amp; }
    else if (phase < 0.10) { s = 19 + Math.sin(((phase - 0.06) / 0.04) * Math.PI) * amp * 0.38; }
    else { s = 19 + Math.sin(t * 45) * pathology * 4; }
    buf[239] = s;

    const w = 240, h = 38;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(0,180,140,0.12)';
    ctx.lineWidth   = 0.5;
    for (let x = 0; x < w; x += 48) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    ctx.beginPath(); ctx.moveTo(0, 19); ctx.lineTo(w, 19); ctx.stroke();

    ctx.strokeStyle = dbs > pathology ? '#00ffcc' : '#ff4d3d';
    ctx.lineWidth   = 1.5;
    ctx.shadowBlur  = 4;
    ctx.shadowColor = '#00ffcc';
    ctx.beginPath();
    for (let i = 0; i < 240; i++) {
      i === 0 ? ctx.moveTo(i, buf[i]) : ctx.lineTo(i, buf[i]);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = 'rgba(0,200,160,0.45)';
    ctx.font      = '8px monospace';
    ctx.fillText(`STN LFP  ${Math.round(freq)} Hz`, 4, 10);
  }

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
    requestAnimationFrame(this._loop);
  }
}

window.motorAssistBrain = new BrainOverlay();
