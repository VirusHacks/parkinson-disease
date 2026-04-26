
import * as THREE from 'three';
import { GUI } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import load_mujoco from '../dist/mujoco_wasm.js';

function createBootLoader() {
  const root = document.getElementById('viewer-loader');
  if (!root) {
    return {
      setPhase() { },
      fail(error) { throw error; },
      complete() { },
    };
  }

  const messageNode = document.getElementById('viewer-loader-message');
  const phaseNode = document.getElementById('viewer-loader-phase');
  const elapsedNode = document.getElementById('viewer-loader-elapsed');
  const barNode = document.getElementById('viewer-loader-bar');
  const start = Date.now();

  const renderElapsed = () => {
    const seconds = Math.floor((Date.now() - start) / 1000);
    elapsedNode.textContent = `${seconds}s elapsed`;
  };

  renderElapsed();
  const timer = window.setInterval(renderElapsed, 1000);

  return {
    setPhase(phase, message, progress) {
      phaseNode.textContent = phase;
      messageNode.textContent = message;
      if (typeof progress === 'number') {
        barNode.style.width = `${Math.max(8, Math.min(100, progress * 100))}%`;
      }
    },
    fail(error) {
      window.clearInterval(timer);
      root.classList.remove('is-hidden');
      root.classList.add('is-error');
      phaseNode.textContent = 'Load failed';
      messageNode.textContent = error?.message || 'The 3D preview failed to initialize.';
      barNode.style.width = '100%';
      elapsedNode.textContent = 'Please refresh and try again.';
      throw error;
    },
    complete() {
      window.clearInterval(timer);
      barNode.style.width = '100%';
      phaseNode.textContent = 'Ready';
      messageNode.textContent = 'Preview ready. Launching the live 3D scene...';
      window.setTimeout(() => {
        root.classList.add('is-hidden');
      }, 250);
    },
  };
}

const bootLoader = createBootLoader();
bootLoader.setPhase('Booting', 'Starting MuJoCo WebAssembly runtime...', 0.08);

// Load the MuJoCo Module
const mujoco = await load_mujoco();
bootLoader.setPhase('Downloading assets', 'Fetching scene files, meshes, and textures for the first preview...', 0.32);

// Set up Emscripten's Virtual File System
var initialScene = "myo_sim/elbow/myo_elbow_combined.xml";
mujoco.FS.mkdir('/working');
mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');
// Download the the examples to MuJoCo's virtual file system
await downloadExampleScenesFolder(mujoco);
bootLoader.setPhase('Building scene', 'Loading the musculoskeletal model into Three.js. This is the slowest step on first load...', 0.68);

export class MuJoCoDemo {
  constructor() {
    this.mujoco = mujoco;

    // Load in the state from XML
    this.model = new mujoco.Model("/working/" + initialScene);
    this.state = new mujoco.State(this.model);
    this.simulation = new mujoco.Simulation(this.model, this.state);

    // Define Random State Variables
    this.params = { scene: initialScene, paused: false, help: false, ctrlnoiserate: 0.0, ctrlnoisestd: 0.0, keyframeNumber: 0 };
    this.mujoco_time = 0.0;
    this.agentMotorState = null;
    // Motion mode is set by viewer-controller when a task is selected:
    //   tap | pinch | tremor | flex | gait | cycle. Drives the per-actuator
    //   waveform shape so each body part performs its UPDRS-style motion.
    this.bodyMotionMode = 'tremor';
    // Gate flag - body holds its rest pose (zero ctrl) until the viewer
    // controller flips this on. Set true on Connect (WS) or Start (SSE),
    // back to false on Disconnect / Stop. Prevents the limb from twitching
    // in the background before the user has actually launched the agent.
    this.bodyActive = false;
    this._lastAgentStateMs = 0;
    this.bodies = {}, this.lights = {};
    this.tmpVec = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.updateGUICallbacks = [];

    this.container = document.createElement('div');
    document.body.appendChild(this.container);

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.001, 100);
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(0.3, 1.5, 1.2);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5);

    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
    this.ambientLight.name = 'AmbientLight';
    this.scene.add(this.ambientLight);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
    this.renderer.setAnimationLoop(this.render.bind(this));

    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(-0.2, 1.4, 0.4);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));

    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);
  }

  async init() {
    // Initialize the three.js Scene using the .xml Model in initialScene
    [this.model, this.state, this.simulation, this.bodies, this.lights] =
      await loadSceneFromURL(mujoco, initialScene, this);

    this.gui = new GUI();
    setupGUI(this);
    window.myoDebugGUI = this.gui;
    if (new URLSearchParams(window.location.search).get('debug') !== '1') {
      this.gui.domElement.style.display = 'none';
    }
  }

  async switchScene(scenePath) {
    this.params.scene = scenePath;
    // Strip the previous body root before loading the new one. Without this
    // every swap stacks another skeleton on top of the previous one (the
    // old bodies/lights/instanced tendon meshes never get GC'd because the
    // group is still parented to the THREE scene). Match what reloadFunc()
    // in mujocoUtils does on the GUI dropdown path.
    const prev = this.scene.getObjectByName('MuJoCo Root');
    if (prev) {
      this.scene.remove(prev);
      prev.traverse((node) => {
        if (node.geometry) { node.geometry.dispose(); }
        if (node.material) {
          const mats = Array.isArray(node.material) ? node.material : [node.material];
          mats.forEach((m) => m && m.dispose && m.dispose());
        }
      });
    }
    this.bodies = {};
    this.lights = {};
    this.mujocoRoot = null;
    [this.model, this.state, this.simulation, this.bodies, this.lights] =
      await loadSceneFromURL(this.mujoco, this.params.scene, this);
    this.simulation.forward();
    for (let i = 0; i < this.updateGUICallbacks.length; i++) {
      this.updateGUICallbacks[i](this.model, this.simulation, this.params);
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  applyMotorState(snapshot) {
    const obs = snapshot?.observation || {};
    const action = snapshot?.action || {};
    this.agentMotorState = {
      target: Number(obs.target_output ?? 0),
      effective: Number(obs.effective_motor_output ?? 0),
      tremor: Math.max(0, Math.min(1, Number(obs.tremor_arv ?? 0))),
      beta: Math.max(0, Math.min(1, Number(obs.beta_arv ?? 0))),
      force: Math.max(0, Math.min(1, Number(obs.force_preserved ?? 0))),
      tracking: Math.max(0, Math.min(1, Number(obs.tracking_accuracy ?? 0))),
      entrainment: Math.max(0, Math.min(1, Number(obs.dbs_entrainment ?? 0))),
      sideEffect: Math.max(0, Math.min(1, Number(obs.side_effect_load ?? 0))),
      dbsAmp: Math.max(0, Number(action.dbs_amplitude ?? obs.dbs_amplitude_ma ?? 0)),
      step: Number(snapshot?.step ?? 0),
    };
    this._lastAgentStateMs = performance.now();
  }

  // Per-mode waveform shape applied to each actuator. Returns a value in
  // roughly [-1, 1] before clamp; the caller scales to ctrl range.
  // Real PD signatures:
  //   tap     - UPDRS finger taps: large rhythmic open/close at ~3 Hz with
  //             decrement (amplitude shrinks with bradykinesia).
  //   pinch   - sustained low-frequency posture with 5 Hz tremor overlay.
  //   tremor  - pure 4–6 Hz rest tremor envelope.
  //   flex    - slow elbow flex/extend at ~0.5 Hz with cogwheel bursts.
  //   gait    - alternating leg cycle ~1 Hz with foot push-off pulse.
  //   cycle   - exercise burst, faster 1.6 Hz cycling on legs.
  _motorWaveform(mode, timeMS, actuatorIdx, s) {
    const t = timeMS * 0.001;
    const freq = (hz) => Math.sin(2 * Math.PI * hz * t + actuatorIdx * 0.7);
    switch (mode) {
      case 'tap': {
        // Bradykinesia: amplitude decrement when beta is high.
        const decrement = 1 - 0.45 * s.beta * (0.5 + 0.5 * Math.sin(t * 0.3));
        return Math.sign(freq(3.0)) * 0.8 * decrement * (actuatorIdx % 3 === 0 ? 1.0 : 0.6);
      }
      case 'pinch': {
        const posture = Math.sin(t * 0.6 + actuatorIdx * 0.4) * 0.5 + 0.3;
        const tremor = Math.sin(2 * Math.PI * 5.2 * t + actuatorIdx) * 0.35 * s.tremor;
        return posture + tremor;
      }
      case 'tremor': {
        // Mostly rest - slight tonic, big 5 Hz oscillation.
        return Math.sin(2 * Math.PI * 5 * t + actuatorIdx * 1.1) * (0.35 + 0.65 * s.tremor) * 0.7;
      }
      case 'flex': {
        const slow = Math.sin(2 * Math.PI * 0.5 * t) * 0.85;
        const cogwheel = Math.sin(2 * Math.PI * 6.5 * t + actuatorIdx) * 0.18 * s.beta;
        return slow + cogwheel;
      }
      case 'gait': {
        // Alternate left/right by actuator parity. Push-off pulse at ~1 Hz.
        const cycle = Math.sin(2 * Math.PI * 1.0 * t + (actuatorIdx % 2 === 0 ? 0 : Math.PI));
        const pushoff = Math.max(0, Math.sin(2 * Math.PI * 1.0 * t)) ** 4 * 0.4;
        return cycle * 0.8 + pushoff * (actuatorIdx % 2 === 0 ? 1 : -1);
      }
      case 'cycle': {
        return Math.sin(2 * Math.PI * 1.6 * t + (actuatorIdx % 2 === 0 ? 0 : Math.PI)) * 0.85;
      }
      default:
        return Math.sin(t * 0.004 * 1000 + actuatorIdx * 0.8) * 0.3;
    }
  }

  _applyAgentMotorControls(timeMS) {
    if (!this.simulation || !this.model || !this.simulation.ctrl) { return; }

    // Body is gated - sit at zero ctrl until the controller (Connect or
    // Start) flips bodyActive on. Avoids "pre-show" twitching.
    if (!this.bodyActive) {
      for (let i = 0; i < this.model.nu; i++) {
        this.simulation.ctrl[i] = 0;
      }
      return;
    }
    if (!this.agentMotorState) {
      // Connected but no state yet - hold rest pose.
      for (let i = 0; i < this.model.nu; i++) {
        this.simulation.ctrl[i] = 0;
      }
      return;
    }
    const s = this.agentMotorState;

    const mode = this.bodyMotionMode || 'tremor';
    const dbsSuppression = 0.20 + 0.80 * s.entrainment;
    // DBS reins in the pathological component but doesn't kill voluntary
    // movement - that's the whole point of well-tuned stimulation.
    const visibleTremor = s.tremor * (1.0 - 0.85 * s.entrainment);
    const dyskinesia = s.sideEffect > 0.35
      ? Math.sin(timeMS * 0.018) * (s.sideEffect - 0.25) * 0.6
      : 0;

    for (let i = 0; i < this.model.nu; i++) {
      const lo = this.model.actuator_ctrlrange ? this.model.actuator_ctrlrange[2 * i] : -1;
      const hi = this.model.actuator_ctrlrange ? this.model.actuator_ctrlrange[2 * i + 1] : 1;
      // Sign alternation gives the limb agonist/antagonist contrast so it
      // doesn't co-contract into stiffness.
      const polarity = i % 2 === 0 ? 1 : -0.65;

      // Voluntary motor command shaped by task mode.
      const taskWave = this._motorWaveform(mode, timeMS, i, s);
      // Tremor overlay sits on top of every actuator with phase scatter.
      const tremorPhase = Math.sin((2 * Math.PI * 5.0 * timeMS) / 1000 + i * 1.7) * visibleTremor * 0.45;
      // Beta-band rigidity drag scales the voluntary command down.
      const stiffness = 1.0 - s.beta * (0.45 - 0.25 * s.entrainment);
      // Effective motor command from agent - fall back to taskWave when no
      // explicit target is being streamed.
      const directDrive = (s.effective || s.target || 0) * (0.35 + s.force * 0.75);
      const drive = directDrive * stiffness + taskWave * (0.6 + 0.4 * s.force);

      const value = Math.max(lo, Math.min(hi, (drive + tremorPhase + dyskinesia) * polarity));
      this.simulation.ctrl[i] = value;
    }
    void dbsSuppression;
  }

  render(timeMS) {
    this.controls.update();

    if (!this.params["paused"]) {
      let timestep = this.model.getOptions().timestep;
      if (timeMS - this.mujoco_time > 35.0) { this.mujoco_time = timeMS; }
      while (this.mujoco_time < timeMS) {
        this._applyAgentMotorControls(timeMS);

        // Jitter the control state with gaussian random noise
        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.simulation.ctrl;
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.simulation.qfrc_applied.length; i++) { this.simulation.qfrc_applied[i] = 0.0; }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
              getPosition(this.simulation.xpos, b, this.bodies[b].position);
              getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
              this.bodies[b].updateWorldMatrix();
            }
          }
          let bodyID = dragged.bodyID;
          this.dragStateManager.update(); // Update the world-space force origin
          let force = toMujocoPos(this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit).multiplyScalar(250));
          let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, bodyID);

          // TODO: Apply pose perturbations (mocap bodies only).
        }

        this.simulation.step();

        this.mujoco_time += timestep * 1000.0;
      }

    } else if (this.params["paused"]) {
      this.dragStateManager.update(); // Update the world-space force origin
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition(this.simulation.xpos, b, this.tmpVec, false); // Get raw coordinate from MuJoCo
        getQuaternion(this.simulation.xquat, b, this.tmpQuat, false); // Get raw coordinate from MuJoCo

        let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
          .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
        if (this.model.body_mocapid[b] >= 0) {
          // Set the root body's mocap position...
          console.log("Trying to move mocap body", b);
          let addr = this.model.body_mocapid[b] * 3;
          let pos = this.simulation.mocap_pos;
          pos[addr + 0] += offset.x;
          pos[addr + 1] += offset.y;
          pos[addr + 2] += offset.z;
        } else {
          // Set the root body's position directly...
          let root = this.model.body_rootid[b];
          let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
          let pos = this.simulation.qpos;
          pos[addr + 0] += offset.x;
          pos[addr + 1] += offset.y;
          pos[addr + 2] += offset.z;

          //// Save the original root body position
          //let x  = pos[addr + 0], y  = pos[addr + 1], z  = pos[addr + 2];
          //let xq = pos[addr + 3], yq = pos[addr + 4], zq = pos[addr + 5], wq = pos[addr + 6];

          //// Clear old perturbations, apply new ones.
          //for (let i = 0; i < this.simulation.qfrc_applied().length; i++) { this.simulation.qfrc_applied()[i] = 0.0; }
          //for (let bi = 0; bi < this.model.nbody(); bi++) {
          //  if (this.bodies[b]) {
          //    getPosition  (this.simulation.xpos (), bi, this.bodies[bi].position);
          //    getQuaternion(this.simulation.xquat(), bi, this.bodies[bi].quaternion);
          //    this.bodies[bi].updateWorldMatrix();
          //  }
          //}
          ////dragStateManager.update(); // Update the world-space force origin
          //let force = toMujocoPos(this.dragStateManager.currentWorld.clone()
          //  .sub(this.dragStateManager.worldHit).multiplyScalar(this.model.body_mass()[b] * 0.01));
          //let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          //// This force is dumped into xrfc_applied
          //this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, b);
          //this.simulation.integratePos(this.simulation.qpos(), this.simulation.qfrc_applied(), 1);

          //// Add extra drag to the root body
          //pos[addr + 0] = x  + (pos[addr + 0] - x ) * 0.1;
          //pos[addr + 1] = y  + (pos[addr + 1] - y ) * 0.1;
          //pos[addr + 2] = z  + (pos[addr + 2] - z ) * 0.1;
          //pos[addr + 3] = xq + (pos[addr + 3] - xq) * 0.1;
          //pos[addr + 4] = yq + (pos[addr + 4] - yq) * 0.1;
          //pos[addr + 5] = zq + (pos[addr + 5] - zq) * 0.1;
          //pos[addr + 6] = wq + (pos[addr + 6] - wq) * 0.1;


        }
      }

      this.simulation.forward();
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition(this.simulation.xpos, b, this.bodies[b].position);
        getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    // Update light transforms.
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.simulation.light_xpos, l, this.lights[l].position);
        getPosition(this.simulation.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Update tendon transforms.
    let numWraps = 0;
    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      let mat = new THREE.Matrix4();
      for (let t = 0; t < this.model.ntendon; t++) {
        let startW = this.simulation.ten_wrapadr[t];
        let r = this.model.tendon_width[t];
        for (let w = startW; w < startW + this.simulation.ten_wrapnum[t] - 1; w++) {
          let tendonStart = getPosition(this.simulation.wrap_xpos, w, new THREE.Vector3());
          let tendonEnd = getPosition(this.simulation.wrap_xpos, w + 1, new THREE.Vector3());
          let tendonAvg = new THREE.Vector3().addVectors(tendonStart, tendonEnd).multiplyScalar(0.5);

          let validStart = tendonStart.length() > 0.01;
          let validEnd = tendonEnd.length() > 0.01;

          if (validStart) { this.mujocoRoot.spheres.setMatrixAt(numWraps, mat.compose(tendonStart, new THREE.Quaternion(), new THREE.Vector3(r, r, r))); }
          if (validEnd) { this.mujocoRoot.spheres.setMatrixAt(numWraps + 1, mat.compose(tendonEnd, new THREE.Quaternion(), new THREE.Vector3(r, r, r))); }
          if (validStart && validEnd) {
            mat.compose(tendonAvg, new THREE.Quaternion().setFromUnitVectors(
              new THREE.Vector3(0, 1, 0), tendonEnd.clone().sub(tendonStart).normalize()),
              new THREE.Vector3(r, tendonStart.distanceTo(tendonEnd), r));
            this.mujocoRoot.cylinders.setMatrixAt(numWraps, mat);
            numWraps++;
          }
        }
      }
      this.mujocoRoot.cylinders.count = numWraps;
      this.mujocoRoot.spheres.count = numWraps > 0 ? numWraps + 1 : 0;
      this.mujocoRoot.cylinders.instanceMatrix.needsUpdate = true;
      this.mujocoRoot.spheres.instanceMatrix.needsUpdate = true;
    }

    // Render!
    this.renderer.render(this.scene, this.camera);
  }
}

try {
  const demo = new MuJoCoDemo();
  bootLoader.setPhase('Initializing viewer', 'Creating the renderer, camera, and simulation state...', 0.84);
  await demo.init();
  bootLoader.setPhase('Finalizing', 'Attaching the interactive preview and controls...', 0.96);
  window.myoDemo = demo;
  window.dispatchEvent(new CustomEvent('myoDemo:ready'));
  bootLoader.complete();
} catch (error) {
  console.error(error);
  bootLoader.fail(error);
}
