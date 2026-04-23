"""
Parkinson's Brain State Calibrator
===================================
Loads pre-computed park-sen simulation outputs and builds a ground-truth
timeline of neural and motor features for the RL environment.

Primary data sources (CSVs from closed-loop DBS simulation):
  - tremor_ARV_Observer          : tremor amplitude envelope (mV, rectified)
  - beta_ARV_Observer            : STN beta-band amplitude (mV, rectified)
  - sEMG_ARV_Observer            : surface EMG envelope (mV)
  - stimulation_Amplitude_Observer : DBS amplitude delivered (mA)
  - stimulation_Pulse_Duration_Observer : DBS pulse width (ms)
  - scheduler_classification      : which sub-controller is active
  - scheduler_output              : DBS command from scheduler
  - Force_amplitude / Force_times : muscle force output (mN)
  - sEMG_values / sEMG_times      : raw surface EMG
  - side_Effects_Observer         : cumulative DBS side-effect load
  - Controller_Bank_Beta_ARV_*    : beta controller error/output/state
  - Motor_Symptom_Sample_Times    : sample timestamps (seconds)

Secondary data (for DBS parameter sweep lookup):
  - Collaterals_Entrained_values.txt : 12x15 entrainment matrix
  - DBS_Amplitude/Pulse_Width interpolation values

The CalibratedBrainState produced here is the authoritative source of
ground-truth Parkinson dynamics for the RL environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PARK_SEN = _HERE / "fleming-model-based-brain"
_RESULTS = _PARK_SEN / "Model_Results"


# ── data containers ───────────────────────────────────────────────────────────

@dataclass
class WindowFeatures:
    """
    Neural and motor features at one simulation timestep.
    All normalized values are in [0, 1] unless otherwise noted.
    Time is in seconds (matching park-sen CSV convention).
    """
    t_s: float                  # simulation time (seconds)

    # ── neural state ──────────────────────────────────────────────────────────
    beta_arv: float             # STN beta ARV (normalized, 0=healthy, 1=max PD)
    tremor_arv: float           # Tremor ARV (normalized, 0=no tremor, 1=max)
    semg_arv: float             # sEMG envelope (normalized)

    # ── DBS delivered ─────────────────────────────────────────────────────────
    dbs_amplitude_ma: float     # mA (raw)
    dbs_pulse_width_ms: float   # ms (raw)

    # ── controller state ──────────────────────────────────────────────────────
    scheduler_class: int        # 0=tremor controller, 1=beta controller
    scheduler_dbs_output: float # DBS amplitude commanded by scheduler (mA)
    beta_ctrl_error: float      # beta controller error (raw)
    beta_ctrl_output: float     # beta controller DBS command (mA)
    side_effect_load: float     # cumulative side-effect proxy (normalized)

    # ── motor output ──────────────────────────────────────────────────────────
    force_amplitude: float      # muscle force (raw mN)
    force_preserved: float      # fraction of healthy force preserved [0,1]
                                # = force / healthy_baseline_force (59752 mN)
    semg_raw_mean: float        # mean raw sEMG in this window

    # ── derived disease state ─────────────────────────────────────────────────
    disease_severity: float     # normalized tremor ARV as severity proxy [0,1]
    beta_suppression: float     # DBS beta suppression achieved so far [0,1]
                                # 0 = no suppression (peak PD), 1 = fully suppressed


@dataclass
class CalibratedBrainState:
    """Full calibration output — authoritative ground truth for the RL env."""

    # Ordered timeline of simulation states (by time)
    timeline: List[WindowFeatures] = field(default_factory=list)

    # ── normalization bounds (from simulation data) ────────────────────────────
    tremor_arv_max: float = 1.0       # max observed tremor ARV
    beta_arv_max: float = 1.0         # max observed beta ARV
    semg_arv_max: float = 1.0         # max observed sEMG ARV
    force_max: float = 1.0            # max observed force amplitude
    side_effect_max: float = 1.0      # max observed side-effect load

    # ── physiological reference points (from simulation) ─────────────────────
    healthy_force_mn: float = 59752.58  # max force = healthy motor output (mN)
    predbs_force_mn: float = 47642.83   # pre-DBS Parkinson force baseline (mN)
                                        # 56% reduction by episode end with DBS

    # ── aggregate baseline stats ───────────────────────────────────────────────
    baseline_tremor_arv: float = 0.0  # median normalized tremor (no DBS)
    baseline_beta_arv: float = 0.0    # median normalized beta (no DBS)
    baseline_semg_arv: float = 0.0    # median normalized sEMG
    baseline_dbs_amplitude: float = 0.0   # median DBS amplitude when active
    baseline_force: float = 0.0       # median force (mN)

    # ── DBS parameter sweep lookup ─────────────────────────────────────────────
    dbs_entrainment: np.ndarray = field(default_factory=lambda: np.zeros((12, 15)))
    dbs_amplitudes_ma: np.ndarray = field(default_factory=lambda: np.zeros(12))
    dbs_pulse_widths_ms: np.ndarray = field(default_factory=lambda: np.zeros(15))


# ── CSV loaders ───────────────────────────────────────────────────────────────

def _csv(name: str) -> Optional[np.ndarray]:
    """Load a single-column CSV. Memory-efficient to prevent OOM on HuggingFace."""
    p = _RESULTS / (name + ".csv")
    if not p.exists():
        print(f"[calibrator] File not found: {p}", flush=True)
        return None
    try:
        print(f"[calibrator] Loading {p.name} ({p.stat().st_size / 1e6:.2f} MB)...", flush=True)
        with open(p, 'r') as f:
            return np.array([float(line) for line in f if line.strip()], dtype=np.float32)
    except Exception as e:
        print(f"[calibrator] Error loading {p.name}: {e}", flush=True)
        return None


def _csv_safe(name: str, fallback_len: int = 0) -> np.ndarray:
    d = _csv(name)
    return d if d is not None else np.zeros(fallback_len)


# ── DBS lookup ────────────────────────────────────────────────────────────────

def _load_dbs_lookup() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    amp_file = _PARK_SEN / "DBS_Amplitude_Interpolation_values.txt"
    pw_file = _PARK_SEN / "DBS_Pulse_Width_Interpolation_values.txt"
    ent_file = _PARK_SEN / "Collaterals_Entrained_values.txt"
    amplitudes = np.loadtxt(str(amp_file))
    pulse_widths = np.loadtxt(str(pw_file))
    raw = np.loadtxt(str(ent_file), delimiter=",")
    return raw / 100.0, amplitudes, pulse_widths


# ── interpolation helpers ─────────────────────────────────────────────────────

def _interp_nearest(times_src: np.ndarray, values_src: np.ndarray, t_target: float) -> float:
    """Return value from src array nearest to t_target."""
    if len(times_src) == 0:
        return 0.0
    idx = int(np.argmin(np.abs(times_src - t_target)))
    return float(values_src[idx])


def _interp_window_mean(times_src: np.ndarray, values_src: np.ndarray,
                         t_lo: float, t_hi: float) -> float:
    """Return mean of values_src where times_src is in [t_lo, t_hi]."""
    mask = (times_src >= t_lo) & (times_src <= t_hi)
    if not mask.any():
        return _interp_nearest(times_src, values_src, (t_lo + t_hi) / 2)
    return float(values_src[mask].mean())


# ── main calibration ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def calibrate(verbose: bool = False) -> CalibratedBrainState:
    """
    Load all park-sen results and return a CalibratedBrainState.
    Cached — computed once per process.
    """
    state = CalibratedBrainState()

    # ── DBS parameter sweep lookup ────────────────────────────────────────────
    try:
        ent, amps, pws = _load_dbs_lookup()
        state.dbs_entrainment = ent
        state.dbs_amplitudes_ma = amps
        state.dbs_pulse_widths_ms = pws
    except Exception as e:
        if verbose:
            print(f"[calibrator] DBS lookup failed: {e}")

    # ── load CSV signals ──────────────────────────────────────────────────────
    motor_t    = _csv_safe("Motor_Symptom_Sample_Times")          # (100,) seconds
    tremor_arv = _csv_safe("tremor_ARV_Observer_values")          # (100,) mV
    beta_arv   = _csv_safe("beta_ARV_Observer_values")            # (100,) mV
    semg_arv   = _csv_safe("sEMG_ARV_Observer_values")            # (100,) mV

    dbs_amp_t  = _csv_safe("stimulation_Amplitude_Observer_times")    # (100,) s
    dbs_amp_v  = _csv_safe("stimulation_Amplitude_Observer_values")   # (100,) mA
    dbs_pw_v   = _csv_safe("stimulation_Pulse_Duration_Observer_values")  # (100,) ms

    sched_t    = _csv_safe("scheduler_sample_times")              # (100,) s
    sched_cls  = _csv_safe("scheduler_classification_values")     # (100,) int
    sched_out  = _csv_safe("scheduler_output_values")             # (100,) mA

    beta_ctrl_t   = _csv_safe("Controller_Bank_Beta_ARV_Controller_Period_1_sample_times")
    beta_ctrl_err = _csv_safe("Controller_Bank_Beta_ARV_Controller_Period_1_error_values")
    beta_ctrl_out = _csv_safe("Controller_Bank_Beta_ARV_Controller_Period_1_output_values")

    se_t   = _csv_safe("side_Effects_Observer_times")             # (6,) s
    se_v   = _csv_safe("side_Effects_Observer_values")            # (6,) normalized

    # Large force/sEMG arrays
    force_t = _csv("Force_times")        # (6.7M,) ms
    force_v = _csv("Force_amplitude_values")  # (6.7M,)
    semg_t  = _csv("sEMG_times")         # (6.7M,) ms
    semg_v  = _csv("sEMG_values")        # (6.7M,)

    n = len(motor_t)
    if n == 0:
        if verbose:
            print("[calibrator] No Motor_Symptom_Sample_Times found")
        return state

    # ── normalization bounds ──────────────────────────────────────────────────
    state.tremor_arv_max = float(tremor_arv.max()) if tremor_arv.max() > 0 else 1.0
    state.beta_arv_max   = float(beta_arv.max())   if beta_arv.max() > 0   else 1.0
    state.semg_arv_max   = float(semg_arv.max())   if semg_arv.max() > 0   else 1.0
    state.force_max      = float(force_v.max())    if force_v is not None and force_v.max() > 0 else 1.0
    state.side_effect_max = float(se_v.max())      if se_v.max() > 0        else 1.0

    tremor_norm = tremor_arv / state.tremor_arv_max
    beta_norm   = beta_arv   / state.beta_arv_max
    semg_norm   = semg_arv   / state.semg_arv_max

    # ── build timeline ────────────────────────────────────────────────────────
    for i, t_s in enumerate(motor_t):
        t_ms = t_s * 1000.0  # convert s → ms for force/sEMG lookup

        # DBS delivered
        dbs_amp = _interp_nearest(dbs_amp_t, dbs_amp_v, t_s) if len(dbs_amp_t) > 0 else 0.0
        dbs_pw  = _interp_nearest(dbs_amp_t, dbs_pw_v,  t_s) if len(dbs_pw_v) > 0 else 0.06

        # Scheduler
        sch_cls = int(round(_interp_nearest(sched_t, sched_cls, t_s))) if len(sched_t) > 0 else 1
        sch_out = _interp_nearest(sched_t, sched_out, t_s)             if len(sched_t) > 0 else 0.0

        # Beta controller
        bc_err = _interp_nearest(beta_ctrl_t, beta_ctrl_err, t_s) if len(beta_ctrl_t) > 0 else 0.0
        bc_out = _interp_nearest(beta_ctrl_t, beta_ctrl_out, t_s) if len(beta_ctrl_t) > 0 else 0.0

        # Side effects
        se_load = _interp_nearest(se_t, se_v / state.side_effect_max, t_s) if len(se_t) > 0 else 0.0

        # Force/sEMG — use 20ms window around t_ms
        if force_t is not None and len(force_t) > 0:
            force_mean = _interp_window_mean(force_t, force_v, t_ms - 10, t_ms + 10)
        else:
            force_mean = 0.0

        if semg_t is not None and len(semg_t) > 0:
            semg_mean = _interp_window_mean(semg_t, semg_v, t_ms - 10, t_ms + 10)
        else:
            semg_mean = 0.0

        force_preserved = float(np.clip(force_mean / 59752.58, 0.0, 1.0))
        # beta_suppression: how much has DBS reduced beta vs its peak value
        # beta_norm[i]=1.0 means peak (no suppression), 0=fully suppressed
        beta_sup = float(np.clip(1.0 - beta_norm[i], 0.0, 1.0))

        wf = WindowFeatures(
            t_s=t_s,
            beta_arv=float(beta_norm[i]),
            tremor_arv=float(tremor_norm[i]),
            semg_arv=float(semg_norm[i]),
            dbs_amplitude_ma=float(dbs_amp),
            dbs_pulse_width_ms=float(dbs_pw),
            scheduler_class=sch_cls,
            scheduler_dbs_output=float(sch_out),
            beta_ctrl_error=float(bc_err),
            beta_ctrl_output=float(bc_out),
            side_effect_load=float(np.clip(se_load, 0.0, 1.0)),
            force_amplitude=float(force_mean),
            force_preserved=force_preserved,
            semg_raw_mean=float(semg_mean),
            disease_severity=float(tremor_norm[i]),
            beta_suppression=beta_sup,
        )
        state.timeline.append(wf)

    state.timeline.sort(key=lambda w: w.t_s)

    # ── baseline stats (first 10 samples = before DBS ramps up) ──────────────
    pre_dbs = [w for w in state.timeline if w.dbs_amplitude_ma < 0.05][:10]
    all_windows = state.timeline

    state.baseline_tremor_arv = float(np.median([w.tremor_arv for w in pre_dbs])) if pre_dbs else float(tremor_norm.mean())
    state.baseline_beta_arv   = float(np.median([w.beta_arv   for w in pre_dbs])) if pre_dbs else float(beta_norm.mean())
    state.baseline_semg_arv   = float(np.median([w.semg_arv   for w in pre_dbs])) if pre_dbs else float(semg_norm.mean())

    active_dbs = [w.dbs_amplitude_ma for w in all_windows if w.dbs_amplitude_ma > 0.1]
    state.baseline_dbs_amplitude = float(np.median(active_dbs)) if active_dbs else 0.0

    nonzero_force = [w.force_amplitude for w in all_windows if w.force_amplitude > 0]
    state.baseline_force = float(np.median(nonzero_force)) if nonzero_force else 0.0

    if verbose:
        print(
            f"[calibrator] {len(state.timeline)} windows loaded from CSVs | "
            f"tremor={state.baseline_tremor_arv:.3f} "
            f"beta={state.baseline_beta_arv:.4f} "
            f"sEMG={state.baseline_semg_arv:.3f} "
            f"DBS_median={state.baseline_dbs_amplitude:.3f}mA "
            f"force_median={state.baseline_force:.1f}"
        )

    return state


# ── DBS effect lookup (parameter sweep table) ─────────────────────────────────

def query_dbs_effect(
    brain_state: CalibratedBrainState,
    amplitude_ma: float,
    pulse_width_ms: float,
) -> float:
    """
    Bilinear interpolation of cortical collateral entrainment fraction [0,1].
    0 = no DBS effect, 1 = full entrainment (maximum Parkinson suppression).
    """
    amps = brain_state.dbs_amplitudes_ma
    pws  = brain_state.dbs_pulse_widths_ms
    mat  = brain_state.dbs_entrainment

    if mat.size == 0 or amps.size == 0 or pws.size == 0:
        return 0.0

    amp_c = float(np.clip(amplitude_ma,   amps[0], amps[-1]))
    pw_c  = float(np.clip(pulse_width_ms, pws[0],  pws[-1]))

    ai = int(np.clip(np.searchsorted(amps, amp_c, side="right") - 1, 0, len(amps) - 2))
    pi = int(np.clip(np.searchsorted(pws,  pw_c,  side="right") - 1, 0, len(pws)  - 2))

    ta = (amp_c - amps[ai]) / (amps[ai+1] - amps[ai]) if amps[ai+1] != amps[ai] else 0.0
    tp = (pw_c  - pws[pi])  / (pws[pi+1]  - pws[pi])  if pws[pi+1]  != pws[pi]  else 0.0

    return float(
        (1-ta)*(1-tp)*mat[ai,pi]   + ta*(1-tp)*mat[ai+1,pi] +
        (1-ta)*tp    *mat[ai,pi+1] + ta*tp    *mat[ai+1,pi+1]
    )


# ── timeline query helpers ────────────────────────────────────────────────────

def get_window_at(brain_state: CalibratedBrainState, t_s: float) -> WindowFeatures:
    """Return the WindowFeatures nearest to t_s (seconds)."""
    if not brain_state.timeline:
        return WindowFeatures(t_s=t_s, beta_arv=0, tremor_arv=0, semg_arv=0,
                              dbs_amplitude_ma=0, dbs_pulse_width_ms=0.06,
                              scheduler_class=1, scheduler_dbs_output=0,
                              beta_ctrl_error=0, beta_ctrl_output=0,
                              side_effect_load=0, force_amplitude=0,
                              force_preserved=0, semg_raw_mean=0,
                              disease_severity=0, beta_suppression=0)
    return min(brain_state.timeline, key=lambda w: abs(w.t_s - t_s))


def get_window_idx(brain_state: CalibratedBrainState, idx: int) -> WindowFeatures:
    """Return the WindowFeatures at position idx in the timeline."""
    tl = brain_state.timeline
    return tl[max(0, min(idx, len(tl) - 1))]


if __name__ == "__main__":
    bs = calibrate(verbose=True)
    print(f"\nPhysiological reference points:")
    print(f"  healthy_force     : {bs.healthy_force_mn:.0f} mN  (max observed = 100% healthy)")
    print(f"  predbs_force      : {bs.predbs_force_mn:.0f} mN  (Parkinson's without DBS = {bs.predbs_force_mn/bs.healthy_force_mn*100:.0f}% of healthy)")
    print(f"  tremor_arv_max    : {bs.tremor_arv_max:.2f} mV")
    print(f"  beta_arv_max      : {bs.beta_arv_max:.6f} mV")
    print(f"\nBaseline (pre-DBS) normalized state:")
    print(f"  tremor severity   : {bs.baseline_tremor_arv:.3f}  (0=none, 1=severe)")
    print(f"  beta oscillation  : {bs.baseline_beta_arv:.4f}  (0=suppressed, 1=peak PD)")
    print(f"\nFull closed-loop trajectory  [t | tremor | beta | DBS_amp | force_preserved | severity | beta_suppression]:")
    print(f"  {'t(s)':>5}  {'tremor':>6}  {'beta':>6}  {'dbs(mA)':>7}  {'force%':>7}  {'severity':>8}  {'beta_sup':>8}")
    for w in bs.timeline[::10]:
        print(f"  {w.t_s:>5.2f}  {w.tremor_arv:>6.3f}  {w.beta_arv:>6.3f}  {w.dbs_amplitude_ma:>7.3f}  {w.force_preserved*100:>6.1f}%  {w.disease_severity:>8.3f}  {w.beta_suppression:>8.3f}")
    print(f"\nDBS entrainment (param sweep):")
    for amp in [0.0, 0.5, 1.0, 2.0, 3.0]:
        ent = query_dbs_effect(bs, amp, 0.13)
        print(f"  {amp:.1f}mA / 0.13ms -> {ent*100:.0f}% entrained")
