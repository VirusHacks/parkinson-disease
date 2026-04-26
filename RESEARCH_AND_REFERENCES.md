# Research and References — MotorAssistEnv

Everything MotorAssistEnv is built on: the biophysical models we calibrated against, the clinical literature behind every reward term, the prior RL-for-DBS work we position ourselves next to, and the AI-safety material that shaped the reward design.

## 1. Scientific lineage

| Source | Role in this project |
|---|---|
| [Fleming et al. 2023 — Motor Network Model](https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model) | **Primary calibration source.** Force, tremor, sEMG, beta, and stimulation relationships in MotorAssistEnv trace back to this work. The 12×15 DBS entrainment surface the agent navigates was published in the companion paper. |
| [Fleming et al. 2020 — Cortical–BG Model](https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model) | Establishes the closed-loop DBS modeling lineage. Motivates beta-band suppression as the primary control target. |
| [cviaai/RL-DBS](https://github.com/cviaai/RL-DBS) | Closest prior art for RL-based DBS control. Grounds the framing that RL-for-DBS is a legitimate research direction, not a speculative one. |
| [MyoHub/myosuite_demo](https://github.com/MyoHub/myosuite_demo) + [MuJoCo-WASM](https://github.com/stillonearth/MuJoCo-WASM) | The 3D visualisation layer. When a judge visits `/viewer`, they see a real arm model jitter with tremor and stabilise as the agent applies DBS — the demo is bundled directly into the repo and driven live by `tremor_arv` from the OpenEnv backend. |
| [DeepMind — Specification gaming](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) | Single biggest influence on the reward-hacking audit and the 15-attack adversarial table in `REWARD_DESIGN.md`. |
| [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) | The hackathon's required interface. Everything in this repo is built around its `reset` / `step` contract. |

## 2. Where each source grounds the codebase

**Biophysical calibration** (`core/calibration.py`, `fleming-model-based-brain/`)
The Fleming 2023 model provides the 100-step physiological anchor (t = 10.02–12.02 s), normalisation bounds, and the 12×15 DBS entrainment surface. The 2020 model establishes why beta-band suppression is the right primary target.

**3D viewer** (`static/myosuite_demo/`, `server/app.py`)
The MyoSuite demo is a WebAssembly-based 3D musculoskeletal arm that runs in the browser with no plugins. We ship it directly inside the repo, mount it at `GET /viewer`, and connect it live to the OpenEnv backend — so every tick, `tremor_arv` from the running environment gets translated into proportional arm jitter. When the agent applies good DBS and tremor drops, the arm visibly steadies. It is a direct window into what the numbers mean for a real patient.

**Clinical reward terms** (`graders/`, `REWARD_DESIGN.md`)
Every weight and formula in the grader cites a specific clinical paper — force weighting from Limousin 1995, 130 Hz optimum from Kühn 2008, beta time-in-range from Tinkhauser 2017, safety threshold from Swann 2018, smoothness penalty from Velisar 2019.

**Anti-hacking design** (`REWARD_DESIGN.md §6–8`)
The DeepMind specification-gaming post directly shaped the 15-attack audit table and the principle "exploiting without solving should not score high."

## 3. Full bibliography

**Software**

1. John-E-Fleming. *Parkinsons_Motor_Network_Model* — https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model
2. John-E-Fleming. *Parkinsons_Cortical_Basal_Ganglia_Network_Model* — https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model
3. cviaai. *RL-DBS* — https://github.com/cviaai/RL-DBS
4. MyoHub. *myosuite* — https://github.com/MyoHub/myosuite
5. MyoHub. *myosuite_demo* — https://github.com/MyoHub/myosuite_demo
6. stillonearth. *MuJoCo-WASM* — https://github.com/stillonearth/MuJoCo-WASM
7. meta-pytorch. *OpenEnv* — https://github.com/meta-pytorch/OpenEnv

**Computational modeling and RL-for-DBS**

8. Fleming, J. E., Senneff, S., & Lowery, M. M. (2023). Multivariable closed-loop control of deep brain stimulation for Parkinson's disease. *J Neural Eng*, 20(5), 056029. https://doi.org/10.1088/1741-2552/acfbfa
9. Fleming, J. E., Dunn, E., & Lowery, M. M. (2020). Simulation of closed-loop DBS control schemes for suppression of pathological beta oscillations. *Front Neurosci*, 14, 166. https://doi.org/10.3389/fnins.2020.00166
10. Krylov, D. et al. (2020). Reinforcement Learning Framework for Deep Brain Stimulation Study. *IJCAI-20*. https://doi.org/10.24963/ijcai.2020/394
11. Krylov, D., Dylov, D. V., & Rosenblum, M. (2020). Reinforcement learning for suppression of collective activity in oscillatory ensembles. *Chaos*, 30(3), 033126. https://doi.org/10.1063/1.5128909
12. Caggiano, V. et al. (2022). MyoSuite: A contact-rich simulation suite for musculoskeletal motor control. *arXiv*. https://doi.org/10.48550/arXiv.2205.13600

**Clinical literature behind reward terms**

| # | Citation | Grounds |
|---|---|---|
| 13 | Limousin P et al. (1995). *Lancet* 345(8942):91–95 | Force weighting, early-step emphasis |
| 14 | Nutt JG & Holford NH (1996). *Ann Neurol* 39(5):561–573 | L-DOPA cycle model (`medication_phase`) |
| 15 | Deuschl G et al. (2006). *NEJM* 355(9):896–908 | Motor distortion coefficients; UPDRS calibration |
| 16 | Kühn AA et al. (2008). *NeuroImage* 36(2):379–387 | 130 Hz frequency optimum; `gamma_arv` warning |
| 17 | Castrioto A et al. (2011). *Arch Neurol* 68(12):1550–1556 | Long-horizon DBS stability |
| 18 | Little S et al. (2013). *Ann Neurol* 74(3):449–457 | aDBS feedback target; beta as primary signal |
| 19 | Priori A et al. (2013). *Exp Neurol* 245:77–86 | Frequency–side-effect coupling |
| 20 | Rosa M et al. (2015). *Mov Disord* 30(7):1003–1005 | sEMG as closed-loop feedback signal |
| 21 | Little S et al. (2016). *Mov Disord* 31(8):1336–1341 | Bilateral aDBS; efficiency term |
| 22 | Tinkhauser G et al. (2017). *Brain* 140(11):2968–2981 | Beta time-in-range metric in `beta_score` |
| 23 | Swann NC et al. (2018). *J Neural Eng* 15(4):046006 | Safety score formula; dyskinesia threshold |
| 24 | Velisar A et al. (2019). *Brain Stimul* 12(4):868–876 | Smoothness penalty |

**AI safety and methodology**

25. Krakovna V et al. (2020). Specification gaming: the flip side of AI ingenuity. *DeepMind blog*. https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
