# Research And References — MotorAssistEnv

This is the one place to go for everything MotorAssistEnv is built on: the biophysical models we calibrated against, the clinical literature that justifies every parameter and reward term, the prior RL-for-DBS work we position ourselves next to, the visualization frameworks we drew inspiration from, and the AI-safety material that shaped the reward design.

If you only have time for one section, read [§1 At A Glance](#1-whats-the-scientific-lineage-of-this-project) and [§7 Recommended Citation Block](#7-whats-the-canonical-citation-block-for-this-repo).

All links and citation details below were checked on April 24, 2026 against the linked repository pages and paper landing pages.

## 1. What's the scientific lineage of this project?

| Source | Type | Why it matters | How to reference it |
|---|---|---|---|
| [John-E-Fleming/Parkinsons_Motor_Network_Model](https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model) | Code + companion paper | Closest scientific parent — provides the motor-network outputs we calibrated the environment from | Cite the companion 2023 *J Neural Eng* paper |
| [John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model](https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model) | Code + companion paper | Earlier cortico-basal-ganglia closed-loop DBS model that the motor work extended | Cite the companion 2020 *Frontiers in Neuroscience* paper |
| [cviaai/RL-DBS](https://github.com/cviaai/RL-DBS) | RL-DBS research repo + papers | Establishes RL-for-DBS as a legitimate research direction; informs reward and controller comparison framing | Cite the IJCAI 2020 paper and the *Chaos* 2020 paper |
| [MyoHub/myosuite](https://github.com/MyoHub/myosuite) | Software framework + paper | Reference point for benchmark-style musculoskeletal control environments | Cite the MyoSuite 2022 arXiv paper |
| [MyoHub/myosuite_demo](https://github.com/MyoHub/myosuite_demo) | Visualization/demo repo | Inspired our browser visualization layer and the choice to keep visualization separate from training-time physics | Reference the repo as software/demo inspiration |
| [DeepMind — *Specification gaming*](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) | AI-safety blog post | Single biggest influence on the reward-hacking audit and the integrity-layer defenses | Cite the blog post directly |
| [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) | RL environment framework | The hackathon's required interface; everything in this repo is built around its `reset` / `step` contract | Reference the framework directly |

Sections 2–5 explain each of these in more depth, plus the clinical literature that grounds every reward term and dynamics parameter.

## 2. What does the Fleming motor-network model contribute?

**Repository:** [John-E-Fleming/Parkinsons_Motor_Network_Model](https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model)

This is the most important external source for the project. The repository simulates a Parkinsonian motor network coupling cortico-basal ganglia circuitry to a motoneuron pool, and produces clinically meaningful outputs — local field potentials, EMG, tremor, force, and stimulation responses — together with a multivariable adaptive control strategy.

Why it matters here:

- It is the core scientific source behind our calibration pipeline.
- The benchmark uses outputs derived from this modeling family to construct the environment's grounded state variables and entrainment surface.
- The force, tremor, sEMG, beta, and stimulation relationships in `MotorAssistEnv` ultimately trace back to this work.

A clean way to describe the dependence in writing:

> MotorAssistEnv is calibrated from outputs of the Fleming motor-network model, which couples cortico-basal ganglia circuitry to a motoneuron pool and generates the LFP, EMG, tremor, force, and stimulation signals that anchor our benchmark dynamics.

The repository's README explicitly asks users to cite the companion paper for academic use.

**Recommended citation**

Fleming, J. E., Senneff, S., & Lowery, M. M. (2023). *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*. Journal of Neural Engineering, 20(5), 056029. https://doi.org/10.1088/1741-2552/acfbfa

**Reference links**

- Repo: https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model
- Paper DOI: https://doi.org/10.1088/1741-2552/acfbfa
- PubMed: https://pubmed.ncbi.nlm.nih.gov/37733003/

## 3. What does the Fleming cortical–basal-ganglia model contribute?

**Repository:** [John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model](https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model)

This is the earlier closed-loop DBS modeling work in the same line. It is a Hodgkin–Huxley-based cortico-basal-ganglia network model with closed-loop control, built to study suppression of pathological beta oscillations.

Why it matters here:

- It establishes the closed-loop DBS modeling lineage the motor-network paper extends.
- It motivates the beta-band suppression target that survives into our benchmark as one of the grader's clinical components.
- It is useful background when explaining *why* pathological beta is a meaningful control target.

**Recommended citation**

Fleming, J. E., Dunn, E., & Lowery, M. M. (2020). *Simulation of closed-loop deep brain stimulation control schemes for suppression of pathological beta oscillations in Parkinson's disease*. Frontiers in Neuroscience, 14, 166. https://doi.org/10.3389/fnins.2020.00166

**Reference links**

- Repo: https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model
- Paper DOI: https://doi.org/10.3389/fnins.2020.00166

## 4. What does RL-DBS contribute?

**Repository:** [cviaai/RL-DBS](https://github.com/cviaai/RL-DBS)

RL-DBS is a reinforcement-learning repository for DBS control aimed at suppressing collective neuronal activity. It emphasizes physically meaningful reward functions and benchmark-style comparison across different physical models.

Why it matters here:

- It is the closest piece of prior art for reinforcement-learning-based DBS control.
- It supports the framing that RL-for-DBS is a legitimate research direction, not a speculative application.
- It informed how we talk about reward design, controller comparison, and benchmark headroom in the README and pitch material.

The repository explicitly asks users to cite two papers — an IJCAI 2020 framework paper and a *Chaos* 2020 paper on suppression of collective activity in oscillatory ensembles.

**Recommended primary citation**

Krylov, D., des Combes, R., Laroche, R., Rosenblum, M., & Dylov, D. V. (2020). *Reinforcement Learning Framework for Deep Brain Stimulation Study*. In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20)* (pp. 2847–2854). https://doi.org/10.24963/ijcai.2020/394

**Recommended secondary citation**

Krylov, D., Dylov, D. V., & Rosenblum, M. (2020). *Reinforcement learning for suppression of collective activity in oscillatory ensembles*. Chaos: An Interdisciplinary Journal of Nonlinear Science, 30(3), 033126. https://doi.org/10.1063/1.5128909

**Reference links**

- Repo: https://github.com/cviaai/RL-DBS
- IJCAI paper: https://www.ijcai.org/proceedings/2020/394
- *Chaos* paper DOI: https://doi.org/10.1063/1.5128909

## 5. What do MyoSuite and the MyoSuite demo contribute?

**Repositories:** [MyoHub/myosuite](https://github.com/MyoHub/myosuite) · [MyoHub/myosuite_demo](https://github.com/MyoHub/myosuite_demo)

MyoSuite is a collection of musculoskeletal RL environments built on MuJoCo and wrapped in the OpenAI Gym API. The companion demo repo is a browser-based interactive demonstration of the same musculoskeletal models.

Why both matter here:

- MyoSuite informed how we frame `MotorAssistEnv` as a *benchmark* rather than a one-off simulator, with grader-driven comparability across runs.
- The demo repo informed the choice to ship a separate web visualization layer that is *visualization for communication*, not training-time physics. Judges and reviewers can see arm motion stabilize without it slowing down the RL loop.

The MyoSuite project asks users to cite its arXiv paper. The demo repo does not have its own paper citation request, so we acknowledge it as software.

**Recommended citation**

Caggiano, V., Wang, H., Durandau, G., Sartori, M., & Kumar, V. (2022). *MyoSuite: A contact-rich simulation suite for musculoskeletal motor control*. arXiv. https://doi.org/10.48550/arXiv.2205.13600

**Suggested software acknowledgements**

- MyoHub. *myosuite* [software repository]. GitHub. https://github.com/MyoHub/myosuite
- MyoHub. *myosuite_demo* [software repository]. GitHub. https://github.com/MyoHub/myosuite_demo

**Reference links**

- MyoSuite repo: https://github.com/MyoHub/myosuite
- MyoSuite docs: https://myosuite.readthedocs.io/en/stable/
- MyoSuite paper: https://arxiv.org/abs/2205.13600
- Demo repo: https://github.com/MyoHub/myosuite_demo

## 6. What clinical literature grounds every parameter and reward term?

This is the literature backing the numerical choices in `parkinsons_Motor/core/patient_profiles.py`, the dose–response curves in the environment, and every component of the grader. Each row says where in our docs the citation appears, so the chain from "this number" → "this paper" is traceable.

| # | Citation | Where it shows up in this repo | What it justifies |
|---|---|---|---|
| C1 | **Limousin P et al. (1995).** *Effect of parkinsonian signs and symptoms of bilateral subthalamic nucleus stimulation*. Lancet 345(8942):91–95. | `REWARD_DESIGN.md` §1 (`force_preserved`) | Functional motor improvement is the primary clinical outcome of STN-DBS |
| C2 | **Nutt JG & Holford NH (1996).** *The response to levodopa in Parkinson's disease: imposing pharmacological law and order*. Ann Neurol 39(5):561–573. | `STATE_ACTION_SPACE.md` §3 (medication on/off cycle) | Medication on/off cycling and the wearing-off pattern modeled in the patient state |
| C3 | **Deuschl G et al. (2006).** *A randomized trial of deep-brain stimulation for Parkinson's disease*. NEJM 355(9):896–908. | `REWARD_DESIGN.md` §1, `TASKS.md` §3, `STATE_ACTION_SPACE.md` §3 | Functional motor improvement after DBS; gait-freezing as an established adverse outcome |
| C4 | **Kühn AA et al. (2008).** *High-frequency stimulation of the subthalamic nucleus suppresses oscillatory β activity in patients with Parkinson's disease in parallel with improvement in motor performance*. NeuroImage 36(2):379–387. | `REWARD_DESIGN.md` §1 (`beta_reduction`), `STATE_ACTION_SPACE.md` §2 (LFP beta) | Beta-band suppression as a biomarker of effective DBS |
| C5 | **Castrioto A et al. (2011).** *Ten-year outcome of subthalamic stimulation in Parkinson disease*. Arch Neurol 68(12):1550–1556. | `TASKS.md` §3 (long-horizon adherence) | Long-horizon side-effect accumulation under chronic DBS |
| C6 | **Olanow CW et al. (2013).** *Continuous intrajejunal infusion of levodopa-carbidopa intestinal gel for patients with advanced Parkinson's disease*. Mov Disord (as cited in `TASKS.md`) | `TASKS.md` §3 (advanced-disease tasks) | Variability and side-effect burden in advanced PD therapy |
| C7 | **Little S et al. (2013).** *Adaptive deep brain stimulation in advanced Parkinson disease*. Ann Neurol 74(3):449–457. | `REWARD_DESIGN.md` §1 (`energy_efficiency`), `STATE_ACTION_SPACE.md` §1 (closed-loop framing) | The clinical case for *adaptive* (closed-loop) DBS over fixed open-loop stim |
| C8 | **Priori A et al. (2013).** *Adaptive deep brain stimulation (aDBS) controlled by local field potential oscillations*. Exp Neurol 245:77–86. | `REWARD_DESIGN.md` §1, `STATE_ACTION_SPACE.md` §1 | LFP-driven adaptive stimulation as the reference clinical paradigm |
| C9 | **Rosa M et al. (2015).** *Adaptive deep brain stimulation in a freely moving Parkinsonian patient*. Mov Disord 30(7):1003–1005. | `REWARD_DESIGN.md` §1, `STATE_ACTION_SPACE.md` §3 | Real-world feasibility evidence for closed-loop DBS |
| C10 | **Little S et al. (2016).** *Bilateral adaptive deep brain stimulation is effective in Parkinson's disease*. Mov Disord 31(8):1336–1341. | `REWARD_DESIGN.md` §1, `TASKS.md` §3 | Energy savings and motor benefit of bilateral aDBS — cited for the energy-efficiency reward weight |
| C11 | **Tinkhauser G et al. (2017).** *Beta burst dynamics in Parkinson's disease OFF and ON dopaminergic medication*. Brain 140(11):2968–2981. | `REWARD_DESIGN.md` §1 (`beta_reduction`), `TASKS.md` §3 (medication interaction), `STATE_ACTION_SPACE.md` §2 | Burst-level beta dynamics as the right granularity for the suppression metric |
| C12 | **Swann NC et al. (2018).** *Adaptive deep brain stimulation for Parkinson's disease using motor cortex sensing*. J Neural Eng 15(4):046006. | `REWARD_DESIGN.md` §1, `STATE_ACTION_SPACE.md` §2 (cortical observations) | Cortical sensing as part of a realistic adaptive DBS observation set |
| C13 | **Velisar A et al. (2019).** *Dual threshold neural closed loop deep brain stimulation in Parkinson disease patients*. Brain Stimul 12(4):868–876. | `REWARD_DESIGN.md` §1 | Threshold-based closed-loop DBS as a baseline comparator for any RL controller |
| C14 | **Fleming JE et al. (2020).** *Simulation of closed-loop deep brain stimulation control schemes for suppression of pathological beta oscillations in Parkinson's disease*. Frontiers in Neuroscience 14, 166. | Source [§3](#3-what-does-the-fleming-corticalbasal-ganglia-model-contribute) | Earlier closed-loop modeling lineage |
| C15 | **Fleming JE et al. (2020).** *PLOS Computational Biology* 16(8):e1008165. | `STATE_ACTION_SPACE.md` §3 | Additional Fleming-group computational modeling cited in the state-space doc |
| C16 | **Fleming JE et al. (2023).** *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*. J Neural Eng 20(5):056029. | Source [§2](#2-what-does-the-fleming-motor-network-model-contribute), `REWARD_DESIGN.md` §1, `TASKS.md` §3 | The primary biophysical source for environment dynamics |

For the purposes of this hackathon, citations C14, C15, and C16 are the ones we are *most* directly dependent on — they are the source of the calibration data.

## 7. What AI safety and methodology references shaped the reward design?

The reward function and grader weren't designed in a vacuum — they were stress-tested against a specific body of work on specification gaming and the OpenEnv hackathon's own reward-design guidance.

| Source | Type | What we used it for |
|---|---|---|
| **DeepMind — *Specification gaming: the flip side of AI ingenuity*** ([blog](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)) | Blog post | Source of the threat model behind the 15-row anti-hacking table in `REWARD_DESIGN.md` §5. Every "what would a clever agent try?" row exists because of this post |
| **OpenEnv project** ([repo](https://github.com/meta-pytorch/OpenEnv)) | RL environment framework | Defines the `reset` / `step` / typed-observation interface and the deterministic-grader contract that the whole project conforms to |
| **OpenEnv hackathon Bible / Help Guide** (`docs/Bible.txt`, `docs/helpguide.txt`) | Internal hackathon docs | Source of the rubric points the reward design and graders are explicitly engineered to satisfy |
| **OpenEnv judging criteria** (`docs/judging_criteria.txt`, `docs/judges.txt`) | Internal hackathon docs | Source of the evaluation lens used in the `REWARD_DESIGN.md` self-audit |

The combination of these four sources is what justifies the *style* of the reward design: dense per-step shaping plus a deterministic episode-end grader, with a deliberate audit pass against known reward-hacking patterns.

## 8. How do these sources map onto specific parts of the project?

| Project area | Primary sources |
|---|---|
| Biophysical state and motor signal grounding | [§2](#2-what-does-the-fleming-motor-network-model-contribute) Fleming motor-network repo + 2023 JNE paper; clinical refs C4, C7, C8, C11, C12 |
| Earlier closed-loop DBS scientific lineage | [§3](#3-what-does-the-fleming-corticalbasal-ganglia-model-contribute) Fleming cortico-basal-ganglia repo + 2020 *Frontiers* paper |
| Reward components (force, beta, energy, side-effects) | Clinical refs C1, C3, C4, C7, C9, C10, C11, C13 |
| Patient-profile parameter calibration | Clinical refs C2, C5, C6, C12; Fleming 2023 (C16) |
| Curriculum / task-difficulty design | Clinical refs C3, C5, C6, C10, C11, C16 |
| RL-for-DBS prior art and benchmark framing | [§4](#4-what-does-rl-dbs-contribute) RL-DBS repo + IJCAI 2020 + *Chaos* 2020 papers |
| Musculoskeletal benchmark framing | [§5](#5-what-do-myosuite-and-the-myosuite-demo-contribute) MyoSuite repo + 2022 arXiv paper |
| Web visualization layer | [§5](#5-what-do-myosuite-and-the-myosuite-demo-contribute) MyoSuite demo repo |
| Reward-hacking audit and integrity layer | [§7](#7-what-ai-safety-and-methodology-references-shaped-the-reward-design) DeepMind specification-gaming post |
| Environment interface and deterministic grading contract | [§7](#7-what-ai-safety-and-methodology-references-shaped-the-reward-design) OpenEnv project + hackathon docs |

## 9. How should we acknowledge all of this?

### Short version (one-paragraph credit)

> MotorAssistEnv is scientifically grounded in the Parkinsonian DBS modeling line developed by John E. Fleming and collaborators — particularly the 2023 motor-network model. It draws on a body of clinical literature on adaptive DBS (Little 2013/2016, Priori 2013, Rosa 2015, Tinkhauser 2017, Swann 2018, Velisar 2019) for parameter calibration and reward-component design. It positions itself relative to RL-DBS as prior art for reinforcement-learning-based DBS control, and to MyoSuite for benchmark-style musculoskeletal environment framing. The reward design is explicitly stress-tested against DeepMind's *Specification gaming* analysis, and the entire interface conforms to the OpenEnv standard.

### README-length version

> MotorAssistEnv is calibrated from the Fleming et al. (2023) biophysical motor-network model of Parkinsonian basal ganglia and motoneuron coupling. Reward components and patient-profile parameters are anchored in published clinical evidence on adaptive DBS — including Limousin (1995), Deuschl (2006), Kühn (2008), Little (2013, 2016), Priori (2013), Rosa (2015), Tinkhauser (2017), Swann (2018), and Velisar (2019). The reward design is audited against DeepMind's *Specification gaming: the flip side of AI ingenuity*, and the environment exposes a strictly OpenEnv-compliant `reset` / `step` interface with a deterministic episode-end grader. Visualization framing is inspired by MyoSuite and MyoSuite Demo; reinforcement-learning-for-DBS positioning follows the RL-DBS line.

## 10. What's the canonical citation block for this repo?

If you only need one compact "References" section for a README, Devpost page, paper, or report, this is it.

**Software references**

1. John-E-Fleming. *Parkinsons_Motor_Network_Model* [software repository]. GitHub. https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model
2. John-E-Fleming. *Parkinsons_Cortical_Basal_Ganglia_Network_Model* [software repository]. GitHub. https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model
3. cviaai. *RL-DBS* [software repository]. GitHub. https://github.com/cviaai/RL-DBS
4. MyoHub. *myosuite* [software repository]. GitHub. https://github.com/MyoHub/myosuite
5. MyoHub. *myosuite_demo* [software repository]. GitHub. https://github.com/MyoHub/myosuite_demo
6. meta-pytorch. *OpenEnv* [software repository]. GitHub. https://github.com/meta-pytorch/OpenEnv

**Computational modeling and RL-for-DBS papers**

7. Fleming, J. E., Senneff, S., & Lowery, M. M. (2023). *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*. Journal of Neural Engineering, 20(5), 056029. https://doi.org/10.1088/1741-2552/acfbfa
8. Fleming, J. E., Dunn, E., & Lowery, M. M. (2020). *Simulation of closed-loop deep brain stimulation control schemes for suppression of pathological beta oscillations in Parkinson's disease*. Frontiers in Neuroscience, 14, 166. https://doi.org/10.3389/fnins.2020.00166
9. Krylov, D., des Combes, R., Laroche, R., Rosenblum, M., & Dylov, D. V. (2020). *Reinforcement Learning Framework for Deep Brain Stimulation Study*. IJCAI-20. https://doi.org/10.24963/ijcai.2020/394
10. Krylov, D., Dylov, D. V., & Rosenblum, M. (2020). *Reinforcement learning for suppression of collective activity in oscillatory ensembles*. Chaos, 30(3), 033126. https://doi.org/10.1063/1.5128909
11. Caggiano, V., Wang, H., Durandau, G., Sartori, M., & Kumar, V. (2022). *MyoSuite: A contact-rich simulation suite for musculoskeletal motor control*. arXiv. https://doi.org/10.48550/arXiv.2205.13600

**Clinical literature behind reward terms and patient calibration**

12. Limousin, P. et al. (1995). *Effect of parkinsonian signs and symptoms of bilateral subthalamic nucleus stimulation*. Lancet, 345(8942), 91–95.
13. Nutt, J. G., & Holford, N. H. G. (1996). *The response to levodopa in Parkinson's disease: imposing pharmacological law and order*. Annals of Neurology, 39(5), 561–573.
14. Deuschl, G. et al. (2006). *A randomized trial of deep-brain stimulation for Parkinson's disease*. New England Journal of Medicine, 355(9), 896–908.
15. Kühn, A. A. et al. (2008). *High-frequency stimulation of the subthalamic nucleus suppresses oscillatory β activity in patients with Parkinson's disease in parallel with improvement in motor performance*. NeuroImage, 36(2), 379–387.
16. Castrioto, A. et al. (2011). *Ten-year outcome of subthalamic stimulation in Parkinson disease*. Archives of Neurology, 68(12), 1550–1556.
17. Olanow, C. W. et al. (2013). *Continuous intrajejunal infusion of levodopa-carbidopa intestinal gel for patients with advanced Parkinson's disease*. Movement Disorders.
18. Little, S. et al. (2013). *Adaptive deep brain stimulation in advanced Parkinson disease*. Annals of Neurology, 74(3), 449–457.
19. Priori, A. et al. (2013). *Adaptive deep brain stimulation (aDBS) controlled by local field potential oscillations*. Experimental Neurology, 245, 77–86.
20. Rosa, M. et al. (2015). *Adaptive deep brain stimulation in a freely moving Parkinsonian patient*. Movement Disorders, 30(7), 1003–1005.
21. Little, S. et al. (2016). *Bilateral adaptive deep brain stimulation is effective in Parkinson's disease*. Movement Disorders, 31(8), 1336–1341.
22. Tinkhauser, G. et al. (2017). *Beta burst dynamics in Parkinson's disease OFF and ON dopaminergic medication*. Brain, 140(11), 2968–2981.
23. Swann, N. C. et al. (2018). *Adaptive deep brain stimulation for Parkinson's disease using motor cortex sensing*. Journal of Neural Engineering, 15(4), 046006.
24. Velisar, A. et al. (2019). *Dual threshold neural closed loop deep brain stimulation in Parkinson disease patients*. Brain Stimulation, 12(4), 868–876.

**AI safety and methodology**

25. Krakovna, V., Uesato, J., Mikulik, V., Rahtz, M., Everitt, T., Kumar, R., Kenton, Z., Leike, J., & Legg, S. (2020). *Specification gaming: the flip side of AI ingenuity*. DeepMind blog. https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/

## 11. Where else are these references used inside the repo?

| Reference | Mentioned in |
|---|---|
| Fleming 2023 (C16) | `REWARD_DESIGN.md`, `TASKS.md`, `STATE_ACTION_SPACE.md`, `PROBLEM.md`, `README.md`, `parkinsons_Motor/README.md`, `docs/JUDGE_PITCH.md`, `CALIBRATION.md` |
| Fleming 2020 *Frontiers* (C14) | `RESEARCH_AND_REFERENCES.md` (this doc), referenced as scientific lineage |
| Fleming 2020 *PLOS CB* (C15) | `STATE_ACTION_SPACE.md` |
| Tinkhauser 2017 (C11) | `REWARD_DESIGN.md`, `TASKS.md`, `STATE_ACTION_SPACE.md` |
| Little 2013 (C7), Little 2016 (C10) | `REWARD_DESIGN.md`, `STATE_ACTION_SPACE.md`, `TASKS.md` |
| Deuschl 2006 (C3) | `REWARD_DESIGN.md`, `TASKS.md`, `STATE_ACTION_SPACE.md` |
| Limousin 1995 (C1), Velisar 2019 (C13) | `REWARD_DESIGN.md` |
| Castrioto 2011 (C5), Olanow 2013 (C6) | `TASKS.md` |
| Nutt & Holford 1996 (C2) | `STATE_ACTION_SPACE.md` |
| Kühn 2008 (C4), Priori 2013 (C8), Rosa 2015 (C9), Swann 2018 (C12) | `REWARD_DESIGN.md`, `STATE_ACTION_SPACE.md` |
| RL-DBS (Krylov 2020 ×2) | `REWARD_DESIGN.md` (anti-hacking framing), this doc |
| MyoSuite (Caggiano 2022) | `ARCHITECTURE.md`, `docs/JUDGE_PITCH.md`, this doc |
| DeepMind *Specification gaming* | `REWARD_DESIGN.md` §5 (anti-hacking table), §6 (integrity layer) |
| OpenEnv project | `README.md`, `PROBLEM.md`, `ARCHITECTURE.md`, `parkinsons_Motor/README.md`, `docs/JUDGE_PITCH.md`, this doc |

## 12. Sources used to compile this document

- Fleming motor-network repo: https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model
- Fleming cortical-basal-ganglia repo: https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model
- RL-DBS repo: https://github.com/cviaai/RL-DBS
- MyoSuite repo: https://github.com/MyoHub/myosuite
- MyoSuite docs: https://myosuite.readthedocs.io/en/stable/
- MyoSuite demo repo: https://github.com/MyoHub/myosuite_demo
- OpenEnv project: https://github.com/meta-pytorch/OpenEnv
- DeepMind *Specification gaming* blog: https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
- Internal repo docs: `docs/Bible.txt`, `docs/helpguide.txt`, `docs/judging_criteria.txt`, `docs/judges.txt`, `REWARD_DESIGN.md`, `TASKS.md`, `STATE_ACTION_SPACE.md`, `PROBLEM.md`, `README.md`
