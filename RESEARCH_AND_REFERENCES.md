# Research And References

This document explains the main research artifacts, repositories, and software projects that informed MotorAssistEnv. It is meant to do two things:

- show the intellectual lineage of the project clearly
- provide a clean citation and acknowledgement guide for README, Devpost, demos, and academic use

All links and citation details below were checked on April 24, 2026 against the linked repository pages and paper landing pages.

## At A Glance

| Source | Type | Why it matters to this project | How we should reference it |
|---|---|---|---|
| [John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model](https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model) | Code + companion paper | Provides the earlier cortico-basal ganglia closed-loop DBS modeling foundation | Cite the companion 2020 Frontiers paper the repo asks users to cite |
| [John-E-Fleming/Parkinsons_Motor_Network_Model](https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model) | Code + companion paper | This is the closest scientific parent of our benchmark; it provides the motor-network outputs we calibrated into the environment | Cite the companion 2023 Journal of Neural Engineering paper the repo asks users to cite |
| [MyoHub/myosuite_demo](https://github.com/MyoHub/myosuite_demo) | Visualization/demo repo | Informed the musculoskeletal browser demo direction and visual communication layer | Reference the repo as software/demo inspiration |
| [MyoHub/myosuite](https://github.com/MyoHub/myosuite) | Software framework + paper | Informed our musculoskeletal visualization framing and benchmark presentation style | Cite the MyoSuite 2022 arXiv paper and reference the repo |
| [cviaai/RL-DBS](https://github.com/cviaai/RL-DBS) | RL-DBS research repo + papers | Helped position our work relative to prior RL-for-DBS efforts and reward/control benchmark thinking | Cite the IJCAI 2020 paper first; optionally also cite the Chaos 2020 paper because the repo requests both |

## 1. Fleming Cortical Basal Ganglia Model

**Repository**

- [GitHub repo](https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model)

**What it is**

This repository presents a Hodgkin-Huxley-based cortico-basal ganglia network model with closed-loop DBS control, built to study suppression of pathological beta oscillations in Parkinson's disease. The repository explicitly identifies its companion paper and asks users to cite that paper for academic use.

**Why it matters for MotorAssistEnv**

- It establishes the earlier closed-loop DBS modeling line that our project builds on.
- It shows the beta-suppression control framing that later evolves into the motor-network work.
- It is useful context for explaining why pathological beta oscillations are a meaningful control target in this benchmark.

**Best way to describe our dependence**

Use wording like:

> Our project is conceptually grounded in the Fleming closed-loop DBS modeling line, beginning with the cortico-basal ganglia network model for pathological beta suppression and extending into the later motor-network model that we calibrate directly.

**How the repo asks to be cited**

The repository says its code is companion to the following paper and that this paper should be cited for academic use. Source: [repo page](https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model), [citation lines](https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model).

**Recommended citation**

Fleming, J. E., Dunn, E., & Lowery, M. M. (2020). *Simulation of closed-loop deep brain stimulation control schemes for suppression of pathological beta oscillations in Parkinson's disease*. Frontiers in Neuroscience, 14, 166. https://doi.org/10.3389/fnins.2020.00166

**Reference links**

- Repo: https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model
- Paper DOI: https://doi.org/10.3389/fnins.2020.00166

## 2. Fleming Motor Network Model

**Repository**

- [GitHub repo](https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model)

**What it is**

This repository contains the motor-network model most directly relevant to our environment. According to the repo description, it simulates a Parkinsonian motor network coupling cortico-basal ganglia circuitry to a motoneuron pool and produces clinically relevant outputs including LFP, EMG, and force. It also includes a multivariable adaptive control strategy for tremor and motor impairment biomarkers.

**Why it matters for MotorAssistEnv**

- This is the core scientific source behind our calibration pipeline.
- Our benchmark uses outputs derived from this modeling family to construct the environment’s grounded state variables and entrainment surface.
- The force, tremor, sEMG, beta, and stimulation relationships in the benchmark ultimately trace back here.

**Best way to describe our dependence**

Use wording like:

> MotorAssistEnv is calibrated from outputs of the Fleming motor-network model, which couples cortico-basal ganglia circuitry to a motoneuron pool and generates the LFP, EMG, tremor, force, and stimulation signals that anchor our benchmark dynamics.

**How the repo asks to be cited**

The repository states that the code is companion to the following paper and that it should be cited for academic use. Source: [repo page](https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model).

**Recommended citation**

Fleming, J. E., Senneff, S., & Lowery, M. M. (2023). *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*. Journal of Neural Engineering, 20(5), 056029. https://doi.org/10.1088/1741-2552/acfbfa

**Reference links**

- Repo: https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model
- Paper DOI: https://doi.org/10.1088/1741-2552/acfbfa
- PubMed landing page: https://pubmed.ncbi.nlm.nih.gov/37733003/

## 3. MyoSuite Demo

**Repository**

- [GitHub repo](https://github.com/MyoHub/myosuite_demo)

**What it is**

This repository is a browser-based interactive demonstration of MyoSuite musculoskeletal models. The repo describes itself as an interactive demo of the musculoskeletal models from the MyoSuite framework.

**Why it matters for MotorAssistEnv**

- It influenced how we thought about making the environment visually legible to judges and readers.
- It helped shape the idea of separating fast benchmark logic from a richer visual demonstration layer.
- It supports our “visualization as communication, not training-time physics” design choice.

**How to reference it**

This repo does not present a separate paper citation request on the repository page in the same way the Fleming and RL-DBS repos do. The safest and cleanest approach is:

- reference the repository directly as software inspiration
- separately cite MyoSuite itself as the underlying framework when discussing musculoskeletal simulation context

**Suggested software acknowledgement**

MyoHub. *myosuite_demo* [software repository]. GitHub. https://github.com/MyoHub/myosuite_demo

**Reference links**

- Repo: https://github.com/MyoHub/myosuite_demo

## 4. MyoSuite

**Repository**

- [GitHub repo](https://github.com/MyoHub/myosuite)

**What it is**

MyoSuite is a collection of musculoskeletal environments and tasks built on MuJoCo and wrapped in the OpenAI Gym API. It is a major reference point for benchmark-style musculoskeletal control environments.

**Why it matters for MotorAssistEnv**

- It informed the benchmark presentation style and environment framing.
- It influenced our separation between the benchmark backend and the musculoskeletal visualization layer.
- It provides precedent for treating biomechanical simulation as a rigorous ML benchmark rather than a purely visual demo.

**How the project asks to be cited**

The MyoSuite documentation and repo point users to cite the project’s arXiv paper. Sources: [repo page](https://github.com/MyoHub/myosuite), [documentation citation section](https://myosuite.readthedocs.io/en/stable/).

**Recommended citation**

Caggiano, V., Wang, H., Durandau, G., Sartori, M., & Kumar, V. (2022). *MyoSuite: A contact-rich simulation suite for musculoskeletal motor control*. arXiv. https://doi.org/10.48550/arXiv.2205.13600

**Suggested software acknowledgement**

MyoHub. *myosuite* [software repository]. GitHub. https://github.com/MyoHub/myosuite

**Reference links**

- Repo: https://github.com/MyoHub/myosuite
- Docs: https://myosuite.readthedocs.io/en/stable/
- Paper: https://arxiv.org/abs/2205.13600

## 5. RL-DBS

**Repository**

- [GitHub repo](https://github.com/cviaai/RL-DBS)

**What it is**

RL-DBS is a reinforcement-learning repository focused on DBS control for suppressing collective neuronal activity. The repo emphasizes physically meaningful reward functions and benchmark-style comparison across different physical models.

**Why it matters for MotorAssistEnv**

- It is a useful prior-art reference for RL in DBS.
- It supports the claim that RL-for-DBS is a legitimate research direction rather than a purely speculative application.
- It helped shape how we talk about reward design, controller comparison, and benchmark headroom.

**How the repo asks to be cited**

The repository explicitly asks users to cite two papers:

- an IJCAI 2020 paper on the RL framework for DBS study
- a Chaos 2020 paper on RL for suppression of collective activity in oscillatory ensembles

Source: [repo page](https://github.com/cviaai/RL-DBS)

**Recommended primary citation**

Krylov, D., des Combes, R., Laroche, R., Rosenblum, M., & Dylov, D. V. (2020). *Reinforcement Learning Framework for Deep Brain Stimulation Study*. In *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20)* (pp. 2847-2854). https://doi.org/10.24963/ijcai.2020/394

**Recommended secondary citation**

Krylov, D., Dylov, D. V., & Rosenblum, M. (2020). *Reinforcement learning for suppression of collective activity in oscillatory ensembles*. Chaos: An Interdisciplinary Journal of Nonlinear Science, 30(3), 033126. https://doi.org/10.1063/1.5128909

**Suggested software acknowledgement**

cviaai. *RL-DBS* [software repository]. GitHub. https://github.com/cviaai/RL-DBS

**Reference links**

- Repo: https://github.com/cviaai/RL-DBS
- IJCAI paper: https://www.ijcai.org/proceedings/2020/394

## How These Sources Map Onto Our Project

| Our project area | Main source(s) |
|---|---|
| Parkinsonian neural and motor signal grounding | Fleming motor-network repo and 2023 JNE paper |
| Earlier closed-loop DBS scientific lineage | Fleming cortical-basal-ganglia repo and 2020 Frontiers paper |
| Musculoskeletal demo and visual communication | myosuite_demo |
| Musculoskeletal benchmark framing | MyoSuite and the MyoSuite paper |
| RL-for-DBS prior art and reward/control benchmark framing | RL-DBS and its cited papers |

## Suggested Acknowledgement Text

### Short version

This project builds on the closed-loop DBS modeling work of John E. Fleming and collaborators, especially the motor-network model for Parkinson's disease and deep brain stimulation. It also draws inspiration from MyoSuite and MyoSuite Demo for musculoskeletal benchmark presentation and visualization, and from RL-DBS as prior art in reinforcement-learning-based DBS control.

### README 

MotorAssistEnv is scientifically grounded in the Parkinsonian DBS modeling line developed by John E. Fleming and collaborators, especially the motor-network model reported in *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease* (Journal of Neural Engineering, 2023). For visual communication and musculoskeletal benchmark presentation, we also drew inspiration from MyoSuite and MyoSuite Demo. For RL-for-DBS prior art, we reference RL-DBS and its accompanying publications on reinforcement learning for DBS study and oscillatory suppression.

## Recommended Citation Block For This Repo

If you need one compact “References” section for a README, Devpost page, or report, this is the cleanest version:

1. Fleming, J. E., Dunn, E., & Lowery, M. M. (2020). *Simulation of closed-loop deep brain stimulation control schemes for suppression of pathological beta oscillations in Parkinson's disease*. Frontiers in Neuroscience, 14, 166. https://doi.org/10.3389/fnins.2020.00166
2. Fleming, J. E., Senneff, S., & Lowery, M. M. (2023). *Multivariable closed-loop control of deep brain stimulation for Parkinson's disease*. Journal of Neural Engineering, 20(5), 056029. https://doi.org/10.1088/1741-2552/acfbfa
3. Caggiano, V., Wang, H., Durandau, G., Sartori, M., & Kumar, V. (2022). *MyoSuite: A contact-rich simulation suite for musculoskeletal motor control*. arXiv. https://doi.org/10.48550/arXiv.2205.13600
4. Krylov, D., des Combes, R., Laroche, R., Rosenblum, M., & Dylov, D. V. (2020). *Reinforcement Learning Framework for Deep Brain Stimulation Study*. IJCAI-20. https://doi.org/10.24963/ijcai.2020/394
5. Krylov, D., Dylov, D. V., & Rosenblum, M. (2020). *Reinforcement learning for suppression of collective activity in oscillatory ensembles*. Chaos, 30(3), 033126. https://doi.org/10.1063/1.5128909

Software references:

- John-E-Fleming. *Parkinsons_Cortical_Basal_Ganglia_Network_Model* [software repository]. GitHub. https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model
- John-E-Fleming. *Parkinsons_Motor_Network_Model* [software repository]. GitHub. https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model
- MyoHub. *myosuite_demo* [software repository]. GitHub. https://github.com/MyoHub/myosuite_demo
- MyoHub. *myosuite* [software repository]. GitHub. https://github.com/MyoHub/myosuite
- cviaai. *RL-DBS* [software repository]. GitHub. https://github.com/cviaai/RL-DBS

## Sources Used For This Document

- Fleming cortical repo: https://github.com/John-E-Fleming/Parkinsons_Cortical_Basal_Ganglia_Network_Model
- Fleming motor repo: https://github.com/John-E-Fleming/Parkinsons_Motor_Network_Model
- MyoSuite repo: https://github.com/MyoHub/myosuite
- MyoSuite docs: https://myosuite.readthedocs.io/en/stable/
- MyoSuite demo repo: https://github.com/MyoHub/myosuite_demo
- RL-DBS repo: https://github.com/cviaai/RL-DBS
