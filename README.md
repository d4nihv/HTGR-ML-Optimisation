# HTGR Reactor Physics-to-ML Pipeline

> A nuclear reactor's coupled point-kinetics / 3-node thermal-core model, taken from a MATLAB/Simulink design study and rebuilt as a scalable, quantitative Python stack — stiff-ODE physics engine → Monte Carlo telemetry → probabilistic ML surrogate wrapped in an autonomous safety agent.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-solve__ivp-8CAAE6?style=flat&logo=scipy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Monte%20Carlo%20Telemetry-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Probabilistic%20Surrogate-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Interactive%20Notebook-F37626?style=flat&logo=jupyter&logoColor=white)

---

## Overview

This repository modernizes the reactor physics core of the **Imperial College London HTGR-5 / HR-5 project** (*"Scaled High Temperature Test Reactor Cogeneration for Zero Emission Hybrid-Desalination"*, March 2026) — a 300 MWth scaled High-Temperature Gas-Cooled Reactor (HTGR) design originally simulated in MATLAB/Simulink — into a modular, dependency-light Python pipeline.

Rather than a line-for-line port, this is a full re-architecture: the original `Calculate_Kinetics.m` and `core_thermals.m` Simulink blocks become a single stiff coupled ODE system solved with SciPy; a 1,000-run Monte Carlo sweep over realistic operating envelopes generates training telemetry with pandas; and a scalable kernel-approximation surrogate (scikit-learn) replaces the O(N³) cost of an exact Gaussian Process, wrapped in a small autonomous agent that makes an explicit accept/flag/reject safety call on every candidate operating point — with a documented reason why, every time.

Every number quoted below is from an actual run of this code, not an estimate.

## Core Architecture

### 1. `src/htgr_physics.py` — Stiff Coupled ODE Physics Engine
A direct translation of the report's 6-group point-kinetics and 3-node fuel/moderator/coolant thermal-core equations into a single 10-state ODE system (`P, C₁..C₆, T_fuel, T_moderator, T_coolant`), integrated with `scipy.integrate.solve_ivp` using the implicit **Radau** method — required because the ~1 ms prompt-neutron lifetime and the ~10²–10³ s moderator thermal timescale span more than five orders of magnitude. Initial conditions aren't guessed off a plot; `steady_state()` solves the 3×3 linear heat-balance system analytically for a self-consistent starting point at any operating condition. The literal source kinetics block has no temperature feedback and diverges under a sustained reactivity step — a Doppler/moderator feedback term was added (clearly labeled as an addition, not extracted report data) to make the model physically stable and usable downstream.

### 2. `src/data_pipeline.py` — Monte Carlo Telemetry Generator
Samples 1,000 operating points across realistic bounds (helium mass flow, coolant inlet temperature, reactivity step size/timing, turbine inlet/back pressure, isentropic efficiency), runs each through the physics engine from its own physically self-consistent steady state, then chains a **CoolProp-based helium turbine expansion** (a Python port of the report's `helium_turbine.m`) onto the reactor-outlet gas to compute turbine work. All 1,000 runs converged — 0 solver failures, 0 NaNs — and the results are structured into a clean pandas DataFrame and written to `data/htgr_telemetry.csv`.

### 3. `src/ml_surrogate.py` — Scalable Probabilistic Surrogate + Autonomous Agent
A full `GaussianProcessRegressor` scales as **O(N³)** in training-set size — it stops being usable well before you're generating tens of thousands of Monte Carlo rows. Each target here instead gets a `StandardScaler → Nystroem(RBF) → BayesianRidge` pipeline: **Nystroem approximates the RBF kernel's feature map with a small number of random components**, and `BayesianRidge` performs closed-form Bayesian linear regression on top of it — giving a genuine posterior **mean and standard deviation** at **O(N·k²)** cost instead of a full kernel-matrix factorization.

`HTGRAgent` wraps the trained surrogate with three independent, explicit checks on every candidate operating point:
- **Safety-margin breach** — predicted peak fuel temperature (mean + 2σ) against a 1650 K limit.
- **High model uncertainty** — a data-driven threshold derived from the surrogate's own held-out predictive-std distribution.
- **Autonomous diagnostic fallback: out-of-distribution / extrapolation guard** — a deterministic training-envelope check, added because Nystroem+BayesianRidge does **not** guarantee its self-reported uncertainty grows with distance from training data the way a true GP's does. When any check fires, the agent doesn't just flag it — it prints a root-cause diagnostic and generates physics-grounded recommendations (increase coolant flow, reduce reactivity insertion, lower inlet temperature) to pull the operating point back into a safe envelope.

### 4. `notebooks/HTGR_Analysis.ipynb` — Interactive Analysis Notebook
The visual walkthrough: an executive-summary narrative, exploratory scatter/heatmap analysis of the Monte Carlo telemetry, a predicted-vs-actual parity plot with **±2σ confidence bands** against a held-out test split, and the agent evaluating one in-distribution and one out-of-distribution scenario live in the notebook output. Built programmatically via `nbformat` and executed end-to-end (`jupyter nbconvert --execute`) before being committed, so every cell's output is verified to run clean.

## Results at a Glance

| | Peak Fuel Temperature | Turbine Work Output |
|---|---|---|
| Test R² | 0.9946 | 1.0000 |
| Test RMSE | 4.85 K | 0.0033 MW |
| Mean predictive σ | 4.88 K | 0.0025 MW |
| 95% CI coverage (±2σ) | 96.5% | 92.0% |

*(`turbine_work_MW` fitting to R²≈1.0 reflects that the telemetry comes from a deterministic simulator with no injected observation noise, not data leakage — turbine work is a smooth composition of the physics engine's output through CoolProp's helium enthalpy relations.)*

- **1,000 / 1,000** Monte Carlo simulations converged.
- Cross-validated hyperparameter search (`GridSearchCV`, 5-fold) over Nystroem `gamma` and `n_components` for each target independently.
- The `HTGRAgent` demo correctly clears a moderate in-distribution scenario (**NOMINAL**, 1485.6 K predicted) and correctly rejects a low-flow / hot-inlet / high-reactivity scenario (**UNSAFE**, 1806.1 K predicted), citing all three of its diagnostic checks simultaneously.

## Quick Start

```bash
git clone <this-repo-url>
cd HTGR-ML-Optimization

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Regenerate the pipeline end-to-end (data/ and models/ are gitignored,
# not committed, so this is the first thing to run on a fresh clone).
python src/data_pipeline.py

# Train + save the surrogate. Must be run as an IMPORTED module, not a
# direct script - see "Known gotcha" below.
python -c "import sys; sys.path.insert(0, 'src'); import ml_surrogate; ml_surrogate.main()"

# Explore the results interactively
jupyter notebook notebooks/HTGR_Analysis.ipynb
```

## Project Structure

```
HTGR-ML-Optimization/
├── src/
│   ├── htgr_physics.py      # Point kinetics + 3-node thermal core (SciPy ODE)
│   ├── data_pipeline.py     # 1,000-run Monte Carlo telemetry -> CSV
│   └── ml_surrogate.py      # Nystroem+BayesianRidge surrogate + HTGRAgent
├── notebooks/
│   └── HTGR_Analysis.ipynb  # Executive summary, EDA, uncertainty plots, agent demo
├── data/                    # Generated Monte Carlo telemetry (gitignored)
├── models/                  # Trained surrogate artifact (gitignored)
├── requirements.txt
└── README.md
```

## Engineering Notes

A few decisions worth calling out explicitly rather than leaving implicit:

- **Added, not invented, physics.** The source MATLAB kinetics block takes reactivity as a bare external input with no temperature feedback. Translated literally, a sustained positive reactivity step causes power — and temperature — to grow without bound; this was confirmed empirically (a first test run hit ~10⁴³ K) before a Doppler/moderator feedback term was added as the default. It's documented in the module as an engineering addition, with the literal feedback-free translation still available (`external_step_reactivity`) for anyone who wants it.
- **The extrapolation guard exists because the uncertainty model has a real limitation.** A Bayesian linear model on a fixed random feature map doesn't inherit a true GP's distance-aware uncertainty growth. `HTGRAgent` therefore never relies on the surrogate's self-reported σ alone for out-of-distribution detection — it also runs a deterministic bounds check against the training envelope, and both signals are surfaced independently in the diagnostic output.
- **Known gotcha: joblib + `__main__`.** Running `python src/ml_surrogate.py` directly makes Python register the `TargetModel`/`HTGRSurrogate` dataclasses under the `__main__` module, so a model saved that way fails to unpickle anywhere else (`AttributeError: Can't get attribute 'TargetModel' on <module '__main__'>` — hit and fixed during development). Always regenerate `models/htgr_surrogate.joblib` via `import ml_surrogate; ml_surrogate.main()`, as shown in Quick Start above.

---

*Based on the reactor design and MATLAB/Simulink model from the Imperial College London HTGR-5 / HR-5 project report, "Scaled High Temperature Test Reactor Cogeneration for Zero Emission Hybrid-Desalination with Independent Water Heat Rejection" (March 2026).*
