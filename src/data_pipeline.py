"""Monte Carlo telemetry pipeline for the HTGR physics engine.

Samples realistic HTGR operating points, runs the coupled point-kinetics /
3-node thermal-core model from ``htgr_physics.py`` for each one, chains a
CoolProp-based helium turbine expansion onto the resulting reactor-outlet
gas (a direct Python port of Appendix B's ``helium_turbine.m``, mirroring
the Core-Thermals -> Helium-Turbine block chain in Fig. 5 of the report),
and writes the combined telemetry to ``data/htgr_telemetry.csv`` as a
pandas DataFrame.

Scope note: the turbine work calculation is a separate, clearly-labelled
post-processing step applied to each run's final reactor-outlet
temperature - it is NOT part of the coupled ODE system built in Step 1
and does not feed back into the reactor's thermal/kinetic response.

Design note on "randomised initial conditions": rather than sampling raw
Tf/Tm/Tc values disconnected from the energy balance (which would inject
an unphysical initial transient into every run and contaminate the peak
extraction), each scenario randomises the *operating point*
(helium flow, coolant inlet temperature) and derives a physically
self-consistent starting state via ``HTGRReactorModel.steady_state()``
before applying its randomised reactivity step.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from htgr_physics import (  # noqa: E402
    HTGRReactorModel,
    KineticsParams,
    OperatingConditions,
    ThermalCoreParams,
    doppler_feedback_reactivity,
)

N_RUNS = 1000
SEED = 42
T_SPAN = (0.0, 400.0)
T_EVAL = np.linspace(*T_SPAN, 401)  # 1 s resolution; enough to resolve the peaks

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "htgr_telemetry.csv"

# ---------------------------------------------------------------------------
# Realistic sampling bounds
# ---------------------------------------------------------------------------
# Helium mass flow: report's scaled operating value is 105 kg/s (Part I,
# Sec. 1.3); swept +/-25% around it.
HELIUM_FLOW_BOUNDS = (78.0, 132.0)  # kg/s

# Coolant (reactor) inlet temperature: compressor-outlet estimate 350 degC =
# 623.15 K (Appendix A, Table 9, primary-loop point 2); swept +/-30 K.
COOLANT_INLET_BOUNDS = (593.15, 653.15)  # K

# Reactivity step magnitude: kept well inside +/-beta_total (0.0075) so every
# run stays sub-prompt-critical; sign varies to cover both power up- and
# down-transients.
RHO_INSERTION_BOUNDS = (-0.0030, 0.0030)
RHO_STEP_TIME_BOUNDS = (50.0, 150.0)  # s; centred on the report's t = 100 s

# Turbine inlet / back pressure (Appendix A, Table 9: reactor outlet ~70 bar,
# gas turbine exit ~45 bar) and isentropic efficiency (not given numerically
# in the report; 0.85-0.92 is a typical industrial range) - sampled
# independently to explore pressure-drop sensitivity across the turbine.
TURBINE_P_IN_BOUNDS = (63e5, 77e5)  # Pa (~63-77 bar)
TURBINE_P_BACK_BOUNDS = (38e5, 50e5)  # Pa (~38-50 bar)
TURBINE_ETA_BOUNDS = (0.85, 0.92)


@dataclass(frozen=True)
class Scenario:
    """One sampled operating point + perturbation."""

    run_id: int
    helium_mass_flow_kgs: float
    coolant_inlet_temp_K: float
    rho_insertion: float
    rho_step_time_s: float
    turbine_p_in_Pa: float
    turbine_p_back_Pa: float
    turbine_eta_is: float


def sample_scenarios(n_runs: int, rng: np.random.Generator) -> list[Scenario]:
    """Uniform-random sampling within the realistic bounds above."""
    scenarios = []
    for i in range(n_runs):
        p_in = rng.uniform(*TURBINE_P_IN_BOUNDS)
        p_back = rng.uniform(*TURBINE_P_BACK_BOUNDS)
        p_back = min(p_back, 0.85 * p_in)  # guarantee a valid expansion (P_back < P_in)

        scenarios.append(
            Scenario(
                run_id=i,
                helium_mass_flow_kgs=float(rng.uniform(*HELIUM_FLOW_BOUNDS)),
                coolant_inlet_temp_K=float(rng.uniform(*COOLANT_INLET_BOUNDS)),
                rho_insertion=float(rng.uniform(*RHO_INSERTION_BOUNDS)),
                rho_step_time_s=float(rng.uniform(*RHO_STEP_TIME_BOUNDS)),
                turbine_p_in_Pa=float(p_in),
                turbine_p_back_Pa=float(p_back),
                turbine_eta_is=float(rng.uniform(*TURBINE_ETA_BOUNDS)),
            )
        )
    return scenarios


def helium_turbine_work(
    t_in_k: float, p_in_pa: float, p_back_pa: float, m_dot: float, eta_is: float
) -> tuple[float, float]:
    """Direct Python/CoolProp port of Appendix B's ``helium_turbine.m``.

    Isentropic expansion of helium from (T_in, P_in) to P_back with
    isentropic efficiency ``eta_is``. Returns (work_W, T_out_K).
    """
    fluid = "Helium"
    h_in = PropsSI("H", "T", t_in_k, "P", p_in_pa, fluid)
    s_in = PropsSI("S", "T", t_in_k, "P", p_in_pa, fluid)
    h_out_isentropic = PropsSI("H", "S", s_in, "P", p_back_pa, fluid)
    h_out = h_in - eta_is * (h_in - h_out_isentropic)
    t_out = PropsSI("T", "H", h_out, "P", p_back_pa, fluid)
    work_w = m_dot * (h_in - h_out)
    return work_w, t_out


def _failed_run_record(scenario: Scenario, sol_message: str) -> dict:
    record = asdict(scenario)
    record["solver_success"] = False
    record["solver_message"] = sol_message
    for col in (
        "initial_fuel_temp_K", "initial_moderator_temp_K", "initial_coolant_temp_K",
        "peak_fuel_temp_K", "peak_moderator_temp_K", "peak_coolant_temp_K",
        "final_power_norm", "final_fuel_temp_K", "final_moderator_temp_K",
        "final_coolant_temp_K", "turbine_pressure_ratio", "turbine_work_MW",
        "turbine_outlet_temp_K",
    ):
        record[col] = np.nan
    return record


def run_scenario(scenario: Scenario) -> dict:
    """Run one coupled kinetics/thermal-core simulation, then chain the
    turbine work calculation onto its final reactor-outlet gas state."""
    operating = OperatingConditions(
        helium_mass_flow=scenario.helium_mass_flow_kgs,
        coolant_inlet_temp=scenario.coolant_inlet_temp_K,
    )
    kinetics = KineticsParams()
    thermal = ThermalCoreParams()

    # Physically self-consistent starting point for THIS scenario's
    # operating conditions (see module docstring).
    probe = HTGRReactorModel(kinetics, thermal, operating, reactivity=lambda t, tf, tm: 0.0)
    y0 = probe.steady_state()

    reactivity = doppler_feedback_reactivity(
        t_step=scenario.rho_step_time_s,
        rho_insertion=scenario.rho_insertion,
        fuel_temp_ref=y0[7],
        moderator_temp_ref=y0[8],
    )
    model = HTGRReactorModel(kinetics, thermal, operating, reactivity=reactivity)

    sol = model.simulate(t_span=T_SPAN, y0=y0, t_eval=T_EVAL)
    if not sol.success:
        return _failed_run_record(scenario, sol.message)

    results = model.unpack(sol)
    record = asdict(scenario)
    record.update(
        {
            "solver_success": True,
            "solver_message": sol.message,
            "initial_fuel_temp_K": float(results["fuel_temp"][0]),
            "initial_moderator_temp_K": float(results["moderator_temp"][0]),
            "initial_coolant_temp_K": float(results["coolant_temp"][0]),
            "peak_fuel_temp_K": float(np.max(results["fuel_temp"])),
            "peak_moderator_temp_K": float(np.max(results["moderator_temp"])),
            "peak_coolant_temp_K": float(np.max(results["coolant_temp"])),
            "final_power_norm": float(results["power"][-1]),
            "final_fuel_temp_K": float(results["fuel_temp"][-1]),
            "final_moderator_temp_K": float(results["moderator_temp"][-1]),
            "final_coolant_temp_K": float(results["coolant_temp"][-1]),
            "turbine_pressure_ratio": scenario.turbine_p_in_Pa / scenario.turbine_p_back_Pa,
        }
    )

    # Turbine work on the post-transient (t = 400 s) reactor-outlet gas.
    # Note: the moderator's ~600+ s thermal time constant means it has not
    # fully re-equilibrated by t = 400 s in most runs - "final" here means
    # "end of the simulated window", not new steady state.
    try:
        work_w, t_out_k = helium_turbine_work(
            t_in_k=record["final_coolant_temp_K"],
            p_in_pa=scenario.turbine_p_in_Pa,
            p_back_pa=scenario.turbine_p_back_Pa,
            m_dot=scenario.helium_mass_flow_kgs,
            eta_is=scenario.turbine_eta_is,
        )
        record["turbine_work_MW"] = work_w / 1e6
        record["turbine_outlet_temp_K"] = t_out_k
    except ValueError:
        # CoolProp can raise outside its valid range for extreme sampled
        # combinations - record as missing rather than crashing the sweep.
        record["turbine_work_MW"] = np.nan
        record["turbine_outlet_temp_K"] = np.nan

    return record


def main() -> None:
    rng = np.random.default_rng(SEED)
    scenarios = sample_scenarios(N_RUNS, rng)

    print(f"Running {N_RUNS} HTGR scenarios...")
    start = time.perf_counter()
    records = []
    for i, scenario in enumerate(scenarios, start=1):
        records.append(run_scenario(scenario))
        if i % 100 == 0:
            elapsed = time.perf_counter() - start
            print(f"  {i}/{N_RUNS} runs complete ({elapsed:.1f}s elapsed)")

    df = pd.DataFrame.from_records(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    elapsed = time.perf_counter() - start
    n_ok = int(df["solver_success"].sum())
    n_turbine_ok = int(df["turbine_work_MW"].notna().sum())
    print(f"\nDone: {len(df)} runs in {elapsed:.1f}s.")
    print(f"  ODE solver succeeded:      {n_ok}/{len(df)}")
    print(f"  Turbine calc succeeded:    {n_turbine_ok}/{len(df)}")
    print(f"  Wrote: {OUTPUT_PATH}")

    numeric_cols = [
        "peak_fuel_temp_K", "peak_moderator_temp_K", "peak_coolant_temp_K",
        "final_coolant_temp_K", "turbine_work_MW", "turbine_outlet_temp_K",
    ]
    print("\nSummary of key columns:")
    print(df[numeric_cols].describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
