"""HTGR reactor physics engine: point kinetics + 3-node thermal core.

Direct translation of the MATLAB/Simulink ``Calculate_Kinetics`` and
``core_thermals`` blocks from the Imperial HTGR-5 scaled-HTTR model into a
single coupled ODE system, integrated with ``scipy.integrate.solve_ivp``.

Reference: "Scaled High Temperature Test Reactor Cogeneration for Zero
Emission Hybrid-Desalination with Independent Water Heat Rejection",
Imperial College London, March 2026 - Part I, Sections 1.3, 2.1-2.3;
Appendix B, Listings 1-2.

Notes on fidelity to the source model
--------------------------------------
* All constants below (beta_i, lambda_i, prompt neutron lifetime, thermal
  capacities, heat-transfer coefficients, helium cp) are copied verbatim
  from the MATLAB appendix listings, including the literal factor of 2 on
  the coolant convective-removal term in ``core_thermals.m``.
* The appendix's ``Calculate_Kinetics`` function takes reactivity as a
  single external input ``rho_in`` — despite the Simulink block diagram
  (Fig. 1) showing Tf/Tm feeding back into the kinetics block, the
  provided code has no internal temperature-feedback term. This port
  matches the code as written: reactivity is a pure function of time,
  supplied via the ``reactivity`` callable. A temperature-feedback
  (Doppler/moderator coefficient) term can be added by passing a callable
  of the form ``rho(t, state)`` — see ``HTGRReactorModel.rhs``.
* The report's Fig. 3 validation plot benchmarks the thermal-core block
  against literature HTTR data at what appears to be a different
  (lower/normalised) operating point than this project's 300 MWth design
  point — the appendix gives no numeric value for the reactivity step
  used to produce that figure. Rather than guess pixel values off a
  scanned plot, initial conditions here are computed *analytically* as
  the true steady state of the 3-node thermal network for whatever
  power/flow/inlet-temperature you specify (see ``steady_state``), and
  the reactivity step size is an explicit, clearly-labelled placeholder
  you should calibrate against your own target transient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

import numpy as np
from scipy.integrate import solve_ivp

N_GROUPS = 6

# Standard 6-group U-235 delayed-neutron data, from Calculate_Kinetics.m
BETA_I = np.array([0.00025, 0.00138, 0.00122, 0.00266, 0.00164, 0.00035])
LAMBDA_I = np.array([0.0766, 0.2825, 0.6154, 1.634, 5.176, 16.72])


@dataclass(frozen=True)
class KineticsParams:
    """Point-kinetics constants (``Calculate_Kinetics.m``)."""

    beta_i: np.ndarray = field(default_factory=lambda: BETA_I.copy())
    lambda_i: np.ndarray = field(default_factory=lambda: LAMBDA_I.copy())
    prompt_neutron_lifetime: float = 1.0e-3  # Lambda, s
    q_nominal: float = 300e6  # W, nominal reactor thermal power

    @property
    def beta_total(self) -> float:
        return float(np.sum(self.beta_i))


@dataclass(frozen=True)
class ThermalCoreParams:
    """3-node thermal-core constants (``core_thermals.m``)."""

    c_fuel: float = 5.0e6  # J/K
    c_moderator: float = 1.0e8  # J/K
    c_coolant: float = 1.0e6  # J/K
    h_fuel_moderator: float = 2.0e5  # W/K
    h_fuel_coolant: float = 4.0e5  # W/K
    h_moderator_coolant: float = 1.5e5  # W/K
    coolant_cp: float = 5193.0  # J/(kg K), helium


@dataclass(frozen=True)
class OperatingConditions:
    """Plant-level operating point (Part I, Section 1.3)."""

    helium_mass_flow: float = 105.0  # kg/s
    coolant_inlet_temp: float = 623.15  # K (350 degC compressor-outlet estimate)


ReactivityFunc = Callable[[float, float, float], float]  # (t, fuel_temp, moderator_temp) -> rho


def external_step_reactivity(t_step: float, rho_insertion: float) -> ReactivityFunc:
    """External reactivity only (rho_ext), exactly as ``Calculate_Kinetics.m``
    consumes its ``rho_in`` argument - no temperature feedback.

    WARNING: with no feedback, a sustained positive step causes power (and
    therefore temperature) to grow without bound - this is correct,
    expected point-kinetics behaviour for an open-loop reactivity
    insertion held constant forever, not a bug. It also means this
    reactivity function is only useful for short transients or negative
    steps. For a self-limiting transient, use
    ``doppler_feedback_reactivity`` instead.
    """

    def _rho(t: float, fuel_temp: float, moderator_temp: float) -> float:
        del fuel_temp, moderator_temp  # unused: no feedback, matches the literal MATLAB
        return rho_insertion if t >= t_step else 0.0

    return _rho


def doppler_feedback_reactivity(
    t_step: float,
    rho_insertion: float,
    fuel_temp_ref: float,
    moderator_temp_ref: float,
    alpha_fuel: float = -3.0e-5,
    alpha_moderator: float = -8.0e-6,
) -> ReactivityFunc:
    """External step reactivity plus linear fuel/moderator temperature
    feedback::

        rho(t) = rho_ext(t) + alpha_fuel * (Tf - Tf_ref)
                             + alpha_moderator * (Tm - Tm_ref)

    The appendix's ``Calculate_Kinetics.m`` listing takes reactivity as a
    single external input with no feedback term, even though the Simulink
    block diagram (Fig. 1) shows Tf/Tm feeding into the kinetics block -
    the feedback arithmetic itself isn't in the given listing. ``alpha_fuel``
    and ``alpha_moderator`` (Doppler and moderator temperature coefficients,
    1/K, both conventionally negative/self-stabilising) are NOT given
    numerically anywhere in the report; the defaults are illustrative
    magnitudes typical of a graphite-moderated HTGR, not extracted report
    data - tune them if you have real coefficients.
    """

    def _rho(t: float, fuel_temp: float, moderator_temp: float) -> float:
        rho_ext = rho_insertion if t >= t_step else 0.0
        return (
            rho_ext
            + alpha_fuel * (fuel_temp - fuel_temp_ref)
            + alpha_moderator * (moderator_temp - moderator_temp_ref)
        )

    return _rho


class ReactorState(NamedTuple):
    """Named view into a single time-slice of the ODE state vector."""

    power: float
    precursors: np.ndarray
    fuel_temp: float
    moderator_temp: float
    coolant_temp: float


class HTGRReactorModel:
    """Coupled point-kinetics / 3-node thermal-core reactor model.

    State vector ``y = [P, C1..C6, Tf, Tm, Tc]`` (10 states):

    * ``P``          normalised reactor power (P = 1 at nominal power)
    * ``C1..C6``     normalised delayed-neutron precursor concentrations
    * ``Tf, Tm, Tc`` fuel, moderator, coolant temperatures [K]
    """

    N_STATES = 1 + N_GROUPS + 3  # power + 6 precursors + 3 temperatures

    def __init__(
        self,
        kinetics: KineticsParams | None = None,
        thermal: ThermalCoreParams | None = None,
        operating: OperatingConditions | None = None,
        reactivity: ReactivityFunc | None = None,
    ) -> None:
        self.kinetics = kinetics or KineticsParams()
        self.thermal = thermal or ThermalCoreParams()
        self.operating = operating or OperatingConditions()

        if reactivity is not None:
            self.reactivity = reactivity
        else:
            # Default: a stable, self-limiting demo transient built around
            # this model's own steady state. See `doppler_feedback_reactivity`
            # docstring - the feedback coefficients are illustrative, not
            # report data. Swap in `external_step_reactivity` for the literal,
            # feedback-free translation of Calculate_Kinetics.m.
            ref = self.steady_state()
            self.reactivity = doppler_feedback_reactivity(
                t_step=100.0,
                rho_insertion=0.0022,
                fuel_temp_ref=ref[7],
                moderator_temp_ref=ref[8],
            )

    # ------------------------------------------------------------------
    # Equilibrium / steady-state helpers
    # ------------------------------------------------------------------
    def equilibrium_precursors(self, power: float = 1.0) -> np.ndarray:
        """Precursor concentrations at equilibrium for a steady ``power``.

        From dC_i/dt = 0: (beta_i / Lambda) * P = lambda_i * C_i
        => C_i = beta_i * P / (lambda_i * Lambda).
        """
        k = self.kinetics
        return (k.beta_i / (k.lambda_i * k.prompt_neutron_lifetime)) * power

    def steady_state(
        self,
        power: float = 1.0,
        coolant_inlet_temp: float | None = None,
        helium_mass_flow: float | None = None,
    ) -> np.ndarray:
        """Solve for the self-consistent equilibrium state at a given power.

        Sets dP/dt = 0 (precursors at their equilibrium values) and
        dTf/dt = dTm/dt = dTc/dt = 0 simultaneously, i.e. solves the 3x3
        linear heat-balance system implied by ``core_thermals.m`` for
        [Tf, Tm, Tc]. Use this to generate physically-consistent initial
        conditions for ``simulate`` at any operating point, rather than
        hard-coding approximate values.
        """
        p = self.thermal
        t_in = self.operating.coolant_inlet_temp if coolant_inlet_temp is None else coolant_inlet_temp
        mdot = self.operating.helium_mass_flow if helium_mass_flow is None else helium_mass_flow
        q_fission = power * self.kinetics.q_nominal
        removal = 2.0 * mdot * p.coolant_cp  # matches the literal "2*mdot*cp" in core_thermals.m

        # Rows: fuel, moderator, coolant steady energy balances.
        # Columns: [Tf, Tm, Tc].
        a = np.array(
            [
                [p.h_fuel_moderator + p.h_fuel_coolant, -p.h_fuel_moderator, -p.h_fuel_coolant],
                [p.h_fuel_moderator, -(p.h_fuel_moderator + p.h_moderator_coolant), p.h_moderator_coolant],
                [p.h_fuel_coolant, p.h_moderator_coolant, -(p.h_fuel_coolant + p.h_moderator_coolant + removal)],
            ]
        )
        b = np.array([q_fission, 0.0, -removal * t_in])
        t_fuel, t_moderator, t_coolant = np.linalg.solve(a, b)

        y0 = np.empty(self.N_STATES)
        y0[0] = power
        y0[1:7] = self.equilibrium_precursors(power)
        y0[7], y0[8], y0[9] = t_fuel, t_moderator, t_coolant
        return y0

    # ------------------------------------------------------------------
    # ODE right-hand sides
    # ------------------------------------------------------------------
    def _point_kinetics(
        self,
        t: float,
        power: float,
        precursors: np.ndarray,
        fuel_temp: float,
        moderator_temp: float,
    ) -> tuple[float, np.ndarray, float]:
        """``Calculate_Kinetics.m``, translated one-to-one (the reactivity
        callable additionally receives Tf/Tm so feedback models can use
        them; the literal appendix code itself has no feedback term)."""
        k = self.kinetics
        rho = self.reactivity(t, fuel_temp, moderator_temp)

        sum_lambda_c = float(np.dot(k.lambda_i, precursors))
        dp_dt = ((rho - k.beta_total) / k.prompt_neutron_lifetime) * power + sum_lambda_c
        dc_dt = (k.beta_i / k.prompt_neutron_lifetime) * power - k.lambda_i * precursors
        q_fission = power * k.q_nominal
        return dp_dt, dc_dt, q_fission

    def _thermal_core(
        self,
        q_fission: float,
        fuel_temp: float,
        moderator_temp: float,
        coolant_temp: float,
    ) -> tuple[float, float, float]:
        """``core_thermals.m``, translated one-to-one (including the
        literal factor of 2 on the coolant convective-removal term)."""
        p = self.thermal
        op = self.operating

        dtf_dt = (
            q_fission
            - p.h_fuel_moderator * (fuel_temp - moderator_temp)
            - p.h_fuel_coolant * (fuel_temp - coolant_temp)
        ) / p.c_fuel

        dtm_dt = (
            p.h_fuel_moderator * (fuel_temp - moderator_temp)
            - p.h_moderator_coolant * (moderator_temp - coolant_temp)
        ) / p.c_moderator

        dtc_dt = (
            p.h_fuel_coolant * (fuel_temp - coolant_temp)
            + p.h_moderator_coolant * (moderator_temp - coolant_temp)
            - 2.0 * op.helium_mass_flow * p.coolant_cp * (coolant_temp - op.coolant_inlet_temp)
        ) / p.c_coolant

        return dtf_dt, dtm_dt, dtc_dt

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Full coupled derivative vector, for ``solve_ivp``."""
        power = y[0]
        precursors = y[1:7]
        fuel_temp, moderator_temp, coolant_temp = y[7], y[8], y[9]

        dp_dt, dc_dt, q_fission = self._point_kinetics(t, power, precursors, fuel_temp, moderator_temp)
        dtf_dt, dtm_dt, dtc_dt = self._thermal_core(q_fission, fuel_temp, moderator_temp, coolant_temp)

        dy = np.empty_like(y)
        dy[0] = dp_dt
        dy[1:7] = dc_dt
        dy[7], dy[8], dy[9] = dtf_dt, dtm_dt, dtc_dt
        return dy

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def simulate(
        self,
        t_span: tuple[float, float] = (0.0, 400.0),
        y0: np.ndarray | None = None,
        t_eval: np.ndarray | None = None,
        method: str = "Radau",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> Any:
        """Integrate the coupled system over ``t_span``.

        Defaults to an implicit stiff solver ("Radau"): the ~1 ms prompt-
        neutron timescale (Lambda) and the ~10^2-10^3 s moderator thermal
        timescale (C_m / h_mc) span more than five orders of magnitude,
        which explicit solvers handle poorly.
        """
        if y0 is None:
            y0 = self.steady_state()
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 2001)

        return solve_ivp(
            fun=self.rhs,
            t_span=t_span,
            y0=y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )

    @staticmethod
    def unpack(sol: Any) -> dict[str, np.ndarray]:
        """Convenience accessor: a ``solve_ivp`` result -> named arrays."""
        y = sol.y
        return {
            "t": sol.t,
            "power": y[0],
            "precursors": y[1:7],
            "fuel_temp": y[7],
            "moderator_temp": y[8],
            "coolant_temp": y[9],
        }

    def state_at(self, sol: Any, index: int) -> ReactorState:
        """Named view of the solution at a given time index."""
        y = sol.y[:, index]
        return ReactorState(
            power=y[0],
            precursors=y[1:7],
            fuel_temp=y[7],
            moderator_temp=y[8],
            coolant_temp=y[9],
        )


if __name__ == "__main__":
    model = HTGRReactorModel()
    solution = model.simulate()

    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")

    results = model.unpack(solution)
    peak_fuel_idx = int(np.argmax(results["fuel_temp"]))
    peak_mod_idx = int(np.argmax(results["moderator_temp"]))

    print(f"Solver: {solution.status} ({solution.message}); {len(solution.t)} output points")
    print(f"Steady-state Tf/Tm/Tc at t=0: "
          f"{results['fuel_temp'][0]:.2f} K / "
          f"{results['moderator_temp'][0]:.2f} K / "
          f"{results['coolant_temp'][0]:.2f} K")
    print(f"Peak fuel temp:      {results['fuel_temp'][peak_fuel_idx]:.2f} K "
          f"at t = {results['t'][peak_fuel_idx]:.1f} s")
    print(f"Peak moderator temp: {results['moderator_temp'][peak_mod_idx]:.2f} K "
          f"at t = {results['t'][peak_mod_idx]:.1f} s")
    print(f"Final coolant temp:  {results['coolant_temp'][-1]:.2f} K")
