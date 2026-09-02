"""Streamlit dashboard: interactively query the HTGR Nystroem+BayesianRidge
surrogate through the same ``HTGRAgent`` used in ``ml_surrogate.py``'s
``__main__`` demo and ``notebooks/HTGR_Analysis.ipynb``.

No model or agent logic lives here - this is a thin UI layer over
``src.ml_surrogate``, imported directly per project convention. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_pipeline import (  # noqa: E402
    COOLANT_INLET_BOUNDS,
    HELIUM_FLOW_BOUNDS,
    RHO_INSERTION_BOUNDS,
)
from src.htgr_physics import KineticsParams, OperatingConditions  # noqa: E402
from src.ml_surrogate import (  # noqa: E402
    FUEL_TEMP_SAFETY_MARGIN_K,
    MODEL_PATH,
    HTGRAgent,
    HTGRSurrogate,
)

# Fixed (non-slider) inputs, held at the report's design values (Appendix A,
# Table 9) - not swept here to keep the dashboard to the 3 controls asked for.
FIXED_RHO_STEP_TIME_S = 100.0
FIXED_TURBINE_P_IN_PA = 70.0e5
FIXED_TURBINE_P_BACK_PA = 45.0e5
FIXED_TURBINE_ETA_IS = 0.885

st.set_page_config(page_title="HTGR Operating Point Explorer", page_icon="⚛️", layout="wide")


@st.cache_resource
def load_surrogate() -> HTGRSurrogate:
    if not MODEL_PATH.exists():
        st.error(
            f"No trained model found at `{MODEL_PATH}`.\n\n"
            "Run this first, from the project root:\n\n"
            '`python -c "from src.ml_surrogate import main; main()"`'
        )
        st.stop()
    return joblib.load(MODEL_PATH)


surrogate = load_surrogate()
agent = HTGRAgent(surrogate)
nominal = OperatingConditions()
beta_total = KineticsParams().beta_total

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Operating Point")

mass_flow = st.sidebar.slider(
    "Helium mass flow (kg/s)",
    min_value=float(HELIUM_FLOW_BOUNDS[0]),
    max_value=float(HELIUM_FLOW_BOUNDS[1]),
    value=float(nominal.helium_mass_flow),
    step=0.5,
)

inlet_temp = st.sidebar.slider(
    "Coolant inlet temperature (K)",
    min_value=float(COOLANT_INLET_BOUNDS[0]),
    max_value=float(COOLANT_INLET_BOUNDS[1]),
    value=float(nominal.coolant_inlet_temp),
    step=0.5,
)

rho_dollar_bounds = (RHO_INSERTION_BOUNDS[0] / beta_total, RHO_INSERTION_BOUNDS[1] / beta_total)
rho_dollars = st.sidebar.slider(
    "Reactivity step ($)",
    min_value=round(rho_dollar_bounds[0], 2),
    max_value=round(rho_dollar_bounds[1], 2),
    value=0.0,
    step=0.01,
    help=f"Dollars = rho / beta_total. beta_total = {beta_total:.4f} for this 6-group model. "
    "0$ = no reactivity perturbation (steady state).",
)
rho_insertion = rho_dollars * beta_total

with st.sidebar.expander("Fixed design parameters"):
    st.caption("Held at the report's design values (Appendix A, Table 9); not exposed as controls.")
    st.write(f"Reactivity step timing: {FIXED_RHO_STEP_TIME_S:.0f} s")
    st.write(f"Turbine inlet pressure: {FIXED_TURBINE_P_IN_PA / 1e5:.0f} bar")
    st.write(f"Turbine back pressure: {FIXED_TURBINE_P_BACK_PA / 1e5:.0f} bar")
    st.write(f"Turbine isentropic efficiency: {FIXED_TURBINE_ETA_IS:.3f}")

# ---------------------------------------------------------------------------
# Query the surrogate + agent
# ---------------------------------------------------------------------------
scenario = {
    "helium_mass_flow_kgs": mass_flow,
    "coolant_inlet_temp_K": inlet_temp,
    "rho_insertion": rho_insertion,
    "rho_step_time_s": FIXED_RHO_STEP_TIME_S,
    "turbine_p_in_Pa": FIXED_TURBINE_P_IN_PA,
    "turbine_p_back_Pa": FIXED_TURBINE_P_BACK_PA,
    "turbine_eta_is": FIXED_TURBINE_ETA_IS,
}
report = agent.evaluate(scenario)
fuel_mean, fuel_std = report.predictions["peak_fuel_temp_K"]
work_mean, work_std = report.predictions["turbine_work_MW"]
fuel_band = agent.confidence_z * fuel_std

# ---------------------------------------------------------------------------
# Main display
# ---------------------------------------------------------------------------
st.title("HTGR Operating Point Explorer")
st.caption(
    "Live query against the Nystroem(RBF)+BayesianRidge surrogate (`src/ml_surrogate.py`) "
    "and the `HTGRAgent` safety wrapper - same model, same logic as the notebook and CLI demo."
)

col1, col2, col3 = st.columns(3)
col1.metric(
    "Predicted Peak Fuel Temp",
    f"{fuel_mean:,.1f} K",
    delta=f"±{fuel_band:.1f} K (2σ)",
    delta_color="off",
)
col2.metric(
    "Predicted Turbine Work",
    f"{work_mean:,.1f} MW",
    delta=f"±{agent.confidence_z * work_std:.2f} MW (2σ)",
    delta_color="off",
)
col3.metric(
    "Model Uncertainty (fuel temp, 2σ)",
    f"±{fuel_band:.1f} K",
    delta=f"flag threshold: {surrogate.targets['peak_fuel_temp_K'].uncertainty_threshold:.1f} K std",
    delta_color="off",
)

# --- Safety gauge -----------------------------------------------------------
gauge_lo, gauge_hi = 1350.0, 1950.0
gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=fuel_mean,
        number={"suffix": " K", "font": {"size": 36}},
        title={"text": f"Peak Fuel Temperature vs. {FUEL_TEMP_SAFETY_MARGIN_K:.0f} K Safety Margin"},
        gauge={
            "axis": {"range": [gauge_lo, gauge_hi]},
            "bar": {"color": "black", "thickness": 0.25},
            "steps": [
                {"range": [gauge_lo, FUEL_TEMP_SAFETY_MARGIN_K], "color": "#c8e6c9"},
                {"range": [FUEL_TEMP_SAFETY_MARGIN_K, gauge_hi], "color": "#ffcdd2"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.9,
                "value": FUEL_TEMP_SAFETY_MARGIN_K,
            },
        },
    )
)
gauge.update_layout(height=300, margin=dict(l=30, r=30, t=60, b=10))
st.plotly_chart(gauge, use_container_width=True)

# --- Agent diagnostic panel --------------------------------------------------
st.subheader("HTGRAgent Diagnostic")

status_box = {"NOMINAL": st.success, "UNCERTAIN": st.warning, "UNSAFE": st.error}.get(report.status, st.info)
status_box(f"**Verdict: {report.status}**")

if report.flags:
    st.markdown("**Root-cause diagnostics:**")
    for flag in report.flags:
        st.markdown(f"- {flag}")
else:
    st.markdown("No issues detected - scenario cleared for the fuel-temperature safety check.")

if report.recommendations:
    st.markdown("**Recommended adjustments:**")
    for rec in report.recommendations:
        st.markdown(f"- {rec}")

with st.expander("Full scenario passed to the surrogate"):
    st.json(scenario)

with st.expander("Surrogate model diagnostics (held-out test set)"):
    for name, tm in surrogate.targets.items():
        st.write(
            f"**{name}** — CV R²={tm.cv_r2:.4f}, Test R²={tm.test_r2:.4f}, "
            f"RMSE={tm.test_rmse:.4f}, mean predictive std={tm.test_mean_std:.4f}, "
            f"95% CI coverage={tm.test_coverage_95:.1%}"
        )
