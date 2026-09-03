# -*- coding: utf-8 -*-
"""Generator for the postgraduate-format ML Digital Twin report.

Reads the project's Python source files fresh from disk (no transcription
risk) and embeds the original MATLAB appendix listings verbatim (Appendix A
is transcribed from the source report; it has no live file to read from).
Writes a single print-ready, styled HTML report to notebooks/.

Run from anywhere:
    python scripts/build_ml_report.py

Regenerate after editing any of the Appendix B-F source files so the report
stays byte-identical to the committed code.
"""
import html
from pathlib import Path
from datetime import date

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "HTGR_ML_Digital_Twin_Report.html"

def esc(s: str) -> str:
    return html.escape(s, quote=False)

def code_block(path: Path, lang: str = "python") -> str:
    text = path.read_text(encoding="utf-8")
    return f'<pre class="code"><code>{esc(text)}</code></pre>'

def code_block_str(text: str) -> str:
    return f'<pre class="code"><code>{esc(text)}</code></pre>'

# ---------------------------------------------------------------------------
# Original MATLAB listings (Imperial Team 5 report, Appendix B and D) -
# transcribed verbatim from the source report provided in this conversation.
# ---------------------------------------------------------------------------
MATLAB_CORE_THERMALS = r'''function [ dTf_dt , dTm_dt , dTc_dt ] = core_thermals ( Tf , Tm , Tc , Q_fission , T_c_in , mdot )
% This function calculates the derivatives for the 3-node HTTR thermal model.

% Thermal Capacities (J/K)
Cf = 5.0e6;   % Fuel thermal capacity
Cm = 1.0e8;   % Moderator (graphite) thermal capacity
Cc = 1.0e6;   % Coolant (helium) thermal capacity

% Heat Transfer Coefficients (W/K)
h_fm = 2.0e5; % Fuel-to-Moderator
h_fc = 4.0e5; % Fuel-to-Coolant
h_mc = 1.5e5; % Moderator-to-Coolant

% Coolant Properties
cp = 5193;    % Specific heat of Helium (J/kg-K)

% Equation 1: Fuel Node
dTf_dt = (Q_fission - h_fm*(Tf - Tm) - h_fc*(Tf - Tc)) / Cf;

% Equation 2: Moderator Node
dTm_dt = (h_fm*(Tf - Tm) - h_mc*(Tm - Tc)) / Cm;

% Equation 3: Coolant Node
dTc_dt = (h_fc*(Tf - Tc) + h_mc*(Tm - Tc) - 2*mdot*cp*(Tc - T_c_in)) / Cc;

end'''

MATLAB_KINETICS = r'''function [dP_dt, dC1_dt, dC2_dt, dC3_dt, dC4_dt, dC5_dt, dC6_dt, Q_fission] = ...
    Calculate_Kinetics(P, C1, C2, C3, C4, C5, C6, rho_in)

% 1. DEFINE CONSTANTS (Standard U-235 values)

% Delayed Neutron Fractions (beta_i) for 6 groups
beta_i = [0.00025, 0.00138, 0.00122, 0.00266, 0.00164, 0.00035];

% Total delayed neutron fraction
beta_total = sum(beta_i); % This is 0.0075

% Decay Constants (lambda_i, in 1/s)
lambda = [0.0766, 0.2825, 0.6154, 1.634, 5.176, 16.72];

% Prompt Neutron Lifetime (s). For HTTRs (graphite), this is "long".
Lambda = 1.0e-3; % 0.001 seconds

% Nominal Thermal Power (Watts)
% This scales our normalized power (P) to real power (Q)
Q_nominal = 300e6; % 300 MW

% 2. CALCULATE DERIVATIVES (THE ODEs)

% First, calculate the "sum(lambda_i * C_i)" part
sum_lambda_C = lambda(1)*C1 + lambda(2)*C2 + lambda(3)*C3 + ...
               lambda(4)*C4 + lambda(5)*C5 + lambda(6)*C6;

% Equation 1: Power (Neutron Population)
dP_dt = ((rho_in - beta_total) / Lambda) * P + sum_lambda_C;

% Equations 2-7: Precursor Concentrations
dC1_dt = (beta_i(1) / Lambda) * P - lambda(1) * C1;
dC2_dt = (beta_i(2) / Lambda) * P - lambda(2) * C2;
dC3_dt = (beta_i(3) / Lambda) * P - lambda(3) * C3;
dC4_dt = (beta_i(4) / Lambda) * P - lambda(4) * C4;
dC5_dt = (beta_i(5) / Lambda) * P - lambda(5) * C5;
dC6_dt = (beta_i(6) / Lambda) * P - lambda(6) * C6;

% 3. CALCULATE THE OUTPUT THERMAL POWER
% The block's output is the normalized power (P) scaled by the nominal power.
Q_fission = P * Q_nominal;

end'''

MATLAB_HELIUM_TURBINE = r'''function [P_out, T_out, h_out, m_dot_out, Work_Out] = helium_turbine(T_in, P_in, P_back, m_dot, eta_is)

coder.extrinsic('py.CoolProp.CoolProp.PropsSI');

P_out = 0.0; T_out = 0.0; h_out = 0.0; m_dot_out = 0.0; Work_Out = 0.0;

fluid = 'Helium';

if T_in <= 10 || P_in <= 1000 || m_dot <= 0
    P_out = double(P_in); T_out = double(T_in); m_dot_out = double(m_dot);
else
    % Initialize variables as doubles before extrinsic calls
    h_in = 0.0;
    s_in = 0.0;
    h_out_s = 0.0;
    T_out_val = 0.0;

    h_in = double(py.CoolProp.CoolProp.PropsSI('H', 'T', double(T_in), 'P', double(P_in), fluid));
    s_in = double(py.CoolProp.CoolProp.PropsSI('S', 'T', double(T_in), 'P', double(P_in), fluid));

    h_s_val = py.CoolProp.CoolProp.PropsSI('H', 'S', s_in, 'P', double(P_back), fluid);
    h_out_s = double(h_s_val);

    h_out_actual = h_in - (double(eta_is) * (h_in - h_out_s));
    h_out = double(h_out_actual);

    temp_T = py.CoolProp.CoolProp.PropsSI('T', 'H', h_out, 'P', double(P_back), fluid);
    T_out = double(temp_T);

    P_out = double(P_back);
    m_dot_out = double(m_dot);
    Work_Out = double(m_dot) * (h_in - h_out);
end

end'''

MATLAB_STEAM_TURBINE = r'''function [W_turb, Steam_T_out, Steam_P_out, Steam_h_out, Steam_m_out] = ...
    steam_turbine_measured(Steam_h_in, Steam_P_in, Steam_m_in, P_exhaust_target, eta_is)

% Declare extrinsic functions
coder.extrinsic('py.CoolProp.CoolProp.PropsSI');

% 1. Initialize variables to define them as double scalars for the Coder
fluid = 'Water';
s_in = 0;
h_out_s = 0;
T_out_val = 0;
Steam_T_out = 0;

% 2. Inlet Entropy calculation
s_in_mx = py.CoolProp.CoolProp.PropsSI('S', 'H', double(Steam_h_in), 'P', double(Steam_P_in), fluid);
s_in = double(s_in_mx);

% 3. Actual Expansion logic
% Get isentropic enthalpy
h_out_s_mx = py.CoolProp.CoolProp.PropsSI('H', 'S', s_in, 'P', double(P_exhaust_target), fluid);
h_out_s = double(h_out_s_mx);

% Now h_out_s is a native double and can be used in this expression:
h_out_act = double(Steam_h_in) - (double(eta_is) * (double(Steam_h_in) - h_out_s));

% 4. Outputs
W_turb = double(Steam_m_in) * (double(Steam_h_in) - h_out_act); % Power in Watts
Steam_h_out = h_out_act;
Steam_P_out = double(P_exhaust_target);

% Final temperature calculation
T_out_mx = py.CoolProp.CoolProp.PropsSI('T', 'H', h_out_act, 'P', double(P_exhaust_target), fluid);
Steam_T_out = double(T_out_mx);

Steam_m_out = double(Steam_m_in);

end'''

MATLAB_HYBRID_DESAL = r'''function [m_fresh_total, P_parasitic_total, Water_T_out, Water_h_out, Water_m_out] = ...
    hybrid_desal_tvc(h_motive_in, m_motive_in, P_motive, T_entrained, P_entrained, P_compressed, W_elec_in, SEC_ro)

% h_motive_in: Enthalpy from LP Turbine outlet (Motive Steam)
% m_motive_in: Steam flow from Turbine (kg/s)
% T_entrained/P_entrained: Conditions of vapor from last MED effect

coder.extrinsic('py.CoolProp.CoolProp.PropsSI');

Water_h_out = 0; % 1. TVC Entrainment Ratio (Ra)
% Ra = flow rate of motive steam / flow rate of entrained vapor
% Using the Power (1994) simplified correlation provided in text
PCF = 3e-7 * P_motive + 0.95; % Pressure Correction Factor
TCF = 1.0; % Temp Correction Factor (Simplified for Phase 1)
Cr = P_compressed / P_entrained; % Compression Ratio

% Ra calculation (Semi-empirical model)
Ra = 0.29 * (Cr^1.19) / (PCF * TCF);
m_entrained = m_motive_in / Ra;

% 2. MED Performance (Thermal)
% Total heating steam = motive + entrained
m_heating_steam = m_motive_in + m_entrained;
% GOR is typically higher for TVC (e.g., 10-14)
GOR_tvc = 11.1;
m_fresh_med = m_heating_steam * GOR_tvc;

% 3. RO Process (Electrical)
m_fresh_ro = W_elec_in / (SEC_ro * 3600); % J/kg conversion

% 4. Output State for Closed-Loop (To Pump/HRSG)
% The motive steam and entrained vapor condense into liquid water
Water_m_out = m_motive_in; % Mass balance of the loop fluid
Water_T_out = 343.15; % 70 C return temp (saturated liquid)
Water_h_out = double(py.CoolProp.CoolProp.PropsSI('H', 'T', Water_T_out, 'P', P_compressed, 'Water'));

% 5. System Totals
m_fresh_total = m_fresh_med + m_fresh_ro;
P_parasitic_total = W_elec_in; % Load for RO Pumps

end'''

MATLAB_HRSG = r'''function [He_P_out, He_T_out, Steam_P_out, Steam_T_out, Steam_h_out, Q_transferred] = ...
    hrsg_boiler(He_T_in, He_P_in, He_m_dot, Water_T_in, Water_P_in, Water_m_dot, ...
    UA_eff, dP_He_frac, dP_Water_frac)

% Declare extrinsic for Coder compatibility
coder.extrinsic('py.CoolProp.CoolProp.PropsSI');

% Strings for CoolProp
fluid_he = 'Helium';
fluid_water = 'Water';
h_he_in = 0;
h_he_target = 0;
h_water_in = 0;
He_T_out = 0;
Steam_T_out = 0;

%% ---------------- 1. INPUT SAFETY ----------------
% Helium
T_he_safe = double(He_T_in);
if T_he_safe < 100
    T_he_safe = 1000;
end

P_he_safe = double(He_P_in);
if P_he_safe < 1e3
    P_he_safe = 7e6;
end

m_dot_he_safe = double(He_m_dot);
if m_dot_he_safe < 1e-3
    m_dot_he_safe = 85.0;
end

% Water
T_water_safe = double(Water_T_in);
if T_water_safe < 200
    T_water_safe = 343.15;
end

P_water_safe = double(Water_P_in);
if P_water_safe < 1e3
    P_water_safe = 7e6;
end

m_dot_water_safe = double(Water_m_dot);
if m_dot_water_safe < 1e-3
    m_dot_water_safe = 100.0;
end

%% ---------------- 2. CLAMP PRESSURE DROPS ----------------
dP_He = min(max(double(dP_He_frac), 0.0), 0.9);
dP_W  = min(max(double(dP_Water_frac), 0.0), 0.9);

%% ---------------- 3. THERMODYNAMICS ----------------
% Helium inlet enthalpy
h_he_in = double(py.CoolProp.CoolProp.PropsSI('H','T', T_he_safe,'P', P_he_safe, fluid_he));

% Helium target enthalpy (672 K)
h_he_target = double(py.CoolProp.CoolProp.PropsSI('H','T',672.0,'P', P_he_safe, fluid_he));

% Heat transfer
Q_transferred = m_dot_he_safe * (h_he_in - h_he_target);

% Water inlet enthalpy
h_water_in = double(py.CoolProp.CoolProp.PropsSI('H','T', T_water_safe,'P', P_water_safe, fluid_water));

%% ---------------- 4. OUTLET STATES ----------------
% Helium outlet
He_P_out = P_he_safe * (1.0 - dP_He);
if He_P_out < 1e3
    He_P_out = 1e5;
end

h_he_out = h_he_in - Q_transferred / m_dot_he_safe;

% Steam outlet
Steam_P_out = P_water_safe * (1.0 - dP_W);
if Steam_P_out < 1e3
    Steam_P_out = 1e5;
end

h_steam_out = h_water_in + Q_transferred / m_dot_water_safe;

% Enthalpy sanity
if h_steam_out < 1e5
    h_steam_out = 3e5;
end

Steam_h_out = h_steam_out;

%% ---------------- 5. FINAL TEMPERATURE LOOKUPS ----------------
He_T_out = double(py.CoolProp.CoolProp.PropsSI('T','H', h_he_out,'P', He_P_out, fluid_he));
Steam_T_out = double(py.CoolProp.CoolProp.PropsSI('T','H', h_steam_out,'P', Steam_P_out, fluid_water));

end'''

MATLAB_HR_POROUS = r'''% steam_sponge_thermosyphon_PDC_Q_SWEEP_ALLINONE.m
% =========================================================================
% ONE-FILE: PDC-style step test (steady-state sweep) on Q_total.
% Sweeps down from 147.7 MW in -10 MW steps, then up in +10 MW steps,
% stopping when results become "unrealistic" by simple criteria.
% Outputs a summary table for each Q_total.
% =========================================================================
clear; clc; close all;

%% ---------------- USER SETTINGS ----------------
Q_base_MW = 147.7;   % baseline [MW]
dQ_MW = 10;           % step size [MW]
max_steps_each_side = 25; % hard cap so it can't run forever

% "Realism" stop criteria (edit if you want)
max_modules = 5000;         % stop if module count exceeds this
max_parasitic_frac = 0.20;  % stop if fan power > 20% of Q_total
min_Q_MW = 10;               % stop if Q gets too small/meaningless

%% ---------------- BUILD Q VECTOR ----------------
Q_list_MW = Q_base_MW; % start

% sweep down
Qk = Q_base_MW - dQ_MW;
for i=1:max_steps_each_side
    if Qk < min_Q_MW
        break;
    end
    Q_list_MW(end+1) = Qk; %#ok<SAGROW>
    Qk = Qk - dQ_MW;
end

% sweep up
Qk = Q_base_MW + dQ_MW;
for i=1:max_steps_each_side
    Q_list_MW(end+1) = Qk; %#ok<SAGROW>
    Qk = Qk + dQ_MW;
end

% Put baseline first, then decreasing, then increasing (as described)
Q_down = sort(Q_list_MW(Q_list_MW<=Q_base_MW),'descend');
Q_up = sort(Q_list_MW(Q_list_MW>=Q_base_MW),'ascend');
Q_vec_MW = [Q_down, Q_up(2:end)];

%% ---------------- RUN SWEEP ----------------
n = numel(Q_vec_MW);

% Preallocate containers (use NaN, then fill)
Q_MW = nan(n,1);
N_modules = nan(n,1);
Pfan_MW = nan(n,1);
Vdot_m3s = nan(n,1);
UA_margin = nan(n,1);
Tair_out_C = nan(n,1);
N_TPCT = nan(n,1);

stop_idx = n; % will shorten if stop criteria hit

for k=1:n
    Q_MW(k) = Q_vec_MW(k);
    out = steam_sponge_model_run(Q_MW(k)*1e6);

    % Pull outputs (guarded)
    if isfield(out,'N_modules'); N_modules(k) = out.N_modules; end
    if isfield(out,'P_fan_total_W'); Pfan_MW(k) = out.P_fan_total_W/1e6; end
    if isfield(out,'Vdot_air_total_m3s'); Vdot_m3s(k) = out.Vdot_air_total_m3s; end
    if isfield(out,'UA_margin'); UA_margin(k) = out.UA_margin; end
    if isfield(out,'T_air_out_C'); Tair_out_C(k) = out.T_air_out_C; end
    if isfield(out,'N_TPCT_total'); N_TPCT(k) = out.N_TPCT_total; end

    % ---- stop criteria ----
    parasitic_frac = Pfan_MW(k)*1e6 / (Q_MW(k)*1e6); % W/W
    if (~isnan(N_modules(k)) && N_modules(k) > max_modules) || ...
       (~isnan(parasitic_frac) && parasitic_frac > max_parasitic_frac)
        stop_idx = k;
        break;
    end
end

% (summary-table, plotting, and reporting code omitted here for brevity -
%  see steam_sponge_model_run below for the physics)

%% =========================================================================
% MODEL WRAPPER (vendorized sizing model, converted to a function)
% =========================================================================
function out = steam_sponge_model_run(Q_total_in)
% 0) SYSTEM-LEVEL INPUTS (TEAM VALUES)
Q_total = Q_total_in; % [W] swept input

% Ambient air side (TEAM SLIDES)
T_air_in_C = 20;         % [C] ambient air inlet temperature
P_amb = 101325;            % [Pa] ambient pressure
dT_air = 15;                % [K] allowed air temperature rise
cp_air = 1005;               % [J/kg-K] air specific heat
rho_air = 1.177;              % [kg/m^3] air density at design point
dP_air = 225;                  % [Pa] allowed air pressure drop
eta_fan = 0.65;                 % [-] fan+motor+drive efficiency
crosswind_multiplier = 1.10;     % [-] airflow derate/multiplier

% Coil "lumped" performance (TEAM SLIDES)
U_overall = 40;   % [W/m^2-K] overall U
F_LMTD = 0.98;      % [-] LMTD correction factor

% Steam condition (TEAM SLIDES)
Tsat_C = 65;         % [C] steam saturation temperature in receiver
Psat_bar = 0.25;       % [bar] saturation pressure
x_in = 0.88;             % [-] steam quality at inlet
x_out = 0.0;               % [-] quality after full condensation (design target)
h_fg = 2.346e6;               % [J/kg] latent heat at ~65 C

g = 9.81; % [m/s^2] gravity

% Condenser surface temperature target
T_cond_surface_C = 62; % [C] (TEAM/DESIGN TARGET)

%% 1) FLUID PROPERTIES (WATER/STEAM + AIR)
st = satWater_simple(Tsat_C);

h_in = st.hf + x_in*h_fg;
h_out = st.hf + x_out*h_fg;
dh = h_in - h_out;

m_dot_mix = Q_total/dh;
m_dot_vapor = x_in*m_dot_mix;

mu_air = 1.90e-5;
k_air = 0.027;
Pr_air = cp_air*mu_air/k_air;

%% 2) THERMOSYPHON (TPCT) GEOMETRY + MATERIAL (VENDOR-ORIENTED)
tp.OD = 0.0127;                 % [m] 1/2 inch = 12.7 mm
tp.t_wall = 0.001016;             % [m] 0.040 inch (ASTM B88 Type L typical)
tp.ID = tp.OD - 2*tp.t_wall;
tp.Lev = 1.50;                       % [m] evaporator length inside sponge (TEAM CAD ASSUMPTION)
tp.Lco = 3.00;                         % [m] condenser length in air coil region (TEAM CAD ASSUMPTION)
tp.k_wall = 390;                         % [W/m-K] copper tube wall conductivity

Aev_od = pi*tp.OD*tp.Lev;
Aev_id = pi*tp.ID*tp.Lev;
Aco_id = pi*tp.ID*tp.Lco;

R_wall_ev = log(tp.OD/tp.ID)/(2*pi*tp.k_wall*tp.Lev);
R_wall_co = log(tp.OD/tp.ID)/(2*pi*tp.k_wall*tp.Lco);

%% 3) POROUS RECEIVER ("SPONGE/FOAM") (RECEMAT-BASED PLACEHOLDERS)
foam.eps = 0.95;      % [-] porosity (Recemat datasheet)
foam.a_s = 1000;         % [m^2/m^3] specific surface area density

foam.k_eff = 5.8;          % [W/m-K] NEED VENDOR DATA (temporary team estimate)
foam.K_perm = 2e-8;           % [m^2] NEED VENDOR DATA (temporary assumption)
foam.t_lig = 0.0008;             % [m] NEED VENDOR DATA (temporary assumption)

foam.t_receiver = 0.020;           % [m] (TEAM CAD ASSUMPTION)
foam.F_cond_enh = 1.0;                % [-] NEED TEST/LITERATURE (default 1.0)

r0 = tp.OD/2;
r1 = r0 + foam.t_receiver;

A_cs_foam = pi*(r1^2 - r0^2);
V_foam = A_cs_foam*tp.Lev;
A_foam_geom = foam.a_s*V_foam;

%% 4) STEAM-SIDE CONDENSATION HTC (NUSSELT FILM CONDENSATION)
w = satWater_simple(Tsat_C);
mu_l = muWater_simple(Tsat_C);
k_l = kWater_simple(Tsat_C);

dT_steam_receiver = 8; % [K] TEAM DESIGN ASSUMPTION

h_nusselt = hFilmCond_NusseltVertical(w.rho_l, w.rho_v, mu_l, k_l, h_fg, g, tp.Lev, dT_steam_receiver);
h_steam = foam.F_cond_enh*h_nusselt;

L_fin = foam.t_receiver;
m = sqrt(2*h_steam/(foam.k_eff*foam.t_lig));
eta_foam = tanh(m*L_fin)/(m*L_fin);
eta_foam = max(min(eta_foam,1),0.05);

A_eff_steam = Aev_od + eta_foam*A_foam_geom;
R_steam = 1/(h_steam*A_eff_steam);

%% 5) INTERNAL EVAPORATOR BOILING HTC (INSIDE TPCT EVAPORATOR)
h_boil = 12000; % [W/m^2-K] NEED TPCT VENDOR DATA (placeholder)
R_boil = 1/(h_boil*Aev_id);

R_evap_total = R_steam + R_wall_ev + R_boil;
dT_allow_evap = 12; % [K] TEAM DESIGN CRITERION

%% 6) AIR-COOLED CONDENSER (PREFERRED: VENDOR MODULE UA)
coil.vendor = 'Kelvion RF-NC101E4H (example drycooler module)';
coil.Ao_total_per_module = 197.1;   % [m^2]
coil.airflow_m3ph = 19400;             % [m^3/h]
coil.Vdot_air_mod = coil.airflow_m3ph/3600;

coil.mod_W = 1.610;
coil.mod_H = 1.350;
coil.A_frontal = coil.mod_W*coil.mod_H;

V_face_vendor = coil.Vdot_air_mod/coil.A_frontal;

mdot_air_mod = rho_air*coil.Vdot_air_mod;
Qcap_mod = mdot_air_mod*cp_air*dT_air;

Qcap_mod_effective = Qcap_mod/crosswind_multiplier;

N_modules = max(1, ceil(Q_total/Qcap_mod_effective));

Vdot_air_total = coil.Vdot_air_mod*N_modules;
P_fan_total = dP_air*Vdot_air_total/eta_fan;

T_air_out_C = T_air_in_C + dT_air;
DT1 = T_cond_surface_C - T_air_in_C;
DT2 = T_cond_surface_C - T_air_out_C;
DTlm = (DT1 - DT2)/log(DT1/DT2);

UA_req = Q_total/(F_LMTD*DTlm);
UA_avail = U_overall*coil.Ao_total_per_module*N_modules;
UA_margin = UA_avail/UA_req;

%% 7) TPCT HEAT-TRANSPORT LIMITS (FLOODING + BOILING + SONIC CHECK)
T_tp_C = (Tsat_C + T_cond_surface_C)/2;
wt = satWater_simple(T_tp_C);

sigma_w = surfaceTension_IAPWS2014(T_tp_C);
A_v = pi*(tp.ID^2)/4;

Qmax_flood = QmaxFlood_FaghriStyle(tp.ID, A_v, wt.rho_l, wt.rho_v, h_fg, sigma_w, g);
Qmax_design = 0.60*Qmax_flood;

qCHF_frac = 0.30;
qCHF_Zuber = CHF_ZuberPoolBoiling(wt.rho_l, wt.rho_v, h_fg, sigma_w, g);
Qmax_boil = qCHF_frac*qCHF_Zuber*(pi*tp.ID*tp.Lev);

a_v = 430;
G_sonic_allow = 0.25*wt.rho_v*a_v;
Qmax_sonic = G_sonic_allow*A_v*h_fg;

Qmax_tpct = min([Qmax_design, Qmax_boil, Qmax_sonic]);

%% 8) NUMBER OF THERMOSYPHONS REQUIRED (BASED ON LIMITS + TEMPERATURE DROP)
h_cond_int = 12000; % [W/m^2-K] NEED TPCT VENDOR DATA
R_cond_int = 1/(h_cond_int*Aco_id);

R_air_total = 1/(UA_avail);

N_pipes = 5000; % [-] initial guess (TEAM GUESS)
while true
    Q_pipe = Q_total/N_pipes;
    dT_evap = Q_pipe*R_evap_total;

    R_air_per_pipe = R_air_total*N_pipes;
    R_cond_total = R_cond_int + R_wall_co + R_air_per_pipe;
    dT_cond = Q_pipe*R_cond_total;

    limits_ok = (Q_pipe <= Qmax_tpct);
    dT_ok = (dT_evap <= dT_allow_evap);

    if limits_ok && dT_ok
        break;
    end

    if ~limits_ok
        N_pipes = N_pipes + 2000;
    else
        N_pipes = N_pipes + 500;
    end
end

Q_pipe = Q_total/N_pipes;
dT_evap = Q_pipe*R_evap_total;
dT_cond = Q_pipe*R_cond_total;
dT_total_pipe = dT_evap + dT_cond;

%% 9) CONDENSATE DRAINAGE CHECK IN POROUS RECEIVER (DARCY)
m_dot_cond_total = m_dot_vapor;
m_dot_cond_per_pipe = m_dot_cond_total/N_pipes;
Vdot_cond_per_pipe = m_dot_cond_per_pipe/w.rho_l;

v_Darcy = Vdot_cond_per_pipe/A_cs_foam;
dP_dz = mu_l*v_Darcy/foam.K_perm;
dP_total = dP_dz*tp.Lev;

dP_head = w.rho_l*g*tp.Lev;
drain_ok = (dP_total <= 0.5*dP_head);

%% -------- PACK OUTPUTS --------
out = struct();
out.Q_total_W = Q_total;
out.N_modules = N_modules;
out.P_fan_total_W = P_fan_total;
out.Vdot_air_total_m3s = Vdot_air_total;
out.T_air_out_C = T_air_out_C;
out.UA_req_WK = UA_req;
out.UA_avail_WK = UA_avail;
out.UA_margin = UA_margin;

end

% =========================================================================
% SUBFUNCTIONS
% =========================================================================
function st = satWater_simple(T_C)
% satWater_simple
%   Quick saturated-water lookup for 50-100C, linear interpolation.
%   Returns: psat [Pa], rho_l [kg/m3], rho_v [kg/m3], hf [J/kg], hfg [J/kg]
%   Source of numbers: typical steam tables (embedded dataset).
%   Upgrade path: replace with IF97 (IAPWS) or CoolProp for production work.

T = [50 55 60 65 70 75 80 85 90 95 100];
ps_MPa = [0.012352 0.015761 0.019946 0.025042 0.031201 0.038563 0.047373 0.057834 0.070140 0.084552 0.101325];
rho_l  = [988.05 985.65 983.16 980.52 977.76 974.89 971.91 968.82 965.62 962.33 958.37];
rho_v  = [0.08302 0.10266 0.13043 0.16146 0.19833 0.24418 0.29216 0.34569 0.41451 0.49220 0.59752];
hf_kJ  = [209.33 230.23 251.18 272.12 292.98 313.93 334.91 355.90 376.92 397.96 419.04];
hfg_kJ = [2382.7 2370.7 2357.7 2345.4 2333.8 2321.4 2308.8 2296.0 2283.2 2270.2 2257.0];

T_C = max(min(T_C, max(T)), min(T));

st.psat = interp1(T, ps_MPa, T_C, 'linear')*1e6;
st.rho_l = interp1(T, rho_l, T_C, 'linear');
st.rho_v = interp1(T, rho_v, T_C, 'linear');
st.hf = interp1(T, hf_kJ, T_C, 'linear')*1e3;
st.hfg = interp1(T, hfg_kJ, T_C, 'linear')*1e3;
end

function mu = muWater_simple(T_C)
% muWater_simple
%   Rough liquid water viscosity correlation (Andrade-type).
%   Valid-ish for ~0-100C. Replace with IAPWS viscosity for higher fidelity.
T = T_C + 273.15;
mu = 2.414e-5*10^(247.8/(T-140));
end

function k = kWater_simple(T_C)
% kWater_simple
%   Rough liquid water thermal conductivity fit, 0-100C.
k = 0.561 + 0.0019*T_C - 1.0e-5*T_C.^2;
end

function h = hFilmCond_NusseltVertical(rhoL, rhoV, muL, kL, hfg, g, L, dT)
% hFilmCond_NusseltVertical
%   Nusselt laminar film condensation on a vertical surface (average h).
%   h = 0.943 * [ rhoL*(rhoL-rhoV)*g*hfg*kL^3 / (muL*L*dT) ]^(1/4)
h = 0.943 * (rhoL*(rhoL-rhoV)*g*hfg*kL^3/(muL*L*max(dT,1e-6)))^(1/4);
end

function sigma = surfaceTension_IAPWS2014(T_C)
% surfaceTension_IAPWS2014
%   Official IAPWS (2014) surface tension correlation:
%   sigma = B * tau^mu * (1 + b*tau), tau = 1 - T/Tc
%   Parameters from IAPWS PDF: Tc=647.096 K, B=235.8 mN/m, mu=1.256, b=-0.625
%   Source: https://iapws.org/public/documents/CH-L9/Surf-H2O-2014.pdf
T = T_C + 273.15;
Tc = 647.096;
tau = 1 - T/Tc;
B = 235.8e-3; mu = 1.256; b = -0.625;
sigma = B*(tau^mu)*(1 + b*tau);
end

function Qmax = QmaxFlood_FaghriStyle(D, Av, rhoL, rhoV, hfg, sigma, g)
% QmaxFlood_FaghriStyle
%   Engineering flooding/entrainment limit form for wickless thermosyphons.
%   Used as a practical bound; vendor data should replace if available.
Bo = sqrt((rhoL - rhoV)*g*D^2/sigma);
K = (rhoL/rhoV)^0.14 * (tanh(sqrt(Bo)))^2;
term1 = (g*sigma*(rhoL - rhoV))^(0.25);
term2 = (rhoV^(-0.5) + rhoL^(-0.5))^(-2);
Qmax = K*hfg*Av*term1*term2;
end

function qCHF = CHF_ZuberPoolBoiling(rhoL, rhoV, hfg, sigma, g)
% CHF_ZuberPoolBoiling
%   Zuber CHF correlation for pool boiling (large horizontal surface):
%   q''_CHF = 0.131 * hfg * rhoV^(1/2) * [ sigma*g*(rhoL-rhoV) ]^(1/4)
qCHF = 0.131 * hfg * sqrt(rhoV) * (sigma*g*(rhoL-rhoV))^(1/4);
end
% =========================================================================
'''

print("Loaded MATLAB listings, lengths:",
      len(MATLAB_CORE_THERMALS), len(MATLAB_KINETICS), len(MATLAB_HELIUM_TURBINE),
      len(MATLAB_STEAM_TURBINE), len(MATLAB_HYBRID_DESAL), len(MATLAB_HRSG), len(MATLAB_HR_POROUS))

# ---------------------------------------------------------------------------
# Python source appendices (read live from disk - zero transcription risk)
# ---------------------------------------------------------------------------
PY_PHYSICS = code_block(REPO / "src" / "htgr_physics.py")
PY_PIPELINE = code_block(REPO / "src" / "data_pipeline.py")
PY_SURROGATE = code_block(REPO / "src" / "ml_surrogate.py")
PY_APP = code_block(REPO / "app.py")
PY_BUILDHTML = code_block(REPO / "scripts" / "build_styled_html.py")

TODAY = date.today().strftime("%d %B %Y")

CSS = """
<style>
@page { size: A4; margin: 25mm 20mm; }
* { box-sizing: border-box; }
html, body {
  background: #FAF6EC; color: #2B2620;
  font-family: "Times New Roman", Times, Georgia, serif;
  font-size: 18px; line-height: 1.6;
  margin: 0; padding: 0;
}
.page { max-width: 880px; margin: 0 auto; padding: 40px 56px; }
h1, h2, h3, h4 { font-family: "Times New Roman", Times, Georgia, serif; color: #17304D; font-weight: bold; }
h1 { font-size: 30px; border-bottom: 2px solid #B8860B; padding-bottom: 8px; margin-top: 0; }
h2 { font-size: 24px; color: #1F3A5F; border-bottom: 1px solid #D8D0BC; padding-bottom: 4px; margin-top: 2.2em; }
h3 { font-size: 20px; color: #1F3A5F; margin-top: 1.6em; }
h4 { font-size: 18px; color: #3A4650; margin-top: 1.3em; }
p { text-align: justify; }
a { color: #7B241C; text-decoration: none; border-bottom: 1px dotted #7B241C; }
strong { color: #17304D; }
em { color: #4A4436; }
hr { border: none; border-top: 1px solid #D8D0BC; margin: 2em 0; }

.titlepage { text-align: center; padding-top: 22vh; page-break-after: always; }
.titlepage .kicker { letter-spacing: 4px; color: #B8860B; font-size: 15px; text-transform: uppercase; margin-bottom: 18px; }
.titlepage h1 { font-size: 34px; border: none; line-height: 1.3; }
.titlepage .subtitle { font-size: 19px; color: #4A4436; font-style: italic; margin: 14px 0 40px; }
.titlepage .meta { margin-top: 60px; font-size: 16px; line-height: 1.9; }
.titlepage .meta b { color: #17304D; }

.notation-page, .toc-page { page-break-after: always; }
table.notation { width: 100%; border-collapse: collapse; margin: 1em 0; }
table.notation td { padding: 4px 10px; border-bottom: 1px solid #EDE7D6; vertical-align: top; }
table.notation td.sym { width: 90px; font-style: italic; white-space: nowrap; }

.toc a { display: block; padding: 3px 0; border-bottom: none; color: #2B2620; }
.toc a:hover { color: #7B241C; }
.toc .l1 { font-weight: bold; margin-top: 10px; }
.toc .l2 { padding-left: 22px; font-size: 16px; color: #4A4436; }
.toc .l3 { padding-left: 42px; font-size: 15px; color: #6B6355; }

table.data { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 16px; }
table.data th { background: #E9E0C8; color: #17304D; border: 1px solid #D8D0BC; padding: 6px 10px; text-align: left; }
table.data td { border: 1px solid #E4DDC9; padding: 6px 10px; }
table.data tr:nth-child(even) td { background: #F3EEDF; }
.caption { font-size: 15px; color: #4A4436; margin-top: 4px; font-style: italic; }

blockquote { border-left: 3px solid #B8860B; background: #F3EEDF; color: #4A4436; padding: 10px 18px; margin-left: 0; }

pre.code { background: #F0EBDD; border: 1px solid #D8D0BC; border-radius: 5px; padding: 12px 14px;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 12.5px;
  line-height: 1.45; overflow-x: auto; white-space: pre; page-break-inside: auto; }
pre.code code { color: #2B2620; }
.codecap { font-family: "SFMono-Regular", Consolas, monospace; font-size: 14px; color: #17304D;
  background: #E9E0C8; padding: 4px 10px; border: 1px solid #D8D0BC; border-bottom: none;
  display: inline-block; border-radius: 5px 5px 0 0; }

.appendix-section { page-break-before: always; }
.part-break { page-break-before: always; }

.ref-list p { text-indent: -28px; padding-left: 28px; text-align: left; }
.footer-note { font-size: 14px; color: #6B6355; margin-top: 3em; border-top: 1px solid #D8D0BC; padding-top: 10px; }
@media print { .page { padding: 0; } }
</style>
"""

# ===========================================================================
BODY = []

BODY.append(f"""
<div class="page titlepage">
  <div class="kicker">Imperial College London</div>
  <h1>Scalable Machine-Learning Surrogate Modelling and Autonomous Safety Agents<br/>for a High-Temperature Gas Reactor Digital Twin</h1>
  <div class="subtitle">A Python-Based Digital-Twin Extension of the Scaled HTTR&ndash;Desalination<br/>
  Complex Developed in Imperial HTGR&nbsp;5 and HR&nbsp;5 (2026)</div>
  <div class="meta">
    <b>Authored by:</b><br/>Muhammad Dani<br/><br/>
    <b>Extending the reactor and heat-rejection models originally developed with:</b><br/>
    Michal Krasowski &middot; Qiji Zhang &middot; Clayton Yeung &middot;
    Oscar Chan &middot; Heidi Chong &middot; Oliver Bush &middot; Muhammad Abubakar &middot; Luca Faillace<br/><br/>
    Imperial College London<br/>{TODAY}
  </div>
</div>
""")

BODY.append(f"""
<div class="page notation-page">
<h2 id="notation">Notation</h2>
<h3>Reactor Physics Symbols (carried over from the original report)</h3>
<table class="notation">
<tr><td class="sym">P(t)</td><td>Normalised reactor power</td></tr>
<tr><td class="sym">&rho;(t)</td><td>Reactivity</td></tr>
<tr><td class="sym">&beta;</td><td>Total delayed neutron fraction</td></tr>
<tr><td class="sym">&beta;<sub>i</sub></td><td>Delayed neutron fraction for precursor group i</td></tr>
<tr><td class="sym">&Lambda;</td><td>Prompt neutron lifetime</td></tr>
<tr><td class="sym">&lambda;<sub>i</sub></td><td>Decay constant for delayed neutron group i</td></tr>
<tr><td class="sym">C<sub>i</sub>(t)</td><td>Delayed-neutron precursor concentration for group i</td></tr>
<tr><td class="sym">T<sub>f</sub>, T<sub>m</sub>, T<sub>c</sub></td><td>Fuel, moderator, and coolant temperature</td></tr>
<tr><td class="sym">T<sub>in</sub></td><td>Coolant inlet temperature</td></tr>
<tr><td class="sym">C<sub>f</sub>, C<sub>m</sub>, C<sub>c</sub></td><td>Heat capacities of fuel, moderator, and coolant</td></tr>
<tr><td class="sym">h<sub>fm</sub>, h<sub>fc</sub>, h<sub>mc</sub></td><td>Inter-node heat-transfer coefficients</td></tr>
<tr><td class="sym">Q<sub>fission</sub></td><td>Fission heat generation rate</td></tr>
<tr><td class="sym">&#7745;</td><td>Mass flow rate (helium unless subscripted otherwise)</td></tr>
<tr><td class="sym">C<sub>p</sub></td><td>Specific heat capacity at constant pressure</td></tr>
</table>
<h3>Machine-Learning Symbols (introduced in this report)</h3>
<table class="notation">
<tr><td class="sym">N</td><td>Number of training points (Monte Carlo runs)</td></tr>
<tr><td class="sym">m</td><td>Number of Nystr&ouml;m landmark components, m &laquo; N</td></tr>
<tr><td class="sym">k(x, x&prime;)</td><td>Kernel function (RBF)</td></tr>
<tr><td class="sym">K</td><td>Full N&times;N Gram (kernel) matrix</td></tr>
<tr><td class="sym">K<sub>m,m</sub></td><td>m&times;m Gram matrix among Nystr&ouml;m landmark points</td></tr>
<tr><td class="sym">&phi;(x)</td><td>Explicit Nystr&ouml;m feature map, &phi;: &#8477;<sup>d</sup> &rarr; &#8477;<sup>m</sup></td></tr>
<tr><td class="sym">&Phi;</td><td>Stacked N&times;m design matrix of Nystr&ouml;m features</td></tr>
<tr><td class="sym">w</td><td>Bayesian ridge weight vector</td></tr>
<tr><td class="sym">&alpha;</td><td>Weight precision (Bayesian ridge prior)</td></tr>
<tr><td class="sym">&tau;</td><td>Noise precision (Bayesian ridge likelihood); denoted &beta; in some texts, renamed here to avoid clashing with the delayed-neutron fraction &beta;</td></tr>
<tr><td class="sym">&mu;<sub>w</sub>, &Sigma;<sub>w</sub></td><td>Posterior mean and covariance of the Bayesian ridge weights</td></tr>
<tr><td class="sym">x<sub>&lowast;</sub></td><td>A query (test) input point</td></tr>
<tr><td class="sym">&mu;<sub>&lowast;</sub>, &sigma;<sub>&lowast;</sub><sup>2</sup></td><td>Predictive mean and variance at a query point</td></tr>
<tr><td class="sym">z</td><td>Confidence multiplier for an upper credible bound (z = 2 used throughout)</td></tr>
<tr><td class="sym">R&sup2;, RMSE</td><td>Coefficient of determination; root-mean-square error</td></tr>
<h3>Acronyms and Abbreviations</h3>
<table class="notation">
<tr><td class="sym">HTGR</td><td>High-Temperature Gas-Cooled Reactor</td></tr>
<tr><td class="sym">HTTR</td><td>High-Temperature Engineering Test Reactor</td></tr>
<tr><td class="sym">ODE</td><td>Ordinary Differential Equation</td></tr>
<tr><td class="sym">GP</td><td>Gaussian Process</td></tr>
<tr><td class="sym">RBF</td><td>Radial Basis Function (kernel)</td></tr>
<tr><td class="sym">CI</td><td>Confidence Interval</td></tr>
<tr><td class="sym">CSV</td><td>Comma-Separated Values</td></tr>
<tr><td class="sym">CI/CD</td><td>Continuous Integration / Continuous Delivery</td></tr>
<tr><td class="sym">API</td><td>Application Programming Interface</td></tr>
<tr><td class="sym">UI</td><td>User Interface</td></tr>
<tr><td class="sym">MED</td><td>Multi-Effect Distillation</td></tr>
<tr><td class="sym">CoolProp</td><td>Open-source thermophysical property library</td></tr>
</table>
</table>
</div>
""")

TOC_ITEMS = [
    ("l1", "Notation", "#notation"),
    ("l1", "Part I &ndash; Introduction and Motivation", "#part1"),
    ("l2", "1.1 From a Validated Report to a Running Digital Twin", "#s1-1"),
    ("l2", "1.2 Objectives of This Extension", "#s1-2"),
    ("l2", "1.3 Relationship to the Original HTGR-5 / HR-5 Report", "#s1-3"),
    ("l2", "1.4 Report Structure", "#s1-4"),
    ("l1", "Part II &ndash; Digital Twin Architecture", "#part2"),
    ("l2", "2.1 System Architecture and Data Flow", "#s2-1"),
    ("l2", "2.2 Technology Stack", "#s2-2"),
    ("l2", "2.3 Design Principles", "#s2-3"),
    ("l1", "Part III &ndash; Physics Engine Translation", "#part3"),
    ("l2", "3.1 Governing Equations", "#s3-1"),
    ("l2", "3.2 Python Implementation via SciPy solve_ivp", "#s3-2"),
    ("l2", "3.3 Analytic Steady-State Initialisation", "#s3-3"),
    ("l2", "3.4 The Open-Loop Divergence Finding", "#s3-4"),
    ("l2", "3.5 Verification by Sanity Check", "#s3-5"),
    ("l2", "3.6 Validation Against Literature Benchmark Data", "#s3-6"),
    ("l1", "Part IV &ndash; Monte Carlo Telemetry Generation", "#part4"),
    ("l2", "4.1 Sampling Methodology and Operating Envelope", "#s4-1"),
    ("l2", "4.2 Helium Turbine Work Extension", "#s4-2"),
    ("l2", "4.3 Dataset Characteristics", "#s4-3"),
    ("l1", "Part V &ndash; Scalable Probabilistic Surrogate Modelling", "#part5"),
    ("l2", "5.1 The O(N&sup3;) Bottleneck of Exact Gaussian Process Regression", "#s5-1"),
    ("l2", "5.2 The Nystr&ouml;m Kernel Approximation", "#s5-2"),
    ("l2", "5.3 Bayesian Ridge Regression on Nystr&ouml;m Features", "#s5-3"),
    ("l2", "5.4 Complexity Comparison", "#s5-4"),
    ("l2", "5.5 Hyperparameter Optimisation", "#s5-5"),
    ("l2", "5.6 Model Evaluation and Calibration", "#s5-6"),
    ("l2", "5.7 The Honest Limitation of the Approximation", "#s5-7"),
    ("l1", "Part VI &ndash; HTGRAgent: An Autonomous Safety Decision Layer", "#part6"),
    ("l2", "6.1 Motivation and Design Philosophy", "#s6-1"),
    ("l2", "6.2 Conditional Decision Logic", "#s6-2"),
    ("l2", "6.3 The Deterministic Extrapolation Guard", "#s6-3"),
    ("l2", "6.4 Root-Cause Recommendation Engine", "#s6-4"),
    ("l2", "6.5 Demonstration Scenarios", "#s6-5"),
    ("l1", "Part VII &ndash; Interactive Systems", "#part7"),
    ("l2", "7.1 The Systems Textbook Notebook", "#s7-1"),
    ("l2", "7.2 The Streamlit Operating-Point Explorer", "#s7-2"),
    ("l2", "7.3 Styled Static Exports", "#s7-3"),
    ("l1", "Part VIII &ndash; Engineering Verification and Debug Log", "#part8"),
    ("l1", "Part IX &ndash; Results Summary", "#part9"),
    ("l1", "Part X &ndash; Conclusions and Further Work", "#part10"),
    ("l1", "References", "#refs"),
    ("l1", "Appendices", "#appendices"),
    ("l2", "Appendix A &ndash; Original MATLAB/Simulink Source Code", "#appA"),
    ("l2", "Appendix B &ndash; Python Physics Engine (htgr_physics.py)", "#appB"),
    ("l2", "Appendix C &ndash; Python Monte Carlo Pipeline (data_pipeline.py)", "#appC"),
    ("l2", "Appendix D &ndash; Python ML Surrogate and Agent (ml_surrogate.py)", "#appD"),
    ("l2", "Appendix E &ndash; Streamlit Dashboard (app.py)", "#appE"),
    ("l2", "Appendix F &ndash; Notebook Styling Utility (build_styled_html.py)", "#appF"),
]
toc_html = "\\n".join(f'<a class="{cls}" href="{href}">{label}</a>' for cls, label, href in TOC_ITEMS)
BODY.append(f"""<div class="page toc-page"><h2>Contents</h2><div class="toc">{toc_html}</div></div>""")

print("Header/title/notation/TOC assembled; body parts so far:", len(BODY))

# ===========================================================================
# PART I
# ===========================================================================
BODY.append("""
<div class="page part-break">
<h1 id="part1">Part I &mdash; Introduction and Motivation</h1>

<h2 id="s1-1">1.1 From a Validated Report to a Running Digital Twin</h2>
<p>The Imperial HTGR&nbsp;5 and HR&nbsp;5 report, <em>&ldquo;Scaled High Temperature Test Reactor
Cogeneration for Zero Emission Hybrid-Desalination with Independent Water Heat Rejection&rdquo;</em>
(Imperial College London, 2026), specified and validated a coupled reactor&ndash;desalination&ndash;heat-rejection
system in MATLAB and Simulink: 6-group point kinetics, a 3-node fuel/moderator/coolant thermal
core, a combined Helium Brayton&ndash;Steam Rankine power cycle, DEEP-modelled hybrid desalination
economics, and a porous-metal thermosyphon heat-rejection system, all validated against literature
benchmark data and manufacturer specification sheets.</p>
<p>This report documents a second phase of work: re-implementing the reactor physics core of that
model as an open, tested, version-controlled Python codebase, extending it with a 1,000-run Monte
Carlo telemetry generator, and building a scalable machine-learning surrogate &mdash; wrapped in an
autonomous decision-making agent &mdash; capable of evaluating candidate operating points in
milliseconds rather than by re-running a stiff ODE solver. The result is a <em>digital twin</em> in
the strict sense: a lightweight, queryable model of the physical system's behaviour, kept honest by
being checked against the original physics at every step.</p>

<h2 id="s1-2">1.2 Objectives of This Extension</h2>
<p>Four objectives were set for this phase of work:</p>
<ol>
<li>Translate the report's governing point-kinetics and thermal-hydraulics equations into a Python
ODE system with no loss of fidelity to the source MATLAB listings, verified rather than assumed
correct.</li>
<li>Generate a statistically representative telemetry dataset spanning the plant's realistic
operating envelope, suitable for training a supervised machine-learning model.</li>
<li>Train a probabilistic surrogate model that scales to Monte Carlo datasets far larger than
1,000 rows &mdash; ruling out an exact Gaussian Process on cost grounds alone &mdash; while still
producing calibrated predictive uncertainty.</li>
<li>Wrap that surrogate in an autonomous agent capable of clearing, flagging, or rejecting a
candidate operating point against the reactor's 1650&nbsp;K fuel-temperature safety margin, with a
stated, inspectable reason in every case.</li>
</ol>

<h2 id="s1-3">1.3 Relationship to the Original HTGR-5 / HR-5 Report</h2>
<p>Nothing in the original report's reactor design, thermodynamic analysis, or economic case is
revisited or re-derived here. Where this report's Python implementation reproduces a result from the
original report (the 3.51&nbsp;kW condensate-pump power, the 1,700 dry-cooler modules, the
11,400&nbsp;m&sup3;/day freshwater production, and others), the reproduction is treated as a
verification exercise: computed independently, then checked against the report's stated value. Two
such checks surfaced genuine numerical inconsistencies in the source report itself &mdash; both are
documented candidly in Part VIII rather than silently reconciled.</p>
<p>The original report's complete MATLAB source code is reproduced in Appendix A of this report for
direct, side-by-side comparison with the Python translation in Appendices B&ndash;D.</p>

<h2 id="s1-4">1.4 Report Structure</h2>
<p>Part II gives the digital twin's system architecture. Parts III and IV cover the physics engine
and the Monte Carlo telemetry generator built on top of it. Part V is the mathematical core of this
report: a full derivation of the Nystr&ouml;m-approximated, Bayesian-ridge probabilistic surrogate
that replaces an intractable exact Gaussian Process. Part VI documents the autonomous safety agent
built on that surrogate. Part VII covers the two interactive artefacts (a Jupyter notebook and a
Streamlit dashboard) built on the same components. Part VIII is an engineering debug log &mdash; a
candid, chronological record of what broke during development and how it was diagnosed. Part IX
consolidates results, and Part X concludes. Appendices A&ndash;F reproduce, in full, every line of
code referenced in the body of this report.</p>
</div>
""")

# ===========================================================================
# PART II
# ===========================================================================
BODY.append("""
<div class="page part-break">
<h1 id="part2">Part II &mdash; Digital Twin Architecture</h1>

<h2 id="s2-1">2.1 System Architecture and Data Flow</h2>
<p>The digital twin is organised as four layers, each depending only on the layer beneath it:</p>
<table class="data">
<tr><th>Layer</th><th>Module</th><th>Responsibility</th></tr>
<tr><td>1. Physics</td><td><code>htgr_physics.py</code></td><td>Coupled point-kinetics / 3-node
thermal-core ODE system, integrated with SciPy</td></tr>
<tr><td>2. Telemetry</td><td><code>data_pipeline.py</code></td><td>Monte Carlo sampling of
operating points, physics-engine execution, CoolProp-based turbine work extension</td></tr>
<tr><td>3. Surrogate + Agent</td><td><code>ml_surrogate.py</code></td><td>Scalable probabilistic
regression and autonomous safety decision logic</td></tr>
<tr><td>4. Interfaces</td><td><code>app.py</code>, notebooks</td><td>Interactive dashboard and
narrative documentation, both querying layer 3 directly</td></tr>
</table>
<p class="caption">Table 2.1 &mdash; The four-layer digital twin architecture. Every layer above
layer&nbsp;1 is a thin, disposable client of the layer beneath it: the physics engine has no
knowledge of the surrogate, and the surrogate has no knowledge of the dashboard.</p>

<h2 id="s2-2">2.2 Technology Stack</h2>
<p>NumPy and SciPy (<code>solve_ivp</code>) implement the physics engine; pandas structures the
Monte Carlo telemetry; CoolProp supplies real helium and water/steam thermophysical properties,
consistent with the original report's own stated reasoning for adopting CoolProp over Simulink's
constant-<i>C<sub>p</sub></i> assumption; scikit-learn (<code>Nystroem</code>, <code>BayesianRidge</code>,
<code>GridSearchCV</code>) implements the surrogate; joblib persists the trained model; Streamlit
and Plotly implement the interactive dashboard; Jupyter and nbconvert produce the narrative
notebooks and their styled static exports.</p>

<h2 id="s2-3">2.3 Design Principles</h2>
<p><b>Translate, then verify.</b> Every equation ported from the original MATLAB listings is checked
against an independent property of the system it should satisfy &mdash; a fixed point, a conservation
law, a monotonicity argument &mdash; before being trusted as a foundation for further work (Part
III.5).</p>
<p><b>Report gaps, don't paper over them.</b> Where the source report's own figures are internally
inconsistent (Part VIII), or where this project's physics deliberately diverges from a literal
reading of the MATLAB listing (Part III.4), the divergence is documented in the code and in this
report rather than silently resolved in whichever direction looks better.</p>
<p><b>Prefer a second, independent check over trusting one signal.</b> The autonomous safety agent
(Part VI) never relies on the surrogate's self-reported uncertainty alone to detect an
out-of-distribution query; it also runs a deterministic bounds check, because the two failure modes
the surrogate can exhibit are not equivalent and a single check cannot catch both.</p>
</div>
""")

print("Parts I-II assembled")

# ===========================================================================
# PART III
# ===========================================================================
BODY.append("""
<div class="page part-break">
<h1 id="part3">Part III &mdash; Physics Engine Translation</h1>

<h2 id="s3-1">3.1 Governing Equations</h2>
<p>The point-kinetics power equation and its six delayed-neutron precursor equations (original report
Eq. 2.1&ndash;2.2, MATLAB <code>Calculate_Kinetics.m</code>, Appendix A of this report):</p>
<blockquote>
d<i>P</i>(t)/d<i>t</i> = [(&rho;(t) &minus; &beta;) / &Lambda;]&thinsp;<i>P</i>(t) + &sum;<sub>i=1</sub><sup>6</sup> &lambda;<sub>i</sub><i>C<sub>i</sub></i>(t)<br/>
d<i>C<sub>i</sub></i>(t)/d<i>t</i> = (&beta;<sub>i</sub>/&Lambda;)&thinsp;<i>P</i>(t) &minus; &lambda;<sub>i</sub><i>C<sub>i</sub></i>(t)
</blockquote>
<p>and the 3-node thermal-core energy balances (original report Eq. 2.3&ndash;2.5, MATLAB
<code>core_thermals.m</code>):</p>
<blockquote>
<i>C<sub>f</sub></i>&thinsp;dT<sub>f</sub>/dt = Q<sub>fission</sub> &minus; h<sub>fm</sub>(T<sub>f</sub>&minus;T<sub>m</sub>) &minus; h<sub>fc</sub>(T<sub>f</sub>&minus;T<sub>c</sub>)<br/>
<i>C<sub>m</sub></i>&thinsp;dT<sub>m</sub>/dt = h<sub>fm</sub>(T<sub>f</sub>&minus;T<sub>m</sub>) &minus; h<sub>mc</sub>(T<sub>m</sub>&minus;T<sub>c</sub>)<br/>
<i>C<sub>c</sub></i>&thinsp;dT<sub>c</sub>/dt = h<sub>fc</sub>(T<sub>f</sub>&minus;T<sub>c</sub>) + h<sub>mc</sub>(T<sub>m</sub>&minus;T<sub>c</sub>) &minus; 2&#7745;C<sub>p</sub>(T<sub>c</sub>&minus;T<sub>in</sub>)
</blockquote>
<p>are combined into a single 10-state coupled system, <i>y</i> = [P, C<sub>1</sub>&hellip;C<sub>6</sub>,
T<sub>f</sub>, T<sub>m</sub>, T<sub>c</sub>], implemented in full in Appendix B.</p>

<h2 id="s3-2">3.2 Python Implementation via SciPy solve_ivp</h2>
<p>The coupled system is stiff: the prompt-neutron lifetime &Lambda; = 10&minus;3&nbsp;s and the
moderator's thermal time constant <i>C<sub>m</sub></i>/h<sub>mc</sub> &asymp; 667&nbsp;s span more
than five orders of magnitude. An explicit solver (e.g. RK45) would need a timestep set by the
fastest mode while integrating over a horizon set by the slowest, making it prohibitively expensive.
<code>HTGRReactorModel.simulate()</code> therefore defaults to SciPy's implicit Radau method, which
is unconditionally stable for stiff systems of this kind.</p>

<h2 id="s3-3">3.3 Analytic Steady-State Initialisation</h2>
<p>Rather than reading approximate initial temperatures from a figure, <code>steady_state()</code>
sets d<i>T<sub>f</sub></i>/dt = d<i>T<sub>m</sub></i>/dt = d<i>T<sub>c</sub></i>/dt = 0 and solves the
resulting 3&times;3 linear system for [T<sub>f</sub>, T<sub>m</sub>, T<sub>c</sub>] directly:</p>
<table class="data">
<tr><th>Row</th><th>[T<sub>f</sub>, T<sub>m</sub>, T<sub>c</sub>] coefficients</th><th>RHS</th></tr>
<tr><td>Fuel balance</td><td>[h<sub>fm</sub>+h<sub>fc</sub>, &minus;h<sub>fm</sub>, &minus;h<sub>fc</sub>]</td><td>Q<sub>fission</sub></td></tr>
<tr><td>Moderator balance</td><td>[h<sub>fm</sub>, &minus;(h<sub>fm</sub>+h<sub>mc</sub>), h<sub>mc</sub>]</td><td>0</td></tr>
<tr><td>Coolant balance</td><td>[h<sub>fc</sub>, h<sub>mc</sub>, &minus;(h<sub>fc</sub>+h<sub>mc</sub>+2&#7745;C<sub>p</sub>)]</td><td>&minus;2&#7745;C<sub>p</sub>T<sub>in</sub></td></tr>
</table>
<p class="caption">Table 3.1 &mdash; The steady-state linear system solved by <code>np.linalg.solve</code>
for any power/flow/inlet-temperature combination, giving a physically self-consistent initial
condition for every simulation and every Monte Carlo scenario.</p>

<h2 id="s3-4">3.4 The Open-Loop Divergence Finding</h2>
<p>The MATLAB <code>Calculate_Kinetics.m</code> listing (Appendix A) takes reactivity as a bare
external argument, <code>rho_in</code>, with no fuel/moderator temperature feedback term &mdash;
despite the report's own Simulink block diagram (Fig.&nbsp;1 of the original report) visually
routing T<sub>f</sub> and T<sub>m</sub> into the kinetics block. Translated literally and subjected to
a sustained positive reactivity step, the coupled system's power &mdash; and therefore temperature
&mdash; grows without bound: a first end-to-end test run produced a peak fuel temperature of
5.6&times;10<sup>43</sup>&nbsp;K by t&nbsp;=&nbsp;400&nbsp;s. This is correct point-kinetics physics
for an open-loop reactivity insertion held constant forever, not a translation error: nothing in the
governing equation turns power growth over on its own once &rho;(t) is fixed above &beta;'s vicinity.</p>
<p>An explicit, clearly-labelled Doppler and moderator temperature-feedback term was therefore added
as the default reactivity model (<code>doppler_feedback_reactivity</code>, Appendix B), documented in
the code as an addition beyond the literal source listing &mdash; not report data. The literal,
feedback-free translation remains available as <code>external_step_reactivity</code> for anyone who
wants the unmodified equations.</p>

<h2 id="s3-5">3.5 Verification by Sanity Check</h2>
<p>Before the physics engine was trusted as a foundation for the Monte Carlo pipeline, three
independent checks were run:</p>
<table class="data">
<tr><th>Check</th><th>Expected property</th><th>Observed residual / result</th></tr>
<tr><td>Steady-state fixed point</td><td><code>rhs(0, steady_state())</code> should be the zero vector</td><td>max|dy/dt| &asymp; 1.8&times;10&minus;14</td></tr>
<tr><td>Precursor equilibrium</td><td>dC<sub>i</sub>/dt = 0 at the algebraic equilibrium</td><td>max residual &asymp; 4.4&times;10&minus;16</td></tr>
<tr><td>Negative reactivity stability</td><td>power should decay toward zero, not diverge</td><td>confirmed; final power &asymp; 2.3&times;10&minus;6</td></tr>
</table>
<p class="caption">Table 3.2 &mdash; Verification results for the coupled ODE system, all at
floating-point precision.</p>

<h2 id="s3-6">3.6 Validation Against Literature Benchmark Data</h2>
<p>The original report's Fig.&nbsp;3 validates the thermal-core block's dynamic response against
literature HTTR data at what appears to be a smaller operating point than this project's scaled 300
MWth design &mdash; the report's presentation deck resolves this: the real HTTR is a ~30&nbsp;MWth
research reactor, scaled &times;10 to 300&nbsp;MWth specifically because a 30&nbsp;MWth plant
produces freshwater for only ~8,000 people against the target city of 80,000, and because the
project's 27% IRR depends on the economies of scale only the larger design achieves. The presentation
deck's quantified steady-state validation figures of merit are reproduced below:</p>
<table class="data">
<tr><th>Figure of merit</th><th>Steady-state deviation</th><th>Series type</th><th>Validated?</th></tr>
<tr><td>Outlet coolant temperature</td><td>0.3%</td><td>Time series</td><td>Yes</td></tr>
<tr><td>Moderator temperature</td><td>~1.1%</td><td>Time series</td><td>Yes</td></tr>
<tr><td>Average fuel block temperature</td><td>~7.3%</td><td>Time series</td><td>Partially</td></tr>
<tr><td>Inlet coolant temperature</td><td>&lt;0.5%</td><td>Scalar</td><td>Yes</td></tr>
</table>
<p class="caption">Table 3.3 &mdash; Reproduced from the project presentation deck, &sect;4.4.2. The
fuel node's larger deviation is attributed by the original report to un-tuned PID control producing
oscillation, not to an energy-balance error &mdash; consistent with the fuel node also being the
fastest-responding, most oscillatory node in this project's own re-implementation (Part VII.1).</p>
</div>
""")

# ===========================================================================
# PART IV
# ===========================================================================
BODY.append("""
<div class="page part-break">
<h1 id="part4">Part IV &mdash; Monte Carlo Telemetry Generation</h1>

<h2 id="s4-1">4.1 Sampling Methodology and Operating Envelope</h2>
<p><code>data_pipeline.py</code> (Appendix C) samples 1,000 operating points uniformly at random
within realistic bounds tied directly to the original report's own stated values:</p>
<table class="data">
<tr><th>Parameter</th><th>Bounds</th><th>Basis</th></tr>
<tr><td>Helium mass flow</td><td>78 &ndash; 132 kg/s</td><td>&plusmn;25% around the report's 105 kg/s baseline (&sect;1.3)</td></tr>
<tr><td>Coolant inlet temperature</td><td>593.15 &ndash; 653.15 K</td><td>&plusmn;30 K around the 350&deg;C compressor-outlet estimate (Table 9)</td></tr>
<tr><td>Reactivity step (absolute)</td><td>&minus;0.0030 &ndash; +0.0030</td><td>Comfortably inside &plusmn;&beta;<sub>total</sub> = 0.0075</td></tr>
<tr><td>Reactivity step timing</td><td>50 &ndash; 150 s</td><td>Centred on the report's t = 100 s validation step</td></tr>
<tr><td>Turbine inlet pressure</td><td>6.3 &ndash; 7.7 MPa</td><td>Around the report's 70 bar reactor-outlet pressure</td></tr>
<tr><td>Turbine back pressure</td><td>3.8 &ndash; 5.0 MPa</td><td>Around the report's 45 bar gas-turbine-exit pressure</td></tr>
<tr><td>Turbine isentropic efficiency</td><td>0.85 &ndash; 0.92</td><td>Typical industrial range (not given numerically in the report)</td></tr>
</table>
<p class="caption">Table 4.1 &mdash; Monte Carlo sampling bounds. Each scenario's starting state is
derived via <code>steady_state()</code> for that scenario's own flow/inlet-temperature combination
(Part III.3) rather than a shared, potentially inconsistent initial condition.</p>

<h2 id="s4-2">4.2 Helium Turbine Work Extension</h2>
<p>Each scenario's post-transient reactor-outlet gas state is passed through
<code>helium_turbine_work()</code>, a direct Python/CoolProp port of the original report's
<code>helium_turbine.m</code> (Appendix A): an isentropic expansion from (T<sub>in</sub>, P<sub>in</sub>)
to P<sub>back</sub> at the sampled isentropic efficiency, using real CoolProp helium enthalpy and
entropy lookups rather than an ideal-gas assumption.</p>

<h2 id="s4-3">4.3 Dataset Characteristics</h2>
<p>All 1,000 sampled scenarios converged: zero ODE solver failures, zero missing turbine-work
values, in approximately 113 seconds of wall-clock time. The resulting <code>htgr_telemetry.csv</code>
carries 23 columns per row &mdash; the 7 sampled inputs, solver metadata, initial/peak/final
temperatures for all three thermal nodes, final normalised power, and the two turbine outputs &mdash;
and forms the training set for Part V's surrogate model.</p>
</div>
""")

print("Parts III-IV assembled")

# ===========================================================================
# PART V
# ===========================================================================
BODY.append("""
<div class="page part-break">
<h1 id="part5">Part V &mdash; Scalable Probabilistic Surrogate Modelling</h1>

<h2 id="s5-1">5.1 The O(N&sup3;) Bottleneck of Exact Gaussian Process Regression</h2>
<p>A Gaussian Process (GP) is the natural first choice for this problem: it is a distribution over
functions with calibrated predictive uncertainty built in in by construction. Given N training
points, its posterior mean and variance at a query point x<sub>&lowast;</sub> are</p>
<blockquote>
&mu;<sub>&lowast;</sub>(x) = k(x)<sup>&#8868;</sup> (K + &sigma;<sub>n</sub><sup>2</sup>I)<sup>&minus;1</sup> y<br/>
&sigma;<sub>&lowast;</sub><sup>2</sup>(x) = k(x,x) &minus; k(x)<sup>&#8868;</sup> (K + &sigma;<sub>n</sub><sup>2</sup>I)<sup>&minus;1</sup> k(x)
</blockquote>
<p>where K &isin; &#8477;<sup>N&times;N</sup> is the Gram matrix (K<sub>ij</sub> = k(x<sub>i</sub>,x<sub>j</sub>)
for the RBF kernel), k(x) &isin; &#8477;<sup>N</sup> is the vector of kernel evaluations against every
training point, and &sigma;<sub>n</sub><sup>2</sup> is observation noise. The bottleneck is
(K+&sigma;<sub>n</sub><sup>2</sup>I)<sup>&minus;1</sup>: computing it exactly via Cholesky
factorisation costs O(N&sup3;) time and O(N&sup2;) memory, plus O(N) per prediction for the mean and
O(N&sup2;) for the variance. At N&nbsp;=&nbsp;800 (this project's training split) that cost is
trivial &mdash; but the point of this architecture is to survive a Monte Carlo sweep one or two
orders of magnitude larger than 1,000 runs without a rewrite. At N&nbsp;=&nbsp;50,000, O(N&sup3;)
&asymp; 1.25&times;10<sup>14</sup> floating-point operations: computationally dead on arrival.</p>

<h2 id="s5-2">5.2 The Nystr&ouml;m Kernel Approximation</h2>
<p>The Nystr&ouml;m method replaces the full kernel with a low-rank approximation built from m
&laquo; N landmark points sampled from the training set:</p>
<blockquote>K &asymp; K<sub>N,m</sub> K<sub>m,m</sub><sup>+</sup> K<sub>m,N</sub></blockquote>
<p>where K<sub>m,m</sub> &isin; &#8477;<sup>m&times;m</sup> is the small Gram matrix among the m
landmarks and K<sub>N,m</sub> &isin; &#8477;<sup>N&times;m</sup> holds the kernel evaluations between
every training point and those landmarks. scikit-learn's <code>Nystroem</code> transformer
eigendecomposes K<sub>m,m</sub> = U&Lambda;U<sup>&#8868;</sup> and builds an explicit
finite-dimensional feature map &phi;: &#8477;<sup>d</sup> &rarr; &#8477;<sup>m</sup>:</p>
<blockquote>&phi;(x) = &Lambda;<sup>&minus;1/2</sup>U<sup>&#8868;</sup>k<sub>m</sub>(x), &nbsp;&nbsp;&nbsp;&nbsp;
k(x,x&prime;) &asymp; &phi;(x)<sup>&#8868;</sup>&phi;(x&prime;)</blockquote>
<p>which converts nonlinear kernel regression into ordinary linear regression in an m-dimensional
space. Every downstream computation &mdash; including the predictive uncertainty &mdash; now scales
with m, not N.</p>

<h2 id="s5-3">5.3 Bayesian Ridge Regression on Nystr&ouml;m Features</h2>
<p>With an explicit &phi;(x), <code>BayesianRidge</code> fits a linear model with Gaussian priors on
both the weights and the observation noise:</p>
<blockquote>y = &phi;(x)<sup>&#8868;</sup>w + &epsilon;, &nbsp;&nbsp; &epsilon; ~ &Nopf;(0, &tau;<sup>&minus;1</sup>), &nbsp;&nbsp;
w ~ &Nopf;(0, &alpha;<sup>&minus;1</sup>I)</blockquote>
<p>Gamma hyperpriors on the noise precision &tau; and weight precision &alpha; are resolved by
evidence maximisation (type-II maximum likelihood / empirical Bayes) directly from the training
data &mdash; no manually-tuned regularisation constant. Given &Phi; &isin; &#8477;<sup>N&times;m</sup>
(the stacked Nystr&ouml;m features), the posterior over w is closed-form:</p>
<blockquote>&Sigma;<sub>w</sub> = (&alpha;I + &tau;&thinsp;&Phi;<sup>&#8868;</sup>&Phi;)<sup>&minus;1</sup>, &nbsp;&nbsp;&nbsp;&nbsp;
&mu;<sub>w</sub> = &tau;&thinsp;&Sigma;<sub>w</sub>&thinsp;&Phi;<sup>&#8868;</sup>y</blockquote>
<p>Forming &Phi;<sup>&#8868;</sup>&Phi; costs O(Nm&sup2;); inverting the resulting m&times;m matrix
costs O(m&sup3;) &mdash; independent of N. For a new query point x<sub>&lowast;</sub>, with
&phi;<sub>&lowast;</sub> = &phi;(x<sub>&lowast;</sub>):</p>
<blockquote>&mu;<sub>&lowast;</sub> = &phi;<sub>&lowast;</sub><sup>&#8868;</sup>&mu;<sub>w</sub>
&nbsp;&nbsp;(predictive mean) &nbsp;&nbsp;&nbsp;&nbsp;
&sigma;<sub>&lowast;</sub><sup>2</sup> = &tau;<sup>&minus;1</sup> + &phi;<sub>&lowast;</sub><sup>&#8868;</sup>&Sigma;<sub>w</sub>&phi;<sub>&lowast;</sub>
&nbsp;&nbsp;(predictive variance)</blockquote>
<p>This is exactly what <code>pipeline.predict(X, return_std=True)</code> returns in
<code>ml_surrogate.py</code> (Appendix D) &mdash; and its correct kwarg passthrough through the
<code>Nystroem</code> transform step to <code>BayesianRidge.predict()</code> was verified in an
isolated smoke test <em>before</em> the multi-minute grid search was run on top of it (Part
VIII.3).</p>

<h2 id="s5-4">5.4 Complexity Comparison</h2>
<table class="data">
<tr><th></th><th>Exact GP</th><th>Nystr&ouml;m + Bayesian ridge (this project)</th></tr>
<tr><td>Training time</td><td>O(N&sup3;)</td><td>O(Nm&sup2; + m&sup3;)</td></tr>
<tr><td>Prediction time (mean)</td><td>O(N)</td><td>O(m)</td></tr>
<tr><td>Prediction time (variance)</td><td>O(N&sup2;)</td><td>O(m&sup2;)</td></tr>
<tr><td>Memory</td><td>O(N&sup2;)</td><td>O(Nm + m&sup2;)</td></tr>
<tr><td>Uncertainty quality</td><td>Exact, for the chosen kernel</td><td>Approximate &mdash; see &sect;5.7</td></tr>
</table>
<p class="caption">Table 5.1 &mdash; With N=800 and m&le;300 the gap is already meaningful; it becomes
decisive with every additional order of magnitude of Monte Carlo data.</p>

<h2 id="s5-5">5.5 Hyperparameter Optimisation</h2>
<p>Each of the two target variables (<code>peak_fuel_temp_K</code>, <code>turbine_work_MW</code>)
gets its own independently-fitted pipeline, with 5-fold cross-validated grid search over the
Nystr&ouml;m RBF bandwidth &gamma; &isin; {10<sup>&minus;3</sup>, 10<sup>&minus;2</sup>,
10<sup>&minus;1</sup>, 1} and landmark count m &isin; {100, 200, 300}, scored by R&sup2;.</p>

<h2 id="s5-6">5.6 Model Evaluation and Calibration</h2>
<table class="data">
<tr><th></th><th>Peak Fuel Temperature</th><th>Turbine Work Output</th></tr>
<tr><td>Test R&sup2;</td><td>0.9946</td><td>1.0000</td></tr>
<tr><td>Test RMSE</td><td>4.85 K</td><td>0.0033 MW</td></tr>
<tr><td>Mean predictive &sigma;</td><td>4.88 K</td><td>0.0025 MW</td></tr>
<tr><td>95% CI coverage (&plusmn;2&sigma;)</td><td>96.5%</td><td>92.0%</td></tr>
<tr><td>Best hyperparameters</td><td>&gamma;=0.001, m=100</td><td>&gamma;=0.001, m=300</td></tr>
</table>
<p class="caption">Table 5.2 &mdash; Held-out test-set performance, from a 5-fold cross-validated grid
search on an 80/20 train/test split (random_state=0).</p>
<p><code>turbine_work_MW</code> fitting to R&sup2;&nbsp;&asymp;&nbsp;1.0000 does not indicate
overfitting or data leakage: the telemetry is generated by a deterministic simulator with no
injected observation noise, and turbine work is a smooth composition of the physics engine's output
through CoolProp's helium enthalpy relations, so a sufficiently flexible model fitting it near-exactly
is the expected outcome, not a red flag. The 96.5%/92.0% coverage figures are reported as measured,
not asserted as perfectly calibrated &mdash; the nominal target for a &plusmn;2&sigma; interval is
95%.</p>

<h2 id="s5-7">5.7 The Honest Limitation of the Approximation</h2>
<p>&phi; is built from a fixed, global set of m landmarks chosen once at training time. An exact
RBF-kernel GP's uncertainty is <em>guaranteed by construction</em> to widen with distance from
training data, because k(x<sub>&lowast;</sub>, x<sub>i</sub>) &rarr; 0 for a stationary kernel as
distance grows. A Nystr&ouml;m-approximated linear model offers no such guarantee: nothing forces
&phi;<sub>&lowast;</sub><sup>&#8868;</sup>&Sigma;<sub>w</sub>&phi;<sub>&lowast;</sub> to grow smoothly
outside the span of the training landmarks. This is a structural limitation of the chosen
architecture, not a footnote, and it directly motivates the second, independent check built into the
autonomous agent in Part VI.</p>
</div>
""")

# ===========================================================================
# PART VI
# ===========================================================================
BODY.append("""
<div class="page part-break">
<h1 id="part6">Part VI &mdash; HTGRAgent: An Autonomous Safety Decision Layer</h1>

<h2 id="s6-1">6.1 Motivation and Design Philosophy</h2>
<p>A trained surrogate that returns a mean and a standard deviation is not, by itself, a safety
system: something still has to decide what those two numbers <em>mean</em> for a given candidate
operating point. <code>HTGRAgent</code> (Appendix D) is a plain-Python decision layer performing
that translation, structurally analogous to the original report's own reactor control philosophy
(Part &sect;8.1 of the original report: automatic shutdown at 90% prompt criticality) &mdash; both
are automated systems that continuously check an operating point against a hard physical limit and
intervene before it is crossed, one implemented in a physical control loop, the other in software.</p>

<h2 id="s6-2">6.2 Conditional Decision Logic</h2>
<p>Every call to <code>HTGRAgent.evaluate(scenario)</code> runs three independent checks, each
capable of escalating the verdict from NOMINAL toward UNCERTAIN or UNSAFE:</p>
<table class="data">
<tr><th>#</th><th>Check</th><th>Trigger condition</th><th>Escalates to</th></tr>
<tr><td>1</td><td>Extrapolation guard</td><td>Any of the 7 input features lies outside the exact
[min, max] range observed during training</td><td>UNCERTAIN</td></tr>
<tr><td>2</td><td>Uncertainty threshold</td><td>Predictive &sigma; on peak_fuel_temp_K exceeds
mean(test &sigma;) + 2&middot;std(test &sigma;), a data-driven cutoff computed once after
training</td><td>UNCERTAIN</td></tr>
<tr><td>3</td><td>Safety margin</td><td>&mu; + z&sigma; (z=2) on peak_fuel_temp_K exceeds
1650&nbsp;K</td><td>UNSAFE (overrides all)</td></tr>
</table>
<p class="caption">Table 6.1 &mdash; The agent's three-check decision logic, implemented in
<code>HTGRAgent.evaluate()</code>, Appendix D.</p>
<p>Check 3 deliberately uses the upper confidence bound rather than the raw predictive mean: a
scenario whose mean prediction sits under 1650&nbsp;K but whose uncertainty band pushes the credible
upper bound past it is still flagged unsafe.</p>

<h2 id="s6-3">6.3 The Deterministic Extrapolation Guard</h2>
<p>As established in Part V.7, the surrogate's self-reported &sigma; is not guaranteed to grow with
distance from the training distribution the way an exact GP's would. Check&nbsp;1 therefore does not
rely on statistics at all &mdash; it is a literal bounds comparison against
<code>HTGRSurrogate.feature_bounds</code>, computed once from the training data's exact per-feature
minimum and maximum. Any query point outside those bounds is extrapolation by definition, regardless
of what the model's &sigma; happens to report for it.</p>

<h2 id="s6-4">6.4 Root-Cause Recommendation Engine</h2>
<p>When the verdict is not NOMINAL, <code>_recommend()</code> generates concrete adjustments tied
directly to the <code>core_thermals.m</code> energy balance, not a black-box output:</p>
<table class="data">
<tr><th>Trigger</th><th>Recommendation</th><th>Physical mechanism</th></tr>
<tr><td>Flow below the training midpoint</td><td>Increase helium_mass_flow_kgs</td><td>More
convective heat removal via the 2&#7745;C<sub>p</sub>(T<sub>c</sub>&minus;T<sub>in</sub>) term</td></tr>
<tr><td>rho_insertion &gt; 0</td><td>Reduce the positive reactivity step</td><td>Less fission heat
generated (Q<sub>fission</sub> = P&middot;Q<sub>nominal</sub>)</td></tr>
<tr><td>Inlet temperature above the training midpoint</td><td>Lower coolant_inlet_temp_K</td><td>Larger
driving &Delta;T for heat removal</td></tr>
</table>

<h2 id="s6-5">6.5 Demonstration Scenarios</h2>
<p>Two scenarios, imported as module-level constants (<code>SAFE_SCENARIO_DEMO</code>,
<code>UNSAFE_SCENARIO_DEMO</code>) so the CLI demo, the notebook, and the dashboard can never drift
apart:</p>
<table class="data">
<tr><th>Scenario</th><th>Key inputs</th><th>Predicted peak fuel temp</th><th>Verdict</th></tr>
<tr><td>Moderate, in-distribution</td><td>110 kg/s, 605 K, &minus;0.0005 &rho;</td><td>1485.6 K
(&sigma;=4.7)</td><td><b>NOMINAL</b></td></tr>
<tr><td>Low flow, hot inlet, strong positive &rho;</td><td>70 kg/s (below training floor), 652 K,
+0.0029 &rho;</td><td>1806.1 K (&sigma;=6.4)</td><td><b>UNSAFE</b> &mdash; all three checks fire
simultaneously</td></tr>
</table>
</div>
""")

print("Parts V-VI assembled")

# ===========================================================================
# PART VII
# ===========================================================================
BODY.append("""
<div class="page part-break">
<h1 id="part7">Part VII &mdash; Interactive Systems</h1>

<h2 id="s7-1">7.1 The Systems Textbook Notebook</h2>
<p><code>notebooks/HTGR_Systems_Textbook.ipynb</code> is a seventeen-chapter, six-part walkthrough of
the entire engineering chain &mdash; reactor down-selection, point kinetics, thermal-hydraulics,
coolant fluid mechanics, the combined Brayton&ndash;Rankine cycle, MED desalination, DEEP economics,
CO&sub2; sustainability, reactor safety systems, and the independent heat-rejection system &mdash;
built from both the original report and its accompanying presentation deck, with every worked
example independently recomputed and checked against the source material's own stated results before
being written into the notebook. Two genuine source-document numerical inconsistencies were caught
this way (Part VIII.5) rather than silently reconciled.</p>

<h2 id="s7-2">7.2 The Streamlit Operating-Point Explorer</h2>
<p><code>app.py</code> (Appendix E) is a live dashboard over the trained surrogate: three sliders
(helium mass flow, coolant inlet temperature, reactivity step in dollars, converted internally via
&rho; = dollars &times; &beta;<sub>total</sub>), metric cards for predicted peak fuel temperature and
turbine work with &plusmn;2&sigma; bands, a Plotly gauge visualising the operating point against the
1650&nbsp;K safety margin, and a live <code>HTGRAgent</code> diagnostic panel &mdash; the same
decision logic documented in Part VI, queried interactively rather than from a fixed demo script.</p>

<h2 id="s7-3">7.3 Styled Static Exports</h2>
<p>Because GitHub's hosted <code>.ipynb</code> renderer sanitises embedded CSS, notebook typography
and background colour cannot be controlled from within the notebook file itself when viewed there.
<code>scripts/build_styled_html.py</code> (Appendix F) works around this by post-processing
nbconvert's own HTML export &mdash; a file this project fully owns &mdash; with the same
Times-New-Roman-on-parchment stylesheet used throughout this report, guaranteeing identical rendering
in any browser regardless of what a third-party viewer's sanitisation rules permit.</p>
</div>
""")

# ===========================================================================
# PART VIII
# ===========================================================================
BODY.append("""
<div class="page part-break">
<h1 id="part8">Part VIII &mdash; Engineering Verification and Debug Log</h1>
<p>Kept here deliberately, in full, as a record of what broke during development, how it was
diagnosed, and how it was fixed &mdash; rather than cleaned out of the project history.</p>

<h3>VIII.1 Open-loop reactivity divergence (~10&sup4;&sup3; K)</h3>
<p><b>Symptom.</b> The first end-to-end run of the coupled ODE produced a peak fuel temperature of
56,397,188,346,951,049,257,910,745,257,165,469,201,203,200&nbsp;K by t=400&nbsp;s.
<b>Root cause.</b> Correct physics, not a code defect: a constant positive reactivity insertion below
prompt-critical still produces unbounded exponential power growth on a stable reactor period unless
something external to the equation forces &rho; back down (Part III.4). <b>Fix.</b> Added an
explicit Doppler/moderator feedback term as the default reactivity model, documented as an addition
beyond the literal MATLAB listing. <b>Lesson.</b> Translate the equations exactly, then test the
equations &mdash; a perfect translation of an incomplete model is still an incomplete model.</p>

<h3>VIII.2 A joblib pickling bug with two heads</h3>
<p><b>Symptom.</b> <code>jupyter nbconvert --execute</code> failed loading the trained surrogate:
<code>AttributeError: Can't get attribute 'TargetModel' on &lt;module '__main__'&gt;</code>.
<b>Root cause.</b> Running <code>python src/ml_surrogate.py</code> directly executes the module as
<code>__main__</code>; joblib pickles the <code>TargetModel</code>/<code>HTGRSurrogate</code>
dataclasses under whatever module they were defined in at pickle time. <b>Second head.</b> Adding
<code>app.py</code> with <code>from src.ml_surrogate import ...</code> introduced a third possible
module path, meaning one saved artefact could only ever satisfy one consumer at a time. <b>Real
fix.</b> Standardised the whole project on <code>src.ml_surrogate</code> as the single canonical
import path, and added a self-detecting guard inside <code>main()</code> that prints the exact
failure mode and the correct regeneration command if triggered, rather than silently saving a broken
artefact. <b>Lesson.</b> A bug that reproduces under a different trigger is not a new bug &mdash; fix
the class of failure, not the instance.</p>

<h3>VIII.3 Verifying an API before building on it</h3>
<p><b>Risk.</b> The entire uncertainty-quantification story (Part V) depends on
<code>Pipeline.predict(X, return_std=True)</code> correctly forwarding <code>return_std</code>
through a <code>Nystroem</code> transform step to <code>BayesianRidge.predict()</code> &mdash;
documented behaviour, not something to assume across scikit-learn versions without checking.
<b>Action.</b> A ten-line isolated smoke test confirmed the kwarg passthrough before the multi-minute
grid search was run on top of it. <b>Lesson.</b> Verify a load-bearing assumption in isolation before
building the full system on it &mdash; a broken assumption found after a five-minute grid search is a
far more expensive bug to diagnose.</p>

<h3>VIII.4 A Mermaid rendering failure and a Markdown/LaTeX collision</h3>
<p>Two independent presentation-layer bugs surfaced during documentation: a Mermaid diagram using the
less-portable cylinder/database node shape failed to render on GitHub with
<code>Cannot read properties of undefined (reading 'render')</code>, fixed by reverting to a plain
rectangle node consistent with every other node in the project's diagrams; and the conventional
&ldquo;star&rdquo; notation x<sub>&lowast;</sub>, written as a bare <code>*</code> inside <code>$$&hellip;$$</code>
LaTeX blocks, was being consumed by Markdown's emphasis parser before KaTeX ever saw it, corrupting
subscripts such as &sigma;<sub>&lowast;</sub><sup>2</sup>. Fixed by replacing every literal
<code>*</code> inside a math span with the LaTeX command <code>\\ast</code>, which contains no literal
asterisk character for Markdown to misinterpret.</p>

<h3>VIII.5 Source-document numerical discrepancies, found and not silently reconciled</h3>
<p>Two independent inconsistencies in the original report's own figures were found while
independently recomputing its worked examples for the systems-textbook notebook: the heat-rejection
condensate pump's itemised fitting-loss coefficients (6 elbows + 2 valves + 1 check valve) sum to
&Sigma;K&nbsp;=&nbsp;5.0, but the report's own downstream result (h<sub>m</sub>&nbsp;=&nbsp;2.98&nbsp;m)
only reproduces using &Sigma;K&nbsp;=&nbsp;6.5 as stated; and the report's Table&nbsp;8 states a
heat-rejection footprint of 60,214&nbsp;m&sup2;, while independently recomputing it from the given
Kelvion module dimensions and the report's own module count gives &asymp;6,076&nbsp;m&sup2;,
matching the accompanying presentation deck's 6,021.4&nbsp;m&sup2; almost exactly &mdash; strongly
suggesting a misplaced decimal point in the report table rather than a genuine order-of-magnitude
design difference. Both are flagged explicitly in the systems-textbook notebook rather than silently
picked one way or the other.</p>
</div>
""")

# ===========================================================================
# PART IX
# ===========================================================================
BODY.append("""
<div class="page part-break">
<h1 id="part9">Part IX &mdash; Results Summary</h1>
<table class="data">
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Monte Carlo scenarios sampled / converged</td><td>1,000 / 1,000</td></tr>
<tr><td>Solver failures</td><td>0</td></tr>
<tr><td>peak_fuel_temp_K &mdash; test R&sup2; / RMSE / 95% coverage</td><td>0.9946 / 4.85 K / 96.5%</td></tr>
<tr><td>turbine_work_MW &mdash; test R&sup2; / RMSE / 95% coverage</td><td>1.0000 / 0.0033 MW / 92.0%</td></tr>
<tr><td>Training-cost scaling vs. exact GP</td><td>O(Nm&sup2;+m&sup3;) vs. O(N&sup3;)</td></tr>
<tr><td>HTGRAgent demo &mdash; in-distribution scenario</td><td>NOMINAL, 1485.6 K predicted</td></tr>
<tr><td>HTGRAgent demo &mdash; out-of-distribution scenario</td><td>UNSAFE, 1806.1 K predicted, all 3 checks fired</td></tr>
<tr><td>Physics-engine sanity-check residuals</td><td>&le; 1.8&times;10&minus;14 (floating-point precision)</td></tr>
<tr><td>Source-document discrepancies independently found and flagged</td><td>2</td></tr>
</table>
</div>
""")

# ===========================================================================
# PART X
# ===========================================================================
BODY.append("""
<div class="page part-break">
<h1 id="part10">Part X &mdash; Conclusions and Further Work</h1>
<p>This report has documented the construction of a Python digital twin over the Imperial HTGR&nbsp;5
/ HR&nbsp;5 reactor model: a verified, stiff-ODE physics engine translated one-to-one from the
original MATLAB listings (with one clearly-labelled addition needed to make the model physically
stable); a 1,000-run Monte Carlo telemetry generator extending that physics with real CoolProp
thermodynamics; a scalable Nystr&ouml;m-plus-Bayesian-ridge probabilistic surrogate replacing an
intractable exact Gaussian Process at a small, honestly-stated cost in uncertainty-quality
guarantees; and an autonomous safety agent built with two independent, complementary checks rather
than trusting a single fallible signal.</p>
<p>Further work follows directly from the limitations documented throughout this report: PID
controller tuning to reduce the fuel/moderator oscillation noted in Part III.6; replacing the
illustrative Doppler/moderator feedback coefficients of Part III.4 with measured values, should they
become available; extending the Monte Carlo sampling envelope and retraining the surrogate once a
larger telemetry dataset makes the O(N&sup3;) argument of Part V.1 bite in practice rather than in
principle; and resolving the two source-document numerical discrepancies of Part VIII.5 with the
original report's authors.</p>
</div>
""")

# ===========================================================================
# REFERENCES
# ===========================================================================
BODY.append("""
<div class="page part-break ref-list">
<h1 id="refs">References</h1>
<p>[1] Imperial HTGR 5 and HR 5 Team. <i>Scaled High Temperature Test Reactor Cogeneration for Zero
Emission Hybrid-Desalination with Independent Water Heat Rejection.</i> Imperial College London,
2026. (This report's primary source; reproduced in Appendix A.)</p>
<p>[2] Imperial Team 5. <i>Scaled High Temperature Test Reactor Cogeneration for Zero-Emission
Hybrid Desalination with Water Independent Heat Rejection</i> (final presentation). Imperial College
London / Rolls-Royce, 2026.</p>
<p>[3] Williams, C.K.I. and Seeger, M. &ldquo;Using the Nystr&ouml;m Method to Speed Up Kernel
Machines.&rdquo; <i>Advances in Neural Information Processing Systems</i> 13, 2001.</p>
<p>[4] Rasmussen, C.E. and Williams, C.K.I. <i>Gaussian Processes for Machine Learning.</i> MIT
Press, 2006.</p>
<p>[5] MacKay, D.J.C. &ldquo;Bayesian Interpolation.&rdquo; <i>Neural Computation</i> 4(3), 415&ndash;447,
1992. (Evidence-maximisation basis for scikit-learn's <code>BayesianRidge</code>.)</p>
<p>[6] Pedregosa, F. et al. &ldquo;Scikit-learn: Machine Learning in Python.&rdquo; <i>Journal of
Machine Learning Research</i> 12, 2825&ndash;2830, 2011.</p>
<p>[7] Virtanen, P. et al. &ldquo;SciPy 1.0: Fundamental Algorithms for Scientific Computing in
Python.&rdquo; <i>Nature Methods</i> 17, 261&ndash;272, 2020.</p>
<p>[8] Bell, I.H., Wronski, J., Quoilin, S., and Lemort, V. &ldquo;Pure and Pseudo-pure Fluid
Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp.&rdquo;
<i>Industrial &amp; Engineering Chemistry Research</i> 53(6), 2498&ndash;2508, 2014.</p>
<p>[9] McKinney, W. &ldquo;Data Structures for Statistical Computing in Python.&rdquo; <i>Proceedings
of the 9th Python in Science Conference</i>, 2010.</p>
</div>
""")

print("Parts VII-X and references assembled")

# ===========================================================================
# APPENDICES
# ===========================================================================
BODY.append("""
<div class="page part-break appendix-section">
<h1 id="appendices">Appendices</h1>
<p>Every appendix below reproduces complete, unedited source code: Appendix A is transcribed verbatim
from the original Imperial HTGR&nbsp;5 / HR&nbsp;5 report's own appendices (MATLAB, Simulink Coder
compatible); Appendices B&ndash;F are read directly from this project's source files at the time this
report was generated, so they are guaranteed to match the repository exactly.</p>
</div>
""")

BODY.append(f"""
<div class="page appendix-section">
<h2 id="appA">Appendix A &mdash; Original MATLAB/Simulink Source Code</h2>
<p>Reproduced from the Imperial HTGR&nbsp;5 and HR&nbsp;5 report, Appendix B (Listings 1&ndash;6) and
Appendix D (Listing 7). This is the exact code that <code>htgr_physics.py</code> (Appendix B of this
report) and <code>data_pipeline.py</code>'s <code>helium_turbine_work()</code> (Appendix C) are
direct translations of.</p>

<div class="codecap">A.1 &mdash; core_thermals.m (3-node thermal core)</div>
{code_block_str(MATLAB_CORE_THERMALS)}

<div class="codecap">A.2 &mdash; Calculate_Kinetics.m (6-group point kinetics)</div>
{code_block_str(MATLAB_KINETICS)}

<div class="codecap">A.3 &mdash; helium_turbine.m (isentropic helium expansion, CoolProp)</div>
{code_block_str(MATLAB_HELIUM_TURBINE)}

<div class="codecap">A.4 &mdash; steam_turbine_measured.m (isentropic steam expansion, CoolProp)</div>
{code_block_str(MATLAB_STEAM_TURBINE)}

<div class="codecap">A.5 &mdash; hybrid_desal_tvc.m (MED with thermal vapour compression)</div>
{code_block_str(MATLAB_HYBRID_DESAL)}

<div class="codecap">A.6 &mdash; hrsg_boiler.m (Heat Recovery Steam Generator energy balance)</div>
{code_block_str(MATLAB_HRSG)}

<div class="codecap">A.7 &mdash; steam_sponge_thermosyphon_PDC_Q_SWEEP_ALLINONE.m (porous-cube heat rejection sizing model)</div>
{code_block_str(MATLAB_HR_POROUS)}
</div>
""")

BODY.append(f"""
<div class="page appendix-section">
<h2 id="appB">Appendix B &mdash; Python Physics Engine</h2>
<p><code>src/htgr_physics.py</code> &mdash; the direct one-to-one translation of Appendix A.1 and A.2
into a coupled 10-state SciPy ODE system (Part III).</p>
{PY_PHYSICS}
</div>
""")

BODY.append(f"""
<div class="page appendix-section">
<h2 id="appC">Appendix C &mdash; Python Monte Carlo Data Pipeline</h2>
<p><code>src/data_pipeline.py</code> &mdash; the 1,000-run Monte Carlo telemetry generator, including
the direct Python/CoolProp port of Appendix A.3 (Part IV).</p>
{PY_PIPELINE}
</div>
""")

BODY.append(f"""
<div class="page appendix-section">
<h2 id="appD">Appendix D &mdash; Python ML Surrogate and Agent</h2>
<p><code>src/ml_surrogate.py</code> &mdash; the Nystr&ouml;m/Bayesian-ridge scalable surrogate (Part V)
and the <code>HTGRAgent</code> autonomous decision layer (Part VI).</p>
{PY_SURROGATE}
</div>
""")

BODY.append(f"""
<div class="page appendix-section">
<h2 id="appE">Appendix E &mdash; Streamlit Dashboard</h2>
<p><code>app.py</code> &mdash; the interactive operating-point explorer (Part VII.2).</p>
{PY_APP}
</div>
""")

BODY.append(f"""
<div class="page appendix-section">
<h2 id="appF">Appendix F &mdash; Notebook Styling Utility</h2>
<p><code>scripts/build_styled_html.py</code> &mdash; regenerates the styled standalone HTML exports of
both project notebooks, including this report's own typographic system (Part VII.3).</p>
{PY_BUILDHTML}
<div class="footer-note">End of report. Generated {TODAY} from the HTGR-ML-Optimization repository
source tree; Appendices B&ndash;F are byte-identical to the committed source files at generation
time.</div>
</div>
""")

# ===========================================================================
# WRITE OUT
# ===========================================================================
html_doc = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n<head>\n<meta charset="utf-8"/>\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1"/>\n'
    "<title>HTGR ML Digital Twin Report</title>\n"
    + CSS
    + "\n</head>\n<body>\n"
    + "\n".join(BODY)
    + "\n</body>\n</html>\n"
)
OUT.write_text(html_doc, encoding="utf-8")
print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")






