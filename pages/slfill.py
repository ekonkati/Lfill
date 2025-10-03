# app.py
# -------------------------------------------------------------------
# Independent SLF (Sanitary Landfill) Designer & Visualizer
# - No dependency on any Excel workbook
# - All computations in pure Python
# - Single page, no sidebar, compact/mobile friendly
# - Visualizations: plan, longitudinal section, cross-section, simple 3D
# - Hydrology (rational method), leachate estimate, drain sizing (Manning),
#   capacity/life, bearing/contact check, quick slope stability (infinite slope),
#   settlement/void ratio option (simplified).
# - Exports only the final, non-cost results to Excel (multi-sheet)
#
# Notes:
# * This is a professional-grade scaffold: formulas are explicit and easy to
#   tune to your local standards, manuals (CPCB, CPHEEO), and project specifics.
# * If you need code-level parity to a previous workbook, paste the exact
#   algebra into the calc_* functions marked below.
# -------------------------------------------------------------------

import io
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- UI: single page, compact, no sidebar ----------------
st.set_page_config(page_title="Independent SLF Designer", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  [data-testid="collapsedControl"], header [data-testid="stToolbar"]{display:none!important;}
  .block-container{padding-top:0.8rem; padding-bottom:1rem; max-width:1400px;}
  .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"]{height:2rem;font-size:.92rem;}
  .stButton>button{height:2.2rem; padding:0 .9rem;}
  .small-note{font-size:.85rem; opacity:.85;}
  .warn{color:#a33;font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.title("Independent SLF Designer & Visualizer")

# ---------------- Data models ----------------
@dataclass
class ProjectInfo:
    name: str = "My SLF Project"
    location: str = "City, State"
    benchmark_rl: float = 0.0  # reference RL for sections

@dataclass
class GeometryInputs:
    length_m: float = 600.0             # site length
    width_m: float = 60.0               # site width (inner flat region)
    depth_below_gl_m: float = 2.0       # excavation below ground level
    bund_top_width_m: float = 5.0       # crest/top width
    outer_slope_h: float = 3.0          # 1 : outer_slope_h
    inner_slope_h: float = 3.0          # 1 : inner_slope_h (above bund on landfill side)
    lift_height_m: float = 5.0
    bench_width_m: float = 4.0
    num_lifts: int = 4
    edge_gap_m: float = 1.0             # gap between bund toe and drain/path
    drain_width_m: float = 1.0

@dataclass
class WasteOps:
    tpd: float = 800.0                  # incoming waste (t/day)
    bulk_density_tpm3: float = 0.85     # placed/compacted density
    daily_cover_pct: float = 10.0       # % of waste volume
    intermediate_cover_pct: float = 5.0 # additional % at phase ends
    final_cover_thk_m: float = 1.0      # final cover thickness (top area)
    settlement_pct: float = 15.0        # final settlement (void collapse) %

@dataclass
class Hydrology:
    runoff_coeff: float = 0.8           # C for rational method
    design_intensity_mmphr: float = 60  # i (mm/hr)
    drainage_slope: float = 0.5/100.0   # S for Manning (m/m)
    manning_n: float = 0.013            # roughness (e.g., lined trapezoidal/rectangular)
    leachate_fraction_runoff: float = 0.25  # fraction of runoff that becomes leachate over waste body

@dataclass
class Geotech:
    phi_deg: float = 30.0               # waste shear friction angle
    cohesion_kpa: float = 5.0           # small effective cohesion
    unit_weight_kNpm3: float = 9.0      # γ (kN/m3) waste mass (saturated ~9-12)
    seismic_kh: float = 0.05            # pseudo-static horizontal coeff (opt)
    foundation_sbc_kPa: float = 200.0   # allowable bearing capacity
    foundation_footprint_factor: float = 0.9  # actual bearing area ratio due to local effects

# ---------------- Calculation helpers ----------------
def calc_stack_height(geom: GeometryInputs) -> float:
    return geom.num_lifts * geom.lift_height_m

def calc_outer_footprint(geom: GeometryInputs) -> Tuple[float, float]:
    """Approximate outer plan envelope: base width grows by H * outer slope on both sides."""
    H = calc_stack_height(geom)
    grow = H * geom.outer_slope_h
    return (geom.length_m, geom.width_m + 2*grow)

def polygonal_volume_from_stair(geom: GeometryInputs) -> float:
    """
    Very practical stacked-berm volume inside bund.
    Simplified as: inner 'terrace' steps. We approximate the interior volume
    as sum of lift layers: each lift adds area = (inner width + k*bench_width)*length*lift_height slope allowance.
    Here we model inner advance by bench width per lift and apply inner slope to the face.
    """
    L = geom.length_m
    H = calc_stack_height(geom)
    # Inner width grows by bench width per lift; start from flat inner width
    w0 = geom.width_m
    area = 0.0
    curw = w0
    for k in range(geom.num_lifts):
        # trapezoid band due to inner slope face approx: horizontal advance = lift_height * inner_slope_h
        dx = geom.lift_height_m * geom.inner_slope_h
        # effective width for this lift (avg): curw + 0.5*(bench + slope advance)
        eff_w = curw + 0.5*(geom.bench_width_m + dx)
        area += eff_w * geom.lift_height_m
        curw += geom.bench_width_m
    vol = area * L  # m3
    # deduct excavation below GL (negative volume) -> actually adds capacity; we include it positively:
    vol += geom.length_m * geom.width_m * geom.depth_below_gl_m
    return max(vol, 0.0)

def calc_waste_capacity_m3(geom: GeometryInputs, waste: WasteOps) -> Dict[str, float]:
    """Compute gross and net capacity incl. covers and settlement."""
    gross_geo_vol = polygonal_volume_from_stair(geom)
    # Covers as volume fractions:
    daily_cover = waste.daily_cover_pct/100.0
    inter_cover = waste.intermediate_cover_pct/100.0
    # Assume final cover is a thickness over top envelope area
    _, Wouter = calc_outer_footprint(geom)
    top_area = geom.length_m * Wouter   # outer top plan (conservative upper bound)
    final_cover_vol = top_area * waste.final_cover_thk_m
    # Settlement reclaim (void collapse increases space for waste; treat as capacity boost)
    settlement_gain = gross_geo_vol * (waste.settlement_pct/100.0)
    net_for_waste = gross_geo_vol * (1.0 - daily_cover - inter_cover) - final_cover_vol + settlement_gain
    net_for_waste = max(net_for_waste, 0.0)
    return {
        "gross_geom_volume_m3": gross_geo_vol,
        "final_cover_volume_m3": final_cover_vol,
        "net_waste_volume_m3": net_for_waste
    }

def calc_life_years(geom: GeometryInputs, waste: WasteOps) -> Dict[str, float]:
    cap = calc_waste_capacity_m3(geom, waste)
    net_m3 = cap["net_waste_volume_m3"]
    # Convert TPD to m3/day using bulk density:
    m3_per_day = waste.tpd / max(waste.bulk_density_tpm3, 1e-6)
    days = net_m3 / max(m3_per_day, 1e-6)
    years = days/365.0
    return {"life_days": days, "life_years": years, **cap}

def rational_Q(C: float, i_mmphr: float, A_m2: float) -> float:
    """Rational method: Q = C i A; i in m/s, A in m2 => Q m3/s."""
    i_mps = (i_mmphr/1000.0) / 3600.0
    return C * i_mps * A_m2

def leachate_estimate(Q_runoff_m3s: float, frac: float) -> float:
    """Very simplified leachate flow as a fraction of runoff over waste body."""
    return Q_runoff_m3s * frac

def manning_rectangular_Q(n: float, A: float, R: float, S: float) -> float:
    """Manning: Q = (1/n) A R^(2/3) S^(1/2)"""
    return (1.0/max(n,1e-6)) * A * (R**(2.0/3.0)) * (math.sqrt(max(S,1e-8)))

def drain_capacity_rectangular(b: float, h: float, n: float, S: float) -> float:
    """Rectangular open channel approx."""
    A = b * h
    P = 2*h + b
    R = A / max(P, 1e-6)
    return manning_rectangular_Q(n, A, R, S)

def bearing_check(geom: GeometryInputs, ge: Geotech, waste: WasteOps) -> Dict[str, float]:
    """Uniform pressure estimate from total weight vs footprint & SBC check."""
    H = calc_stack_height(geom)
    # Outer footprint area (base bearing area) times empirical factor
    L, Wouter = calc_outer_footprint(geom)
    area = L * Wouter * ge.foundation_footprint_factor
    # Approximate total waste volume * unit weight + final cover:
    cap = calc_waste_capacity_m3(geom, waste)
    waste_vol = cap["net_waste_volume_m3"]
    # Convert unit weight (kN/m3) -> kPa when divided by area (m2) with height integration
    # Take resultant load as γ * volume (kN)
    total_load_kN = ge.unit_weight_kNpm3 * waste_vol
    q_kPa = (total_load_kN / max(area,1e-6))  # kN/m2 = kPa
    fos_bearing = ge.foundation_sbc_kPa / max(q_kPa, 1e-6)
    return {"bearing_pressure_kPa": q_kPa, "SBC_kPa": ge.foundation_sbc_kPa, "FOS_bearing": fos_bearing, "footprint_area_m2": area}

def infinite_slope_fos(phi_deg: float, c_kPa: float, gamma_kNpm3: float, z_m: float, beta_deg: float, kh: float=0.0) -> float:
    """
    Infinite slope FOS (drained, simple):
    FOS = (c/(γ z sinβ cosβ) + tanφ / tanβ) * (1 - kh * (cosβ / sinβ))
    This is a simplistic check; refine with your preferred method as needed.
    """
    beta = math.radians(max(min(beta_deg, 89.0), 1.0))
    phi = math.radians(phi_deg)
    term1 = (c_kPa / max(gamma_kNpm3 * z_m * math.sin(beta) * math.cos(beta), 1e-6))
    term2 = (math.tan(phi) / max(math.tan(beta), 1e-6))
    seismic_mod = max(1.0 - kh * (math.cos(beta)/max(math.sin(beta),1e-6)), 0.1)
    return (term1 + term2) * seismic_mod

def stability_checks(geom: GeometryInputs, gt: Geotech) -> Dict[str, float]:
    """
    Quick stability proxies:
    - Outer slope β_o = arctan(1/outer_slope_h)
    - Inner slope β_i = arctan(1/inner_slope_h)
    Evaluate FoS at representative depths.
    """
    z = max(calc_stack_height(geom)/2.0, 1.0)  # representative slice depth
    beta_o = math.degrees(math.atan(1.0/max(geom.outer_slope_h,1e-6)))
    beta_i = math.degrees(math.atan(1.0/max(geom.inner_slope_h,1e-6)))
    fos_outer = infinite_slope_fos(gt.phi_deg, gt.cohesion_kpa, gt.unit_weight_kNpm3, z, beta_o, gt.seismic_kh)
    fos_inner = infinite_slope_fos(gt.phi_deg, gt.cohesion_kpa, gt.unit_weight_kNpm3, z, beta_i, gt.seismic_kh)
    return {"beta_outer_deg": beta_o, "beta_inner_deg": beta_i, "FOS_outer": fos_outer, "FOS_inner": fos_inner, "rep_depth_m": z}

# ---------------- Inputs ----------------
with st.expander("1) Project & Site", expanded=True):
    p = ProjectInfo(
        name = st.text_input("Project name", "My SLF Project"),
        location = st.text_input("Location", "City, State"),
        benchmark_rl = st.number_input("Benchmark RL (m)", value=0.0, format="%.3f"),
    )

with st.expander("2) Geometry & Phasing", expanded=True):
    g = GeometryInputs(
        length_m = st.number_input("Site length L (m)", value=600.0, min_value=10.0, step=10.0),
        width_m = st.number_input("Inner width W (m)", value=60.0, min_value=10.0, step=1.0),
        depth_below_gl_m = st.number_input("Depth below GL (m)", value=2.0, min_value=0.0, step=0.1, format="%.2f"),
        bund_top_width_m = st.number_input("Bund crest/top width (m)", value=5.0, min_value=1.0, step=0.5, format="%.2f"),
        outer_slope_h = st.number_input("Outer slope (1 : H)", value=3.0, min_value=1.0, step=0.5, format="%.2f"),
        inner_slope_h = st.number_input("Inner slope (1 : H)", value=3.0, min_value=1.0, step=0.5, format="%.2f"),
        lift_height_m = st.number_input("Lift/bench height (m)", value=5.0, min_value=1.0, step=0.5, format="%.2f"),
        bench_width_m = st.number_input("Bench width (m)", value=4.0, min_value=1.0, step=0.5, format="%.2f"),
        num_lifts = st.number_input("Number of lifts", value=4, min_value=1, step=1),
        edge_gap_m = st.number_input("Edge gap to drain/path (m)", value=1.0, min_value=0.0, step=0.5, format="%.2f"),
        drain_width_m = st.number_input("Peripheral drain width (m)", value=1.0, min_value=0.5, step=0.5, format="%.2f"),
    )

with st.expander("3) Waste & Operations", expanded=True):
    w = WasteOps(
        tpd = st.number_input("Waste throughput (TPD)", value=800.0, min_value=10.0, step=10.0),
        bulk_density_tpm3 = st.number_input("Bulk density (t/m³)", value=0.85, min_value=0.3, step=0.05, format="%.2f"),
        daily_cover_pct = st.number_input("Daily cover (%)", value=10.0, min_value=0.0, step=1.0, format="%.1f"),
        intermediate_cover_pct = st.number_input("Intermediate cover (%)", value=5.0, min_value=0.0, step=1.0, format="%.1f"),
        final_cover_thk_m = st.number_input("Final cover thickness (m)", value=1.0, min_value=0.0, step=0.1, format="%.2f"),
        settlement_pct = st.number_input("Settlement allowance (%)", value=15.0, min_value=0.0, step=1.0, format="%.1f"),
    )

with st.expander("4) Hydrology & Leachate", expanded=True):
    h = Hydrology(
        runoff_coeff = st.number_input("Runoff coefficient C (0-1)", value=0.8, min_value=0.0, max_value=1.0, step=0.05),
        design_intensity_mmphr = st.number_input("Design rainfall intensity i (mm/hr)", value=60.0, min_value=1.0, step=1.0),
        drainage_slope = st.number_input("Drain longitudinal slope S (m/m)", value=0.5/100.0, min_value=0.0001, step=0.0001, format="%.4f"),
        manning_n = st.number_input("Manning n (lined channel)", value=0.013, min_value=0.010, step=0.001, format="%.3f"),
        leachate_fraction_runoff = st.number_input("Leachate fraction of runoff", value=0.25, min_value=0.0, max_value=1.0, step=0.05),
    )

with st.expander("5) Geotechnical / Stability", expanded=True):
    gt = Geotech(
        phi_deg = st.number_input("Waste friction angle φ (deg)", value=30.0, min_value=10.0, max_value=45.0, step=1.0),
        cohesion_kpa = st.number_input("Waste cohesion c (kPa)", value=5.0, min_value=0.0, step=0.5, format="%.1f"),
        unit_weight_kNpm3 = st.number_input("Unit weight γ (kN/m³)", value=9.0, min_value=6.0, max_value=18.0, step=0.5, format="%.1f"),
        seismic_kh = st.number_input("Pseudo-static kh", value=0.05, min_value=0.0, max_value=0.25, step=0.01, format="%.2f"),
        foundation_sbc_kPa = st.number_input("Allowable SBC (kPa)", value=200.0, min_value=50.0, step=10.0),
        foundation_footprint_factor = st.number_input("Footprint efficiency factor (0-1)", value=0.9, min_value=0.5, max_value=1.0, step=0.05),
    )

# ---------------- Calculations ----------------
H_total = calc_stack_height(g)
L_outer, W_outer = calc_outer_footprint(g)
cap = calc_waste_capacity_m3(g, w)
life = calc_life_years(g, w)

# Hydrology: use outer envelope area for runoff (conservative)
A_runoff = L_outer * W_outer
Q_runoff = rational_Q(h.runoff_coeff, h.design_intensity_mmphr, A_runoff)  # m3/s
Q_leachate = leachate_estimate(Q_runoff, h.leachate_fraction_runoff)
# Drain capacity per linear meter (rectangular) -> pick drain height = drain width (square-ish)
drain_h = g.drain_width_m
Q_drain = drain_capacity_rectangular(g.drain_width_m, drain_h, h.manning_n, h.drainage_slope)

# Bearing & Stability checks
bearing = bearing_check(g, gt, w)
stab = stability_checks(g, gt)

# ---------------- Results (cards) ----------------
st.markdown("### 6) Key Results (non-cost)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Stack Height H (m)", f"{H_total:.2f}")
    st.metric("Outer Plan (L×W) m", f"{L_outer:.1f} × {W_outer:.1f}")
with c2:
    st.metric("Gross Geom. Volume (m³)", f"{cap['gross_geom_volume_m3']:.0f}")
    st.metric("Final Cover Vol (m³)", f"{cap['final_cover_volume_m3']:.0f}")
with c3:
    st.metric("Net Waste Volume (m³)", f"{cap['net_waste_volume_m3']:.0f}")
    st.metric("Estimated Life (years)", f"{life['life_years']:.2f}")
with c4:
    st.metric("Runoff Q (m³/s)", f"{Q_runoff:.4f}")
    st.metric("Leachate Q est. (m³/s)", f"{Q_leachate:.4f}")

c5, c6, c7 = st.columns(3)
with c5:
    st.metric("Drain cap (m³/s) ~ per channel", f"{Q_drain:.4f}")
with c6:
    st.metric("Bearing Pressure (kPa)", f"{bearing['bearing_pressure_kPa']:.1f}")
    st.metric("FOS (Bearing vs SBC)", f"{bearing['FOS_bearing']:.2f}")
with c7:
    st.metric("FOS Outer Slope", f"{stab['FOS_outer']:.2f}")
    st.metric("FOS Inner Slope", f"{stab['FOS_inner']:.2f}")

# ---------------- Visualizations ----------------
st.markdown("### 7) Visualizations")

def section_profile(geom: GeometryInputs, base_rl: float=0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a schematic longitudinal section profile through centerline.
    """
    H = calc_stack_height(geom)
    z0 = base_rl - geom.depth_below_gl_m
    # Outer slope up to bund crest (left)
    x = [0.0]; z = [z0]
    x.append(H*geom.outer_slope_h); z.append(z0+H)
    x.append(H*geom.outer_slope_h + geom.bund_top_width_m/2); z.append(z0+H)
    # Inner benches (terrace)
    bx = [x[-1]]; bz = [z[-1]]
    curx, curz = bx[0], bz[0]
    for _ in range(geom.num_lifts):
        curx += geom.bench_width_m
        bx.append(curx); bz.append(curz)
        curz += geom.lift_height_m
        bx.append(curx); bz.append(curz)
    # Right crest and outer descent
    xr = [bx[-1]]; zr = [bz[-1]]
    xr.append(xr[-1] + geom.bund_top_width_m/2); zr.append(zr[-1])
    xr.append(xr[-1] + H*geom.outer_slope_h); zr.append(z0)
    X = x + bx + xr
    Z = z + bz + zr
    return np.array(X), np.array(Z)

sx, sz = section_profile(g, p.benchmark_rl)
v1, v2 = st.columns([1,1])
with v1:
    fig1, ax1 = plt.subplots(figsize=(7,3))
    ax1.plot(sx, sz, linewidth=2)
    ax1.axhline(p.benchmark_rl, linestyle="--", linewidth=1)
    ax1.set_xlabel("Horizontal (m)")
    ax1.set_ylabel("Elevation (m)")
    ax1.set_title("Longitudinal Section (schematic)")
    ax1.grid(True, linewidth=0.3)
    st.pyplot(fig1)

with v2:
    # For cross-section, reuse same profile (schematic intent)
    fig2, ax2 = plt.subplots(figsize=(7,3))
    ax2.plot(sx, sz, linewidth=2)
    ax2.axhline(p.benchmark_rl, linestyle="--", linewidth=1)
    ax2.set_xlabel("Horizontal (m)")
    ax2.set_ylabel("Elevation (m)")
    ax2.set_title("Cross Section (schematic)")
    ax2.grid(True, linewidth=0.3)
    st.pyplot(fig2)

# Plan view envelope
fig3, ax3 = plt.subplots(figsize=(7,3.5))
base_rect = plt.Rectangle((0, 0), g.length_m, g.width_m, fill=False, linewidth=1.5, label="Inner base")
outer_rect = plt.Rectangle((0, -(W_outer-g.width_m)/2), L_outer, W_outer, fill=False, linestyle="--", label="Outer envelope")
ax3.add_patch(base_rect); ax3.add_patch(outer_rect)
ax3.set_xlim(-5, g.length_m+5)
ax3.set_ylim(-max(W_outer, g.width_m)/1.8, max(W_outer, g.width_m)/1.8)
ax3.set_aspect("equal")
ax3.set_title("Plan View (Inner base vs Outer envelope)")
ax3.set_xlabel("Length (m)"); ax3.set_ylabel("Width (m)")
ax3.grid(True, linewidth=0.3)
ax3.legend(loc="upper right")
st.pyplot(fig3)

# Simple 3D box (outer envelope)
fig4 = plt.figure(figsize=(7,4))
ax4 = fig4.add_subplot(111, projection='3d')
X = [0, L_outer, L_outer, 0, 0]
Y = [-(W_outer/2), -(W_outer/2), (W_outer/2), (W_outer/2), -(W_outer/2)]
Z = [p.benchmark_rl, p.benchmark_rl, p.benchmark_rl, p.benchmark_rl, p.benchmark_rl]
ax4.plot(X, Y, Z, linewidth=1.2)
ax4.plot(X, Y, [p.benchmark_rl+H_total]*5, linewidth=1.2)
for i in range(4):
    ax4.plot([X[i], X[i]], [Y[i], Y[i]], [p.benchmark_rl, p.benchmark_rl+H_total], linewidth=1.0)
ax4.set_title("3D Preview (Outer Envelope)")
ax4.set_xlabel("X (m)"); ax4.set_ylabel("Y (m)"); ax4.set_zlabel("Z (m)")
st.pyplot(fig4)

# ---------------- Export final results (non-cost, independent) ----------------
st.markdown("### 8) Export Final Results (Excel)")
if st.button("⬇️ Export results to Excel"):
    # Prepare sheets
    inputs_df = pd.DataFrame({
        "Category": ["Project","Project","Geometry","Geometry","Geometry","Geometry","Geometry","Geometry","Geometry","Geometry","Geometry","WasteOps","WasteOps","WasteOps","WasteOps","WasteOps","WasteOps","Hydrology","Hydrology","Hydrology","Hydrology","Hydrology","Geotech","Geotech","Geotech","Geotech","Geotech","Geotech"],
        "Name": ["Project Name","Location","Length L (m)","Inner Width W (m)","Depth below GL (m)","Bund top width (m)","Outer slope (1:H)","Inner slope (1:H)","Lift height (m)","Bench width (m)","# Lifts","TPD","Bulk density (t/m3)","Daily cover (%)","Intermediate cover (%)","Final cover thk (m)","Settlement (%)","Runoff C","Design intensity (mm/hr)","Drain slope S (m/m)","Manning n","Leachate fraction","φ (deg)","c (kPa)","γ (kN/m3)","kh","SBC (kPa)","Footprint factor"],
        "Value": [p.name, p.location, g.length_m, g.width_m, g.depth_below_gl_m, g.bund_top_width_m, g.outer_slope_h, g.inner_slope_h, g.lift_height_m, g.bench_width_m, g.num_lifts, w.tpd, w.bulk_density_tpm3, w.daily_cover_pct, w.intermediate_cover_pct, w.final_cover_thk_m, w.settlement_pct, h.runoff_coeff, h.design_intensity_mmphr, h.drainage_slope, h.manning_n, h.leachate_fraction_runoff, gt.phi_deg, gt.cohesion_kpa, gt.unit_weight_kNpm3, gt.seismic_kh, gt.foundation_sbc_kPa, gt.foundation_footprint_factor]
    })

    capacity_df = pd.DataFrame({
        "Metric": ["Outer Length (m)","Outer Width (m)","Total Height H (m)", "Gross geometric volume (m³)","Final cover volume (m³)","Net waste volume (m³)","Life (days)","Life (years)"],
        "Value": [L_outer, W_outer, H_total, cap["gross_geom_volume_m3"], cap["final_cover_volume_m3"], cap["net_waste_volume_m3"], life["life_days"], life["life_years"]]
    })

    hydro_df = pd.DataFrame({
        "Metric": ["Runoff Area (m²)","C (–)","i (mm/hr)","Q_runoff (m³/s)","Leachate fraction","Q_leachate (m³/s)","Drain width (m)","Drain height (m)","Drain slope S","Manning n","Drain capacity ~ (m³/s)"],
        "Value": [A_runoff, h.runoff_coeff, h.design_intensity_mmphr, Q_runoff, h.leachate_fraction_runoff, Q_leachate, g.drain_width_m, drain_h, h.drainage_slope, h.manning_n, Q_drain]
    })

    geotech_df = pd.DataFrame({
        "Metric": ["Bearing pressure (kPa)","Allowable SBC (kPa)","FOS (bearing)","Outer slope β (deg)","Inner slope β (deg)","Rep. depth (m)","FOS outer slope","FOS inner slope"],
        "Value": [bearing["bearing_pressure_kPa"], bearing["SBC_kPa"], bearing["FOS_bearing"], stab["beta_outer_deg"], stab["beta_inner_deg"], stab["rep_depth_m"], stab["FOS_outer"], stab["FOS_inner"]]
    })

    # Make an in-memory xlsx
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        inputs_df.to_excel(xw, sheet_name="Inputs", index=False)
        capacity_df.to_excel(xw, sheet_name="Capacity & Life", index=False)
        hydro_df.to_excel(xw, sheet_name="Hydrology & Drains", index=False)
        geotech_df.to_excel(xw, sheet_name="Geotech & Stability", index=False)
    st.download_button("Download SLF_Results.xlsx", data=buf.getvalue(),
                       file_name="SLF_Results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------- Footnotes ----------------
st.markdown("""
<div class="small-note">
This independent app does not read any external Excel. You can tune formulas inside the code blocks:
<br>• <b>polygonal_volume_from_stair</b> (geometry/volume logic)
<br>• <b>calc_waste_capacity_m3</b> & <b>calc_life_years</b> (capacity & life)
<br>• <b>rational_Q</b>, <b>drain_capacity_rectangular</b> (hydrology & drains)
<br>• <b>bearing_check</b>, <b>stability_checks</b> (bearing & slope FoS)
<br><br>
If you want this to mirror a specific Excel’s cell-by-cell math, paste those exact formulae into these functions — the UI and exports already support it.
</div>
""", unsafe_allow_html=True)
