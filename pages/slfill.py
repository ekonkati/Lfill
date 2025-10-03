# app.py
# Independent SLF Designer & Visualizer
# - Single page, no sidebar; compact multi-column inputs
# - Geometry: bund (outer slope), benches (lift height + bench width), optional top-width cap
# - Exact cross-section polygon (bund -> benches -> bund), Plotly visuals (section, plan, 3D)
# - Calculations (transparent): cross-section areas, volumes, capacity/life, hydrology, quick stability
# - BOQ, Abstract, Summary tables
# - Export final results (no costs) to Excel

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from io import BytesIO

st.set_page_config(page_title="SLF Designer (Independent)", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  [data-testid="collapsedControl"], header [data-testid="stToolbar"]{display:none!important;}
  .block-container{padding-top:.6rem; padding-bottom:.8rem; max-width:1300px;}
  .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"]{height:2rem;font-size:.92rem;}
  .stButton > button{height:2.2rem; padding:0 .9rem;}
  .formline {display:flex; gap:.75rem; align-items:flex-end; flex-wrap:wrap;}
  .formline > div {flex:1 1 180px;}
  .small-note{font-size:.85rem; opacity:.85;}
</style>
""", unsafe_allow_html=True)

# --------------------- Data models ---------------------
@dataclass
class Geometry:
    length_m: float = 600.0            # extrusion length for volume (m)
    inner_base_w_m: float = 60.0       # inner base width at GL (m)
    depth_bg_m: float = 2.0            # excavation depth below GL (m)
    bund_top_w_m: float = 5.0          # bund crest width (m)
    outer_slope_h: float = 3.0         # outer slope 1:H
    lift_h_m: float = 5.0              # bench step height (m)
    bench_w_m: float = 4.0             # bench tread width (m)
    n_lifts: int = 4                   # number of bench steps per side
    cap_top_ratio: float = 0.30        # optional: limit top width <= cap_ratio * half base (0..1). 0 disables.

@dataclass
class WasteOps:
    tpd: float = 800.0
    bulk_density_tpm3: float = 0.85    # t/m3
    daily_cover_pct: float = 10.0
    intermediate_cover_pct: float = 5.0
    final_cover_thk_m: float = 1.0
    settlement_pct: float = 15.0       # % regain of capacity due to settlement

@dataclass
class Hydro:
    runoff_coeff: float = 0.8
    intensity_mmphr: float = 60.0
    drain_n: float = 0.013
    drain_slope: float = 0.005         # m/m
    drain_width_m: float = 1.0
    drain_height_m: float = 1.0

@dataclass
class Geotech:
    phi_deg: float = 30.0
    cohesion_kpa: float = 5.0
    gamma_kNpm3: float = 9.0
    kh: float = 0.05
    SBC_kpa: float = 200.0
    footprint_factor: float = 0.90

# --------------------- Geometry builders ---------------------
def build_section_points(
    inner_base_w: float, depth_bg: float, bund_top_w: float,
    outer_slope_h: float, lift_h: float, bench_w: float, n_lifts: int,
    cap_top_ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build symmetric stepped section:
    - Start at BL (y=-depth_bg), rise along outer slope to GL+H, crest (half bund),
      then benches (horizontal treads + vertical risers), then mirror to the right, then outer slope down.
    Returns X (horizontal) and Z (elevation) arrays for a single closed polyline (open at GL to show profile).
    """
    H = n_lifts * lift_h
    z0 = -depth_bg

    # Left outer slope toe to crest
    xL0, zL0 = 0.0, z0
    xL1, zL1 = H * outer_slope_h, z0 + H           # slope up to crest level
    xL2, zL2 = xL1 + bund_top_w/2.0, zL1           # left half crest

    # Benches from inner crest edge
    bx, bz = [xL2], [zL2]
    curx, curz = xL2, zL2
    for _ in range(n_lifts):
        # bench tread
        curx += bench_w
        bx.append(curx); bz.append(curz)
        # vertical rise (schematic) for the inner slope between benches
        curz += lift_h
        bx.append(curx); bz.append(curz)

    # Optional top-width cap (limit extra inward tread after crest)
    if cap_top_ratio and cap_top_ratio > 0:
        half_base = xL2 - xL0  # horizontal from left BL to inner crest (slope + half crest)
        max_top = half_base * cap_top_ratio
        top_w = max(0.0, bx[-1] - xL2)
        if top_w > max_top:
            shrink = top_w - max_top
            bx[-1] -= shrink
            bx[-2] -= shrink
            curx = bx[-1]

    xR2, zR2 = curx + bund_top_w/2.0, curz         # right half crest
    xR1, zR1 = xR2 + H * outer_slope_h, z0         # right outer slope down to BL

    # Construct final polyline
    X = [xL0, xL1, xL2] + bx[1:] + [xR2, xR1]
    Z = [zL0, zL1, zL2] + bz[1:] + [zR2, zR1]
    return np.array(X), np.array(Z)

def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """Shoelace formula (signed). Use abs for area."""
    return 0.5 * float(np.sum(x*np.roll(y,-1) - np.roll(x,-1)*y))

def clip_polygon_at_gl(X: np.ndarray, Z: np.ndarray, gl: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the profile polygon into parts below GL and above GL for area accounting.
    Returns (X_below, Z_below, X_above, Z_above) as open polylines; we form areas via trapezoids with GL.
    """
    # Ensure broken into segments across GL
    pts = list(zip(X, Z))
    below, above = [], []
    for i in range(len(pts)-1):
        (x1,z1),(x2,z2) = pts[i], pts[i+1]
        if (z1<=gl and z2<=gl):
            below += [(x1,z1)]
            if i==len(pts)-2: below += [(x2,z2)]
        elif (z1>=gl and z2>=gl):
            above += [(x1,z1)]
            if i==len(pts)-2: above += [(x2,z2)]
        else:
            # crosses GL -> find intersection
            t = (gl - z1) / (z2 - z1 + 1e-12)
            xi = x1 + t*(x2 - x1)
            # push until GL point
            if z1 < gl:
                below += [(x1,z1), (xi, gl)]
                above += [(xi, gl)]
                if i==len(pts)-2: above += [(x2,z2)]
            else:
                above += [(x1,z1), (xi, gl)]
                below += [(xi, gl)]
                if i==len(pts)-2: below += [(x2,z2)]
    Xb, Zb = (np.array([p[0] for p in below]), np.array([p[1] for p in below])) if below else (np.array([]), np.array([]))
    Xa, Za = (np.array([p[0] for p in above]), np.array([p[1] for p in above])) if above else (np.array([]), np.array([]))
    return Xb, Zb, Xa, Za

def area_below_gl_rect(inner_base_w: float, depth_bg: float) -> float:
    """Excavation area (BL→GL) assuming vertical cut for inner base."""
    return inner_base_w * depth_bg

def area_above_gl_profile(X: np.ndarray, Z: np.ndarray) -> float:
    """Area above GL via polygonal decomp to GL baseline."""
    Xb, Zb, Xa, Za = clip_polygon_at_gl(X, Z, 0.0)
    # integrate area above GL as sum of trapezoids to GL baseline
    if Xa.size < 2: return 0.0
    area = 0.0
    for i in range(len(Xa)-1):
        x1,z1 = Xa[i], Za[i]
        x2,z2 = Xa[i+1], Za[i+1]
        base = x2 - x1
        # trapezoid height is (z1 + z2)/2 with GL as zero baseline
        area += base * ( (max(z1,0)+max(z2,0)) / 2.0 )
    return abs(area)

# --------------------- domain calcs ---------------------
def rational_flow(C: float, i_mmphr: float, A_m2: float) -> float:
    """Q = C i A; i in mm/hr -> m/s, A in m2, Q in m3/s."""
    i_mps = (i_mmphr/1000.0) / 3600.0
    return C * i_mps * A_m2

def manning_rect_Q(n: float, b: float, h: float, S: float) -> float:
    A = b*h
    P = b + 2*h
    R = A / max(P,1e-9)
    return (1.0/max(n,1e-9)) * A * (R**(2.0/3.0)) * (S**0.5)

def infinite_slope_fos(phi_deg, c_kPa, gamma, z_m, beta_deg, kh):
    beta = math.radians(max(min(beta_deg, 89.0), 1.0))
    phi = math.radians(phi_deg)
    term1 = (c_kPa / max(gamma*z_m*math.sin(beta)*math.cos(beta), 1e-6))
    term2 = (math.tan(phi) / max(math.tan(beta), 1e-6))
    seismic = max(1.0 - kh*(math.cos(beta)/max(math.sin(beta),1e-6)), 0.1)
    return (term1 + term2) * seismic

# --------------------- UI inputs ---------------------
st.header("Inputs")

# Geometry row 1
col = st.container()
with col:
    st.markdown('<div class="formline">', unsafe_allow_html=True)
    L = st.number_input("Length L (m)", value=600.0, min_value=10.0, step=10.0, format="%.2f")
    Wb = st.number_input("Inner base width W (m)", value=60.0, min_value=5.0, step=1.0, format="%.2f")
    depth_bg = st.number_input("Depth below GL (m)", value=2.0, min_value=0.0, step=0.1, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

# Geometry row 2
with col:
    st.markdown('<div class="formline">', unsafe_allow_html=True)
    bund_top = st.number_input("Bund crest width (m)", value=5.0, min_value=1.0, step=0.5, format="%.2f")
    outer_H = st.number_input("Outer slope 1:H", value=3.0, min_value=1.0, step=0.5, format="%.2f")
    lift_h = st.number_input("Lift height (m)", value=5.0, min_value=1.0, step=0.5, format="%.2f")
    bench_w = st.number_input("Bench width (m)", value=4.0, min_value=1.0, step=0.5, format="%.2f")
    n_lifts = st.number_input("# Lifts", value=4, min_value=1, step=1)
    cap_ratio = st.number_input("Cap top width ratio (0=off)", value=0.30, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

# Waste row
with col:
    st.markdown('<div class="formline">', unsafe_allow_html=True)
    tpd = st.number_input("Waste throughput (TPD)", value=800.0, min_value=10.0, step=10.0)
    rho = st.number_input("Bulk density (t/m³)", value=0.85, min_value=0.3, step=0.05, format="%.2f")
    daily_cover = st.number_input("Daily cover (%)", value=10.0, min_value=0.0, max_value=40.0, step=1.0, format="%.1f")
    inter_cover = st.number_input("Intermediate cover (%)", value=5.0, min_value=0.0, max_value=30.0, step=1.0, format="%.1f")
    final_cover = st.number_input("Final cover thickness (m)", value=1.0, min_value=0.0, step=0.1, format="%.2f")
    settlement = st.number_input("Settlement allowance (%)", value=15.0, min_value=0.0, max_value=50.0, step=1.0, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

# Hydro row
with col:
    st.markdown('<div class="formline">', unsafe_allow_html=True)
    runoff_c = st.number_input("Runoff coeff C", value=0.8, min_value=0.0, max_value=1.0, step=0.05)
    intensity = st.number_input("Rain intensity i (mm/hr)", value=60.0, min_value=1.0, step=1.0)
    drain_n = st.number_input("Manning n", value=0.013, min_value=0.010, step=0.001, format="%.3f")
    drain_S = st.number_input("Drain slope S (m/m)", value=0.005, min_value=0.0001, step=0.0001, format="%.4f")
    drain_b = st.number_input("Drain width (m)", value=1.0, min_value=0.5, step=0.5, format="%.2f")
    drain_h = st.number_input("Drain height (m)", value=1.0, min_value=0.5, step=0.5, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

# Geotech row
with col:
    st.markdown('<div class="formline">', unsafe_allow_html=True)
    phi = st.number_input("φ (deg)", value=30.0, min_value=10.0, max_value=45.0, step=1.0)
    coh = st.number_input("c (kPa)", value=5.0, min_value=0.0, step=0.5, format="%.1f")
    gamma = st.number_input("γ (kN/m³)", value=9.0, min_value=6.0, max_value=18.0, step=0.5, format="%.1f")
    kh = st.number_input("kh (pseudo-static)", value=0.05, min_value=0.0, max_value=0.25, step=0.01, format="%.2f")
    SBC = st.number_input("Allowable SBC (kPa)", value=200.0, min_value=50.0, step=10.0)
    footprint_fac = st.number_input("Footprint efficiency (0-1)", value=0.90, min_value=0.5, max_value=1.0, step=0.05, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------- Build section & visuals ---------------------
X, Z = build_section_points(
    inner_base_w=Wb, depth_bg=depth_bg, bund_top_w=bund_top,
    outer_slope_h=outer_H, lift_h=lift_h, bench_w=bench_w, n_lifts=n_lifts,
    cap_top_ratio=cap_ratio
)

H_total = n_lifts * lift_h
# Outer width grows by outer slope: add both sides -> outer plan width:
outer_width = Wb + 2*(H_total*outer_H)

# Cross-section visuals (Plotly)
st.header("Visualizations")

# Cross-section
fig_cs = go.Figure()
fig_cs.add_trace(go.Scatter(x=X, y=Z, mode="lines", name="Profile"))
fig_cs.add_trace(go.Scatter(x=[min(X)-5, max(X)+5], y=[0,0], mode="lines",
                            line=dict(dash="dash"), name="GL"))
fig_cs.update_layout(
    title="Cross Section (Bund + Benches + Outer Slopes)",
    xaxis_title="Horizontal (m)", yaxis_title="Elevation (m)",
    template="simple_white", height=360, width=1000
)
st.plotly_chart(fig_cs, use_container_width=True)

# Plan view (inner base vs outer envelope)
fig_plan = go.Figure()
fig_plan.add_shape(type="rect", x0=0, y0=-Wb/2, x1=L, y1=+Wb/2, line=dict(width=2), name="Inner base")
fig_plan.add_shape(type="rect", x0=0, y0=-outer_width/2, x1=L, y1=+outer_width/2, line=dict(dash="dash"), name="Outer envelope")
fig_plan.update_layout(
    title="Plan View (Inner Base vs Outer Envelope)",
    xaxis_title="Length (m)", yaxis_title="Width (m)",
    template="simple_white", height=360, width=1000
)
st.plotly_chart(fig_plan, use_container_width=True)

# Simple 3D (outer envelope box; stepped 3D available later if you want)
fig3d = go.Figure()
# base rectangle
fig3d.add_trace(go.Scatter3d(x=[0,L,L,0,0], y=[-outer_width/2,-outer_width/2,outer_width/2,outer_width/2,-outer_width/2], z=[0,0,0,0,0], mode="lines", name="Base"))
# top rectangle
fig3d.add_trace(go.Scatter3d(x=[0,L,L,0,0], y=[-outer_width/2,-outer_width/2,outer_width/2,outer_width/2,-outer_width/2], z=[H_total]*5, mode="lines", name="Top"))
# pillars
for (x,y) in [(0,-outer_width/2),(L,-outer_width/2),(L,outer_width/2),(0,outer_width/2)]:
    fig3d.add_trace(go.Scatter3d(x=[x,x], y=[y,y], z=[0,H_total], mode="lines", showlegend=False))
fig3d.update_layout(
    title="3D Preview (Outer Envelope)",
    scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
    height=420, template="simple_white"
)
st.plotly_chart(fig3d, use_container_width=True)

# --------------------- Calculations (transparent) ---------------------
st.header("Detailed Calculations")

# 1) Cross-section areas
A_exc = area_below_gl_rect(Wb, depth_bg)                    # BL→GL
A_above = area_above_gl_profile(X, Z)                       # GL→top from profile
A_total = A_exc + A_above

# 2) Volumes by extrusion along L
V_exc = A_exc * L
V_fill = A_above * L
V_total_geom = A_total * L

# 3) Capacity & life adjustments
daily_frac = daily_cover/100.0
inter_frac = inter_cover/100.0
m3_per_day = tpd / max(rho, 1e-6)
top_area_for_final_cover = L * outer_width  # conservative (outer plan)
V_final_cover = top_area_for_final_cover * final_cover
V_settlement_gain = V_total_geom * (settlement/100.0)
V_net_waste = V_total_geom * (1.0 - daily_frac - inter_frac) - V_final_cover + V_settlement_gain
V_net_waste = max(V_net_waste, 0.0)
life_days = V_net_waste / max(m3_per_day, 1e-6)
life_years = life_days / 365.0

# 4) Hydrology (rational)
A_runoff = L * outer_width
Q_runoff = rational_flow(runoff_c, intensity, A_runoff)
Q_drain = manning_rect_Q(drain_n, drain_b, drain_h, drain_S)

# 5) Bearing & slope checks (quick)
bearing_area = L * outer_width * footprint_fac
total_load_kN = gamma * (V_net_waste / max(1.0, 1.0))  # kN (γ in kN/m3 times volume)
q_kPa = total_load_kN / max(bearing_area, 1e-6)
FOS_bearing = SBC / max(q_kPa, 1e-6)
beta_outer = math.degrees(math.atan(1.0/max(outer_H,1e-6)))
rep_depth = max(H_total/2.0, 1.0)
FOS_outer = infinite_slope_fos(phi, coh, gamma, rep_depth, beta_outer, kh)

# --------------------- Working tables ---------------------
calc_rows = [
    ("Total height H (m)", H_total),
    ("Outer envelope width (m)", outer_width),
    ("A_exc (BL→GL) (m²)", A_exc),
    ("A_above (GL→Top) (m²)", A_above),
    ("A_total (m²)", A_total),
    ("V_exc (m³)", V_exc),
    ("V_fill (m³)", V_fill),
    ("V_total_geom (m³)", V_total_geom),
    ("Top plan area for final cover (m²)", top_area_for_final_cover),
    ("V_final_cover (m³)", V_final_cover),
    ("Settlement gain (m³)", V_settlement_gain),
    ("V_net_waste (m³)", V_net_waste),
    ("Waste density (t/m³)", rho),
    ("TPD", tpd),
    ("m³/day (TPD/ρ)", m3_per_day),
    ("Life (days)", life_days),
    ("Life (years)", life_years),
    ("Runoff area (m²)", A_runoff),
    ("Q_runoff (m³/s)", Q_runoff),
    ("Drain capacity ~ (m³/s)", Q_drain),
    ("Bearing area eff (m²)", bearing_area),
    ("Net bearing q (kPa)", q_kPa),
    ("FOS (bearing)", FOS_bearing),
    ("Outer slope β (deg)", beta_outer),
    ("Rep. depth (m)", rep_depth),
    ("FOS_outer (quick)", FOS_outer),
]
df_working = pd.DataFrame(calc_rows, columns=["Metric", "Value"])

# BOQ-style quantities (no costs)
boq_items = [
    ("Excavation BL→GL", "W * depth * L", f"{Wb:.2f} * {depth_bg:.2f} * {L:.2f}", V_exc),
    ("Fill GL→Top (bund+benches)", "A_above * L", f"{A_above:.2f} * {L:.2f}", V_fill),
    ("Final cover", "Outer plan * thk", f"{(L*outer_width):.2f} * {final_cover:.2f}", V_final_cover),
]
df_boq = pd.DataFrame(boq_items, columns=["Item", "Formula", "Substitution", "Qty (m³)"])

# Abstract (rolled up)
df_abs = pd.DataFrame({
    "Category": ["Excavation", "Fill", "Final Cover", "Net Waste Volume", "Life (years)"],
    "Quantity": [V_exc, V_fill, V_final_cover, V_net_waste, life_years]
})

# Summary (high level)
df_sum = pd.DataFrame({
    "Key": [
        "Outer envelope (L×W) m",
        "Total height (m)",
        "Net waste volume (m³)",
        "Life (years)",
        "Runoff Q (m³/s)",
        "Drain capacity (m³/s)",
        "Bearing pressure (kPa)",
        "FOS (bearing)",
        "FOS (outer slope, quick)"
    ],
    "Value": [
        f"{L:.1f} × {outer_width:.1f}",
        f"{H_total:.2f}",
        f"{V_net_waste:.0f}",
        f"{life_years:.2f}",
        f"{Q_runoff:.4f}",
        f"{Q_drain:.4f}",
        f"{q_kPa:.1f}",
        f"{FOS_bearing:.2f}",
        f"{FOS_outer:.2f}"
    ]
})

st.subheader("Working (line-by-line)")
st.dataframe(df_working, use_container_width=True)
st.subheader("BOQ (quantities only, no costs)")
st.dataframe(df_boq, use_container_width=True)
st.subheader("Abstract")
st.dataframe(df_abs, use_container_width=True)
st.subheader("Summary")
st.dataframe(df_sum, use_container_width=True)

# --------------------- Export ---------------------
st.header("Export Final Results (Excel)")
if st.button("⬇️ Download Excel"):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
        pd.DataFrame({
            "Input": ["L","W","depth_bg","bund_top","outer_H","lift_h","bench_w","#lifts","cap_ratio",
                      "TPD","ρ","daily%","inter%","final_cover(m)","settlement%","C","i(mm/hr)","n","S","drain_w","drain_h",
                      "φ","c(kPa)","γ","kh","SBC","footprint_fac"],
            "Value": [L,Wb,depth_bg,bund_top,outer_H,lift_h,bench_w,n_lifts,cap_ratio,
                      tpd,rho,daily_cover,inter_cover,final_cover,settlement,runoff_c,intensity,drain_n,drain_S,drain_b,drain_h,
                      phi,coh,gamma,kh,SBC,footprint_fac]
        }).to_excel(xw, sheet_name="Inputs", index=False)
        df_working.to_excel(xw, sheet_name="Working", index=False)
        df_boq.to_excel(xw, sheet_name="BOQ", index=False)
        df_abs.to_excel(xw, sheet_name="Abstract", index=False)
        df_sum.to_excel(xw, sheet_name="Summary", index=False)
    st.download_button("Download SLF_Results.xlsx", data=out.getvalue(),
                       file_name="SLF_Results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown('<div class="small-note">No costing is shown or exported. If you need step-wise bench volumes and face areas per lift for a more granular BOQ, say the word and I’ll add those rows explicitly.</div>', unsafe_allow_html=True)
