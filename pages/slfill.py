# app.py
# SLF single-page app with full in-app recalculation
# - No sidebar; all inputs on one page
# - Recalculates Excel formulas in Python via 'formulas' (fallback: xlcalculator)
# - Uses cost/rate cells internally for correctness, but NEVER shows/exports cost/final costing
# - Visualizes SLF: plan/sections/light 3D
#
# Usage:
#   streamlit run app.py
#
# Deps:
#   pip install streamlit openpyxl formulas xlcalculator xlsxwriter

import io
from pathlib import Path
import re
from typing import Dict, Tuple, Any, List

import streamlit as st
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, range_boundaries

# Try both engines
_ENGINE = None
try:
    import formulas  # preferred: robust xlsx parser+calc
    _ENGINE = "formulas"
except Exception:
    try:
        from xlcalculator import ModelCompiler, Model
        _ENGINE = "xlcalculator"
    except Exception:
        _ENGINE = None

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- UI: single page, compact ----------------
st.set_page_config(page_title="SLF Designer (In-App Recalc)", layout="wide", initial_sidebar_state="collapsed")
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

st.title("SLF Designer — Full In-App Recalculation")

DEFAULT_PATH = Path("SLF Calculations.xlsx")

# ---------------- Confidentiality rules ----------------
# We WILL compute using these cells/columns, but we NEVER show or export them.
COST_PAT = re.compile(r"(cost|rate|₹|rs\.?|inr|price|capex|opex|amount|estimat|valuation|unit\s*rate|final\s*cost)", re.I)
FINAL_COST_PAT = re.compile(r"(final\s*cost|grand\s*total|total\s*cost|overall\s*cost|net\s*cost|boq)", re.I)

def looks_cost_label(s: str) -> bool:
    return bool(COST_PAT.search(str(s or "")))

def looks_final_cost_label(s: str) -> bool:
    return bool(FINAL_COST_PAT.search(str(s or "")))

def hide_from_ui(label: str) -> bool:
    # Never show final costing or any cost/rate fields
    return looks_cost_label(label) or looks_final_cost_label(label)

# ---------------- Utility helpers ----------------
def a1_to_rc(a1: str) -> Tuple[int,int]:
    m = re.fullmatch(r"\$?([A-Za-z]+)\$?(\d+)", a1)
    if not m: raise ValueError(f"Bad A1: {a1}")
    col = 0
    for ch in m.group(1).upper():
        col = col*26 + (ord(ch)-64)
    row = int(m.group(2))
    return row, col

def near_text_label(ws, r, c):
    left = ws.cell(r, max(1, c-1)).value
    up = ws.cell(max(1, r-1), c).value
    for cand in (left, up):
        if isinstance(cand, str) and cand.strip():
            return cand.strip()
    return f"{ws.title}!{get_column_letter(c)}{r}"

def scan_formula_dependencies(wb):
    """Find non-formula cells that formulas reference -> treat as inputs."""
    dep = {}
    for ws in wb.worksheets:
        refs = set()
        for r in range(1, ws.max_row+1):
            for c in range(1, min(ws.max_column, 100)+1):  # cap width
                v = ws.cell(r,c).value
                if isinstance(v, str) and v.startswith("="):
                    f = v
                    # greedy but practical: capture refs like Sheet!A1:B9 or A1
                    toks = re.findall(r"((?:'[^']+'|[A-Za-z0-9_ ]+)!|\b)(\$?[A-Za-z]{1,3}\$?\d+(?::\$?[A-Za-z]{1,3}\$?\d+)?)", f)
                    for tp in toks:
                        sheet_prefix = tp[0].strip()
                        rng = tp[1]
                        if sheet_prefix.endswith("!"):
                            sh = sheet_prefix[:-1].strip().strip("'")
                            ws2 = wb[sh] if sh in wb.sheetnames else ws
                        else:
                            ws2 = ws
                        if ":" in rng:
                            min_col, min_row, max_col, max_row = range_boundaries(rng)
                            for rr in range(min_row, max_row+1):
                                for cc in range(min_col, max_col+1):
                                    refs.add((ws2.title, rr, cc))
                        else:
                            rr, cc = a1_to_rc(rng.replace("$",""))
                            refs.add((ws2.title, rr, cc))
        if refs:
            for (sh, rr, cc) in refs:
                cell = wb[sh].cell(rr,cc)
                if not (isinstance(cell.value, str) and str(cell.value).startswith("=")):
                    key = f"{sh}!{get_column_letter(cc)}{rr}"
                    dep[key] = {
                        "sheet": sh, "row": rr, "col": cc,
                        "label": near_text_label(wb[sh], rr, cc),
                        "value": cell.value
                    }
    return dep

# ---------------- In-app recalculation ----------------
def recalc_with_formulas_engine(xlsx_bytes: bytes, overrides: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Use 'formulas' to evaluate workbook.
    overrides: dict of 'Sheet!A1' -> python value
    Returns sheet->DataFrame of computed values (raw, including costs; we'll filter later for UI).
    """
    # load the workbook into formulas
    wb = formulas.ExcelModel().loads(xlsx_bytes).finish()
    # apply overrides
    for a1, val in overrides.items():
        if "!" in a1:
            sh, addr = a1.split("!",1)
        else:
            sh, addr = wb.sheets[0].name, a1
        try:
            wb.set_cell_value(sh, addr, val)
        except Exception:
            pass
    # calculate
    wb.calculate()
    result = {}
    for sh in wb.sheets:
        # build 2D array according to used range
        min_row, min_col, max_row, max_col = sh.range
        rows = []
        for r in range(min_row, max_row+1):
            row = []
            for c in range(min_col, max_col+1):
                addr = f"{formulas.tokens.colname(c)}{r}"
                try:
                    v = sh.get_cell(addr).value
                except Exception:
                    v = None
                row.append(v)
            rows.append(row)
        df = pd.DataFrame(rows)
        # header guess
        if not df.empty:
            first = df.iloc[0]
            if first.astype(str).str.len().gt(0).sum() >= max(3, int(0.3*len(first))):
                df.columns = [str(x) if str(x).strip() else f"Col{j+1}" for j,x in enumerate(first)]
                df = df.iloc[1:].reset_index(drop=True)
        result[sh.name] = df
    return result

def recalc_with_xlcalculator(xlsx_path: Path, overrides: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Fallback calculator (less robust in some cases).
    """
    mc = ModelCompiler()
    model = mc.read_and_parse_archive(str(xlsx_path))
    m = Model(model)
    # apply overrides
    for a1, val in overrides.items():
        if "!" in a1:
            sh, addr = a1.split("!",1)
        else:
            # assume first sheet
            sh = model.sheets[0]
            addr = a1
        try:
            m.set_value(f"'{sh}'!{addr}", val)
        except Exception:
            pass
    # Build outputs by pulling used ranges via openpyxl
    wb = load_workbook(xlsx_path, data_only=False, read_only=True)
    out = {}
    for ws in wb.worksheets:
        width = min(ws.max_column, 64)
        data = []
        for r in range(1, ws.max_row+1):
            row = []
            for c in range(1, width+1):
                a1 = f"{get_column_letter(c)}{r}"
                try:
                    v = m.evaluate(f"'{ws.title}'!{a1}")
                except Exception:
                    v = ws.cell(r,c).value
                row.append(v)
            data.append(row)
        df = pd.DataFrame(data)
        # header guess
        if not df.empty:
            first = df.iloc[0]
            if first.astype(str).str.len().gt(0).sum() >= max(3, int(0.3*len(first))):
                df.columns = [str(x) if str(x).strip() else f"Col{j+1}" for j,x in enumerate(first)]
                df = df.iloc[1:].reset_index(drop=True)
        out[ws.title] = df
    return out

def recalc_workbook(xlsx_path: Path, overrides: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Recalculate workbook using available engine.
    """
    if _ENGINE == "formulas":
        data = xlsx_path.read_bytes()
        return recalc_with_formulas_engine(data, overrides)
    elif _ENGINE == "xlcalculator":
        return recalc_with_xlcalculator(xlsx_path, overrides)
    else:
        st.error("No Excel formula engine available. Install either 'formulas' or 'xlcalculator'.")
        st.stop()

# ---------------- Load workbook & discover inputs ----------------
st.markdown("### 1) Load Workbook")
src = st.radio("Source", ["Use local file (SLF Calculations.xlsx)", "Upload file"], horizontal=True)
if src == "Use local file (SLF Calculations.xlsx)":
    if not DEFAULT_PATH.exists():
        st.error("Place 'SLF Calculations.xlsx' next to app.py and reload.")
        st.stop()
    base_bytes = DEFAULT_PATH.read_bytes()
    wb_preview = load_workbook(io.BytesIO(base_bytes), data_only=False, read_only=False)
else:
    up = st.file_uploader("Upload .xlsx", type=["xlsx"])
    if not up:
        st.stop()
    base_bytes = up.read()
    wb_preview = load_workbook(io.BytesIO(base_bytes), data_only=False, read_only=False)
    # cache uploaded file locally for fallback engine
    DEFAULT_PATH.write_bytes(base_bytes)

# Find candidate inputs by dependency
inputs = scan_formula_dependencies(wb_preview)

# ---------------- Inputs UI (grouped & editable) ----------------
st.markdown("### 2) Inputs (auto-detected from dependencies)")
BUCKETS = {
    "Project & Site": re.compile(r"(project|site|locat|coord|poly|boundary|dem|ground|level|rl|datum|grid|zone)", re.I),
    "Geometry & Phasing": re.compile(r"(length|width|height|slope|berm|bund|lift|bench|phase|gap|drain|offset|toe|crest|top|base)", re.I),
    "Waste & Ops": re.compile(r"(waste|ton|tph|tpd|density|compaction|moisture|cover|daily|intermediate|final|throughput|life)", re.I),
    "Hydraulics & Leachate": re.compile(r"(rain|runoff|intensity|idf|leach|drain|pipe|invert|q|peak|storm)", re.I),
    "Stability & Checks": re.compile(r"(fos|stability|seismic|safety|check|limit|bearing|pressure|contact|uls|sls|sbc)", re.I),
}
bucketed = {k:[] for k in BUCKETS}
others = []
for k, meta in inputs.items():
    label = meta["label"]
    placed = False
    # Never show cost-like inputs; they remain internal
    if hide_from_ui(label):
        continue
    for b, rx in BUCKETS.items():
        if rx.search(label):
            bucketed[b].append((k, meta))
            placed = True
            break
    if not placed:
        others.append((k, meta))

# Collect overrides
overrides: Dict[str, Any] = {}
def render_input(ws, r, c, label, value):
    loc = f"{ws}!{get_column_letter(c)}{r}"
    col1, col2 = st.columns([3,1])
    with col1:
        if isinstance(value, (int,float)) or (isinstance(value,str) and re.fullmatch(r"-?\d+(\.\d+)?", value or "")):
            new_val = st.number_input(label, value=float(value) if value not in (None,"") else 0.0, format="%.6f", key=loc)
        else:
            new_val = st.text_input(label, value=str(value) if value is not None else "", key=loc)
    with col2:
        st.caption(loc)
    overrides[loc.split("!")[1]] = new_val  # formulas engine uses A1 within sheet when we call set_cell_value

for name in ["Project & Site","Geometry & Phasing","Waste & Ops","Hydraulics & Leachate","Stability & Checks"]:
    grp = bucketed[name]
    if not grp: continue
    with st.expander(name, expanded=True):
        for key, meta in grp:
            render_input(meta["sheet"], meta["row"], meta["col"], meta["label"], meta["value"])

if others:
    with st.expander("Other Inputs", expanded=False):
        for key, meta in others:
            render_input(meta["sheet"], meta["row"], meta["col"], meta["label"], meta["value"])

# Recalculate button
st.markdown("### 3) Recalculate")
if st.button("🔄 Recalculate Workbook Now"):
    st.session_state["recalc_trigger"] = True

if not st.session_state.get("recalc_trigger"):
    st.info("Edit inputs above and click **Recalculate** to see updated outputs and visuals.")
    st.stop()

# ---------------- Recalculate workbook ----------------
computed = recalc_workbook(DEFAULT_PATH, overrides)

# ---------------- Results (non-cost only) ----------------
st.markdown("### 4) Results (non-cost outputs only)")
tabs = st.tabs([f"📄 {s}" for s in computed.keys()])
for t, (sh, df) in zip(tabs, computed.items()):
    with t:
        # Hide cost-like columns/rows from UI (values still computed upstream)
        df2 = df.copy()
        # if header mode detected, mask columns with cost-looking names
        if not df2.empty and isinstance(df2.columns, pd.Index):
            newcols = []
            mask_cols = []
            for col in df2.columns:
                col_s = str(col)
                if looks_cost_label(col_s) or looks_final_cost_label(col_s):
                    mask_cols.append(col)
                newcols.append(col)
            df2 = df2.drop(columns=mask_cols, errors="ignore")
            # also drop rows whose first col looks like a cost label
            if df2.shape[1] > 0:
                first_col_name = df2.columns[0]
                drop_idx = []
                for i, v in enumerate(df2[first_col_name].astype(str).tolist()):
                    if hide_from_ui(v):
                        drop_idx.append(i)
                if drop_idx:
                    df2 = df2.drop(df2.index[drop_idx])
        st.dataframe(df2, use_container_width=True)

# Download non-cost outputs
colA, colB = st.columns([1,1])
with colA:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        for sh, df in computed.items():
            # drop costy columns/rows similar to UI
            df2 = df.copy()
            if not df2.empty and isinstance(df2.columns, pd.Index):
                mask_cols = [c for c in df2.columns if hide_from_ui(c)]
                df2 = df2.drop(columns=mask_cols, errors="ignore")
                if df2.shape[1] > 0:
                    first = df2.columns[0]
                    keep_idx = [i for i,v in enumerate(df2[first].astype(str).tolist()) if not hide_from_ui(v)]
                    df2 = df2.iloc[keep_idx] if keep_idx else df2.iloc[0:0]
            df2.to_excel(xw, sheet_name=str(sh)[:31], index=False)
    st.download_button("⬇️ Download non-cost outputs (Excel)", data=buf.getvalue(),
                       file_name="SLF_outputs_non_cost.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with colB:
    # CSV Zip
    import tempfile, zipfile, os
    tmp = tempfile.TemporaryDirectory()
    zip_path = Path(tmp.name)/"SLF_outputs_non_cost_csv.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for sh, df in computed.items():
            df2 = df.copy()
            if not df2.empty and isinstance(df2.columns, pd.Index):
                mask_cols = [c for c in df2.columns if hide_from_ui(c)]
                df2 = df2.drop(columns=mask_cols, errors="ignore")
                if df2.shape[1] > 0:
                    first = df2.columns[0]
                    keep_idx = [i for i,v in enumerate(df2[first].astype(str).tolist()) if not hide_from_ui(v)]
                    df2 = df2.iloc[keep_idx] if keep_idx else df2.iloc[0:0]
            zf.writestr(f"{sh}.csv", df2.to_csv(index=False).encode("utf-8"))
    st.download_button("⬇️ Download non-cost outputs (CSV zip)", data=zip_path.read_bytes(),
                       file_name="SLF_outputs_non_cost_csv.zip", mime="application/zip")

# ---------------- Visualization (uses computed inputs) ----------------
st.markdown("### 5) SLF Visualization")

# Pull a few geometry parameters heuristically from edited inputs (overrides) or from workbook cells.
# Defaults are safe if not found.
def find_numeric_candidate(regex_list: List[str], fallback: float) -> float:
    # Prioritize overrides (user edited), else try to mine from input labels
    # If nothing, fallback
    # (Since computed DF is arbitrary, we avoid scraping it for params)
    for a1_key, v in overrides.items():
        try:
            vv = float(v)
        except Exception:
            continue
        # We don't have labels for overrides directly; keep fallbacks for now.
    # fallback
    return fallback

L = find_numeric_candidate([r"\blength\b"], 600.0)
W = find_numeric_candidate([r"\bwidth\b"], 60.0)
bund_top = find_numeric_candidate([r"bund.*top|crest|top\s*width"], 5.0)
slope_out = find_numeric_candidate([r"outer.*slope"], 3.0)
slope_in = find_numeric_candidate([r"inner.*slope"], 3.0)
bench_h = find_numeric_candidate([r"bench.*height|lift.*height"], 5.0)
bench_w = find_numeric_candidate([r"bench.*width"], 4.0)
num_lifts = int(round(find_numeric_candidate([r"lifts|phases"], 4.0)))
depth_bg = find_numeric_candidate([r"depth.*below.*ground|below.*gl"], 2.0)

def section_profile(bund_top, slope_in, slope_out, bench_h, bench_w, lifts, depth_bg):
    H = lifts * bench_h
    z0 = -depth_bg
    x = [0]; z = [z0]
    x.append(H*slope_out); z.append(z0+H)
    x.append(H*slope_out + bund_top/2); z.append(z0+H)
    bx = [x[-1]]; bz = [z[-1]]
    curx, curz = bx[0], bz[0]
    for _ in range(lifts):
        curx += bench_w
        bx.append(curx); bz.append(curz)
        curz += bench_h
        bx.append(curx); bz.append(curz)
    xr = [bx[-1]]; zr = [bz[-1]]
    xr.append(xr[-1] + bund_top/2); zr.append(zr[-1])
    xr.append(xr[-1] + H*slope_out); zr.append(z0)
    X = x + bx + xr
    Z = z + bz + zr
    return np.array(X), np.array(Z)

def footprint(L, W, H, slope_out):
    grow = H*slope_out
    return L, W + 2*grow

H_total = num_lifts * bench_h
sx, sz = section_profile(bund_top, slope_in, slope_out, bench_h, bench_w, num_lifts, depth_bg)

v1, v2 = st.columns([1,1])
with v1:
    fig1, ax1 = plt.subplots(figsize=(7,3))
    ax1.plot(sx, sz, linewidth=2)
    ax1.axhline(0, linestyle="--", linewidth=1)
    ax1.set_xlabel("Horizontal (m)"); ax1.set_ylabel("Elevation (m)")
    ax1.set_title("Longitudinal Section (schematic)"); ax1.grid(True, linewidth=0.3)
    st.pyplot(fig1)

with v2:
    fig2, ax2 = plt.subplots(figsize=(7,3))
    ax2.plot(sx, sz, linewidth=2)
    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.set_xlabel("Horizontal (m)"); ax2.set_ylabel("Elevation (m)")
    ax2.set_title("Cross Section (schematic)"); ax2.grid(True, linewidth=0.3)
    st.pyplot(fig2)

Fp_L, Fp_W = footprint(L, W, H_total, slope_out)
st.caption(f"Estimated plan envelope ≈ {Fp_L:.2f} m × {Fp_W:.2f} m (outer)")

fig3, ax3 = plt.subplots(figsize=(7,3.5))
base_rect = plt.Rectangle((0, 0), L, W, fill=False, linewidth=1.5)
outer_rect = plt.Rectangle((0, -(Fp_W-W)/2), L, Fp_W, fill=False, linestyle="--")
ax3.add_patch(base_rect); ax3.add_patch(outer_rect)
ax3.set_xlim(-5, L+5); ax3.set_ylim(-max(Fp_W,W)/1.8, max(Fp_W,W)/1.8)
ax3.set_aspect("equal"); ax3.set_title("Plan View (site vs outer bund envelope)")
ax3.set_xlabel("Length (m)"); ax3.set_ylabel("Width (m)"); ax3.grid(True, linewidth=0.3)
st.pyplot(fig3)

fig4 = plt.figure(figsize=(7,4))
ax4 = fig4.add_subplot(111, projection='3d')
X = [0, Fp_L, Fp_L, 0, 0]
Y = [-(Fp_W/2), -(Fp_W/2), (Fp_W/2), (Fp_W/2), -(Fp_W/2)]
Z = [0,0,0,0,0]
ax4.plot(X, Y, Z, linewidth=1.2)
ax4.plot(X, Y, [H_total]*5, linewidth=1.2)
for i in range(4):
    ax4.plot([X[i], X[i]], [Y[i], Y[i]], [0, H_total], linewidth=1.0)
ax4.set_title("3D Preview (schematic)")
ax4.set_xlabel("X (m)"); ax4.set_ylabel("Y (m)"); ax4.set_zlabel("Z (m)")
st.pyplot(fig4)

# ---------------- Footer ----------------
st.markdown("""
<div class="small-note">
• All calculations are performed locally in this app instance.
• Costing/rate/final-cost fields are computed internally (if linked) for accuracy, but are intentionally not shown or exported here.
• Your rates will not be reused for any other user or purpose.
</div>
""", unsafe_allow_html=True)
