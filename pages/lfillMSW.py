import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Landfill Design & Cost Calculator",
    page_icon="♻️",
    layout="wide"
)

# --- App Title and Description ---
st.title("♻️ Landfill Design & Cost Calculator")
st.markdown("""
This application replicates the logic of a complex spreadsheet for landfill design.
Adjust the parameters in the sidebar to see how they affect the landfill's capacity, lifespan, and cost.
- **NGL**: Natural Ground Level
- **ABL**: Above Ground Level
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Parameters")

# --- General Inputs ---
st.sidebar.subheader("General Inputs")
quantity_tpd = st.sidebar.number_input("Waste Quantity (TPD - Tons Per Day)", min_value=0.0, value=1000.0, step=50.0)
duration_years = st.sidebar.number_input("Duration (Years)", min_value=1, value=25, step=1)
density = st.sidebar.number_input("Compacted Waste Density (Tons/cum)", min_value=0.1, value=1.1, step=0.05)

# --- Landfill Dimensions ---
st.sidebar.subheader("Landfill Dimensions (at NGL)")
length_m = st.sidebar.number_input("Overall Length (m)", min_value=0.0, value=480.0, step=10.0)
width_m = st.sidebar.number_input("Overall Width (m)", min_value=0.0, value=420.0, step=10.0)
cost_per_sqm = st.sidebar.number_input("Cost per Sqm (INR)", min_value=0.0, value=350.0, step=10.0)

# --- Below NGL Parameters ---
st.sidebar.subheader("Below Natural Ground Level (NGL)")
depth_below_ngl = st.sidebar.number_input("Depth below NGL (m)", min_value=0.0, value=5.0, step=0.5)
bund_width = st.sidebar.number_input("Bund Width (m)", min_value=0.0, value=5.0, step=1.0)
bund_height = st.sidebar.number_input("Bund Height (m)", min_value=0.0, value=2.0, step=0.5)
external_slope = st.sidebar.number_input("External Slope (1:X)", min_value=0.1, value=2.5, step=0.1)
internal_slope = st.sidebar.number_input("Internal Slope (1:X)", min_value=0.1, value=2.0, step=0.1)

# --- Above NGL Parameters ---
st.sidebar.subheader("Above Ground Level (ABL) Layers")
with st.sidebar.expander("Layer 1 Parameters"):
    height_abl1 = st.sidebar.number_input("Height ABL 1 (m)", min_value=0.0, value=5.0, step=0.5, key='h1')
    side_slope_abl1 = st.sidebar.number_input("Side Slope ABL 1 (1:X)", min_value=0.1, value=3.0, step=0.1, key='s1')
    berm_abl1 = st.sidebar.number_input("Berm ABL 1 (m)", min_value=0.0, value=5.0, step=0.5, key='b1')

with st.sidebar.expander("Layer 2 Parameters"):
    height_abl2 = st.sidebar.number_input("Height ABL 2 (m)", min_value=0.0, value=5.0, step=0.5, key='h2')
    side_slope_abl2 = st.sidebar.number_input("Side Slope ABL 2 (1:X)", min_value=0.1, value=3.5, step=0.1, key='s2')
    berm_abl2 = st.sidebar.number_input("Berm ABL 2 (m)", min_value=0.0, value=5.0, step=0.5, key='b2')

with st.sidebar.expander("Layer 3 Parameters"):
    height_abl3 = st.sidebar.number_input("Height ABL 3 (m)", min_value=0.0, value=5.0, step=0.5, key='h3')
    side_slope_abl3 = st.sidebar.number_input("Side Slope ABL 3 (1:X)", min_value=0.1, value=3.5, step=0.1, key='s3')
    berm_abl3 = st.sidebar.number_input("Berm ABL 3 (m)", min_value=0.0, value=5.0, step=0.5, key='b3')

with st.sidebar.expander("Layer 4 Parameters"):
    height_abl4 = st.sidebar.number_input("Height ABL 4 (m)", min_value=0.0, value=5.0, step=0.5, key='h4')
    side_slope_abl4 = st.sidebar.number_input("Side Slope ABL 4 (1:X)", min_value=0.1, value=4.0, step=0.1, key='s4')
    berm_abl4 = st.sidebar.number_input("Berm ABL 4 (m)", min_value=0.0, value=5.0, step=0.5, key='b4')

with st.sidebar.expander("Layer 5 Parameters"):
    height_abl5 = st.sidebar.number_input("Height ABL 5 (m)", min_value=0.0, value=5.0, step=0.5, key='h5')
    side_slope_abl5 = st.sidebar.number_input("Side Slope ABL 5 (1:X)", min_value=0.1, value=4.0, step=0.1, key='s5')

# --- Calculations ---
# This section mirrors the logic from the JSON file.

# Basic quantities
tpa = quantity_tpd * 365
total_quantity_tons = tpa * duration_years
total_volume_required_cum = total_quantity_tons / density if density > 0 else 0

# Base Dimensions
landfill_area_sqm = length_m * width_m

# Below NGL calculations
base_width_b20 = bund_width + (bund_height * external_slope) + (bund_height * internal_slope)
top_width_c20 = bund_width

base_length_b21 = length_m - (2 * base_width_b20)
base_breadth_b22 = width_m - (2 * base_width_b20)
top_length_c21 = length_m - (2 * top_width_c20)
top_breadth_c22 = width_m - (2 * top_width_c20)

base_area_b24 = base_length_b21 * base_breadth_b22
top_area_c24 = top_length_c21 * top_breadth_c22

volume_below_ngl = ((base_area_b24 + top_area_c24) / 2) * depth_below_ngl if depth_below_ngl > 0 else 0

# Above NGL calculations (Layer by layer)
total_volume_abl = 0

# Layer 1
l1_top_len = top_length_c21
l1_top_bre = top_breadth_c22
l1_bot_len = l1_top_len - (2 * height_abl1 * side_slope_abl1)
l1_bot_bre = l1_top_bre - (2 * height_abl1 * side_slope_abl1)
l1_top_area = l1_top_len * l1_top_bre
l1_bot_area = l1_bot_len * l1_bot_bre
vol_abl1 = ((l1_top_area + l1_bot_area) / 2) * height_abl1 if height_abl1 > 0 else 0
total_volume_abl += vol_abl1

# Layer 2
l2_top_len = l1_bot_len - (2 * berm_abl1)
l2_top_bre = l1_bot_bre - (2 * berm_abl1)
l2_bot_len = l2_top_len - (2 * height_abl2 * side_slope_abl2)
l2_bot_bre = l2_top_bre - (2 * height_abl2 * side_slope_abl2)
l2_top_area = l2_top_len * l2_top_bre
l2_bot_area = l2_bot_len * l2_bot_bre
vol_abl2 = ((l2_top_area + l2_bot_area) / 2) * height_abl2 if height_abl2 > 0 else 0
total_volume_abl += vol_abl2

# Layer 3
l3_top_len = l2_bot_len - (2 * berm_abl2)
l3_top_bre = l2_bot_bre - (2 * berm_abl2)
l3_bot_len = l3_top_len - (2 * height_abl3 * side_slope_abl3)
l3_bot_bre = l3_top_bre - (2 * height_abl3 * side_slope_abl3)
l3_top_area = l3_top_len * l3_top_bre
l3_bot_area = l3_bot_len * l3_bot_bre
vol_abl3 = ((l3_top_area + l3_bot_area) / 2) * height_abl3 if height_abl3 > 0 else 0
total_volume_abl += vol_abl3

# Layer 4
l4_top_len = l3_bot_len - (2 * berm_abl3)
l4_top_bre = l3_bot_bre - (2 * berm_abl3)
l4_bot_len = l4_top_len - (2 * height_abl4 * side_slope_abl4)
l4_bot_bre = l4_top_bre - (2 * height_abl4 * side_slope_abl4)
l4_top_area = l4_top_len * l4_top_bre
l4_bot_area = l4_bot_len * l4_bot_bre
vol_abl4 = ((l4_top_area + l4_bot_area) / 2) * height_abl4 if height_abl4 > 0 else 0
total_volume_abl += vol_abl4

# Layer 5
l5_top_len = l4_bot_len - (2 * berm_abl4)
l5_top_bre = l4_bot_bre - (2 * berm_abl4)
l5_bot_len = l5_top_len - (2 * height_abl5 * side_slope_abl5)
l5_bot_bre = l5_top_bre - (2 * height_abl5 * side_slope_abl5)
l5_top_area = l5_top_len * l5_top_bre
l5_bot_area = l5_bot_len * l5_bot_bre
vol_abl5 = ((l5_top_area + l5_bot_area) / 2) * height_abl5 if height_abl5 > 0 else 0
total_volume_abl += vol_abl5

# Final Totals
total_landfill_volume = volume_below_ngl + total_volume_abl
total_capacity_tons = total_landfill_volume * density
lifespan_years = total_capacity_tons / tpa if tpa > 0 else 0
total_cost = landfill_area_sqm * cost_per_sqm
cost_per_ton = total_cost / total_capacity_tons if total_capacity_tons > 0 else 0
leachate_approx = tpa * 0.20 # Assumes 20% leachate generation

# --- Main Panel for Displaying Results ---

st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Landfill Volume", f"{total_landfill_volume:,.0f} cum")
col2.metric("Total Waste Capacity", f"{total_capacity_tons:,.0f} Tons")
col3.metric("Calculated Lifespan", f"{lifespan_years:.2f} Years")
col4.metric("Total Land Area", f"{landfill_area_sqm:,.0f} Sqm")

st.header("Cost Analysis")
col1a, col2a, col3a = st.columns(3)
col1a.metric("Total Land Cost", f"₹ {total_cost:,.2f}")
col2a.metric("Cost per Ton of Waste", f"₹ {cost_per_ton:.2f}")
col3a.metric("Est. Annual Leachate", f"{leachate_approx:,.0f} Tons")

st.header("Detailed Volume Breakdown")
st.table({
    "Component": ["Volume Below NGL", "Volume ABL 1", "Volume ABL 2", "Volume ABL 3", "Volume ABL 4", "Volume ABL 5", "Total ABL Volume"],
    "Volume (cubic meters)": [
        f"{volume_below_ngl:,.2f}",
        f"{vol_abl1:,.2f}",
        f"{vol_abl2:,.2f}",
        f"{vol_abl3:,.2f}",
        f"{vol_abl4:,.2f}",
        f"{vol_abl5:,.2f}",
        f"{total_volume_abl:,.2f}"
    ]
})

st.header("Design Requirements vs. Capacity")
st.markdown(f"**Total Volume Required:** `{total_volume_required_cum:,.0f} cum` (based on waste quantity and duration)")
st.markdown(f"**Total Volume Designed:** `{total_landfill_volume:,.0f} cum`")

if total_landfill_volume < total_volume_required_cum:
    st.warning("The designed landfill volume is LESS than the required volume. Consider increasing dimensions, height, or depth.")
else:
    st.success("The designed landfill volume is SUFFICIENT for the required capacity.")
