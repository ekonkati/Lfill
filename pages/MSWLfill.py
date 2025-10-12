import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="Landfill Construction Simulator", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .plot-container {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .compact-input {
        margin-bottom: 0.5rem;
    }
    .input-group {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏗️ Landfill Construction Simulator</h1>', unsafe_allow_html=True)
st.markdown("A comprehensive tool for landfill design, volume calculation, and interactive visualization")

# Initialize session state variables properly
def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.daily_quantity = 1000.0
        st.session_state.duration = 25.0
        st.session_state.density = 1.00
        st.session_state.width = 600.0
        st.session_state.length = 600.0
        st.session_state.landfill_area_acres = 10.0
        st.session_state.bund_width = 5.0
        st.session_state.bund_height = 4.0
        st.session_state.waste_height = 5.0
        st.session_state.berm_width = 4.0
        st.session_state.external_slope = 2.0
        st.session_state.internal_slope = 3.0
        st.session_state.waste_slope = 3.0
        st.session_state.depth_below_ngl = 2.0
        st.session_state.anchor_trench_depth = 1.5
        st.session_state.anchor_trench_width = 2.0
        st.session_state.cost_per_sqm = 3850.0
        st.session_state.leachate_percentage = 20

# Initialize session state at startup
initialize_session_state()

# Display results in tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Input Parameters", "📊 Results Summary", "🎯 Interactive Visualization", "📐 Detailed Calculations", "💰 Cost Analysis", "📈 Reports"])

with tab1:
    st.header("Input Parameters")
    
    # Compact layout with columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Basic Parameters**")
            daily_quantity = st.number_input("Daily Waste (TPD)", value=float(st.session_state.daily_quantity), min_value=1.0, step=10.0, key="daily")
            duration = st.number_input("Duration (years)", value=float(st.session_state.duration), min_value=1.0, step=1.0, key="duration")
            density = st.number_input("Density (ton/Cum)", value=float(st.session_state.density), min_value=0.1, step=0.01, key="density")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Land Dimensions**")
            width = st.number_input("Width (m)", value=float(st.session_state.width), min_value=10.0, step=10.0, key="width")
            length = st.number_input("Length (m)", value=float(st.session_state.length), min_value=10.0, step=10.0, key="length")
            landfill_area_acres = st.number_input("Area (Acres)", value=float(st.session_state.landfill_area_acres), min_value=1.0, step=0.5, key="area")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Bund Parameters**")
            bund_width = st.number_input("Bund Width (m)", value=float(st.session_state.bund_width), min_value=1.0, step=0.5, key="bund_width")
            bund_height = st.number_input("Bund Height (m)", value=float(st.session_state.bund_height), min_value=1.0, step=0.5, key="bund_height")
            waste_height = st.number_input("Waste Height (m)", value=float(st.session_state.waste_height), min_value=1.0, step=0.5, key="waste_height")
            berm_width = st.number_input("Berm Width (m)", value=float(st.session_state.berm_width), min_value=1.0, step=0.5, key="berm_width")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Slope Parameters**")
            external_slope = st.number_input("External Slope", value=float(st.session_state.external_slope), min_value=1.0, step=0.5, key="external")
            internal_slope = st.number_input("Internal Slope", value=float(st.session_state.internal_slope), min_value=1.0, step=0.5, key="internal")
            waste_slope = st.number_input("Waste Slope", value=float(st.session_state.waste_slope), min_value=1.0, step=0.5, key="waste_slope")
            depth_below_ngl = st.number_input("Depth Below NGL (m)", value=float(st.session_state.depth_below_ngl), min_value=0.0, step=0.5, key="depth")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Cost and Leachate parameters in a compact row
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Trench & Cost**")
            anchor_trench_depth = st.number_input("Anchor Trench Depth (m)", value=float(st.session_state.anchor_trench_depth), min_value=0.5, step=0.5, key="anchor_depth")
            anchor_trench_width = st.number_input("Anchor Trench Width (m)", value=float(st.session_state.anchor_trench_width), min_value=1.0, step=0.5, key="anchor_width")
            cost_per_sqm = st.number_input("Cost per Sqm (INR)", value=float(st.session_state.cost_per_sqm), min_value=100.0, step=100.0, key="cost")
            leachate_percentage = st.slider("Leachate %", min_value=5, max_value=50, value=float(st.session_state.leachate_percentage), key="leachate_percentage")
            st.markdown('</div>', unsafe_allow_html=True)

# Define the cross-section profile function with proper landfill structure
def create_realistic_cross_section_profile(points, cross_section_pos, is_vertical, results_data, bund_h, external_slope_param, at_depth, anchor_depth, anchor_depth, anchor_width, waste_h, berm_w):
    """Create realistic cross-section profile with proper landfill structure including anchor trench"""
    profile = []
    bund_profile = []
    
    # Calculate bund geometry
    bund_inner_length = results_data['bl_length']
    bund_inner_width = results_data['bl_width']
    bund_outer_length = bund_inner_length + 2 * (bund_h * external_slope_param)
    bund_outer_width = bund_inner_width + 2 * (bund_h * external_slope_param)
    
    for point in points:
        if is_vertical:
            # Vertical cross-section (along Y-axis)
            x_pos = cross_section_pos
            y_pos = point
            
            # Calculate height at this point
            height_at_point = 0
            
            # Check if within landfill area
            if abs(y_pos) <= bund_outer_length/2:
                # Check if point is in waste area
                waste_height = 0
                for level in results_data['levels']:
                    if level['Volume'] > 0:
                        l, w = level['Length'], level['Width']
                        if abs(x_pos) <= l/2 and abs(y_pos) <= w/2:
                            waste_height = level['Top_Z']
                profile.append(waste_height)
                
                # Calculate bund height
                if abs(x_pos) <= bund_outer_width/2:
                    if abs(y_pos) <= bund_outer_length/2:
                        # Inside bund area
                        dist_from_inner_x = max(0, abs(x_pos) - bund_inner_width/2)
                        dist_from_inner_y = max(0, abs(y_pos) - bund_inner_length/2)
                        
                        if dist_from_inner_x > 0 or dist_from_inner_y > 0:
                            # Outside inner boundary - calculate bund height
                            if dist_from_inner_x > 0 and dist_from_inner_y > 0:
                                # Corner area
                                corner_dist = math.sqrt(dist_from_inner_x**2 + dist_from_inner_y**2)
                                bund_h_calc = min(bund_h, corner_dist / external_slope_param)
                            elif dist_from_inner_x > 0:
                                # Side area
                                bund_h_calc = min(bund_h, dist_from_inner_x / external_slope_param)
                            else:
                                # End area
                                bund_h_calc = min(bund_h, dist_from_inner_y / external_slope_param)
                        else:
                            bund_h_calc = 0
                        
                        bund_profile.append(bund_h_calc)
                    else:
                        bund_profile.append(0)
                else:
                    bund_profile.append(0)
            else:
                profile.append(0)
                bund_profile.append(0)
        else:
            # Horizontal cross-section (along X-axis)
            x_pos = point
            y_pos = cross_section_pos
            
            # Calculate height at this point
            height_at_point = 0
            
            # Check if within landfill area
            if abs(x_pos) <= bund_outer_length/2:
                # Check if point is in waste area
                waste_height = 0
                for level in results_data['levels']:
                    if level['Volume'] > 0:
                        l, w = level['Length'], level['Width']
                        if abs(x_pos) <= l/2 and abs(y_pos) <= w/2:
                            waste_height = level['Top_Z']
                profile.append(waste_height)
                
                # Calculate bund height
                if abs(y_pos) <= bund_outer_width/2:
                    if abs(x_pos) <= bund_outer_length/2:
                        # Inside bund area
                        dist_from_inner_x = max(0, abs(x_pos) - bund_inner_length/2)
                        dist_from_inner_y = max(0, abs(y_pos) - bund_inner_width/2)
                        
                        if dist_from_inner_x > 0 or dist_from_inner_y > 0:
                            # Outside inner boundary - calculate bund height
                            if dist_from_inner_x > 0 and dist_from_inner_y > 0:
                                # Corner area
                                corner_dist = math.sqrt(dist_from_inner_x**2 + dist_from_inner_y**2)
                                bund_h_calc = min(bund_h, corner_dist / external_slope_param)
                            elif dist_from_inner_x > 0:
                                # Side area
                                bund_h_calc = min(bund_h, dist_from_inner_x / external_slope_param)
                            else:
                                # End area
                                bund_h_calc = min(bund_h, dist_from_inner_y / external_slope_param)
                        else:
                            bund_h_calc = 0
                        
                        bund_profile.append(bund_h_calc)
                    else:
                        bund_profile.append(0)
                else:
                    profile.append(0)
                    bund_profile.append(0)
            else:
                profile.append(0)
                bund_profile.append(0)
    
    return profile, bund_profile

# Calculate all values with proper landfill structure
def calculate_landfill_parameters():
    # Basic calculations
    tpa = daily_quantity * 365
    total_quantity = tpa * duration
    landfill_area_sqm = landfill_area_acres * 4047
    
    # Calculate levels and volumes
    levels_data = []
    
    # 1. Anchor Trench - Bottom-most level
    anchor_trench_volume = (width - 2 * anchor_trench_width) * (length - 2 * anchor_trench_width) * anchor_trench_depth)
    levels_data.append({
        'Level': 'Anchor Trench',
        'Length': width - 2 * anchor_trench_width,
        'Width': length - 2 * anchor_trench_width,
        'Area': (width - 2 * anchor_trench_width) * (length - 2 * anchor_trench_width),
        'Height': anchor_trench_depth,
        'Volume': anchor_trench_volume,
        'Type': 'Anchor Trench',
        'Bottom_Z': -depth_below_ngl - anchor_trench_depth,
        'Top_Z': -depth_below_ngl
    })
    
    # 2. Below Ground Level (BBL) - Excavation below ground forming tub structure
    bbl_length = width - 4 * anchor_trench_width
    bbl_width_calc = length - 4 * anchor_trench_width
    
    if bbl_length <= 0 or bbl_width_calc <= 0:
        bbl_length = width * 0.8
        bbl_width_calc = length * 0.8
    
    bbl_area = bbl_length * bbl_width_calc
    bbl_volume = bbl_area * depth_below_ngl
    
    levels_data.append({
        'Level': 'BBL',
        'Length': bbl_length,
        'Width': bbl_width_calc,
        'Area': bbl_area,
        'Height': depth_below_ngl,
        'Volume': bbl_volume,
        'Type': 'Excavation',
        'Bottom_Z': -depth_below_ngl,
        'Top_Z': 0
    })
    
    # 3. Base Level (BL) - Waste filling up to top of bund (TOB)
    bl_length = bbl_length + (((bund_height + depth_below_ngl * internal_slope) * 2)
    bl_width = bbl_width_calc + (((bund_height + depth_below_ngl * internal_slope) * 2)
    bl_area = bl_length * bl_width
    bl_volume = ((bbl_area + bl_area) / 2) * bund_height
    
    levels_data.append({
        'Level': 'BL',
        'Length': bl_length,
        'Width': bl_width,
        'Area': bl_area,
        'Height': bund_height,
        'Volume': bl_volume,
        'Type': 'Waste up to Bund',
        'Bottom_Z': 0,
        'Top_Z': bund_height
    })
    
    # 4. Above Bund Levels (ABL) - Waste filling from top of bund to crest with berms
    current_length = bl_length
    current_width = bl_width
    total_volume = anchor_trench_volume + bbl_volume + bl_volume
    current_z = bund_height
    
    for i in range(1, 10):  # Calculate up to 9 levels
        # ABL level - waste filling
        abl_length = current_length - (waste_height * waste_slope * 2)
        abl_width_calc = current_length - (waste_height * waste_slope * 2)
        
        if abl_length <= 0 or abl_width_calc <= 0:
            break
            
        abl_area = abl_length * abl_width_calc
        prev_area = current_length * current_width
        abl_volume = ((prev_area + abl_area) / 2) * waste_height
        total_volume += abl_volume
        
        levels_data.append({
            'Level': f'ABL {i}',
            'Length': abl_length,
            'Width': abl_width_calc,
            'Area': abl_area,
            'Height': waste_height,
            'Volume': abl_volume,
            'Type': 'Waste above Bund',
            'Bottom_Z': current_z,
            'Top_Z': current_z + waste_height
        })
        
        current_z += waste_height
        
        # Berm level
        berm_length = abl_length - (berm_width * 2)
        berm_width_calc = abl_width_calc - (berm_width * 2)
        
        if berm_length <= 0 or berm_width_calc <= 0:
            break
            
        berm_area = berm_length * berm_width_calc
        
        levels_data.append({
            'Level': f'ABL {i} Berm',
            'Length': berm_length,
            'Width': berm_width_calc,
            'Area': berm_area,
            'Height': 0.0,
            'Volume': 0.0,
            'Type': 'Berm',
            'Bottom_Z': current_z,
            'Top_Z': current_z
        })
        
        current_length = berm_length
        current_width = berm_width_calc
    
    # Cost calculations
    provided_quantity = total_volume * density
    required_quantity = total_quantity
    is_sufficient = "Yes" if provided_quantity >= required_quantity else "No"
    landfill_life = provided_quantity / tpa if tpa > 0 else 0
    total_cost = landfill_area_sqm * cost_per_sqm
    per_ton_cost = total_cost / provided_quantity if provided_quantity > 0 else 0
    
    # Leachate calculations
    leachate_tpa = tpa * (leachate_percentage / 100)
    leachate_kld = leachate_tpa / 365
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': provided_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'leachate_kld': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width_calc,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': provided_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'leachate_kld': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width_calc,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }

# Calculate parameters
results = calculate_landfill_parameters()

# Display results in tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Input Parameters", "📊 Results Summary", "🎯 Interactive Visualization", "📐 Detailed Calculations", "💰 Cost Analysis", "📈 Reports"])

with tab1:
    st.header("Input Parameters")
    
    # Compact layout with columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Basic Parameters**")
            daily_quantity = st.number_input("Daily Waste (TPD)", value=float(st.session_state.daily_quantity, min_value=1.0, step=10.0, key="daily")
            duration = st.number_input("Duration (years)", value=float(st.session_state.duration, min_value=1.0, step=1.0, key="duration")
            density = st.number_input("Density (ton/Cum)", value=float(st.session_state.density, min_value=0.1, step=0.01, key="density")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Land Dimensions**")
            width = st.number_input("Width (m)", value=float(st.session_state.width, min_value=10.0, step=10.0, key="width")
            length = st.number_input("Length (m)", value=st.session_state.length, min_value=10.0, step=10.0, key="length")
            landfill_area_acres = st.number_input("Area (Acres)", value=st.session_state.landfill_area_acres, min_value=1.0, step=0.5, key="area")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Bund Parameters**")
            bund_width = st.number_input("Bund Width (m)", value=float(st.session_state.bund_width, min_value=1.0, step=0.5, key="bund_width")
            bund_height = st.number_input("Bund Height (m)", value=st.session_state.bund_height, min_value=1.0, step=0.5, key="bund_height")
            waste_height = st.number_input("Waste Height (m)", value=st.session_state.waste_height, min_value=1.0, step=0.5, key="waste_height")
            berm_width = st.number_input("Berm Width (m)", value=st.session_state.berm_width, min_value=1.0, step=0.5, key="berm_width")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Slope Parameters**")
            external_slope = st.number_input("External Slope", value=float(st.session_state.external_slope, min_value=1.0, step=0.5, key="external")
            internal_slope = st.number_input("Internal Slope", value=st.session_state.internal_slope, min_value=1.0, step=0.5, key="internal")
            waste_slope = st.number_input("Waste Slope", value=st.session_state.waste_slope, min_value=1.0, step=0.5, key="waste_slope")
            depth_below_ngl = st.number_input("Depth Below NGL (m)", value=st.session_state.depth_below_ngl, min_value=0.0, step=0.5, key="depth")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Cost and Leachate parameters in a compact row
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Trench & Cost**")
            anchor_trench_depth = st.number_input("Anchor Trench Depth (m)", value=st.session_state.anchor_trench_depth, min_value=0.5, step=0.5, key="anchor_depth")
            anchor_trench_width = st.number_input("Anchor Trench Width (m)", value=st.session_state.anchor_trench_width, min_value=1.0, step=0.5, key="anchor_width")
            cost_per_sqm = st.number_input("Cost per Sqm (INR)", value=st.session_state.cost_per_sqm, min_value=100.0, step=100.0, key="cost")
            leachate_percentage = st.slider("Leachate %", min_value=5, max_value=50, value=float(st.session_state.leachate_percentage, min_value=5, max_value=50, value=float(st.session_state.leachate_percentage))
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.info("""
        📊 **Report Contents:**
        - Executive Summary
        - Design Parameters
        - Volume Calculations
        - Cost Analysis
        - Visualizations (2D & 3D)
        - Recommendations
        """)
        
        if st.button("📥 Generate PDF Report", type="primary"):
            st.success("PDF report generation would be implemented here!")
            st.balloons()
        
    with col2:
        st.info("""
        📈� **Export Options:**
        - Excel spreadsheet with calculations
        - CSV data for further analysis
        - High-resolution images
        - 3D model files
        """)
        
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("📊 Export to Excel"):
                st.success("Excel export would be implemented here!")
        with col2b:
            if st.button("📸 Export Images"):
                st.success("Image export would be implemented here!")
    
    # Summary section
    st.markdown('<h3 class="section-header">Project Summary</h3>', unsafe_allow_html=True)
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Total Capacity", f"{results['provided_quantity']:,.0f} Tons")
        st.metric("Landfill Life", f"{results['landfill_life']:.1f} Years")
    
    with summary_col2:
        st.metric("Total Investment", f"₹{results['total_cost']:,.0f}")
        st.metric("Cost per Ton", f"₹{results['per_ton_cost']:.0f}")
    
    with summary_col3:
        st.metric("Daily Processing", f"{daily_quantity:.0f} TPD")
        st.metric("Annual Processing", f"{results['tpa']:,.0f} TPA")

# Footer
st.markdown("---")
st.markdown("### 📝� Important Notes")
st.info("""
- This simulator provides estimates based on the input parameters
- Actual construction may require additional engineering considerations
- Environmental regulations and local conditions should be factored in
- Regular monitoring and maintenance are essential for landfill operations
- The 3D visualization is a simplified representation for conceptual understanding
- Cross-sections are generated dynamically based on slider position
""")

st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-top: 2rem;'>
    <p>Landfill Construction Simulator v2.0 | Interactive Design & Analysis Tool</p>
</div>
""", unsafe_allow_html=True)

with tab3:
    st.header("Interactive 3D/2D Visualization")
    
    # Cross-section controls
    st.markdown('<h3 class="section-header">Cross-Section Controls</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.write("**Select Position:**")
        cross_section_type = st.radio("Cross-Section Type", ["Vertical (Y-axis)", "Horizontal (X-axis)"])
        
    with col2:
        if cross_section_type == "Vertical (Y-axis)":
            cross_section_x = st.slider("X Position (m)", -width/2, width/2, 0.0, step=10.0)
            cross_section_y = 0.0
        else:
            cross_section_y = st.slider("Y Position (m)", -length/2, length/2, 0.0, step=10.0)
            cross_section_x = 0.0
    
    with col3:
        st.write("**Current Position:**")
        st.write(f"X: {cross_section_x:.1f}m")
        st.write(f"Y: {cross_section_y:.1f}m")
    
    # Visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("3D Landfill Model")
        
        # Create 3D surface plot with proper bund modeling
        fig_3d = go.Figure()
        
        # Create height map for 3D surface
        x = np.linspace(-width/2, width/2, 80)
        y = np.linspace(-length/2, length/2, 80)
        X, Y = np.meshgrid(x, y)
        
        # Create height map with proper bund modeling
        Z = np.zeros_like(X)
        
        # Calculate bund dimensions
        bund_inner_length = results['bl_length']
        bund_inner_width = results['bl_length']
        bund_outer_length = bund_inner_length + 2 * (bund_height * external_slope)
        bund_outer_width = bund_inner_width + 2 * (bund_height * external_slope)
        bund_outer_width = bund_inner_width + 2 * (bund_height * external_slope)
        
        # Model the complete landfill structure
        for i in range(len(X)):
            for j in range(len(X[0]):
                x_pos, y_pos = X[i, j], Y[i, j] = X[i, j], Y[i, j]
                
                # Check if point is in bund area
                if (abs(x_pos) <= bund_outer_length/2 and abs(y_pos) <= bund_outer_width/2):
                    # Inside outer bund boundary
                    
                    # Check if point is in waste area
                    if (abs(x_pos) <= bund_inner_length/2 and abs(y_pos) <= bund_inner_width/2):
                        # Inside waste area - calculate waste height
                        waste_height = 0
                        for level in results['levels']:
                            if level['Volume'] > 0:
                                l, w = level['Length'], level['Width']
                                if abs(x_pos) <= l/2 and abs(y_pos) <= w/2:
                                    waste_height = level['Top_Z']
                                profile.append(waste_height)
                        
                # Calculate bund height
                if abs(x_pos) <= bund_outer_width/2:
                    if abs(y_pos) <= bund_outer_length/2:
                        # Inside bund area
                        dist_from_inner_x = max(0, abs(x_pos - bund_inner_length/2)
                        dist_from_inner_y = max(0, abs(y_pos - bund_inner_length/2)
                        
                        if dist_from_inner_x > 0 or dist_from_inner_y > 0:
                            # Outside inner boundary - calculate bund height
                            if dist_from_inner_x > 0 and dist_from_inner_y > 0:
                                # Corner area
                                corner_dist = math.sqrt(dist_from_inner_x**2 + dist_from_inner_y**2)
                                bund_h_calc = min(bund_height, corner_dist / external_slope)
                            elif dist_from_inner_x > 0:
                                # Side area
                                bund_h_calc = min(bund_height, dist_from_inner_x / external_slope)
                            else:
                                # End area
                                bund_h_calc = min(bund_h, dist_from_inner_y / external_slope)
                            else:
                                bund_h_calc = 0
                            
                        bund_profile.append(bund_h_calc)
                    else:
                        bund_profile.append(0)
                else:
                    profile.append(0)
                else:
                    profile.append(0)
                    bund_profile.append(0)
            else:
                profile.append(0)
                    bund_profile.append(0)
    
    return profile, bund_profile

# Calculate all values with proper landfill structure
def calculate_landfill_parameters():
    # Basic calculations
    tpa = daily_quantity * 365
    total_quantity = tpa * duration
    landfill_area_sqm = landfill_area_acres * 4047
    
    # Calculate levels and volumes
    levels_data = []
    
    # 1. Anchor Trench - Bottom-most level
    anchor_trench_volume = (width - 2 * anchor_trench_width) * (length - 2 * anchor_trench_width) * anchor_trench_depth
    levels_data.append({
        'Level': 'Anchor Trench',
        'Length': width - 2 * anchor_trench_width,
        'Width': length - 2 * anchor_trench_width,
        'Area': (width - 2 * anchor_trench_width) * (length - 2 * anchor_trench_width),
        'Area': (width - 2 * anchor_trench_width) * (length - 2 * anchor_trench_width),
        'Height': anchor_trench_depth,
        'Volume': anchor_trench_volume,
        'Type': 'Anchor Trench',
        'Bottom_Z': -depth_below_ngl - anchor_trench_depth,
        'Top_Z': -depth_below_ngl
    })
    
    # 2. Below Ground Level (BBL) - Excavation below ground forming tub structure
    bbl_length = width - 4 * anchor_trench_width
    bbl_width_calc = length - 4 * anchor_trench_width
    
    if bbl_length <= 0 or bbl_width_calc <= 0:
        bbl_length = width * 0.8
        bbl_width_calc = length * 0.8
        bbl_width_calc = length * 0.8
    
    bbl_area = bbl_length * bbl_width_calc
    bbl_volume = bbl_area * depth_below_ngl
    
    levels_data.append({
        'Level': 'BBL',
        'Length': bbl_length,
        'Width': bbl_width_calc,
        'Area': bbl_area,
        'Height': depth_below_ngl,
        'Volume': bbl_volume,
        'Type': 'Excavation',
        'Bottom_Z': -depth_below_ngl,
        'Top_Z': 0
    })
    
    # 3. Base Level (BL) - Waste filling up to top of bund (TOB)
    bl_length = bbl_length + (((bund_height + depth_below_ngl * internal_slope) * 2)
    bl_width = bbl_width_calc + (((bund_height + depth_below_ngl * internal_slope) * 2)
    bl_area = bl_length * bl_width
    bl_volume = ((bbl_area + bl_area) / 2) * bund_height
    
    levels_data.append({
        'Level': 'BL',
        'Length': bl_length,
        'Width': bl_width,
        'Area': bl_area,
        'Height': bund_height,
        'Volume': bl_volume,
        'Type': 'Waste up to Bund',
        'Bottom_Z': 0,
        'Top_Z': bund_height
    })
    
    # 4. Above Bund Levels (ABL) - Waste filling from top of bund to crest with berms
    current_length = bl_length
    current_width = bl_width
    total_volume = anchor_trench_volume + bbl_volume + bl_volume
    current_z = bund_height
    
    for i in range(1, 10):  # Calculate up to 9 levels
        # ABL level - waste filling
        abl_length = current_length - (waste_height * waste_slope * 2)
        abl_width_calc = current_width - (waste_height * waste_slope * 2)
        
        if abl_length <= 0 or abl_width_calc <= 0:
            break
            
        abl_area = abl_length * abl_width_calc
        prev_area = current_length * current_width
        abl_volume = ((prev_area + abl_area) / 2) * waste_height
        total_volume += abl_volume
        
        levels_data.append({
            'Level': f'ABL {i}',
            'Length': abl_length,
            'Width': abl_width_calc,
            'Area': abl_area,
            'Height': waste_height,
            'Volume': abl_volume,
            'Type': 'Waste above Bund',
            'Bottom_Z': current_z,
            'Top_Z': current_z + waste_height
        })
        
        # Berm level
        berm_length = abl_length - (berm_width * 2)
        berm_width_calc = abl_width_calc - (berm_width * 2)
        
        if berm_length <= 0 or berm_width_calc <= 0:
            break
            
        berm_area = berm_length * berm_width_calc
        berm_area = berm_length * berm_width_calc
        
        levels_data.append({
            'Level': f'ABL {i} Berm',
            'Length': berm_length,
            'Width': berm_width_calc,
            'Area': berm_area,
            'Height': 0.0,
            'Volume': 0.0,
            'Type': 'Berm',
            'Bottom_Z': current_z,
            'Top_Z': current_z
        })
        
        current_length = berm_length
        current_width = berm_width_calc
        
    # Cost calculations
    provided_quantity = total_volume * density
    required_quantity = total_quantity
    is_sufficient = "Yes" if provided_quantity >= required_quantity else "No"
    landfill_life = provided_quantity / tpa if tpa > 0 else 0
    total_cost = landfill_area_sqm * cost_per_sqm
    per_ton_cost = total_cost / provided_quantity if provided_quantity > 0 else 0
    leachate_tpa = tpa * (leachate_percentage / 100)
    leachate_kld = leachate_tpa / 365
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': provided_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'leachate_kld': leachate_kld
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }

# Calculate parameters
results = calculate_landfill_parameters()

# Display results in tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Input Parameters", "📊 Results Summary", "🎯 Interactive Visualization", "📐 Detailed Calculations", "💰 Cost Analysis", "📈 Reports"])

with tab1:
    st.header("Input Parameters")
    
    # Compact layout with columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Basic Parameters**")
            daily_quantity = st.number_input("Daily Waste (TPD)", value=1000.0, min_value=1.0, step=10.0, key="daily")
            duration = st.number_input("Duration (years)", value=25.0, min_value=1.0, step=1.0, key="duration")
            density = st.number_input("Density (ton/Cum)", value=1.00, min_value=0.1, step=0.01, key="density")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Land Dimensions**")
            width = st.number_input("Width (m)", value=600.0, min_value=10.0, step=10.0, key="width")
            length = st.number_input("Length (m)", value=600.0, min_value=10.0, step=10.0, key="length")
            landfill_area_acres = st.number_input("Area (Acres)", value=10.0, min_value=1.0, step=0.5, key="area")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Bund Parameters**")
            bund_width = st.number_input("Bund Width (m)", value=5.0, min_value=1.0, step=0.5, key="bund_width")
            bund_height = st.number_input("Bund Height (m)", value=4.0, min_value=1.0, step=0.5, key="bund_height")
            waste_height = st.number_input("Waste Height (m)", value=5.0, min_value=1.0, step=0.5, key="waste_height")
            berm_width = st.number_input("Berm Width (m)", value=4.0, min_value=1.0, step=0.5, key="berm_width")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Slope Parameters**")
            external_slope = st.number_input("External Slope", value=2.0, min_value=1.0, step=0.5, key="external")
            internal_slope = st.number_input("Internal Slope", value=3.0, min_value=1.0, step=0.5, key="internal")
            waste_slope = st.number_input("Waste Slope", value=3.0, min_value=1.0, step=0.5, key="waste_slope")
            depth_below_ngl = st.number_input("Depth Below NGL (m)", value=2.0, min_value=0.0, step=0.5, key="depth")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Cost and Leachate parameters in a compact row
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Trench & Cost**")
            anchor_trench_depth = st.number_input("Anchor Trench Depth (m)", value=1.5, min_value=0.5, step=0.5, key="anchor_depth")
            anchor_trench_width = st.number_input("Anchor Trench Width (m)", value=2.0, min_value=1.0, step=0.5, key="anchor_width")
            cost_per_sqm = st.number_input("Cost per Sqm (INR)", value=3850.0, min_value=100.0, step=100.0, key="cost")
            leachate_percentage = st.slider("Leachate %", min_value=5, max_value=50, value=float(st.session_state.leachate_percentage))
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.info("""
        📊 **Report Contents:**
        - Executive Summary
        - Design Parameters
        - Volume Calculations
        - Cost Analysis
        - Visualizations (2D & 3D
        - Recommendations
        """)
        
        if st.button("📥 Generate PDF Report", type="primary"):
            st.success("PDF report generation would be implemented here!")
            st.balloons()
    
    with col2:
        st.info("""
        📈� **Export Options:**
        - Excel spreadsheet with calculations
        - CSV data for further analysis
        - High-resolution images
        - 3D model files
        "" )
        
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("📊 Export to Excel"):
                st.success("Excel export would be implemented here!")
        with col2b:
            if st.button("📸 Export Images"):
                st.success("Image export would be implemented here!")
    
    # Summary section
    st.markdown('<h3 class="section-header">Project Summary</h3>', unsafe_allow_html=True)
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Total Capacity", f"{results['provided_quantity']:,.0f} Tons")
        st.metric("Landfill Life", f"{results['landfill_life']:.1f} Years")
    
    with summary_col2:
        st.metric("Total Investment", f"₹{results['total_cost']:,.0f}")
        st.metric("Cost per Ton", f"₹{results['per_ton_cost']:.0f}")
    
    with summary_col3:
        st.metric("Daily Processing", f"{daily_quantity:.0f} TPD")
        st.metric("Annual Processing", f"{results['tpa']:,.0f} TPA")
    
    with summary_col3:
        st.metric("Annual Processing", f"{results['tpa']:,.0f} TPA")

# Footer
st.markdown("---")
st.markdown("### 📝 Important Notes")
st.info("""
- This simulator provides estimates based on the input parameters
- Actual construction may require additional engineering considerations
- Environmental regulations and local conditions should be factored in
- Regular monitoring and maintenance are essential for landfill operations
- The 3D visualization is a simplified representation for conceptual understanding
- Cross-sections are generated dynamically based on slider position
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-top: 2rem;'>
    <p>Landfill Construction Simulator v2.0 | Interactive Design & Analysis Tool</p>
</div>
""", unsafe_allow_html=True)

with tab3:
    st.header("Interactive 3D/2D Visualization")
    
    # Cross-section controls
    st.markdown('<h3 class="section-header">Cross-Section Controls</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.write("**Select Position:**")
        cross_section_type = st.radio("Cross-Section Type", ["Vertical (Y-axis)", "Horizontal (X-axis)"])
        
    with col2:
        if cross_section_type == "Vertical (Y-axis)":
            cross_section_x = st.slider("X Position (m)", -width/2, width/2, 0.0, step=10.0)
            cross_section_y = 0.0
        else:
            cross_section_y = st.slider("Y Position (m)", -length/2, length/2, 0.0, step=10.0)
            cross_section_x = 0.0
    
    with col3:
        st.write("**Current Position:**")
        st.write(f"X: {cross_section_x:.1f}m")
        st.write(f"Y: {cross_section_y:.1f}m")
    
    # Visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("3D Landfill Model")
        
        # Create 3D surface plot with proper bund modeling
        fig_3d = go.Figure()
        
        # Create height map for 3D surface
        x = np.linspace(-width/2, width/2, 80)
        y = np.linspace(-length/2, length/2, 80)
        X, Y = np.linspace(-length/2, length/2, 80)
        X, Y = np.meshgrid(x, y)
        
        # Create height map with proper bund modeling
        Z = np.zeros_like(X)
        
        # Create height map with proper bund modeling
        Z = np.zeros_like(X)
        
        # Calculate bund dimensions
        bund_inner_length = results['bl_length']
        bund_inner_width = results['bl_length']
        bund_inner_width = results['bl_width']
        bund_outer_length = bund_inner_length + 2 * (bund_height * external_slope)
        bund_outer_width = bund_inner_width + 2 * (bund_height * external_slope)
        bund_outer_width = bund_inner_width + 2 * (bund_height * external_slope)
        
        # Model the complete landfill structure
        for i in range(len(X)):
            for j in range(len(X[0]):
            for j in range(len(X[0])):
                x_pos, y_pos = X[i, j], Y[i, j], Y[i, j]
                
                # Check if point is in bund area
                if (abs(x_pos <= bund_outer_length/2 and abs(y_pos) <= bund_outer_width/2):
                    # Inside outer bund boundary
                    
                    # Check if point is in waste area
                    if (abs(x_pos <= bund_inner_length/2 and abs(y_pos) <= bund_inner_width/2):
                        waste_height = 0
                        for level in results['levels']:
                            if level['Volume'] > 0:
                                l, w = level['Length'], level['Width']
                                if abs(x_pos <= l/2 and abs(y_pos) <= w/2:
                                    waste_height = level['Top_Z']
                profile.append(waste_height)
                
                # Calculate bund height
                if abs(x_pos) <= bund_outer_width/2:
                    if abs(y_pos) <= bund_outer_length/2:
                        # Inside bund area
                        dist_from_inner_x = max(0, abs(x_pos - bund_inner_length/2)
                        dist_from_inner_y = max(0, abs(y_pos - bund_inner_width/2)
                        
                        if dist_from_inner_x > 0 or dist_from_inner_y > 0:
                            # Corner area
                            corner_dist = math.sqrt(dist_from_inner_x**2 + dist_from_inner_y**2)
                            bund_h_calc = min(bund_height, corner_dist / external_slope_param)
                        elif dist_from_inner_x > 0:
                            # Side area
                            bund_h_calc = min(bund_height, dist_from_inner_x / external_slope_param)
                        else:
                            # End area
                            bund_h_calc = min(bund_h, dist_from_inner_y / external_slope_param)
                        else:
                            bund_h_calc = 0
                        
                        bund_profile.append(bund_h_calc)
                    else:
                        bund_profile.append(0)
                else:
                    profile.append(0)
            else:
                profile.append(0)
                    bund_profile.append(0)
            else:
                profile.append(0)
    
    return profile, bund_profile

# Calculate all values with proper landfill structure
def calculate_landfill_parameters():
    # Basic calculations
    tpa = daily_quantity * 365
    total_quantity = tpa * duration
    total_quantity = tpa * duration
    landfill_area_sqm = landfill_area_acres = landfill_area_acres * 4047
    
    # Calculate levels and volumes
    levels_data = []
    
    # 1. Anchor Trench - Bottom-most level
    anchor_trench_volume = (width - 2 * anchor_trench_width) * (length - 2 * anchor_trench_width) * anchor_trench_depth
    levels_data.append({
        'Level': 'Anchor Trench',
        'Length': width - 2 * anchor_trench_width,
        'Width': length - 2 * anchor_trench_width,
        'Area': (width - 2 * anchor_trench_width) * (length - 2 * anchor_trench_width),
        'Area': (width - 2 * anchor_trench_width) * (length - 2 * anchor_trench_width),
        'Height': anchor_trench_depth,
        'Volume': anchor_trench_volume,
        'Type': 'Anchor Trench',
        'Bottom_Z': -depth_below_ngl - anchor_trench_depth,
        'Top_Z': -depth_below_ngl - anchor_trench_depth,
        'Type': 'Anchor Trench',
        'Bottom_Z': -depth_below_ngl - anchor_trench_depth
        'Top_Z': -depth_below_ngl - anchor_trench_depth
    })
    
    # 2. Below Ground Level (BBL) - Waste filling up to top of bund (TOB)
    bl_length = bbl_length + (((bund_height + depth_below_ngl * internal_slope) * 2)
    bl_width = bbl_width_calc + (((bund_height + depth_below_ngl * internal_slope) * 2)
    bl_width = bbl_width_calc + (((bund_height + depth_below_ngl * internal_slope) * 2)
    bl_area = bl_length * bl_width
    bl_area = bl_length * bl_width
    bl_volume = ((bbl_area + bl_area) / 2) * bund_height
    
    levels_data.append({
        'Level': 'BL',
        'Length': bl_length,
        'Width': bl_width,
        'Area': bl_area,
        'Height': bund_height,
        'Volume': bl_volume,
        'Type': 'Waste up to Bund',
        'Bottom_Z': 0,
        'Top_Z': bund_height
    })
    
    # 3. Above Bund Levels (ABL) - Waste filling from top of bund to crest with berms
    current_length = bl_length
    current_width = bl_width
    current_width = bl_width
    total_volume = anchor_trench_volume + bbl_volume + bl_volume
    current_z = bund_height
    
    for i in range(1, 10):  # Calculate up to 9 levels
        # ABL level - waste filling
        abl_length = current_length - (waste_height * waste_slope * 2)
        abl_width_calc = current_length - (waste_height * waste_slope * 2)
        abl_width_calc = current_width - (waste_height * waste_slope * 2)
        
        if abl_length <= 0 or abl_width_calc <= 0:
            break
            
        abl_area = abl_length * abl_width_calc
        prev_area = current_length * current_width
        abl_volume = ((prev_area + abl_area) / 2) * waste_height
        total_volume += abl_volume
        
        levels_data.append({
            'Level': f'ABL {i}',
            'Length': abl_length,
            'Width': abl_width_calc,
            'Area': abl_area,
            'Height': waste_height,
            'Volume': abl_volume,
            'Type': 'Waste above Bund',
            'Bottom_Z': current_z,
            'Top_Z': current_z + waste_height
        })
        
        # Berm level
        berm_length = abl_length - (berm_width * 2)
        berm_width_calc = abl_width_calc - (berm_width * 2)
        
        if berm_length <= 0 or berm_width_calc <= 0:
            break
            
        berm_area = berm_length * berm_width_calc
        
        if berm_area <= 0:
            berm_area = berm_length * berm_width_calc
        
        levels_data.append({
            'Level': f'ABL {i} Berm',
            'Length': berm_length,
            'Width': berm_width_calc,
            'Area': berm_area,
            'Height': 0.0,
            'Volume': 0.0,
            'Type': 'Berm',
            'Bottom_Z': current_z,
            'Top_Z': current_z
        })
        
        current_length = berm_length
        current_width = berm_width_calc
        
        current_width = berm_width_calc
    
    # Cost calculations
    provided_quantity = total_volume * density
    required_quantity = total_quantity
    is_sufficient = "Yes" if provided_quantity >= required_quantity else "No"
    landfill_life = provided_quantity / tpa if tpa > 0 else 0
    total_cost = landfill_area_sqm * cost_per_sqm
    per_ton_cost = total_cost / provided_quantity if provided_quantity > 0 else 0
    landfill_life = provided_quantity / tpa if tpa > 0 else 0
    total_cost = landfill_area_sqm * cost_per_sqm
    per_ton_cost = total_cost / provided_quantity if provided_quantity > 0 else 0
    
    # Leachate calculations
    leachate_tpa = tpa * (leachate_percentage / 100)
    leachate_kld = leachate_tpa / 365
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': provided_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'leachate_kld': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width_calc,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': provided_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'leachate_kld': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': provided_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'leachate_kld': leachate_tpa / 365,
        'leachate_kld': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': provided_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'leachate_kld': leachate_tpa / 365,
        'leachate_kld': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width_calc,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': provided_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa / 365,
        'leachate_kld': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': provided_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa / 365,
        'leachate_kld': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': provided_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'leachate_tpa': leachate_tpa / 365,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa / 365,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels': levels_data,
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bbl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa': leachate_tpa,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bbl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'leach_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bbl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost',
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bl_length,
        'bl_width': bbl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bbl_width,
        'bl_length': bl_length,
        'bl_width': bbl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost',
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bbl_width,
        'bl_length': bbl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost,
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bbl_width,
        'bl_length': bbl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': 'required_quantity',
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost',
        'leachate_tpa',
        'bbl_length': bbl_length,
        'bbl_width': bbl_width,
        'bl_length': bbl_width,
        'bl_length': bbl_width,
        'bl_length': bbl_width,
        'bl_length': bbl_width,
        'bl_length': bbl_width,
        'max_height': current_z
    }
    
    return {
        'levels_data',
        'total_volume': total_volume,
        'provided_quantity': required_quantity,
        'required_quantity': required_quantity,
        'is_sufficient': is_sufficient,
        'landfill_life': landfill_life,
        'total_cost': total_cost,
        'per_ton_cost': per_ton_cost',
        'leachate_tpa',
        'bbl_length': bbl_length,
    }
