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
    .layer-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 10px 0;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 5px;
        padding: 5px 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏗️ Landfill Construction Simulator</h1>', unsafe_allow_html=True)
st.markdown("A comprehensive tool for landfill design, volume calculation, and interactive visualization")

# Initialize session state variables to avoid undefined variable errors
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
    st.session_state.berm_width_param = 4.0  # Renamed to avoid conflict
    st.session_state.external_slope = 2.0
    st.session_state.internal_slope = 3.0
    st.session_state.waste_slope = 3.0
    st.session_state.depth_below_ngl = 2.0
    st.session_state.cost_per_sqm = 3850.0
    st.session_state.leachate_percentage = 20
    # Additional parameters for more detailed cross-section
    st.session_state.liner_thickness = 0.5
    st.session_state.drain_thickness = 0.3
    st.session_state.cover_thickness = 0.6
    st.session_state.topsoil_thickness = 0.3
    st.session_state.soil_bund_height = 2.0
    st.session_state.soil_bund_top_width = 2.0
    st.session_state.excavation_slope = 1.5
    st.session_state.excavation_depth = 3.0  # New parameter for excavation depth
    st.session_state.bottom_width = 100.0  # New parameter for bottom width
    st.session_state.bottom_length = 100.0  # New parameter for bottom length

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
            daily_quantity = st.number_input("Daily Waste (TPD)", value=st.session_state.daily_quantity, min_value=1.0, step=10.0, key="daily")
            duration = st.number_input("Duration (years)", value=st.session_state.duration, min_value=1.0, step=1.0, key="duration")
            density = st.number_input("Density (ton/Cum)", value=st.session_state.density, min_value=0.1, step=0.01, key="density")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Land Dimensions**")
            width = st.number_input("Width (m)", value=st.session_state.width, min_value=10.0, step=10.0, key="width")
            length = st.number_input("Length (m)", value=st.session_state.length, min_value=10.0, step=10.0, key="length")
            landfill_area_acres = st.number_input("Area (Acres)", value=st.session_state.landfill_area_acres, min_value=1.0, step=0.5, key="area")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Bund Parameters**")
            bund_width = st.number_input("Bund Width (m)", value=st.session_state.bund_width, min_value=1.0, step=0.5, key="bund_width")
            bund_height = st.number_input("Bund Height (m)", value=st.session_state.bund_height, min_value=1.0, step=0.5, key="bund_height")
            waste_height = st.number_input("Waste Height (m)", value=st.session_state.waste_height, min_value=1.0, step=0.5, key="waste_height")
            berm_width_param = st.number_input("Berm Width (m)", value=st.session_state.berm_width_param, min_value=1.0, step=0.5, key="berm_width")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Slope Parameters**")
            external_slope = st.number_input("External Slope", value=st.session_state.external_slope, min_value=1.0, step=0.5, key="external")
            internal_slope = st.number_input("Internal Slope", value=st.session_state.internal_slope, min_value=1.0, step=0.5, key="internal")
            waste_slope = st.number_input("Waste Slope", value=st.session_state.waste_slope, min_value=1.0, step=0.5, key="waste")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Excavation and Soil Bund parameters
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Excavation Parameters**")
            excavation_depth = st.number_input("Excavation Depth (m)", value=st.session_state.excavation_depth, min_value=0.5, step=0.5, key="excavation_depth")
            excavation_slope = st.number_input("Excavation Slope", value=st.session_state.excavation_slope, min_value=1.0, step=0.5, key="excavation_slope")
            bottom_width = st.number_input("Bottom Width (m)", value=st.session_state.bottom_width, min_value=10.0, step=10.0, key="bottom_width")
            bottom_length = st.number_input("Bottom Length (m)", value=st.session_state.bottom_length, min_value=10.0, step=10.0, key="bottom_length")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Soil Bund & Cover Parameters**")
            soil_bund_height = st.number_input("Soil Bund Height (m)", value=st.session_state.soil_bund_height, min_value=0.5, step=0.5, key="soil_bund_height")
            soil_bund_top_width = st.number_input("Soil Bund Top Width (m)", value=st.session_state.soil_bund_top_width, min_value=1.0, step=0.5, key="soil_bund_top_width")
            liner_thickness = st.number_input("Liner Thickness (m)", value=st.session_state.liner_thickness, min_value=0.1, step=0.1, key="liner")
            drain_thickness = st.number_input("Drain Thickness (m)", value=st.session_state.drain_thickness, min_value=0.1, step=0.1, key="drain")
            cover_thickness = st.number_input("Cover Thickness (m)", value=st.session_state.cover_thickness, min_value=0.1, step=0.1, key="cover")
            topsoil_thickness = st.number_input("Topsoil Thickness (m)", value=st.session_state.topsoil_thickness, min_value=0.1, step=0.1, key="topsoil")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Cost and Leachate parameters
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Cost & Leachate**")
            cost_per_sqm = st.number_input("Cost per Sqm (INR)", value=st.session_state.cost_per_sqm, min_value=100.0, step=100.0, key="cost")
            leachate_percentage = st.slider("Leachate %", min_value=5, max_value=50, value=st.session_state.leachate_percentage, key="leachate")
            st.markdown('</div>', unsafe_allow_html=True)

# Define the capsule-shaped cross-section profile function
def create_capsule_cross_section_profile(points, cross_section_pos, is_vertical, results_data):
    """Create capsule-shaped cross-section profile with all landfill components"""
    
    # Initialize all profile arrays
    natural_ground = []
    excavation_profile = []
    liner_profile = []
    drain_profile = []
    waste_profile = []
    soil_bund_profile = []
    cover_profile = []
    topsoil_profile = []
    
    # Calculate dimensions for the capsule-shaped landfill
    # Bottom dimensions
    bottom_half_width = bottom_width / 2
    bottom_half_length = bottom_length / 2
    
    # Top dimensions at ground level (inner edge of soil bund)
    top_ground_width = bottom_width + 2 * (excavation_depth * excavation_slope)
    top_ground_length = bottom_length + 2 * (excavation_depth * excavation_slope)
    top_ground_half_width = top_ground_width / 2
    top_ground_half_length = top_ground_length / 2
    
    # Top dimensions at top of soil bund
    top_bund_width = top_ground_width - 2 * (soil_bund_height * 2.0)  # Inward slope of 1:2
    top_bund_length = top_ground_length - 2 * (soil_bund_height * 2.0)
    top_bund_half_width = top_bund_width / 2
    top_bund_half_length = top_bund_length / 2
    
    # Calculate above-ground landfill dimensions
    ag_top_width = top_bund_width + 2 * (bund_height * external_slope)
    ag_top_length = top_bund_length + 2 * (bund_height * external_slope)
    ag_top_half_width = ag_top_width / 2
    ag_top_half_length = ag_top_length / 2
    
    for point in points:
        if is_vertical:
            # Vertical cross-section (along Y-axis)
            x_pos = cross_section_pos
            y_pos = point
            
            # Natural ground level (always 0)
            natural_ground.append(0)
            
            # Calculate excavation profile (tub shape)
            if abs(x_pos) <= top_ground_half_width:
                # Calculate width at this x position based on depth
                if abs(x_pos) <= bottom_half_width:
                    # At the bottom - full depth
                    excavation_depth_at_x = excavation_depth
                else:
                    # Sloping sides - calculate depth based on distance from bottom edge
                    dist_from_bottom_edge = abs(x_pos) - bottom_half_width
                    excavation_depth_at_x = max(0, excavation_depth - (dist_from_bottom_edge / excavation_slope))
                
                excavation_profile.append(-excavation_depth_at_x)
                
                # Calculate liner (at bottom of excavation)
                if excavation_depth_at_x > 0:
                    liner_profile.append(-excavation_depth_at_x)
                    drain_profile.append(-excavation_depth_at_x + drain_thickness)
                else:
                    liner_profile.append(0)
                    drain_profile.append(0)
            else:
                excavation_profile.append(0)
                liner_profile.append(0)
                drain_profile.append(0)
            
            # Calculate waste height in the tub
            if abs(x_pos) <= top_ground_half_width:
                # Calculate width at this x position based on height
                if abs(x_pos) <= bottom_half_width:
                    # At the bottom - full height to top of soil bund
                    waste_height_at_x = soil_bund_height
                else:
                    # Sloping sides - calculate height based on distance from bottom edge
                    dist_from_bottom_edge = abs(x_pos) - bottom_half_width
                    max_dist = top_ground_half_width - bottom_half_width
                    waste_height_at_x = soil_bund_height * (1 - (dist_from_bottom_edge / max_dist))
                
                waste_profile.append(waste_height_at_x)
                
                # Calculate cover and topsoil (on top of waste)
                if waste_height_at_x > 0:
                    cover_profile.append(waste_height_at_x + cover_thickness)
                    topsoil_profile.append(waste_height_at_x + cover_thickness + topsoil_thickness)
                else:
                    cover_profile.append(0)
                    topsoil_profile.append(0)
            else:
                waste_profile.append(0)
                cover_profile.append(0)
                topsoil_profile.append(0)
            
            # Calculate soil bund height (inward sloping)
            if abs(x_pos) <= top_ground_half_width:
                if abs(x_pos) > top_bund_half_width:
                    # In the soil bund area
                    dist_from_inner_edge = abs(x_pos) - top_bund_half_width
                    soil_bund_h_at_x = soil_bund_height * (1 - (dist_from_inner_edge / (top_ground_half_width - top_bund_half_width)))
                    soil_bund_profile.append(soil_bund_h_at_x)
                else:
                    soil_bund_profile.append(0)
            else:
                soil_bund_profile.append(0)
                
        else:
            # Horizontal cross-section (along X-axis)
            x_pos = point
            y_pos = cross_section_pos
            
            # Natural ground level (always 0)
            natural_ground.append(0)
            
            # Calculate excavation profile (tub shape)
            if abs(y_pos) <= top_ground_half_length:
                # Calculate width at this y position based on depth
                if abs(y_pos) <= bottom_half_length:
                    # At the bottom - full depth
                    excavation_depth_at_y = excavation_depth
                else:
                    # Sloping sides - calculate depth based on distance from bottom edge
                    dist_from_bottom_edge = abs(y_pos) - bottom_half_length
                    excavation_depth_at_y = max(0, excavation_depth - (dist_from_bottom_edge / excavation_slope))
                
                excavation_profile.append(-excavation_depth_at_y)
                
                # Calculate liner (at bottom of excavation)
                if excavation_depth_at_y > 0:
                    liner_profile.append(-excavation_depth_at_y)
                    drain_profile.append(-excavation_depth_at_y + drain_thickness)
                else:
                    liner_profile.append(0)
                    drain_profile.append(0)
            else:
                excavation_profile.append(0)
                liner_profile.append(0)
                drain_profile.append(0)
            
            # Calculate waste height in the tub
            if abs(y_pos) <= top_ground_half_length:
                # Calculate width at this y position based on height
                if abs(y_pos) <= bottom_half_length:
                    # At the bottom - full height to top of soil bund
                    waste_height_at_y = soil_bund_height
                else:
                    # Sloping sides - calculate height based on distance from bottom edge
                    dist_from_bottom_edge = abs(y_pos) - bottom_half_length
                    max_dist = top_ground_half_length - bottom_half_length
                    waste_height_at_y = soil_bund_height * (1 - (dist_from_bottom_edge / max_dist))
                
                waste_profile.append(waste_height_at_y)
                
                # Calculate cover and topsoil (on top of waste)
                if waste_height_at_y > 0:
                    cover_profile.append(waste_height_at_y + cover_thickness)
                    topsoil_profile.append(waste_height_at_y + cover_thickness + topsoil_thickness)
                else:
                    cover_profile.append(0)
                    topsoil_profile.append(0)
            else:
                waste_profile.append(0)
                cover_profile.append(0)
                topsoil_profile.append(0)
            
            # Calculate soil bund height (inward sloping)
            if abs(y_pos) <= top_ground_half_length:
                if abs(y_pos) > top_bund_half_length:
                    # In the soil bund area
                    dist_from_inner_edge = abs(y_pos) - top_bund_half_length
                    soil_bund_h_at_y = soil_bund_height * (1 - (dist_from_inner_edge / (top_ground_half_length - top_bund_half_length)))
                    soil_bund_profile.append(soil_bund_h_at_y)
                else:
                    soil_bund_profile.append(0)
            else:
                soil_bund_profile.append(0)
    
    return {
        'natural_ground': natural_ground,
        'excavation': excavation_profile,
        'liner': liner_profile,
        'drain': drain_profile,
        'waste': waste_profile,
        'soil_bund': soil_bund_profile,
        'cover': cover_profile,
        'topsoil': topsoil_profile
    }

# Define the above-ground landfill profile function
def create_above_ground_profile(points, cross_section_pos, is_vertical, results_data):
    """Create above-ground landfill profile with berms"""
    
    # Initialize all profile arrays
    ag_waste_profile = []
    ag_bund_profile = []
    ag_berm_profile = []
    ag_cover_profile = []
    ag_topsoil_profile = []
    
    # Calculate dimensions for the above-ground landfill
    # Base dimensions at top of soil bund
    base_width = bottom_width + 2 * (excavation_depth * excavation_slope) - 2 * (soil_bund_height * 2.0)
    base_length = bottom_length + 2 * (excavation_depth * excavation_slope) - 2 * (soil_bund_height * 2.0)
    base_half_width = base_width / 2
    base_half_length = base_length / 2
    
    # Calculate levels and dimensions for above-ground landfill
    current_width = base_width
    current_length = base_length
    current_half_width = base_half_width
    current_half_length = base_half_length
    current_z = soil_bund_height
    
    # Store level information for reference
    ag_levels = []
    
    for i in range(1, 10):  # Calculate up to 9 levels
        # AGL level - waste filling
        new_width = current_width - (waste_height * waste_slope * 2)
        new_length = current_length - (waste_height * waste_slope * 2)
        new_half_width = new_width / 2
        new_half_length = new_length / 2
        
        if new_width <= 0 or new_length <= 0:
            break
            
        ag_levels.append({
            'level': i,
            'bottom_z': current_z,
            'top_z': current_z + waste_height,
            'bottom_half_width': current_half_width,
            'top_half_width': new_half_width,
            'bottom_half_length': current_half_length,
            'top_half_length': new_half_length
        })
        
        current_z += waste_height
        current_width = new_width
        current_length = new_length
        current_half_width = new_half_width
        current_half_length = new_half_length
        
        # Berm level
        berm_width_calc = current_width - (berm_width_param * 2)  # Use the parameter variable
        berm_length = current_length - (berm_width_param * 2)
        berm_half_width = berm_width_calc / 2
        berm_half_length = berm_length / 2
        
        if berm_width_calc <= 0 or berm_length <= 0:
            break
            
        ag_levels.append({
            'level': i,
            'type': 'berm',
            'z': current_z,
            'half_width': berm_half_width,
            'half_length': berm_half_length
        })
        
        current_width = berm_width_calc
        current_length = berm_length
        current_half_width = berm_half_width
        current_half_length = berm_half_length
    
    for point in points:
        if is_vertical:
            # Vertical cross-section (along Y-axis)
            x_pos = cross_section_pos
            y_pos = point
            
            # Initialize heights
            ag_waste_height = 0
            ag_bund_height = 0
            ag_berm_height = 0
            
            # Check if point is within the above-ground landfill area
            if abs(x_pos) <= base_half_width:
                # Calculate waste height at this position
                for level in ag_levels:
                    if 'type' not in level or level['type'] != 'berm':
                        # Waste level
                        if abs(x_pos) <= level['top_half_width']:
                            ag_waste_height = level['top_z']
                        elif abs(x_pos) <= level['bottom_half_width']:
                            # On the slope - interpolate height
                            dist_from_bottom = abs(x_pos) - level['top_half_width']
                            total_slope_width = level['bottom_half_width'] - level['top_half_width']
                            if total_slope_width > 0:
                                slope_ratio = 1 - (dist_from_bottom / total_slope_width)
                                ag_waste_height = level['bottom_z'] + (level['top_z'] - level['bottom_z']) * slope_ratio
                    else:
                        # Berm level
                        if abs(x_pos) <= level['half_width']:
                            ag_berm_height = level['z']
                
                # Calculate bund height (outside waste area)
                if abs(x_pos) > base_half_width and abs(x_pos) <= (base_half_width + bund_height * external_slope):
                    dist_from_edge = abs(x_pos) - base_half_width
                    ag_bund_height = soil_bund_height + bund_height * (1 - (dist_from_edge / (bund_height * external_slope)))
                
                # Set profiles
                ag_waste_profile.append(ag_waste_height)
                ag_bund_profile.append(ag_bund_height)
                ag_berm_profile.append(ag_berm_height)
                
                # Calculate cover and topsoil
                if ag_waste_height > 0:
                    ag_cover_profile.append(ag_waste_height + cover_thickness)
                    ag_topsoil_profile.append(ag_waste_height + cover_thickness + topsoil_thickness)
                elif ag_berm_height > 0:
                    ag_cover_profile.append(ag_berm_height + cover_thickness)
                    ag_topsoil_profile.append(ag_berm_height + cover_thickness + topsoil_thickness)
                else:
                    ag_cover_profile.append(0)
                    ag_topsoil_profile.append(0)
            else:
                ag_waste_profile.append(0)
                ag_bund_profile.append(0)
                ag_berm_profile.append(0)
                ag_cover_profile.append(0)
                ag_topsoil_profile.append(0)
        else:
            # Horizontal cross-section (along X-axis)
            x_pos = point
            y_pos = cross_section_pos
            
            # Initialize heights
            ag_waste_height = 0
            ag_bund_height = 0
            ag_berm_height = 0
            
            # Check if point is within the above-ground landfill area
            if abs(y_pos) <= base_half_length:
                # Calculate waste height at this position
                for level in ag_levels:
                    if 'type' not in level or level['type'] != 'berm':
                        # Waste level
                        if abs(y_pos) <= level['top_half_length']:
                            ag_waste_height = level['top_z']
                        elif abs(y_pos) <= level['bottom_half_length']:
                            # On the slope - interpolate height
                            dist_from_bottom = abs(y_pos) - level['top_half_length']
                            total_slope_width = level['bottom_half_length'] - level['top_half_length']
                            if total_slope_width > 0:
                                slope_ratio = 1 - (dist_from_bottom / total_slope_width)
                                ag_waste_height = level['bottom_z'] + (level['top_z'] - level['bottom_z']) * slope_ratio
                    else:
                        # Berm level
                        if abs(y_pos) <= level['half_length']:
                            ag_berm_height = level['z']
                
                # Calculate bund height (outside waste area)
                if abs(y_pos) > base_half_length and abs(y_pos) <= (base_half_length + bund_height * external_slope):
                    dist_from_edge = abs(y_pos) - base_half_length
                    ag_bund_height = soil_bund_height + bund_height * (1 - (dist_from_edge / (bund_height * external_slope)))
                
                # Set profiles
                ag_waste_profile.append(ag_waste_height)
                ag_bund_profile.append(ag_bund_height)
                ag_berm_profile.append(ag_berm_height)
                
                # Calculate cover and topsoil
                if ag_waste_height > 0:
                    ag_cover_profile.append(ag_waste_height + cover_thickness)
                    ag_topsoil_profile.append(ag_waste_height + cover_thickness + topsoil_thickness)
                elif ag_berm_height > 0:
                    ag_cover_profile.append(ag_berm_height + cover_thickness)
                    ag_topsoil_profile.append(ag_berm_height + cover_thickness + topsoil_thickness)
                else:
                    ag_cover_profile.append(0)
                    ag_topsoil_profile.append(0)
            else:
                ag_waste_profile.append(0)
                ag_bund_profile.append(0)
                ag_berm_profile.append(0)
                ag_cover_profile.append(0)
                ag_topsoil_profile.append(0)
    
    return {
        'waste': ag_waste_profile,
        'bund': ag_bund_profile,
        'berm': ag_berm_profile,
        'cover': ag_cover_profile,
        'topsoil': ag_topsoil_profile
    }

# Calculate all values
def calculate_landfill_parameters():
    # Basic calculations
    tpa = daily_quantity * 365
    total_quantity = tpa * duration
    landfill_area_sqm = landfill_area_acres * 4047
    
    # Calculate dimensions for the capsule-shaped landfill
    # Bottom dimensions
    bottom_area = bottom_width * bottom_length
    
    # Top dimensions at ground level (inner edge of soil bund)
    top_ground_width = bottom_width + 2 * (excavation_depth * excavation_slope)
    top_ground_length = bottom_length + 2 * (excavation_depth * excavation_slope)
    top_ground_area = top_ground_width * top_ground_length
    
    # Top dimensions at top of soil bund
    top_bund_width = top_ground_width - 2 * (soil_bund_height * 2.0)  # Inward slope of 1:2
    top_bund_length = top_ground_length - 2 * (soil_bund_height * 2.0)
    top_bund_area = top_bund_width * top_bund_length
    
    # Calculate levels and volumes
    levels_data = []
    
    # 1. Below Ground Level (BBL) - Excavation below ground in tub shape
    bbl_volume = (bottom_area + top_ground_area) / 2 * excavation_depth
    
    levels_data.append({
        'Level': 'BBL',
        'Length': bottom_length,
        'Width': bottom_width,
        'Area': bottom_area,
        'Height': excavation_depth,
        'Volume': bbl_volume,
        'Type': 'Excavation',
        'Bottom_Z': -excavation_depth,
        'Top_Z': 0
    })
    
    # 2. Waste in Tub (WT) - Waste filling in tub shape up to top of soil bund
    wt_volume = (bottom_area + top_bund_area) / 2 * soil_bund_height
    
    levels_data.append({
        'Level': 'WT',
        'Length': bottom_length,
        'Width': bottom_width,
        'Area': bottom_area,
        'Height': soil_bund_height,
        'Volume': wt_volume,
        'Type': 'Waste in Tub',
        'Bottom_Z': 0,
        'Top_Z': soil_bund_height
    })
    
    # 3. Above Ground Levels (AGL) - Waste filling from top of soil bund to crest with berms
    current_width = top_bund_width
    current_length = top_bund_length
    total_volume = bbl_volume + wt_volume
    current_z = soil_bund_height
    
    for i in range(1, 10):  # Calculate up to 9 levels
        # AGL level - waste filling
        new_width = current_width - (waste_height * waste_slope * 2)
        new_length = current_length - (waste_height * waste_slope * 2)
        
        if new_width <= 0 or new_length <= 0:
            break
            
        new_area = new_width * new_length
        prev_area = current_width * current_length
        agl_volume = ((prev_area + new_area) / 2) * waste_height
        total_volume += agl_volume
        
        levels_data.append({
            'Level': f'AGL {i}',
            'Length': new_length,
            'Width': new_width,
            'Area': new_area,
            'Height': waste_height,
            'Volume': agl_volume,
            'Type': 'Above Ground Waste',
            'Bottom_Z': current_z,
            'Top_Z': current_z + waste_height
        })
        
        current_z += waste_height
        
        # Berm level
        berm_width_calc = new_width - (berm_width_param * 2)  # Use the parameter variable
        berm_length = new_length - (berm_width_param * 2)
        
        if berm_width_calc <= 0 or berm_length <= 0:
            break
            
        berm_area = berm_width_calc * berm_length
        
        levels_data.append({
            'Level': f'AGL {i} Berm',
            'Length': berm_length,
            'Width': berm_width_calc,
            'Area': berm_area,
            'Height': 0.0,
            'Volume': 0.0,
            'Type': 'Berm',
            'Bottom_Z': current_z,
            'Top_Z': current_z
        })
        
        current_width = berm_width_calc
        current_length = berm_length
    
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
        'leachate_kld': leachate_kld,
        'tpa': tpa,
        'bottom_width': bottom_width,
        'bottom_length': bottom_length,
        'top_ground_width': top_ground_width,
        'top_ground_length': top_ground_length,
        'top_bund_width': top_bund_width,
        'top_bund_length': top_bund_length,
        'max_height': current_z
    }

# Calculate parameters - this will be executed when the script runs
results = calculate_landfill_parameters()

with tab2:
    st.header("Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Volume", f"{results['total_volume']:,.2f} Cum")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Provided Quantity", f"{results['provided_quantity']:,.2f} Tons")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Required Quantity", f"{results['required_quantity']:,.2f} Tons")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Capacity Sufficient", results['is_sufficient'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Landfill Life", f"{results['landfill_life']:.2f} Years")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Cost", f"₹{results['total_cost']:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Cost per Ton", f"₹{results['per_ton_cost']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Leachate Generation", f"{results['leachate_kld']:.2f} KLD")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Status indicators
    st.markdown('<h3 class="section-header">Project Status</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        if results['is_sufficient'] == "Yes":
            st.success("✅ The landfill design meets the required capacity!")
        else:
            st.error("❌ The landfill design does not meet the required capacity. Consider increasing area or dimensions.")
    
    with col2:
        if results['landfill_life'] >= duration:
            st.success(f"✅ Landfill life ({results['landfill_life']:.1f} years) exceeds project duration ({duration:.0f} years)")
        else:
            st.warning(f"⚠️ Landfill life ({results['landfill_life']:.1f} years) is less than project duration ({duration:.0f} years)")

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
        st.subheader("3D Capsule-Shaped Landfill Model")
        
        # Create 3D surface plot with capsule-shaped modeling
        fig_3d = go.Figure()
        
        # Create height map for 3D surface
        x = np.linspace(-width/2, width/2, 100)
        y = np.linspace(-length/2, length/2, 100)
        X, Y = np.meshgrid(x, y)
        
        # Create multiple height maps for different components
        Z_ground = np.zeros_like(X)  # Natural ground
        Z_excavation = np.zeros_like(X)  # Excavation
        Z_waste = np.zeros_like(X)  # Waste
        Z_soil_bund = np.zeros_like(X)  # Soil bund
        Z_ag_waste = np.zeros_like(X)  # Above-ground waste
        Z_ag_bund = np.zeros_like(X)  # Above-ground bund
        
        # Calculate dimensions for the capsule-shaped landfill
        bottom_half_width = bottom_width / 2
        bottom_half_length = bottom_length / 2
        
        # Top dimensions at ground level (inner edge of soil bund)
        top_ground_width = bottom_width + 2 * (excavation_depth * excavation_slope)
        top_ground_length = bottom_length + 2 * (excavation_depth * excavation_slope)
        top_ground_half_width = top_ground_width / 2
        top_ground_half_length = top_ground_length / 2
        
        # Top dimensions at top of soil bund
        top_bund_width = top_ground_width - 2 * (soil_bund_height * 2.0)  # Inward slope of 1:2
        top_bund_length = top_ground_length - 2 * (soil_bund_height * 2.0)
        top_bund_half_width = top_bund_width / 2
        top_bund_half_length = top_bund_length / 2
        
        # Model the complete capsule-shaped landfill structure
        for i in range(len(X)):
            for j in range(len(X[0])):
                x_pos, y_pos = X[i, j], Y[i, j]
                
                # Calculate excavation (tub shape)
                if abs(x_pos) <= top_ground_half_width and abs(y_pos) <= top_ground_half_length:
                    # Calculate width at this position based on depth
                    if abs(x_pos) <= bottom_half_width and abs(y_pos) <= bottom_half_length:
                        # At the bottom - full depth
                        excavation_depth_at_pos = excavation_depth
                    else:
                        # Sloping sides - calculate depth based on distance from bottom edge
                        dist_from_bottom_edge_x = max(0, abs(x_pos) - bottom_half_width)
                        dist_from_bottom_edge_y = max(0, abs(y_pos) - bottom_half_length)
                        
                        # Use the maximum distance from the bottom edge
                        max_dist = max(dist_from_bottom_edge_x, dist_from_bottom_edge_y)
                        excavation_depth_at_pos = max(0, excavation_depth - (max_dist / excavation_slope))
                    
                    Z_excavation[i, j] = -excavation_depth_at_pos
                
                # Calculate waste in tub
                if abs(x_pos) <= top_ground_half_width and abs(y_pos) <= top_ground_half_length:
                    # Calculate width at this position based on height
                    if abs(x_pos) <= bottom_half_width and abs(y_pos) <= bottom_half_length:
                        # At the bottom - full height to top of soil bund
                        waste_height_at_pos = soil_bund_height
                    else:
                        # Sloping sides - calculate height based on distance from bottom edge
                        dist_from_bottom_edge_x = max(0, abs(x_pos) - bottom_half_width)
                        dist_from_bottom_edge_y = max(0, abs(y_pos) - bottom_half_length)
                        
                        # Use the maximum distance from the bottom edge
                        max_dist_x = top_ground_half_width - bottom_half_width
                        max_dist_y = top_ground_half_length - bottom_half_length
                        max_dist = max(dist_from_bottom_edge_x, dist_from_bottom_edge_y)
                        max_total_dist = max(max_dist_x, max_dist_y)
                        
                        if max_total_dist > 0:
                            waste_height_at_pos = soil_bund_height * (1 - (max_dist / max_total_dist))
                        else:
                            waste_height_at_pos = soil_bund_height
                    
                    Z_waste[i, j] = waste_height_at_pos
                
                # Calculate soil bund (inward sloping)
                if abs(x_pos) <= top_ground_half_width and abs(y_pos) <= top_ground_half_length:
                    if abs(x_pos) > top_bund_half_width or abs(y_pos) > top_bund_half_width:
                        # In the soil bund area
                        dist_from_inner_edge_x = max(0, abs(x_pos) - top_bund_half_width)
                        dist_from_inner_edge_y = max(0, abs(y_pos) - top_bund_half_length)
                        
                        # Use the maximum distance from the inner edge
                        max_dist = max(dist_from_inner_edge_x, dist_from_inner_edge_y)
                        max_total_dist_x = top_ground_half_width - top_bund_half_width
                        max_total_dist_y = top_ground_half_length - top_bund_half_length
                        max_total_dist = max(max_total_dist_x, max_total_dist_y)
                        
                        if max_total_dist > 0:
                            soil_bund_h_at_pos = soil_bund_height * (1 - (max_dist / max_total_dist))
                        else:
                            soil_bund_h_at_pos = 0
                        
                        Z_soil_bund[i, j] = soil_bund_h_at_pos
                
                # Calculate above-ground waste
                if abs(x_pos) <= top_bund_half_width and abs(y_pos) <= top_bund_half_length:
                    # Calculate current dimensions at this height
                    current_width = top_bund_width
                    current_length = top_bund_length
                    current_z = soil_bund_height
                    
                    # Calculate waste height at this position
                    for level in range(1, 10):  # Calculate up to 9 levels
                        new_width = current_width - (waste_height * waste_slope * 2)
                        new_length = current_length - (waste_height * waste_slope * 2)
                        
                        if new_width <= 0 or new_length <= 0:
                            break
                        
                        new_half_width = new_width / 2
                        new_half_length = new_length / 2
                        
                        if abs(x_pos) <= new_half_width and abs(y_pos) <= new_half_length:
                            Z_ag_waste[i, j] = current_z + waste_height
                        elif abs(x_pos) <= current_width/2 and abs(y_pos) <= current_length/2:
                            # On the slope - interpolate height
                            dist_from_inner_x = max(0, abs(x_pos) - new_half_width)
                            dist_from_inner_y = max(0, abs(y_pos) - new_half_length)
                            
                            # Use the maximum distance from the inner edge
                            max_dist = max(dist_from_inner_x, dist_from_inner_y)
                            max_total_dist_x = current_width/2 - new_half_width
                            max_total_dist_y = current_length/2 - new_half_length
                            max_total_dist = max(max_total_dist_x, max_total_dist_y)
                            
                            if max_total_dist > 0:
                                slope_ratio = 1 - (max_dist / max_total_dist)
                                Z_ag_waste[i, j] = current_z + waste_height * slope_ratio
                        
                        current_z += waste_height
                        current_width = new_width
                        current_length = new_length
                        
                        # Berm level
                        berm_width_calc = current_width - (berm_width_param * 2)  # Use the parameter variable
                        berm_length = current_length - (berm_width_param * 2)
                        
                        if berm_width_calc <= 0 or berm_length <= 0:
                            break
                        
                        current_width = berm_width_calc
                        current_length = berm_length
                
                # Calculate above-ground bund
                if (abs(x_pos) > top_bund_half_width or abs(y_pos) > top_bund_half_width) and \
                   (abs(x_pos) <= top_bund_half_width + bund_height * external_slope and \
                    abs(y_pos) <= top_bund_half_length + bund_height * external_slope):
                    # Outside waste area but within bund area
                    dist_from_edge_x = max(0, abs(x_pos) - top_bund_half_width)
                    dist_from_edge_y = max(0, abs(y_pos) - top_bund_half_length)
                    
                    # Use the maximum distance from the edge
                    max_dist = max(dist_from_edge_x, dist_from_edge_y)
                    Z_ag_bund[i, j] = soil_bund_height + bund_height * (1 - (max_dist / (bund_height * external_slope)))
        
        # Add natural ground
        fig_3d.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_ground,
                colorscale=[[0, 'green'], [1, 'green']],
                showscale=False,
                opacity=0.3,
                hoverinfo='skip',
                name='Natural Ground'
            )
        )
        
        # Add excavation
        fig_3d.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_excavation,
                colorscale=[[0, '#8B4513'], [1, '#A0522D']],
                showscale=False,
                opacity=0.8,
                hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Z: %{z:.1f}m<extra></extra>',
                name='Excavation'
            )
        )
        
        # Add waste in tub
        fig_3d.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_waste,
                colorscale=[[0, '#654321'], [1, '#8B4513']],
                showscale=False,
                opacity=0.8,
                hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Z: %{z:.1f}m<extra></extra>',
                name='Waste in Tub'
            )
        )
        
        # Add soil bund
        fig_3d.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_soil_bund,
                colorscale=[[0, '#D2691E'], [1, '#DEB887']],
                showscale=False,
                opacity=0.8,
                hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Z: %{z:.1f}m<extra></extra>',
                name='Soil Bund'
            )
        )
        
        # Add above-ground waste
        fig_3d.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_ag_waste,
                colorscale=[[0, '#654321'], [1, '#8B4513']],
                showscale=False,
                opacity=0.8,
                hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Z: %{z:.1f}m<extra></extra>',
                name='Above-Ground Waste'
            )
        )
        
        # Add above-ground bund
        fig_3d.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_ag_bund,
                colorscale=[[0, '#D2691E'], [1, '#DEB887']],
                showscale=False,
                opacity=0.8,
                hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Z: %{z:.1f}m<extra></extra>',
                name='Above-Ground Bund'
            )
        )
        
        # Add cross-section line
        if cross_section_type == "Vertical (Y-axis)":
            # Vertical line at X position
            line_y = np.linspace(-length/2, length/2, 20)
            line_x = [cross_section_x] * len(line_y)
            line_z = np.linspace(-excavation_depth, results['max_height'], 20)
            
            fig_3d.add_trace(
                go.Scatter3d(
                    x=line_x, y=line_y, z=line_z,
                    mode='lines',
                    line=dict(color='red', width=5),
                    name='Cross-Section',
                    hoverinfo='skip'
                )
            )
        else:
            # Horizontal line at Y position
            line_x = np.linspace(-width/2, width/2, 20)
            line_y = [cross_section_y] * len(line_x)
            line_z = np.linspace(-excavation_depth, results['max_height'], 20)
            
            fig_3d.add_trace(
                go.Scatter3d(
                    x=line_x, y=line_y, z=line_z,
                    mode='lines',
                    line=dict(color='red', width=5),
                    name='Cross-Section',
                    hoverinfo='skip'
                )
            )
        
        # Update layout
        fig_3d.update_layout(
            title="3D Capsule-Shaped Landfill Model",
            scene=dict(
                xaxis_title="Length (m)",
                yaxis_title="Width (m)",
                zaxis_title="Height (m)",
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.3)  # Keep height scale realistic
            ),
            height=600,
            showlegend=True
        )
        
        # Display 3D plot
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Capsule-Shaped Cross-Section View")
        
        # Create 2D cross-section with capsule-shaped modeling
        fig_2d = go.Figure()
        
        # Only create cross-section if we have the required variables
        try:
            # Ensure cross_section_x and cross_section_y are single values
            cs_x = float(cross_section_x) if isinstance(cross_section_x, (int, float, np.number)) else 0.0
            cs_y = float(cross_section_y) if isinstance(cross_section_y, (int, float, np.number)) else 0.0
            
            # Ensure is_vertical is a boolean
            is_vert = cross_section_type == "Vertical (Y-axis)"
            
            if is_vert:
                # Vertical cross-section (along Y-axis)
                st.write("📐 Vertical Cross-Section (Along Y-axis)")
                
                # Generate detailed cross-section profile
                y_points = np.linspace(-length/2, length/2, 200)
                
                # Create capsule-shaped profiles for below-ground components
                bg_profiles = create_capsule_cross_section_profile(
                    points=y_points,
                    cross_section_pos=cs_x,
                    is_vertical=is_vert,
                    results_data=results
                )
                
                # Create above-ground profiles
                ag_profiles = create_above_ground_profile(
                    points=y_points,
                    cross_section_pos=cs_x,
                    is_vertical=is_vert,
                    results_data=results
                )
                
                # Add natural ground level
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=bg_profiles['natural_ground'],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name='Natural Ground',
                    hoverinfo='skip'
                ))
                
                # Add excavation profile
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=bg_profiles['excavation'],
                    mode='lines',
                    line=dict(color='#8B4513', width=2),
                    name='Excavation',
                    fill='tonexty',
                    fillcolor='rgba(139, 69, 19, 0.3)',
                    hovertemplate='Y: %{x:.1f}m<br>Excavation Depth: %{y:.1f}m<extra></extra>'
                ))
                
                # Add liner (bottom layer)
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=bg_profiles['liner'],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Liner',
                    hovertemplate='Y: %{x:.1f}m<br>Liner Depth: %{y:.1f}m<extra></extra>'
                ))
                
                # Add drain layer
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=bg_profiles['drain'],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Drain Layer',
                    hovertemplate='Y: %{x:.1f}m<br>Drain Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add waste in tub
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=bg_profiles['waste'],
                    mode='lines',
                    line=dict(color='#654321', width=3),
                    name='Waste in Tub',
                    fill='tonexty',
                    fillcolor='rgba(101, 67, 33, 0.5)',
                    hovertemplate='Y: %{x:.1f}m<br>Waste Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add soil bund
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=bg_profiles['soil_bund'],
                    mode='lines',
                    line=dict(color='#D2691E', width=3),
                    name='Soil Bund',
                    hovertemplate='Y: %{x:.1f}m<br>Soil Bund Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add above-ground waste
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=ag_profiles['waste'],
                    mode='lines',
                    line=dict(color='#8B4513', width=3),
                    name='Above-Ground Waste',
                    fill='tonexty',
                    fillcolor='rgba(139, 69, 19, 0.5)',
                    hovertemplate='Y: %{x:.1f}m<br>AG Waste Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add above-ground bund
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=ag_profiles['bund'],
                    mode='lines',
                    line=dict(color='#DEB887', width=3),
                    name='Above-Ground Bund',
                    hovertemplate='Y: %{x:.1f}m<br>AG Bund Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add berms
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=ag_profiles['berm'],
                    mode='lines',
                    line=dict(color='#CD853F', width=2),
                    name='Berms',
                    hovertemplate='Y: %{x:.1f}m<br>Berm Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add cover layer
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=ag_profiles['cover'],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    name='Cover Layer',
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.3)',
                    hovertemplate='Y: %{x:.1f}m<br>Cover Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add topsoil layer
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=ag_profiles['topsoil'],
                    mode='lines',
                    line=dict(color='brown', width=2),
                    name='Topsoil',
                    fill='tonexty',
                    fillcolor='rgba(139, 69, 19, 0.3)',
                    hovertemplate='Y: %{x:.1f}m<br>Topsoil Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add annotations for different components
                fig_2d.add_annotation(
                    x=0, y=-excavation_depth/2,
                    text="Excavation Pit",
                    showarrow=False,
                    font=dict(color="white", size=10),
                    bgcolor="rgba(139, 69, 19, 0.7)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4
                )
                
                fig_2d.add_annotation(
                    x=0, y=soil_bund_height/2,
                    text="Waste in Tub",
                    showarrow=False,
                    font=dict(color="white", size=10),
                    bgcolor="rgba(101, 67, 33, 0.7)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4
                )
                
                fig_2d.update_layout(
                    title="Capsule-Shaped Vertical Cross-Section",
                    xaxis_title="Distance along Y-axis (m)",
                    yaxis_title="Height (m)",
                    height=600,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
            else:
                # Horizontal cross-section (along X-axis)
                st.write("📐 Horizontal Cross-Section (Along X-axis)")
                
                # Generate detailed cross-section profile
                x_points = np.linspace(-width/2, width/2, 200)
                
                # Create capsule-shaped profiles for below-ground components
                bg_profiles = create_capsule_cross_section_profile(
                    points=x_points,
                    cross_section_pos=cs_y,
                    is_vertical=is_vert,
                    results_data=results
                )
                
                # Create above-ground profiles
                ag_profiles = create_above_ground_profile(
                    points=x_points,
                    cross_section_pos=cs_y,
                    is_vertical=is_vert,
                    results_data=results
                )
                
                # Add natural ground level
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=bg_profiles['natural_ground'],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name='Natural Ground',
                    hoverinfo='skip'
                ))
                
                # Add excavation profile
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=bg_profiles['excavation'],
                    mode='lines',
                    line=dict(color='#8B4513', width=2),
                    name='Excavation',
                    fill='tonexty',
                    fillcolor='rgba(139, 69, 19, 0.3)',
                    hovertemplate='X: %{x:.1f}m<br>Excavation Depth: %{y:.1f}m<extra></extra>'
                ))
                
                # Add liner (bottom layer)
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=bg_profiles['liner'],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name='Liner',
                    hovertemplate='X: %{x:.1f}m<br>Liner Depth: %{y:.1f}m<extra></extra>'
                ))
                
                # Add drain layer
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=bg_profiles['drain'],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Drain Layer',
                    hovertemplate='X: %{x:.1f}m<br>Drain Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add waste in tub
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=bg_profiles['waste'],
                    mode='lines',
                    line=dict(color='#654321', width=3),
                    name='Waste in Tub',
                    fill='tonexty',
                    fillcolor='rgba(101, 67, 33, 0.5)',
                    hovertemplate='X: %{x:.1f}m<br>Waste Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add soil bund
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=bg_profiles['soil_bund'],
                    mode='lines',
                    line=dict(color='#D2691E', width=3),
                    name='Soil Bund',
                    hovertemplate='X: %{x:.1f}m<br>Soil Bund Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add above-ground waste
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=ag_profiles['waste'],
                    mode='lines',
                    line=dict(color='#8B4513', width=3),
                    name='Above-Ground Waste',
                    fill='tonexty',
                    fillcolor='rgba(139, 69, 19, 0.5)',
                    hovertemplate='X: %{x:.1f}m<br>AG Waste Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add above-ground bund
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=ag_profiles['bund'],
                    mode='lines',
                    line=dict(color='#DEB887', width=3),
                    name='Above-Ground Bund',
                    hovertemplate='X: %{x:.1f}m<br>AG Bund Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add berms
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=ag_profiles['berm'],
                    mode='lines',
                    line=dict(color='#CD853F', width=2),
                    name='Berms',
                    hovertemplate='X: %{x:.1f}m<br>Berm Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add cover layer
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=ag_profiles['cover'],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    name='Cover Layer',
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.3)',
                    hovertemplate='X: %{x:.1f}m<br>Cover Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add topsoil layer
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=ag_profiles['topsoil'],
                    mode='lines',
                    line=dict(color='brown', width=2),
                    name='Topsoil',
                    fill='tonexty',
                    fillcolor='rgba(139, 69, 19, 0.3)',
                    hovertemplate='X: %{x:.1f}m<br>Topsoil Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add annotations for different components
                fig_2d.add_annotation(
                    x=0, y=-excavation_depth/2,
                    text="Excavation Pit",
                    showarrow=False,
                    font=dict(color="white", size=10),
                    bgcolor="rgba(139, 69, 19, 0.7)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4
                )
                
                fig_2d.add_annotation(
                    x=0, y=soil_bund_height/2,
                    text="Waste in Tub",
                    showarrow=False,
                    font=dict(color="white", size=10),
                    bgcolor="rgba(101, 67, 33, 0.7)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4
                )
                
                fig_2d.update_layout(
                    title="Capsule-Shaped Horizontal Cross-Section",
                    xaxis_title="Distance along X-axis (m)",
                    yaxis_title="Height (m)",
                    height=600,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
            
            # Display 2D cross-section
            st.plotly_chart(fig_2d, use_container_width=True)
            
            # Add component information
            st.subheader("Component Details at Cross-Section")
            
            component_info = {
                'Component': ['Excavation Depth', 'Liner', 'Drain Layer', 'Waste in Tub', 'Soil Bund', 
                             'Above-Ground Waste', 'Cover', 'Topsoil'],
                'Thickness/Height (m)': [f"{excavation_depth:.2f}", f"{liner_thickness:.2f}", f"{drain_thickness:.2f}", 
                                         f"{soil_bund_height:.2f}", f"{soil_bund_height:.2f}",
                                         f"{results['max_height'] - soil_bund_height:.2f}", f"{cover_thickness:.2f}", f"{topsoil_thickness:.2f}"],
                'Function': ['Waste containment', 'Leak prevention', 'Leachate collection', 'Waste storage', 
                            'Containment', 'Waste storage', 'Environmental protection', 'Vegetation support']
            }
            st.dataframe(pd.DataFrame(component_info), use_container_width=True)
            
            # Add level information
            st.subheader("Level Information at Cross-Section")
            level_info = []
            for level in results['levels']:
                if level['Volume'] > 0:
                    l, w = level['Length'], level['Width']
                    if abs(cs_x) <= l/2 and abs(cs_y) <= w/2:
                        level_info.append({
                            'Level': level['Level'],
                            'Type': level['Type'],
                            'Height': f"{level['Bottom_Z']:.1f} - {level['Top_Z']:.1f} m",
                            'Volume': f"{level['Volume']:.2f} Cum"
                        })
            
            if level_info:
                st.dataframe(pd.DataFrame(level_info), use_container_width=True)
            else:
                st.info("No landfill levels at this location")
                
        except Exception as e:
            st.error(f"Error generating cross-section: {str(e)}")
            st.info("Please try adjusting the cross-section position or parameters.")
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.header("Detailed Calculations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Volume Calculations</h3>', unsafe_allow_html=True)
        volume_df = pd.DataFrame(results['levels'])
        volume_df['Cumulative Volume'] = volume_df['Volume'].cumsum()
        volume_df['Cumulative Volume'] = volume_df['Cumulative Volume'].round(2)
        volume_df['Volume'] = volume_df['Volume'].round(2)
        volume_df['Area'] = volume_df['Area'].round(2)
        st.dataframe(volume_df, use_container_width=True)
        
        st.markdown('<h3 class="section-header">Leachate Calculations</h3>', unsafe_allow_html=True)
        leachate_data = {
            'Parameter': ['Waste per Annum', 'Leachate %', 'Leachate TPA', 'Leachate KLD'],
            'Value': [f"{results['tpa']:,.2f} TPA", f"{leachate_percentage}%", 
                     f"{results['leachate_tpa']:,.2f} TPA", f"{results['leachate_kld']:.2f} KLD"]
        }
        st.dataframe(pd.DataFrame(leachate_data), use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">Design Parameters</h3>', unsafe_allow_html=True)
        design_params = {
            'Parameter': ['Daily Quantity', 'Duration', 'Density', 'Landfill Area', 'Bottom Width', 'Bottom Length', 'Total Levels', 'Max Height'],
            'Value': [f"{daily_quantity:.0f} TPD", f"{duration:.0f} years", f"{density:.2f} ton/Cum",
                     f"{landfill_area_acres:.1f} Acres", f"{bottom_width:.2f} m", f"{bottom_length:.2f} m",
                     len([l for l in results['levels'] if l['Volume'] > 0]), f"{results['max_height']:.2f} m"]
        }
        st.dataframe(pd.DataFrame(design_params), use_container_width=True)
        
        st.markdown('<h3 class="section-header">Slope Analysis</h3>', unsafe_allow_html=True)
        slope_data = {
            'Slope Type': ['External', 'Internal', 'Waste', 'Excavation', 'Soil Bund'],
            'Ratio (H:V)': [f"1:{external_slope:.1f}", f"1:{internal_slope:.1f}", f"1:{waste_slope:.1f}", 
                           f"1:{excavation_slope:.1f}", "1:2.0"],
            'Angle (degrees)': [f"{math.degrees(math.atan(1/external_slope)):.1f}°",
                               f"{math.degrees(math.atan(1/internal_slope)):.1f}°",
                               f"{math.degrees(math.atan(1/waste_slope)):.1f}°",
                               f"{math.degrees(math.atan(1/excision_slope)):.1f}" if 'excision_slope' in locals() else f"{math.degrees(math.atan(1/excision_slope)):.1f}°",
                               f"{math.degrees(math.atan(1/2.0)):.1f}°"]
        }
        st.dataframe(pd.DataFrame(slope_data), use_container_width=True)
        
        # Volume breakdown by type
        st.markdown('<h3 class="section-header">Volume Breakdown by Type</h3>', unsafe_allow_html=True)
        excavation_volume = sum(l['Volume'] for l in results['levels'] if l['Type'] == 'Excavation')
        waste_in_tub = sum(l['Volume'] for l in results['levels'] if l['Type'] == 'Waste in Tub')
        above_ground_waste = sum(l['Volume'] for l in results['levels'] if l['Type'] == 'Above Ground Waste')
        
        volume_breakdown = {
            'Component': ['Excavation', 'Waste in Tub', 'Above Ground Waste', 'Total'],
            'Volume (Cum)': [excavation_volume, waste_in_tub, above_ground_waste, results['total_volume']],
            'Percentage': [
                f"{excavation_volume/results['total_volume']*100:.1f}%" if results['total_volume'] > 0 else "0%",
                f"{waste_in_tub/results['total_volume']*100:.1f}%" if results['total_volume'] > 0 else "0%",
                f"{above_ground_waste/results['total_volume']*100:.1f}%" if results['total_volume'] > 0 else "0%",
                "100%"
            ]
        }
        st.dataframe(pd.DataFrame(volume_breakdown), use_container_width=True)

with tab5:
    st.header("Cost Analysis")
    
    # Cost breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Cost Breakdown</h3>', unsafe_allow_html=True)
        land_cost = landfill_area_acres * 4047 * cost_per_sqm
        development_cost = results['total_cost'] * 0.3
        total_cost_calculated = land_cost + development_cost
        
        cost_data = {
            'Item': ['Land Area Cost', 'Development Cost (30%)', 'Total Cost'],
            'Amount (INR)': [
                f"₹{land_cost:,.2f}",
                f"₹{development_cost:,.2f}",
                f"₹{total_cost_calculated:,.2f}"
            ]
        }
        st.dataframe(pd.DataFrame(cost_data), use_container_width=True)
        
        # Cost per unit
        st.markdown('<h3 class="section-header">Unit Costs</h3>', unsafe_allow_html=True)
        unit_costs = {
            'Metric': ['Cost per Ton', 'Cost per Cum', 'Cost per Sqm', 'Annual Cost'],
            'Value': [
                f"₹{results['per_ton_cost']:.2f}",
                f"₹{results['per_ton_cost']/density:.2f}",
                f"₹{cost_per_sqm:.2f}",
                f"₹{total_cost_calculated/results['landfill_life']:.2f}" if results['landfill_life'] > 0 else "N/A"
            ]
        }
        st.dataframe(pd.DataFrame(unit_costs), use_container_width=True)
    
    with col2:
        # Cost visualization
        st.markdown('<h3 class="section-header">Cost Distribution</h3>', unsafe_allow_html=True)
        
        fig_cost = go.Figure(data=[
            go.Bar(
                name='Land Cost',
                x=['Total Cost'],
                y=[land_cost],
                text=f"₹{land_cost:,.0f}",
                textposition='auto'
            ),
            go.Bar(
                name='Development Cost',
                x=['Total Cost'],
                y=[development_cost],
                text=f"₹{development_cost:,.0f}",
                textposition='auto'
            )
        ])
        
        fig_cost.update_layout(
            barmode='stack',
            title="Cost Breakdown",
            yaxis_title="Cost (INR)",
            height=400
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # ROI Analysis
        st.markdown('<h3 class="section-header">Economic Indicators</h3>', unsafe_allow_html=True)
        if daily_quantity > 0:
            annual_revenue = daily_quantity * 365 * 500  # Assuming ₹500 per ton
            payback_period = total_cost_calculated / annual_revenue if annual_revenue > 0 else 0
            roi = (annual_revenue * results['landfill_life'] - total_cost_calculated) / total_cost_calculated * 100 if total_cost_calculated > 0 else 0
            
            metrics = {
                'Indicator': ['Annual Revenue (Est.)', 'Payback Period', 'ROI'],
                'Value': [
                    f"₹{annual_revenue:,.2f}",
                    f"{payback_period:.2f} years" if payback_period > 0 else "N/A",
                    f"{roi:.2f}%"
                ]
            }
            st.dataframe(pd.DataFrame(metrics), use_container_width=True)

with tab6:
    st.header("Reports and Export")
    
    st.markdown('<h3 class="section-header">Generate Comprehensive Report</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
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
        📈 **Export Options:**
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
st.markdown("### 📝 Important Notes")
st.info("""
- This simulator provides estimates based on the input parameters
- Actual construction may require additional engineering considerations
- Environmental regulations and local conditions should be factored in
- Regular monitoring and maintenance are essential for landfill operations
- The 3D visualization is a simplified representation for conceptual understanding
- Cross-sections are generated dynamically based on slider position
- The capsule-shaped design provides better containment and environmental protection
- Soil bunds provide additional containment and stability
- Proper liner and drainage systems are essential for leachate management
""")

st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-top: 2rem;'>
    <p>Landfill Construction Simulator v3.0 | Capsule-Shaped Design with Enhanced Visualization</p>
</div>
""", unsafe_allow_html=True)
