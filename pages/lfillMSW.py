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
    st.session_state.berm_width = 4.0
    st.session_state.external_slope = 2.0
    st.session_state.internal_slope = 3.0
    st.session_state.waste_slope = 3.0
    st.session_state.depth_below_ngl = 2.0
    st.session_state.cost_per_sqm = 3850.0
    st.session_state.leachate_percentage = 20

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
            berm_width = st.number_input("Berm Width (m)", value=st.session_state.berm_width, min_value=1.0, step=0.5, key="berm_width")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Slope Parameters**")
            external_slope = st.number_input("External Slope", value=st.session_state.external_slope, min_value=1.0, step=0.5, key="external")
            internal_slope = st.number_input("Internal Slope", value=st.session_state.internal_slope, min_value=1.0, step=0.5, key="internal")
            waste_slope = st.number_input("Waste Slope", value=st.session_state.waste_slope, min_value=1.0, step=0.5, key="waste")
            depth_below_ngl = st.number_input("Depth Below NGL (m)", value=st.session_state.depth_below_ngl, min_value=0.0, step=0.5, key="depth")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Cost and Leachate parameters in a compact row
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            st.markdown("**Cost & Leachate**")
            cost_per_sqm = st.number_input("Cost per Sqm (INR)", value=st.session_state.cost_per_sqm, min_value=100.0, step=100.0, key="cost")
            leachate_percentage = st.slider("Leachate %", min_value=5, max_value=50, value=st.session_state.leachate_percentage, key="leachate")
            st.markdown('</div>', unsafe_allow_html=True)

# Define the cross-section profile function with parameters instead of relying on global variables
def create_realistic_cross_section_profile(points, cross_section_pos, is_vertical, results_data, bund_h, external_slope_param):
    """Create realistic cross-section profile with proper bund structure"""
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
            
            # Check if within landfill area
            if abs(y_pos) <= bund_outer_length/2:
                # Calculate waste height
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
            
            # Check if within landfill area
            if abs(x_pos) <= bund_outer_length/2:
                # Calculate waste height
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
                    bund_profile.append(0)
            else:
                profile.append(0)
                bund_profile.append(0)
    
    return profile, bund_profile

# Calculate all values
def calculate_landfill_parameters():
    # Basic calculations
    tpa = daily_quantity * 365
    total_quantity = tpa * duration
    landfill_area_sqm = landfill_area_acres * 4047
    
    # Calculate levels and volumes
    levels_data = []
    
    # 1. Below Ground Level (BBL) - Excavation below ground
    drain_width = 1.0
    drain_depth = 1.0
    
    bbl_length = width - ((drain_width + (depth_below_ngl * internal_slope) + drain_depth + 1) * 2)
    bbl_width_calc = length - ((drain_width + (depth_below_ngl * internal_slope) + drain_depth + 1) * 2)
    
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
    
    # 2. Base Level (BL) - Waste filling up to top of bund
    bl_length = bbl_length + (((bund_height + depth_below_ngl) * internal_slope) * 2)
    bl_width = bbl_width_calc + (((bund_height + depth_below_ngl) * internal_slope) * 2)
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
    total_volume = bbl_volume + bl_volume
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
        'leachate_kld': leachate_kld,
        'tpa': tpa,
        'bbl_length': bbl_length,
        'bbl_width': bbl_width_calc,
        'bl_length': bl_length,
        'bl_width': bl_width,
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
            cross_section_y = 0
        else:
            cross_section_y = st.slider("Y Position (m)", -length/2, length/2, 0.0, step=10.0)
            cross_section_x = 0
    
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
        bund_inner_width = results['bl_width']
        bund_outer_length = bund_inner_length + 2 * (bund_height * external_slope)
        bund_outer_width = bund_inner_width + 2 * (bund_height * external_slope)
        
        # Model the complete landfill structure
        for i in range(len(X)):
            for j in range(len(X[0])):
                x_pos, y_pos = X[i, j], Y[i, j]
                
                # Check if point is in bund area
                if (abs(x_pos) <= bund_outer_length/2 and abs(y_pos) <= bund_outer_width/2):
                    # Inside outer bund boundary
                    
                    # Check if point is in waste area
                    if (abs(x_pos) <= bund_inner_length/2 and abs(y_pos) <= bund_inner_width/2):
                        # Inside waste area - calculate waste height
                        for level in results['levels']:
                            if level['Volume'] > 0:
                                l, w = level['Length'], level['Width']
                                if abs(x_pos) <= l/2 and abs(y_pos) <= w/2:
                                    Z[i, j] = level['Top_Z']
                    else:
                        # In bund area - calculate bund height with proper slopes
                        dist_from_center_x = abs(x_pos)
                        dist_from_center_y = abs(y_pos)
                        
                        # Distance from inner boundary
                        dist_from_inner_x = max(0, dist_from_center_x - bund_inner_length/2)
                        dist_from_inner_y = max(0, dist_from_center_y - bund_inner_width/2)
                        
                        # Calculate height based on slopes
                        if dist_from_inner_x > 0 and dist_from_inner_y > 0:
                            # Corner area - use diagonal distance
                            corner_dist = math.sqrt(dist_from_inner_x**2 + dist_from_inner_y**2)
                            Z[i, j] = min(bund_height, corner_dist / external_slope)
                        elif dist_from_inner_x > 0:
                            # Side area
                            Z[i, j] = min(bund_height, dist_from_inner_x / external_slope)
                        elif dist_from_inner_y > 0:
                            # End area
                            Z[i, j] = min(bund_height, dist_from_inner_y / external_slope)
                        else:
                            # Inside inner boundary
                            Z[i, j] = 0
        
        # Add waste surface
        fig_3d.add_trace(
            go.Surface(
                x=X, y=Y, z=Z,
                colorscale=[[0, '#8B4513'], [0.3, '#A0522D'], [0.6, '#CD853F'], [1, '#DEB887']],
                showscale=True,
                colorbar=dict(title="Height (m)"),
                hovertemplate='X: %{x:.1f}m<br>Y: %{y:.1f}m<br>Z: %{z:.1f}m<extra></extra>',
                name='Landfill'
            )
        )
        
        # Add ground level plane
        xx, yy = np.meshgrid(np.linspace(-width/2, width/2, 10), np.linspace(-length/2, length/2, 10))
        zz = np.zeros_like(xx)
        fig_3d.add_trace(
            go.Surface(
                x=xx, y=yy, z=zz,
                colorscale=[[0, 'green'], [1, 'green']],
                showscale=False,
                opacity=0.3,
                hoverinfo='skip',
                name='Ground Level'
            )
        )
        
        # Add cross-section line
        if cross_section_type == "Vertical (Y-axis)":
            # Vertical line at X position
            line_y = np.linspace(-length/2, length/2, 20)
            line_x = [cross_section_x] * len(line_y)
            line_z = np.linspace(-depth_below_ngl, results['max_height'], 20)
            
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
            line_z = np.linspace(-depth_below_ngl, results['max_height'], 20)
            
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
            title="3D Landfill Model with Proper Bund Structure",
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
        st.subheader("Cross-Section View")
        
        # Create 2D cross-section with proper bund modeling
        fig_2d = go.Figure()
        
        # Only create cross-section if we have the required variables
        try:
            if cross_section_type == "Vertical (Y-axis)":
                # Vertical cross-section (along Y-axis)
                st.write("📐 Vertical Cross-Section (Along Y-axis)")
                
                # Generate detailed cross-section profile
                y_points = np.linspace(-length/2, length/2, 200)
                
                # Create realistic profiles with proper parameters
                waste_profile, bund_profile = create_realistic_cross_section_profile(
                    y_points, cross_section_x, cross_section_y, is_vertical=True, 
                    results_data=results, bund_h=bund_height, external_slope_param=external_slope
                )
                
                # Add ground level
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=[0]*len(y_points),
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name='Ground Level',
                    hoverinfo='skip'
                ))
                
                # Add bund profile with proper structure
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=bund_profile,
                    mode='lines',
                    line=dict(color='#D2691E', width=3),
                    name='Bund Profile',
                    hovertemplate='Y: %{x:.1f}m<br>Bund Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add waste profile
                fig_2d.add_trace(go.Scatter(
                    x=y_points, y=waste_profile,
                    mode='lines',
                    line=dict(color='#654321', width=3),
                    name='Waste Profile',
                    fill='tonexty',
                    fillcolor='rgba(101, 67, 33, 0.5)',
                    hovertemplate='Y: %{x:.1f}m<br>Waste Height: %{y:.1f}m<extra></extra>'
                ))
                
                fig_2d.update_layout(
                    title="Vertical Cross-Section with Proper Bund Structure",
                    xaxis_title="Distance along Y-axis (m)",
                    yaxis_title="Height (m)",
                    height=400,
                    showlegend=True
                )
                
            else:
                # Horizontal cross-section (along X-axis)
                st.write("📐 Horizontal Cross-Section (Along X-axis)")
                
                # Generate detailed cross-section profile
                x_points = np.linspace(-width/2, width/2, 200)
                
                # Create realistic profiles with proper parameters
                waste_profile, bund_profile = create_realistic_cross_section_profile(
                    x_points, cross_section_x, cross_section_y, is_vertical=False, 
                    results_data=results, bund_h=bund_height, external_slope_param=external_slope
                )
                
                # Add ground level
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=[0]*len(x_points),
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    name='Ground Level',
                    hoverinfo='skip'
                ))
                
                # Add bund profile with proper structure
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=bund_profile,
                    mode='lines',
                    line=dict(color='#D2691E', width=3),
                    name='Bund Profile',
                    hovertemplate='X: %{x:.1f}m<br>Bund Height: %{y:.1f}m<extra></extra>'
                ))
                
                # Add waste profile
                fig_2d.add_trace(go.Scatter(
                    x=x_points, y=waste_profile,
                    mode='lines',
                    line=dict(color='#654321', width=3),
                    name='Waste Profile',
                    fill='tonexty',
                    fillcolor='rgba(101, 67, 33, 0.5)',
                    hovertemplate='X: %{x:.1f}m<br>Waste Height: %{y:.1f}m<extra></extra>'
                ))
                
                fig_2d.update_layout(
                    title="Horizontal Cross-Section with Proper Bund Structure",
                    xaxis_title="Distance along X-axis (m)",
                    yaxis_title="Height (m)",
                    height=400,
                    showlegend=True
                )
            
            # Display 2D cross-section
            st.plotly_chart(fig_2d, use_container_width=True)
            
            # Add level information
            st.subheader("Level Information at Cross-Section")
            level_info = []
            for level in results['levels']:
                if level['Volume'] > 0:
                    l, w = level['Length'], level['Width']
                    if abs(cross_section_x) <= l/2 and abs(cross_section_y) <= w/2:
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
            'Parameter': ['Daily Quantity', 'Duration', 'Density', 'Landfill Area', 'Base Width', 'Total Levels', 'Max Height'],
            'Value': [f"{daily_quantity:.0f} TPD", f"{duration:.0f} years", f"{density:.2f} ton/Cum",
                     f"{landfill_area_acres:.1f} Acres", f"{bund_width + (bund_height * external_slope) + (bund_height * internal_slope):.2f} m",
                     len([l for l in results['levels'] if l['Volume'] > 0]), f"{results['max_height']:.2f} m"]
        }
        st.dataframe(pd.DataFrame(design_params), use_container_width=True)
        
        st.markdown('<h3 class="section-header">Slope Analysis</h3>', unsafe_allow_html=True)
        slope_data = {
            'Slope Type': ['External', 'Internal', 'Waste'],
            'Ratio (H:V)': [f"1:{external_slope:.1f}", f"1:{internal_slope:.1f}", f"1:{waste_slope:.1f}"],
            'Angle (degrees)': [f"{math.degrees(math.atan(1/external_slope)):.1f}°",
                               f"{math.degrees(math.atan(1/internal_slope)):.1f}°",
                               f"{math.degrees(math.atan(1/waste_slope)):.1f}°"]
        }
        st.dataframe(pd.DataFrame(slope_data), use_container_width=True)
        
        # Volume breakdown by type
        st.markdown('<h3 class="section-header">Volume Breakdown by Type</h3>', unsafe_allow_html=True)
        excavation_volume = sum(l['Volume'] for l in results['levels'] if l['Type'] == 'Excavation')
        waste_up_to_bund = sum(l['Volume'] for l in results['levels'] if l['Type'] == 'Waste up to Bund')
        waste_above_bund = sum(l['Volume'] for l in results['levels'] if l['Type'] == 'Waste above Bund')
        
        volume_breakdown = {
            'Component': ['Excavation Below Ground', 'Waste up to Bund', 'Waste above Bund', 'Total'],
            'Volume (Cum)': [excavation_volume, waste_up_to_bund, waste_above_bund, results['total_volume']],
            'Percentage': [
                f"{excavation_volume/results['total_volume']*100:.1f}%" if results['total_volume'] > 0 else "0%",
                f"{waste_up_to_bund/results['total_volume']*100:.1f}%" if results['total_volume'] > 0 else "0%",
                f"{waste_above_bund/results['total_volume']*100:.1f}%" if results['total_volume'] > 0 else "0%",
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
""")

st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-top: 2rem;'>
    <p>Landfill Construction Simulator v2.0 | Interactive Design & Analysis Tool</p>
</div>
""", unsafe_allow_html=True)
