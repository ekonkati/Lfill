import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap
import math

st.set_page_config(page_title="Landfill Construction Simulator", layout="wide")

st.title("🏗️ Landfill Construction Simulator")
st.markdown("A comprehensive tool for landfill design, volume calculation, and visualization")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Basic Parameters
st.sidebar.subheader("Basic Parameters")
daily_quantity = st.sidebar.number_input("Daily Waste Quantity (TPD)", value=1000.0, min_value=1.0, step=10.0)
duration = st.sidebar.number_input("Duration (years)", value=25.0, min_value=1.0, step=1.0)
density = st.sidebar.number_input("Density (ton/Cum)", value=1.00, min_value=0.1, step=0.01)
landfill_area_acres = st.sidebar.number_input("Landfill Area (Acres)", value=10.0, min_value=1.0, step=0.5)

# Dimension Parameters
st.sidebar.subheader("Dimensions")
width = st.sidebar.number_input("Width (m)", value=600.0, min_value=10.0, step=10.0)
length = st.sidebar.number_input("Length (m)", value=600.0, min_value=10.0, step=10.0)
bund_width = st.sidebar.number_input("Bund Width (m)", value=5.0, min_value=1.0, step=0.5)
bund_height = st.sidebar.number_input("Bund Height (m)", value=4.0, min_value=1.0, step=0.5)
external_slope = st.sidebar.number_input("External Slope (H:V)", value=2.0, min_value=1.0, step=0.5)
internal_slope = st.sidebar.number_input("Internal Slope (H:V)", value=3.0, min_value=1.0, step=0.5)
waste_slope = st.sidebar.number_input("Waste Slope (H:V)", value=3.0, min_value=1.0, step=0.5)
waste_height = st.sidebar.number_input("Height of Waste (m)", value=5.0, min_value=1.0, step=0.5)
berm_width = st.sidebar.number_input("Berm Width (m)", value=4.0, min_value=1.0, step=0.5)
depth_below_ngl = st.sidebar.number_input("Depth below NGL (m)", value=2.0, min_value=0.0, step=0.5)

# Cost Parameters
st.sidebar.subheader("Cost Parameters")
cost_per_sqm = st.sidebar.number_input("Cost per Sqm (INR)", value=3850.0, min_value=100.0, step=100.0)

# Leachate Parameters
st.sidebar.subheader("Leachate Parameters")
leachate_percentage = st.sidebar.slider("Leachate Percentage (%)", min_value=5, max_value=50, value=20)

# Calculate all values
def calculate_landfill_parameters():
    # Basic calculations
    tpa = daily_quantity * 365
    total_quantity = tpa * duration
    landfill_area_sqm = landfill_area_acres * 4047
    
    # Calculate levels and volumes
    levels_data = []
    
    # 1. Below Ground Level (BBL) - Excavation below ground
    # According to Excel: BBL length = (B9-((B20+(B24*B15)+B22+1)*2))
    # Where B9 = width, B20 = drain width, B24 = depth below NGL, B15 = internal slope, B22 = drain depth
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
        'Type': 'Excavation'
    })
    
    # 2. Base Level (BL) - Waste filling up to top of bund
    # According to Excel: BL length = (B27+(((B12+B24)*B15)*2))
    # Where B27 = BBL length, B12 = bund height, B24 = depth below NGL, B15 = internal slope
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
        'Type': 'Waste up to Bund'
    })
    
    # 3. Above Bund Levels (ABL) - Waste filling from top of bund to crest with berms
    current_length = bl_length
    current_width = bl_width
    total_volume = bbl_volume + bl_volume
    
    for i in range(1, 10):  # Calculate up to 9 levels
        # ABL level - waste filling
        # According to Excel: ABL length = (B28-(5*$B$17)*2)
        # Where B28 = previous level length, B17 = waste slope
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
            'Type': 'Waste above Bund'
        })
        
        # Berm level
        # According to Excel: ABL i Berm length = (B32-4*2)
        # Where B32 = ABL length, 4 = berm width
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
            'Type': 'Berm'
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
        'bl_width': bl_width
    }

# Calculate parameters
results = calculate_landfill_parameters()

# Display results in tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Results Summary", "📐 2D Visualization", "🎯 3D Visualization", "📋 Detailed Calculations", "💰 Cost Analysis"])

with tab1:
    st.header("Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Volume", f"{results['total_volume']:,.2f} Cum")
        st.metric("Provided Quantity", f"{results['provided_quantity']:,.2f} Tons")
    
    with col2:
        st.metric("Required Quantity", f"{results['required_quantity']:,.2f} Tons")
        st.metric("Capacity Sufficient", results['is_sufficient'])
    
    with col3:
        st.metric("Landfill Life", f"{results['landfill_life']:.2f} Years")
        st.metric("Total Cost", f"₹{results['total_cost']:,.2f}")
    
    with col4:
        st.metric("Cost per Ton", f"₹{results['per_ton_cost']:.2f}")
        st.metric("Leachate Generation", f"{results['leachate_kld']:.2f} KLD")
    
    # Status indicators
    st.subheader("Project Status")
    if results['is_sufficient'] == "Yes":
        st.success("✅ The landfill design meets the required capacity!")
    else:
        st.error("❌ The landfill design does not meet the required capacity. Consider increasing area or dimensions.")
    
    if results['landfill_life'] >= duration:
        st.success(f"✅ Landfill life ({results['landfill_life']:.1f} years) exceeds project duration ({duration:.0f} years)")
    else:
        st.warning(f"⚠️ Landfill life ({results['landfill_life']:.1f} years) is less than project duration ({duration:.0f} years)")

with tab2:
    st.header("2D Cross-Section Visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cross-section view
    ax1.set_title("Landfill Cross-Section (Side View)")
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Height (m)")
    ax1.grid(True, alpha=0.3)
    
    # Draw landfill profile
    x_points = []
    y_points = []
    
    # Get dimensions from results
    bbl_length = results['bbl_length']
    bbl_width = results['bbl_width']
    bl_length = results['bl_length']
    bl_width = results['bl_width']
    
    # Calculate total width for proper scaling
    total_width = width + 2 * (bund_height * external_slope + bund_width + waste_height * waste_slope * 5)
    
    # Start from leftmost point
    current_x = 0
    current_y = -depth_below_ngl
    
    # 1. Left side of excavation below ground
    # From left edge to bottom of excavation
    x_points.extend([current_x, current_x + depth_below_ngl * internal_slope])
    y_points.extend([current_y, current_y + depth_below_ngl])
    
    # Bottom of excavation
    current_x += depth_below_ngl * internal_slope
    current_y += depth_below_ngl
    x_points.extend([current_x, current_x + bbl_length])
    y_points.extend([current_y, current_y])
    
    # Right side of excavation
    current_x += bbl_length
    x_points.extend([current_x, current_x + depth_below_ngl * internal_slope])
    y_points.extend([current_y, current_y - depth_below_ngl])
    
    # Ground level
    current_x += depth_below_ngl * internal_slope
    current_y -= depth_below_ngl
    x_points.extend([current_x, current_x + bund_height * external_slope])
    y_points.extend([current_y, current_y + bund_height])
    
    # Top of bund
    current_x += bund_height * external_slope
    current_y += bund_height
    x_points.extend([current_x, current_x + bund_width])
    y_points.extend([current_y, current_y])
    
    # Internal slope to waste
    current_x += bund_width
    x_points.extend([current_x, current_x + bund_height * internal_slope])
    y_points.extend([current_y, current_y - bund_height])
    
    # Waste layers above bund
    current_x += bund_height * internal_slope
    current_y -= bund_height
    
    num_waste_layers = min(5, len([l for l in results['levels'] if 'ABL' in l['Level'] and l['Volume'] > 0]))
    
    for i in range(num_waste_layers):
        waste_horizontal = waste_height * waste_slope
        x_points.extend([current_x, current_x + waste_horizontal])
        y_points.extend([current_y, current_y + waste_height])
        current_x += waste_horizontal
        current_y += waste_height
        
        if i < num_waste_layers - 1:  # Add berm except for last layer
            x_points.extend([current_x, current_x + berm_width])
            y_points.extend([current_y, current_y])
            current_x += berm_width
    
    # Right side (mirror of left side)
    right_x = current_x
    right_y = current_y
    
    # Right slopes
    x_points.extend([right_x, right_x + waste_height * waste_slope])
    y_points.extend([right_y, right_y - waste_height])
    
    x_points.extend([right_x + waste_height * waste_slope, right_x + waste_height * waste_slope + bund_width])
    y_points.extend([right_y - waste_height, right_y - waste_height])
    
    x_points.extend([right_x + waste_height * waste_slope + bund_width, right_x + waste_height * waste_slope + bund_width + bund_height * internal_slope])
    y_points.extend([right_y - waste_height, right_y - 2 * bund_height])
    
    x_points.extend([right_x + waste_height * waste_slope + bund_width + bund_height * internal_slope, right_x + waste_height * waste_slope + bund_width + bund_height * internal_slope + bund_width])
    y_points.extend([right_y - 2 * bund_height, right_y - 2 * bund_height])
    
    x_points.extend([right_x + waste_height * waste_slope + bund_width + bund_height * internal_slope + bund_width, right_x + waste_height * waste_slope + bund_width + bund_height * internal_slope + bund_width + bund_height * external_slope])
    y_points.extend([right_y - 2 * bund_height, right_y - 3 * bund_height])
    
    # Close the shape
    x_points.append(x_points[0])
    y_points.append(y_points[0])
    
    # Fill and plot
    ax1.fill(x_points, y_points, color='#8B4513', alpha=0.6, label='Waste Fill')
    ax1.plot(x_points, y_points, 'k-', linewidth=2)
    
    # Add ground level line
    max_x = max(x_points)
    ax1.plot([0, max_x], [0, 0], 'g--', linewidth=1, label='Ground Level')
    
    # Add bund top line
    bund_left_x = depth_below_ngl * internal_slope + bbl_length + depth_below_ngl * internal_slope + bund_height * external_slope
    bund_right_x = bund_left_x + bund_width
    ax1.plot([bund_left_x, bund_right_x], [bund_height, bund_height], 'b-', linewidth=2, label='Bund Top')
    
    # Add labels for different sections
    ax1.text(bbl_length/2, -depth_below_ngl/2, "Excavation\nBelow Ground", ha='center', va='center')
    ax1.text(bund_left_x + bund_width/2, bund_height/2, "Bund", ha='center', va='center')
    
    # Add legend
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')
    
    # Top view
    ax2.set_title("Landfill Top View")
    ax2.set_xlabel("Length (m)")
    ax2.set_ylabel("Width (m)")
    ax2.grid(True, alpha=0.3)
    
    # Create custom brown colors
    brown_colors = ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#F4A460', '#D2691E', '#BC8F8F']
    
    # Draw concentric rectangles for each level
    for i, level in enumerate(results['levels']):
        if level['Volume'] > 0:
            color = brown_colors[i % len(brown_colors)]
            rect = Rectangle(
                (width/2 - level['Length']/2, length/2 - level['Width']/2),
                level['Length'], level['Width'],
                fill=False, edgecolor=color, linewidth=2,
                label=level['Level']
            )
            ax2.add_patch(rect)
    
    ax2.set_xlim(0, width + 100)
    ax2.set_ylim(0, length + 100)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_aspect('equal', adjustable='box')
    
    st.pyplot(fig)
    
    # Level dimensions table
    st.subheader("Level Dimensions")
    levels_df = pd.DataFrame(results['levels'])
    st.dataframe(levels_df, use_container_width=True)

with tab3:
    st.header("3D Visualization")
    
    # Create 3D visualization
    fig = go.Figure()
    
    # Generate mesh for 3D landfill
    def create_landfill_3d():
        # Create a stepped pyramid shape
        levels = []
        current_height = -depth_below_ngl
        
        for level in results['levels']:
            if level['Volume'] > 0:
                l, w = level['Length'], level['Width']
                h = level['Height']
                
                # Create vertices for the rectangular prism at this level
                vertices = [
                    [-l/2, -w/2, current_height],
                    [l/2, -w/2, current_height],
                    [l/2, w/2, current_height],
                    [-l/2, w/2, current_height],
                    [-l/2, -w/2, current_height + h],
                    [l/2, -w/2, current_height + h],
                    [l/2, w/2, current_height + h],
                    [-l/2, w/2, current_height + h]
                ]
                levels.append(vertices)
                current_height += h
        
        return levels
    
    # Create 3D surface plot
    x = np.linspace(-width/2, width/2, 30)
    y = np.linspace(-length/2, length/2, 30)
    X, Y = np.meshgrid(x, y)
    
    # Create height map
    Z = np.zeros_like(X)
    current_height = -depth_below_ngl
    
    for level in results['levels']:
        if level['Volume'] > 0:
            l, w, h = level['Length'], level['Width'], level['Height']
            mask = (np.abs(X) <= l/2) & (np.abs(Y) <= w/2)
            Z[mask] = current_height + h
            current_height += h
    
    # Add surface
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, '#8B4513'], [0.5, '#A0522D'], [1, '#CD853F']],
            showscale=True,
            colorbar=dict(title="Height (m)")
        )
    )
    
    # Add wireframe edges for better visualization
    levels_3d = create_landfill_3d()
    
    for level_vertices in levels_3d:
        # Bottom edges
        for i in range(4):
            next_i = (i + 1) % 4
            fig.add_trace(
                go.Scatter3d(
                    x=[level_vertices[i][0], level_vertices[next_i][0]],
                    y=[level_vertices[i][1], level_vertices[next_i][1]],
                    z=[level_vertices[i][2], level_vertices[next_i][2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                )
            )
        
        # Top edges
        for i in range(4):
            next_i = (i + 1) % 4
            fig.add_trace(
                go.Scatter3d(
                    x=[level_vertices[i+4][0], level_vertices[next_i+4][0]],
                    y=[level_vertices[i+4][1], level_vertices[next_i+4][1]],
                    z=[level_vertices[i+4][2], level_vertices[next_i+4][2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                )
            )
        
        # Vertical edges
        for i in range(4):
            fig.add_trace(
                go.Scatter3d(
                    x=[level_vertices[i][0], level_vertices[i+4][0]],
                    y=[level_vertices[i][1], level_vertices[i+4][1]],
                    z=[level_vertices[i][2], level_vertices[i+4][2]],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                )
            )
    
    # Add ground level plane
    xx, yy = np.meshgrid(np.linspace(-width/2, width/2, 10), np.linspace(-length/2, length/2, 10))
    zz = np.zeros_like(xx)
    fig.add_trace(
        go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, 'green'], [1, 'green']],
            showscale=False,
            opacity=0.2
        )
    )
    
    fig.update_layout(
        title="3D Landfill Visualization",
        scene=dict(
            xaxis_title="Length (m)",
            yaxis_title="Width (m)",
            zaxis_title="Height (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=700,
        width=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive controls info
    st.info("🖱️ Use your mouse to rotate, zoom, and pan the 3D visualization")

with tab4:
    st.header("Detailed Calculations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volume Calculations")
        volume_df = pd.DataFrame(results['levels'])
        volume_df['Cumulative Volume'] = volume_df['Volume'].cumsum()
        volume_df['Cumulative Volume'] = volume_df['Cumulative Volume'].round(2)
        volume_df['Volume'] = volume_df['Volume'].round(2)
        volume_df['Area'] = volume_df['Area'].round(2)
        st.dataframe(volume_df, use_container_width=True)
        
        st.subheader("Leachate Calculations")
        leachate_data = {
            'Parameter': ['Waste per Annum', 'Leachate %', 'Leachate TPA', 'Leachate KLD'],
            'Value': [f"{results['tpa']:,.2f} TPA", f"{leachate_percentage}%", 
                     f"{results['leachate_tpa']:,.2f} TPA", f"{results['leachate_kld']:.2f} KLD"]
        }
        st.dataframe(pd.DataFrame(leachate_data), use_container_width=True)
    
    with col2:
        st.subheader("Design Parameters")
        design_params = {
            'Parameter': ['Daily Quantity', 'Duration', 'Density', 'Landfill Area', 'Base Width', 'Total Levels'],
            'Value': [f"{daily_quantity:.0f} TPD", f"{duration:.0f} years", f"{density:.2f} ton/Cum",
                     f"{landfill_area_acres:.1f} Acres", f"{bund_width + (bund_height * external_slope) + (bund_height * internal_slope):.2f} m",
                     len([l for l in results['levels'] if l['Volume'] > 0])]
        }
        st.dataframe(pd.DataFrame(design_params), use_container_width=True)
        
        st.subheader("Slope Analysis")
        slope_data = {
            'Slope Type': ['External', 'Internal', 'Waste'],
            'Ratio (H:V)': [f"1:{external_slope:.1f}", f"1:{internal_slope:.1f}", f"1:{waste_slope:.1f}"],
            'Angle (degrees)': [f"{math.degrees(math.atan(1/external_slope)):.1f}°",
                               f"{math.degrees(math.atan(1/internal_slope)):.1f}°",
                               f"{math.degrees(math.atan(1/waste_slope)):.1f}°"]
        }
        st.dataframe(pd.DataFrame(slope_data), use_container_width=True)
        
        # Volume breakdown by type
        st.subheader("Volume Breakdown by Type")
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
        st.subheader("Cost Breakdown")
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
        st.subheader("Unit Costs")
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
        st.subheader("Cost Distribution")
        
        fig = go.Figure(data=[
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
        
        fig.update_layout(
            barmode='stack',
            title="Cost Breakdown",
            yaxis_title="Cost (INR)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI Analysis
        st.subheader("Economic Indicators")
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

# Footer
st.markdown("---")
st.markdown("### 📝 Notes")
st.info("""
- This simulator provides estimates based on the input parameters
- Actual construction may require additional engineering considerations
- Environmental regulations and local conditions should be factored in
- Regular monitoring and maintenance are essential for landfill operations
- The 3D visualization is a simplified representation for conceptual understanding
""")

# Download report button
if st.button("📥 Generate Report"):
    st.success("Report generation feature would be implemented here with PDF export functionality")
    st.info("In a production environment, this would generate a comprehensive PDF report with all calculations and visualizations.")
