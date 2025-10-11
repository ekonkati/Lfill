import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math

st.set_page_config(page_title="Landfill Construction Simulator", layout="wide")

st.title("🏗️ Landfill Construction Simulator")
st.markdown("A comprehensive tool for landfill design, volume calculation, and visualization")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Basic Parameters
st.sidebar.subheader("Basic Parameters")
daily_quantity = st.sidebar.number_input("Daily Waste Quantity (TPD)", value=1000, min_value=1)
duration = st.sidebar.number_input("Duration (years)", value=25, min_value=1)
density = st.sidebar.number_input("Density (ton/Cum)", value=1.00, min_value=0.1, step=0.01)
landfill_area_acres = st.sidebar.number_input("Landfill Area (Acres)", value=10, min_value=1)

# Dimension Parameters
st.sidebar.subheader("Dimensions")
width = st.sidebar.number_input("Width (m)", value=600, min_value=10)
length = st.sidebar.number_input("Length (m)", value=600, min_value=10)
bund_width = st.sidebar.number_input("Bund Width (m)", value=5, min_value=1)
bund_height = st.sidebar.number_input("Bund Height (m)", value=4, min_value=1)
external_slope = st.sidebar.number_input("External Slope (H:V)", value=2, min_value=1, step=0.5)
internal_slope = st.sidebar.number_input("Internal Slope (H:V)", value=3, min_value=1, step=0.5)
waste_slope = st.sidebar.number_input("Waste Slope (H:V)", value=3, min_value=1, step=0.5)
waste_height = st.sidebar.number_input("Height of Waste (m)", value=5, min_value=1)
berm_width = st.sidebar.number_input("Berm Width (m)", value=4, min_value=1)
depth_below_ngl = st.sidebar.number_input("Depth below NGL (m)", value=2, min_value=0)

# Cost Parameters
st.sidebar.subheader("Cost Parameters")
cost_per_sqm = st.sidebar.number_input("Cost per Sqm (INR)", value=3850, min_value=100)

# Leachate Parameters
st.sidebar.subheader("Leachate Parameters")
leachate_percentage = st.sidebar.slider("Leachate Percentage (%)", min_value=5, max_value=50, value=20)

# Calculate all values
def calculate_landfill_parameters():
    # Basic calculations
    tpa = daily_quantity * 365
    total_quantity = tpa * duration
    landfill_area_sqm = landfill_area_acres * 4047
    
    # Base width calculation
    base_width = bund_width + (bund_height * external_slope) + (bund_height * internal_slope)
    
    # Calculate levels and volumes
    levels_data = []
    
    # Below Ground Level (BGL)
    bbl_length = width - ((2 + (depth_below_ngl * internal_slope) + 1) * 2)
    bbl_width = length - ((2 + (depth_below_ngl * internal_slope) + 1) * 2)
    bbl_area = bbl_length * bbl_width
    bbl_volume = bbl_area * depth_below_ngl
    
    # Base Level (BL)
    bl_length = bbl_length + (((bund_height + depth_below_ngl) * internal_slope) * 2)
    bl_width = bbl_width + (((bund_height + depth_below_ngl) * internal_slope) * 2)
    bl_area = bl_length * bl_width
    bl_volume = ((bbl_area + bl_area) / 2) * bund_height
    
    levels_data.append({
        'Level': 'BBL',
        'Length': bbl_length,
        'Width': bbl_width,
        'Area': bbl_area,
        'Height': depth_below_ngl,
        'Volume': bbl_volume
    })
    
    levels_data.append({
        'Level': 'BL',
        'Length': bl_length,
        'Width': bl_width,
        'Area': bl_area,
        'Height': bund_height,
        'Volume': bl_volume
    })
    
    # Above Bund Levels (ABL)
    current_length = bl_length
    current_width = bl_width
    total_volume = bbl_volume + bl_volume
    
    for i in range(1, 10):  # Calculate up to 9 levels
        # ABL level
        abl_length = current_length - (5 * waste_slope * 2)
        abl_width = current_width - (5 * waste_slope * 2)
        
        if abl_length <= 0 or abl_width <= 0:
            break
            
        abl_area = abl_length * abl_width
        prev_area = current_length * current_width
        abl_volume = ((prev_area + abl_area) / 2) * waste_height
        total_volume += abl_volume
        
        levels_data.append({
            'Level': f'ABL {i}',
            'Length': abl_length,
            'Width': abl_width,
            'Area': abl_area,
            'Height': waste_height,
            'Volume': abl_volume
        })
        
        # Berm level
        berm_length = abl_length - (berm_width * 2)
        berm_width_calc = abl_width - (berm_width * 2)
        
        if berm_length <= 0 or berm_width_calc <= 0:
            break
            
        berm_area = berm_length * berm_width_calc
        
        levels_data.append({
            'Level': f'ABL {i} Berm',
            'Length': berm_length,
            'Width': berm_width_calc,
            'Area': berm_area,
            'Height': 0,
            'Volume': 0
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
        'tpa': tpa
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
        st.success(f"✅ Landfill life ({results['landfill_life']:.1f} years) exceeds project duration ({duration} years)")
    else:
        st.warning(f"⚠️ Landfill life ({results['landfill_life']:.1f} years) is less than project duration ({duration} years)")

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
    
    # Start from left
    current_x = 0
    current_y = 0
    
    # Left slope
    x_points.extend([current_x, current_x + bund_height * external_slope])
    y_points.extend([current_y, current_y + bund_height])
    
    # Top of bund
    current_x += bund_height * external_slope
    current_y = bund_height
    x_points.extend([current_x, current_x + bund_width])
    y_points.extend([current_y, current_y])
    
    # Internal slope to waste
    current_x += bund_width
    x_points.extend([current_x, current_x + bund_height * internal_slope])
    y_points.extend([current_y, current_y - bund_height])
    
    # Waste layers
    current_x += bund_height * internal_slope
    current_y = 0
    for i in range(min(5, len([l for l in results['levels'] if 'ABL' in l['Level']]))):
        waste_width = 5 * waste_slope
        x_points.extend([current_x, current_x + waste_width])
        y_points.extend([current_y, current_y + waste_height])
        current_x += waste_width
        current_y += waste_height
        
        if i < 4:  # Add berm
            x_points.extend([current_x, current_x + berm_width])
            y_points.extend([current_y, current_y])
            current_x += berm_width
    
    # Right side (mirror)
    right_x = x_points[-1]
    right_y = y_points[-1]
    x_points.extend([right_x + waste_height * waste_slope, right_x + waste_height * waste_slope + bund_width])
    y_points.extend([right_y, right_y])
    x_points.extend([right_x + waste_height * waste_slope + bund_width, right_x + waste_height * waste_slope + bund_width + bund_height * internal_slope])
    y_points.extend([right_y, right_y - bund_height])
    x_points.extend([right_x + waste_height * waste_slope + bund_width + bund_height * internal_slope, right_x + waste_height * waste_slope + bund_width + bund_height * internal_slope + bund_width])
    y_points.extend([right_y - bund_height, right_y - bund_height])
    x_points.extend([right_x + waste_height * waste_slope + bund_width + bund_height * internal_slope + bund_width, right_x + waste_height * waste_slope + bund_width + bund_height * internal_slope + bund_width + bund_height * external_slope])
    y_points.extend([right_y - bund_height, right_y - 2 * bund_height])
    
    # Close the shape
    x_points.append(x_points[0])
    y_points.append(y_points[0])
    
    ax1.fill(x_points, y_points, color='brown', alpha=0.6, label='Waste')
    ax1.plot(x_points, y_points, 'k-', linewidth=2)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Top view
    ax2.set_title("Landfill Top View")
    ax2.set_xlabel("Length (m)")
    ax2.set_ylabel("Width (m)")
    ax2.grid(True, alpha=0.3)
    
    # Draw concentric rectangles for each level
    colors = plt.cm.Brown(np.linspace(0.3, 0.9, len(results['levels'])))
    
    for i, level in enumerate(results['levels']):
        if level['Volume'] > 0:
            rect = plt.Rectangle(
                (width/2 - level['Length']/2, length/2 - level['Width']/2),
                level['Length'], level['Width'],
                fill=False, edgecolor=colors[i], linewidth=2,
                label=level['Level']
            )
            ax2.add_patch(rect)
    
    ax2.set_xlim(0, width + 100)
    ax2.set_ylim(0, length + 100)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_aspect('equal')
    
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
    def create_landfill_mesh():
        vertices = []
        faces = []
        
        # Base level
        base_length = results['levels'][0]['Length']
        base_width = results['levels'][0]['Width']
        
        # Create vertices for each level
        levels_vertices = []
        current_height = 0
        
        for level in results['levels']:
            if level['Volume'] > 0:
                l, w = level['Length'], level['Width']
                h = level['Height']
                
                # Create 8 vertices for the rectangular prism
                level_verts = [
                    [-l/2, -w/2, current_height],
                    [l/2, -w/2, current_height],
                    [l/2, w/2, current_height],
                    [-l/2, w/2, current_height],
                    [-l/2, -w/2, current_height + h],
                    [l/2, -w/2, current_height + h],
                    [l/2, w/2, current_height + h],
                    [-l/2, w/2, current_height + h]
                ]
                levels_vertices.append(level_verts)
                current_height += h
        
        return levels_vertices
    
    # Create simplified 3D representation
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'scatter3d'}]],
        subplot_titles=['3D Surface View', '3D Wireframe View']
    )
    
    # Surface plot
    x = np.linspace(-width/2, width/2, 50)
    y = np.linspace(-length/2, length/2, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create a simplified height map
    Z = np.zeros_like(X)
    center_x, center_y = 0, 0
    
    for i in range(len(x)):
        for j in range(len(y)):
            dist_from_center = max(abs(x[i]), abs(y[j]))
            if dist_from_center < width/2:
                # Create a stepped profile
                level_height = 0
                for level in results['levels']:
                    if level['Volume'] > 0:
                        if dist_from_center < level['Length']/2 and dist_from_center < level['Width']/2:
                            level_height += level['Height']
                Z[j, i] = level_height
    
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z, colorscale='Earth', showscale=True),
        row=1, col=1
    )
    
    # Wireframe representation
    levels_mesh = create_landfill_mesh()
    
    for i, level_verts in enumerate(levels_mesh):
        if i < len(levels_mesh) - 1:
            # Draw edges
            for j in range(4):
                next_j = (j + 1) % 4
                # Bottom edges
                fig.add_trace(
                    go.Scatter3d(
                        x=[level_verts[j][0], level_verts[next_j][0]],
                        y=[level_verts[j][1], level_verts[next_j][1]],
                        z=[level_verts[j][2], level_verts[next_j][2]],
                        mode='lines',
                        line=dict(color='black', width=2),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                # Top edges
                fig.add_trace(
                    go.Scatter3d(
                        x=[level_verts[j+4][0], level_verts[next_j+4][0]],
                        y=[level_verts[j+4][1], level_verts[next_j+4][1]],
                        z=[level_verts[j+4][2], level_verts[next_j+4][2]],
                        mode='lines',
                        line=dict(color='black', width=2),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                # Vertical edges
                fig.add_trace(
                    go.Scatter3d(
                        x=[level_verts[j][0], level_verts[j+4][0]],
                        y=[level_verts[j][1], level_verts[j+4][1]],
                        z=[level_verts[j][2], level_verts[j+4][2]],
                        mode='lines',
                        line=dict(color='black', width=2),
                        showlegend=False
                    ),
                    row=1, col=2
                )
    
    fig.update_layout(
        title="3D Landfill Visualization",
        scene=dict(
            xaxis_title="Length (m)",
            yaxis_title="Width (m)",
            zaxis_title="Height (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive parameters
    st.subheader("Interactive 3D View")
    st.info("Use your mouse to rotate, zoom, and pan the 3D visualization")

with tab4:
    st.header("Detailed Calculations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volume Calculations")
        volume_df = pd.DataFrame(results['levels'])
        volume_df['Cumulative Volume'] = volume_df['Volume'].cumsum()
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
            'Value': [f"{daily_quantity} TPD", f"{duration} years", f"{density} ton/Cum",
                     f"{landfill_area_acres} Acres", f"{bund_width + (bund_height * external_slope) + (bund_height * internal_slope):.2f} m",
                     len([l for l in results['levels'] if l['Volume'] > 0])]
        }
        st.dataframe(pd.DataFrame(design_params), use_container_width=True)
        
        st.subheader("Slope Analysis")
        slope_data = {
            'Slope Type': ['External', 'Internal', 'Waste'],
            'Ratio (H:V)': [f"1:{external_slope}", f"1:{internal_slope}", f"1:{waste_slope}"],
            'Angle (degrees)': [f"{math.degrees(math.atan(1/external_slope)):.1f}°",
                               f"{math.degrees(math.atan(1/internal_slope)):.1f}°",
                               f"{math.degrees(math.atan(1/waste_slope)):.1f}°"]
        }
        st.dataframe(pd.DataFrame(slope_data), use_container_width=True)

with tab5:
    st.header("Cost Analysis")
    
    # Cost breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cost Breakdown")
        cost_data = {
            'Item': ['Land Area', 'Development Cost', 'Total Cost'],
            'Amount (INR)': [
                f"₹{landfill_area_acres * 4047 * cost_per_sqm:,.2f}",
                f"₹{results['total_cost'] * 0.3:,.2f}",
                f"₹{results['total_cost']:,.2f}"
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
                f"₹{results['total_cost']/results['landfill_life']:.2f}"
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
                y=[results['total_cost'] * 0.7],
                text=f"₹{results['total_cost'] * 0.7:,.0f}",
                textposition='auto'
            ),
            go.Bar(
                name='Development Cost',
                x=['Total Cost'],
                y=[results['total_cost'] * 0.3],
                text=f"₹{results['total_cost'] * 0.3:,.0f}",
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
            roi = (annual_revenue * results['landfill_life'] - results['total_cost']) / results['total_cost'] * 100
            
            metrics = {
                'Indicator': ['Annual Revenue (Est.)', 'Payback Period', 'ROI'],
                'Value': [
                    f"₹{annual_revenue:,.2f}",
                    f"{results['total_cost']/annual_revenue:.2f} years",
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
""")

# Download report button
if st.button("📥 Generate Report"):
    st.success("Report generation feature would be implemented here with PDF export functionality")
