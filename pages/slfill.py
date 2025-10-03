import math
import plotly.graph_objects as go

# ---- Example Inputs ----
depth_bg      = 2.0   # depth below GL
bund_top_w    = 5.0
outer_slope_h = 3.0   # outer slope 1:3
inner_slope_h = 3.0   # inner slope 1:3 (schematic steps)
lift_h        = 5.0
bench_w       = 4.0
n_lifts       = 4

# ---- Build stepped profile ----
H = n_lifts * lift_h
z0 = -depth_bg
xL0, zL0 = 0.0, z0
xL1, zL1 = H * outer_slope_h, z0 + H
xL2, zL2 = xL1 + bund_top_w/2, zL1

bx, bz = [xL2], [zL2]
curx, curz = xL2, zL2
for k in range(n_lifts):
    curx += bench_w; bx.append(curx); bz.append(curz)    # bench
    curz += lift_h;   bx.append(curx); bz.append(curz)   # vertical rise

xR2, zR2 = curx + bund_top_w/2, curz
xR1, zR1 = xR2 + H * outer_slope_h, z0

X = [xL0, xL1, xL2] + bx[1:] + [xR2, xR1]
Z = [zL0, zL1, zL2] + bz[1:] + [zR2, zR1]

# ---- Plotly Figure ----
fig = go.Figure()

fig.add_trace(go.Scatter(x=X, y=Z, mode="lines+markers", name="Landfill Profile"))
fig.add_trace(go.Scatter(
    x=[min(X)-5, max(X)+5], y=[0,0],
    mode="lines", line=dict(dash="dash"), name="Ground Level"
))

fig.update_layout(
    title="Landfill Cross Section (schematic)",
    xaxis_title="Horizontal (m)",
    yaxis_title="Elevation (m)",
    template="simple_white",
    width=800, height=400
)

fig.show()
