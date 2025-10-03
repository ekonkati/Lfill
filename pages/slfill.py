import math
import plotly.graph_objects as go

# ------------------ INPUTS (edit these) ------------------
depth_bg      = 2.0   # depth below ground level (BL is at -depth_bg)
bund_top_w    = 5.0   # bund crest/top width (each side shares half)
outer_slope_h = 3.0   # outer slope 1:outer_slope_h (upwards from BL to GL+H)
inner_slope_h = 3.0   # inner (landfill) slope 1:inner_slope_h between benches (schematic step)
lift_h        = 5.0   # vertical rise per lift
bench_w       = 4.0   # horizontal bench width
n_lifts       = 4     # number of lifts
limit_top_to_base_ratio = None  # e.g. 0.30 to cap top width to 30% of base. None = no cap.

# ------------------ PROFILE CONSTRUCTION ------------------
H = n_lifts * lift_h                     # total stack height above GL
z0 = -depth_bg                           # BL (below ground)
# Left outer slope: from BL up to GL+H with slope 1:outer_slope_h
xL0, zL0 = 0.0, z0
xL1, zL1 = H * outer_slope_h, z0 + H     # toe to crest (left)
# Left half of bund crest
xL2, zL2 = xL1 + bund_top_w / 2.0, zL1

# Inner benches (steps) start from bund inner edge:
bx = [xL2]
bz = [zL2]
curx, curz = xL2, zL2
for k in range(n_lifts):
    # Horizontal bench
    curx += bench_w
    bx.append(curx);  bz.append(curz)
    # Vertical riser (schematic); true face would be slope 1:inner_slope_h
    curz += lift_h
    bx.append(curx);  bz.append(curz)

# Optional: cap top width as a fraction of base width (schematic control)
if limit_top_to_base_ratio is not None and limit_top_to_base_ratio > 0:
    base_w  = xL2 - xL0  # half base (outer slope horiz) + half-crest
    max_top = base_w * limit_top_to_base_ratio
    top_w   = max(0.0, bx[-1] - xL2)
    if top_w > max_top:
        # pull back last bench end to respect cap
        shrink = top_w - max_top
        bx[-1] -= shrink
        bx[-2] -= shrink
        curx    = bx[-1]

# Right half of bund crest (mirror)
xR2, zR2 = curx + bund_top_w / 2.0, curz
# Right outer slope down to BL
xR1, zR1 = xR2 + H * outer_slope_h, z0

# Concatenate full polyline (left slope → left crest → benches → right crest → right slope)
X = [xL0, xL1, xL2] + bx[1:] + [xR2, xR1]
Z = [zL0, zL1, zL2] + bz[1:] + [zR2, zR1]

# ------------------ PLOTLY FIG ------------------
fig = go.Figure()

# Landfill stepped profile
fig.add_trace(go.Scatter(x=X, y=Z, mode="lines", name="Landfill Profile"))

# Ground Level (GL) reference
fig.add_trace(go.Scatter(
    x=[min(X)-5, max(X)+5], y=[0,0],
    mode="lines", line=dict(dash="dash"), name="GL"
))

# Labels for key points
fig.add_annotation(x=xL2, y=zL2, text="Bund Crest (Left)", showarrow=True, yshift=12)
fig.add_annotation(x=xR2, y=zR2, text="Bund Crest (Right)", showarrow=True, yshift=12)
fig.add_annotation(x=0, y=z0, text="BL", showarrow=True, yshift=-12)

fig.update_layout(
    title="Landfill Cross Section (Bund + Benches + Outer Slopes)",
    xaxis_title="Horizontal (m)",
    yaxis_title="Elevation (m)",
    xaxis=dict(scaleanchor=None, constrain="domain"),
    template="simple_white",
    width=900, height=420
)

# Make axes a bit roomy
fig.update_xaxes(range=[min(X)-10, max(X)+10], zeroline=True)
fig.update_yaxes(range=[min(-depth_bg-1, min(Z)-2), max(Z)+3], zeroline=True)

fig.show()
