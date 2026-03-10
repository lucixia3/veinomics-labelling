import numpy as np
import plotly.graph_objects as go
from .utils import vox2phys


LABEL_INFO = {
    1:  ("VOG",         "gold"),
    2:  ("STS",         "orange"),
    3:  ("ICV",         "cyan"),
    4:  ("RBVR",        "lightgreen"),
    6:  ("SSS",         "red"),
    7:  ("TransvSig-R", "deepskyblue"),
    8:  ("TransvSig-L", "royalblue"),
    9:  ("Cortical-R",  "magenta"),
    10: ("Cortical-L",  "purple"),
    11: ("VCM-R",       "lime"),
    12: ("VCM-L",       "greenyellow"),
}


def _plane_surface(centroid, normal, half_size=70, n_grid=25):
    t1 = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.array([0.0, 1.0, 0.0])
    else:
        t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2)
    s = np.linspace(-half_size, half_size, n_grid)
    S, T = np.meshgrid(s, s)
    X = centroid[0] + S * t1[0] + T * t2[0]
    Y = centroid[1] + S * t1[1] + T * t2[1]
    Z = centroid[2] + S * t1[2] + T * t2[2]
    return X, Y, Z


def render_html(cleaned, cta, planes, torcular_plane, torcular, z_threshold, xyz_sss, output_path):
    fig = go.Figure()

    for lbl, (name, color) in LABEL_INFO.items():
        pts_vox = np.argwhere(cleaned == lbl)
        if pts_vox.shape[0] == 0:
            continue
        if pts_vox.shape[0] > 6000:
            pts_vox = pts_vox[np.random.choice(pts_vox.shape[0], 6000, replace=False)]
        xyz = vox2phys(pts_vox, cta.origin, cta.spacing, cta.direction)
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers", marker=dict(size=2, color=color, opacity=0.7), name=name,
        ))

    for i, (plane, col) in enumerate(zip(planes, ["rgba(255,80,80,0.20)", "rgba(80,80,255,0.20)"])):
        X, Y, Z = _plane_surface(plane["centroid"], plane["normal"])
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.20,
                                 colorscale=[[0, col], [1, col]], showscale=False,
                                 name=f"SSS plane {i+1}"))

    X, Y, Z = _plane_surface(torcular, torcular_plane["normal"])
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.30,
                             colorscale=[[0, "rgba(255,200,0,0.30)"], [1, "rgba(255,200,0,0.30)"]],
                             showscale=False, name="Torcular plane"))

    fig.add_trace(go.Scatter3d(
        x=[torcular[0]], y=[torcular[1]], z=[torcular[2]],
        mode="markers", marker=dict(size=8, color="yellow", symbol="diamond"),
        name="Torcular Herophili",
    ))

    xg, yg = np.meshgrid(
        np.linspace(xyz_sss[:, 0].min() - 40, xyz_sss[:, 0].max() + 40, 20),
        np.linspace(xyz_sss[:, 1].min() - 40, xyz_sss[:, 1].max() + 40, 20),
    )
    fig.add_trace(go.Surface(x=xg, y=yg, z=np.full_like(xg, z_threshold), opacity=0.18,
                             colorscale=[[0, "rgba(0,200,0,0.18)"], [1, "rgba(0,200,0,0.18)"]],
                             showscale=False, name="Z threshold (torcular)"))

    fig.update_layout(scene=dict(aspectmode="data"), title="Venous segmentation — veinomics-label")
    fig.write_html(output_path)
    print("HTML saved:", output_path)
