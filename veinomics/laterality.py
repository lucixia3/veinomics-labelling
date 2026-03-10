import numpy as np


def sss_laterality_planes(xyz_sss):
    global_centroid = np.mean(xyz_sss, axis=0)
    centered = xyz_sss - global_centroid

    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    main_dir = Vt[0]
    projections = centered @ main_dir
    med_proj = np.median(projections)

    planes = []
    for mask in [projections <= med_proj, projections > med_proj]:
        pts = xyz_sss[mask]
        if pts.shape[0] < 3:
            planes.append({"centroid": global_centroid, "normal": np.array([1.0, 0.0, 0.0])})
            continue
        c = np.mean(pts, axis=0)
        _, _, Vt_half = np.linalg.svd(pts - c, full_matrices=False)
        n = np.cross(Vt_half[0], np.array([0.0, 0.0, 1.0]))
        n_mag = np.linalg.norm(n)
        n = np.array([1.0, 0.0, 0.0]) if n_mag < 1e-6 else n / n_mag
        if n[0] < 0:
            n = -n
        planes.append({"centroid": c, "normal": n})

    torcular = xyz_sss[np.argmin(xyz_sss[:, 2])]
    n_torc = np.cross(main_dir, np.array([0.0, 0.0, 1.0]))
    n_torc_mag = np.linalg.norm(n_torc)
    n_torc = np.array([1.0, 0.0, 0.0]) if n_torc_mag < 1e-6 else n_torc / n_torc_mag
    if n_torc[0] < 0:
        n_torc = -n_torc
    torcular_plane = {"centroid": torcular, "normal": n_torc}

    return planes, torcular_plane, torcular, main_dir, global_centroid, med_proj, float(torcular[2])


def classify_venous(xyz_pt, planes, torcular_plane, torcular, main_dir, global_centroid, med_proj, z_threshold):
    is_above = xyz_pt[2] > z_threshold
    if is_above:
        proj = np.dot(xyz_pt - global_centroid, main_dir)
        plane = planes[0] if proj <= med_proj else planes[1]
        side_val = np.dot(xyz_pt - plane["centroid"], plane["normal"])
    else:
        side_val = np.dot(xyz_pt - torcular, torcular_plane["normal"])
    is_right = side_val > 0
    if is_above:
        return 9 if is_right else 10
    return 7 if is_right else 8
