import os
import subprocess
import numpy as np
import nibabel as nib
import ants
import argparse
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    label as cc_label,
    binary_closing,
)

# ==========================================================
# NNUNET ENVIRONMENT VARIABLES
# ==========================================================

os.environ["nnUNet_raw"] = r"C:\nnunet\nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"C:\nnunet\nnUNet_preprocessed"
os.environ["nnUNet_results"] = r"C:\nnunet\nnUNet_results"

# ==========================================================
# UTILIDADES
# ==========================================================

def run_command(cmd: str) -> None:
    print("\nRunning nnU-Net...\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError("nnU-Net inference failed")


def prepare_input(cta_path: str, work_dir: str, prefix: str) -> str:
    inference_input = os.path.join(work_dir, f"Inference_{prefix}")
    os.makedirs(inference_input, exist_ok=True)
    img = nib.load(cta_path)
    new_path = os.path.join(inference_input, f"{prefix}_0000.nii.gz")
    nib.save(img, new_path)
    return inference_input


def restore_space(original_cta: str, prediction_path: str, output_path: str) -> None:
    original = nib.load(original_cta)
    pred = nib.load(prediction_path)
    new_img = nib.Nifti1Image(pred.get_fdata(), original.affine, original.header)
    nib.save(new_img, output_path)


def vox2phys(ijk, origin, spacing, direction):
    ijk = np.asarray(ijk, dtype=np.float64)
    scaled = ijk * np.array(spacing)
    return scaled @ np.array(direction).T + np.array(origin)


# ==========================================================
# PLANOS SSS PARA LATERALIDAD
# ==========================================================

def sss_laterality_planes(xyz_sss):
    """
    Calcula dos planos de lateralidad a partir de la geometría del SSS.

    El SSS se divide por su punto medio (según su eje principal) en dos mitades:
    - Mitad 1: desde un extremo hasta el centro
    - Mitad 2: desde el centro hasta el otro extremo

    Para cada mitad se ajusta un plano óptimo (SVD sobre todos los puntos
    de esa mitad), maximizando el número de puntos que toca.

    Returns:
        planes       : lista de 2 dict con 'centroid', 'normal'
        main_dir     : dirección principal del SSS (vector unitario)
        global_centroid : centroide global del SSS
        med_proj     : valor de corte de proyección (separa las dos mitades)
    """
    global_centroid = np.mean(xyz_sss, axis=0)
    centered = xyz_sss - global_centroid

    # Dirección principal del SSS completo
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    main_dir = Vt[0]

    # Proyectar todos los puntos sobre el eje principal
    projections = centered @ main_dir
    med_proj = np.median(projections)

    mask1 = projections <= med_proj
    mask2 = projections > med_proj

    planes = []
    for mask in [mask1, mask2]:
        pts = xyz_sss[mask]
        if pts.shape[0] < 3:
            # Fallback: plano sagital simple
            planes.append({
                "centroid": global_centroid,
                "normal": np.array([1.0, 0.0, 0.0]),
            })
            continue

        c = np.mean(pts, axis=0)
        _, _, Vt_half = np.linalg.svd(pts - c, full_matrices=False)
        v1 = Vt_half[0]  # dirección principal de esta mitad del SSS

        # Normal al plano: v1 × Z_hat
        # El SSS corre en Y (AP), así que v1×Z ≈ X → separa izquierda/derecha
        z_hat = np.array([0.0, 0.0, 1.0])
        n = np.cross(v1, z_hat)
        n_mag = np.linalg.norm(n)
        if n_mag < 1e-6:
            n = np.array([1.0, 0.0, 0.0])
        else:
            n = n / n_mag

        # Convención: normal apunta hacia +X (derecha del paciente)
        if n[0] < 0:
            n = -n

        planes.append({"centroid": c, "normal": n})

    # Tórcula = voxel del SSS con menor Z (el punto más inferior del seno)
    torcular = xyz_sss[np.argmin(xyz_sss[:, 2])]

    # Normal del plano torcular: usa la dirección GLOBAL del SSS (más robusta que SVD local)
    # SSS corre en Y (AP) → main_dir ≈ [0,±1,0] → main_dir × Z_hat ≈ [±1,0,0]  (izq/der)
    z_hat = np.array([0.0, 0.0, 1.0])
    n_torc = np.cross(main_dir, z_hat)
    n_torc_mag = np.linalg.norm(n_torc)
    if n_torc_mag < 1e-6:
        n_torc = np.array([1.0, 0.0, 0.0])
    else:
        n_torc = n_torc / n_torc_mag
    if n_torc[0] < 0:
        n_torc = -n_torc

    # El plano pasa por la tórcula con esa normal
    torcular_plane = {"centroid": torcular, "normal": n_torc}

    # Umbral horizontal: Z de la tórcula
    z_threshold = float(torcular[2])

    return planes, torcular_plane, torcular, main_dir, global_centroid, med_proj, z_threshold


def classify_venous(xyz_pt, planes, torcular_plane, torcular, main_dir, global_centroid, med_proj, z_threshold):
    """
    Clasifica un punto venoso genérico en uno de 4 labels:

    Por ENCIMA del Z de la tórcula → venas corticales
        Lateralidad: plano del half del SSS más cercano (9=der, 10=izq)

    Por DEBAJO del Z de la tórcula → transverso+sigmoide
        Lateralidad: plano de la tórcula con centroide = endpoint del SSS (7=der, 8=izq)
    """
    is_above = xyz_pt[2] > z_threshold

    if is_above:
        # Corticales: usar el half del SSS al que proyecta el punto
        proj = np.dot(xyz_pt - global_centroid, main_dir)
        plane = planes[0] if proj <= med_proj else planes[1]
        side_val = np.dot(xyz_pt - plane["centroid"], plane["normal"])
    else:
        # Transverso+sigmoide: usar el plano centrado en la tórcula
        side_val = np.dot(xyz_pt - torcular, torcular_plane["normal"])

    is_right = side_val > 0

    if is_above and is_right:
        return 9
    elif is_above and not is_right:
        return 10
    elif not is_above and is_right:
        return 7
    else:
        return 8


# ==========================================================
# INFERENCIA ARTERIA/VENA
# ==========================================================

def run_artven(inference_input: str, output_dir: str, dataset_id: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        f'nnUNetv2_predict '
        f'-i "{inference_input}" '
        f'-o "{output_dir}" '
        f'-d {dataset_id} '
        f'-c 3d_fullres '
        f'-p nnUNetResEncUNetLPlans '
        f'-chk checkpoint_best.pth '
        f'-f all '
        f'-device cuda'
    )
    run_command(cmd)


# ==========================================================
# INFERENCIA TOPBRAIN
# ==========================================================

def run_topbrain(inference_input: str, output_dir: str, dataset_id: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        f'nnUNetv2_predict '
        f'-i "{inference_input}" '
        f'-o "{output_dir}" '
        f'-d {dataset_id} '
        f'-c 3d_fullres '
        f'-p nnUNetResEncUNetLPlans '
        f'-chk checkpoint_best.pth '
        f'-f all '
        f'-device cuda'
    )
    run_command(cmd)


# ==========================================================
# RENDER HTML CON PLANOS
# ==========================================================

LABEL_INFO = {
    1:  ("VOG",          "gold"),
    2:  ("STS",          "orange"),
    3:  ("ICV",          "cyan"),
    4:  ("RBVR",         "lightgreen"),
    6:  ("SSS",          "red"),
    7:  ("TransvSig-R",  "deepskyblue"),
    8:  ("TransvSig-L",  "royalblue"),
    9:  ("Cortical-R",   "magenta"),
    10: ("Cortical-L",   "purple"),
}


def _plane_surface(centroid, normal, half_size=70, n_grid=25):
    """Devuelve arrays X, Y, Z para un go.Surface de un plano infinito."""
    z_hat = np.array([0.0, 0.0, 1.0])
    t1 = np.cross(normal, z_hat)
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
    import plotly.graph_objects as go
    fig = go.Figure()

    # --- Estructuras segmentadas ---
    for lbl, (name, color) in LABEL_INFO.items():
        pts_vox = np.argwhere(cleaned == lbl)
        if pts_vox.shape[0] == 0:
            continue
        if pts_vox.shape[0] > 6000:
            idx = np.random.choice(pts_vox.shape[0], 6000, replace=False)
            pts_vox = pts_vox[idx]
        xyz = vox2phys(pts_vox, cta.origin, cta.spacing, cta.direction)
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers",
            marker=dict(size=2, color=color, opacity=0.7),
            name=name,
        ))

    # --- Planos SSS (lateralidad corticales) ---
    plane_colors = ["rgba(255,80,80,0.20)", "rgba(80,80,255,0.20)"]
    for i, (plane, pcolor) in enumerate(zip(planes, plane_colors)):
        X, Y, Z = _plane_surface(plane["centroid"], plane["normal"])
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.20,
            colorscale=[[0, pcolor], [1, pcolor]],
            showscale=False,
            name=f"Plano SSS mitad {i+1}",
        ))

    # --- Plano de la tórcula (lateralidad transversos) ---
    X, Y, Z = _plane_surface(torcular, torcular_plane["normal"])
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        opacity=0.30,
        colorscale=[[0, "rgba(255,200,0,0.30)"], [1, "rgba(255,200,0,0.30)"]],
        showscale=False,
        name="Plano tórcula (transversos)",
    ))

    # --- Punto tórcula ---
    fig.add_trace(go.Scatter3d(
        x=[torcular[0]], y=[torcular[1]], z=[torcular[2]],
        mode="markers",
        marker=dict(size=8, color="yellow", symbol="diamond"),
        name="Tórcula (endpoint SSS)",
    ))

    # --- Plano horizontal Z (arriba=corticales / abajo=transverso) ---
    x_min, x_max = xyz_sss[:, 0].min() - 40, xyz_sss[:, 0].max() + 40
    y_min, y_max = xyz_sss[:, 1].min() - 40, xyz_sss[:, 1].max() + 40
    xg, yg = np.meshgrid(
        np.linspace(x_min, x_max, 20),
        np.linspace(y_min, y_max, 20),
    )
    zg = np.full_like(xg, z_threshold)
    fig.add_trace(go.Surface(
        x=xg, y=yg, z=zg,
        opacity=0.18,
        colorscale=[[0, "rgba(0,200,0,0.18)"], [1, "rgba(0,200,0,0.18)"]],
        showscale=False,
        name="Plano horizontal Z (tórcula)",
    ))

    fig.update_layout(
        scene=dict(aspectmode="data"),
        title="Segmentación venosa + planos SSS",
    )
    fig.write_html(output_path)
    print("HTML guardado:", output_path)


# ==========================================================
# FUSIÓN: ARTVEN + TOPBRAIN + LATERALIDAD POR PLANOS SSS
# ==========================================================

def fuse_results(cta_path: str, artven_path: str, topbrain_path: str, final_output: str) -> None:
    print("\n=== FUSIÓN ARTVEN + TOPBRAIN + LATERALIDAD SSS ===")

    cta      = ants.image_read(cta_path)
    artven   = ants.image_read(artven_path)
    topbrain = ants.image_read(topbrain_path)

    art_arr = artven.numpy()
    top_arr = topbrain.numpy()

    # Máscara venosa desde artven (label 2)
    ven_raw = (art_arr == 2)

    # Labels Topbrain (1–6)
    LBL_VOG  = 1
    LBL_STS  = 2
    LBL_ICV  = 3
    LBL_RBVR = 4
    LBL_LBVR = 5
    LBL_SSS  = 6

    # Label placeholder para venoso genérico (se reclasifica con planos)
    LBL_VEN_PENDING = 7

    print("Labels únicos en Topbrain:", np.unique(top_arr))

    final = np.zeros_like(top_arr, dtype=np.uint8)
    structure = np.ones((3, 3, 3))

    # -------------------------------------------------------
    # 1) Expandir Topbrain (dilatación 1 voxel)
    # -------------------------------------------------------
    expanded_top = np.zeros_like(top_arr)
    icv_dilated_mask = binary_dilation(top_arr == LBL_ICV, structure=structure, iterations=1)

    for label_val in [LBL_VOG, LBL_STS, LBL_ICV, LBL_RBVR, LBL_LBVR, LBL_SSS]:
        mask = (top_arr == label_val)
        if np.sum(mask) == 0:
            continue
        dilated = binary_dilation(mask, structure=structure, iterations=1)
        expanded_top[dilated] = label_val

    expanded_top[icv_dilated_mask] = LBL_ICV
    final[expanded_top > 0] = expanded_top[expanded_top > 0]

    # -------------------------------------------------------
    # 2) Relleno por distancia: venosos sin asignar cerca de Topbrain
    # -------------------------------------------------------
    remaining = ven_raw & (final == 0)

    if np.sum(expanded_top > 0) > 0:
        dist, indices = distance_transform_edt(
            expanded_top == 0,
            return_indices=True
        )
        sx, sy, sz = indices
        FILL_DISTANCE = 3

        for i, j, k in np.argwhere(remaining):
            if dist[i, j, k] <= FILL_DISTANCE:
                nearest_label = expanded_top[sx[i, j, k], sy[i, j, k], sz[i, j, k]]
                if nearest_label == LBL_VOG:
                    nearest_label = LBL_VEN_PENDING
                final[i, j, k] = nearest_label

    # -------------------------------------------------------
    # 3) Venosos restantes → pendiente de clasificar
    # -------------------------------------------------------
    remaining2 = ven_raw & (final == 0)
    final[remaining2] = LBL_VEN_PENDING

    # -------------------------------------------------------
    # 4) Limpieza de islas pequeñas
    # -------------------------------------------------------
    cleaned = np.zeros_like(final)
    MIN_ISLAND_SIZE = 200

    for label_val in np.unique(final):
        if label_val == 0:
            continue
        mask = (final == label_val)
        mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
        comp, ncomp = cc_label(mask)
        for comp_id in range(1, ncomp + 1):
            region = (comp == comp_id)
            if np.sum(region) >= MIN_ISLAND_SIZE:
                cleaned[region] = label_val

    cleaned[np.round(top_arr).astype(np.int32) == int(LBL_ICV)] = LBL_ICV

    # Componentes pendientes que tocan ICV o RBVR → absorber
    pend_mask = (cleaned == LBL_VEN_PENDING)
    if np.any(pend_mask):
        comp7, ncomp7 = cc_label(pend_mask, structure=np.ones((3, 3, 3)))
        for comp_id in range(1, ncomp7 + 1):
            comp_mask = (comp7 == comp_id)
            if not np.any(comp_mask):
                continue
            comp_dil = binary_dilation(comp_mask, structure=np.ones((3, 3, 3)), iterations=1)
            touches_3 = np.any(comp_dil & (cleaned == LBL_ICV))
            touches_4 = np.any(comp_dil & (cleaned == LBL_RBVR))
            touches_1 = np.any(comp_dil & (cleaned == LBL_VOG))

            if touches_3:
                cleaned[comp_mask] = LBL_ICV
            elif touches_1 or touches_4:
                cleaned[comp_mask] = LBL_RBVR

    # -------------------------------------------------------
    # 5) Label 5 → 4 (sin lateralidad para RBVR/LBVR)
    # -------------------------------------------------------
    cleaned[cleaned == LBL_LBVR] = LBL_RBVR

    # -------------------------------------------------------
    # 6) Lateralidad por planos SSS — solo para venoso genérico pendiente
    # -------------------------------------------------------
    sss_pts_vox = np.argwhere(top_arr == LBL_SSS)
    xyz_sss = None
    planes = None

    if sss_pts_vox.shape[0] >= 6:
        xyz_sss = vox2phys(sss_pts_vox, cta.origin, cta.spacing, cta.direction)
        planes, torcular_plane, torcular, main_dir, global_centroid, med_proj, z_threshold = \
            sss_laterality_planes(xyz_sss)

        print(f"  Plano SSS mitad 1 — normal: {planes[0]['normal'].round(3)}, "
              f"centroide: {planes[0]['centroid'].round(1)}")
        print(f"  Plano SSS mitad 2 — normal: {planes[1]['normal'].round(3)}, "
              f"centroide: {planes[1]['centroid'].round(1)}")
        print(f"  Tórcula (endpoint SSS): {torcular.round(1)}")
        print(f"  Umbral horizontal Z (tórcula): {z_threshold:.1f} mm")

        # Clasificar venoso pendiente → 7/8/9/10
        pts_vox = np.argwhere(cleaned == LBL_VEN_PENDING)
        if pts_vox.shape[0] > 0:
            xyz_pts = vox2phys(pts_vox, cta.origin, cta.spacing, cta.direction)
            for idx, (i, j, k) in enumerate(pts_vox):
                cleaned[i, j, k] = classify_venous(
                    xyz_pts[idx], planes, torcular_plane, torcular,
                    main_dir, global_centroid, med_proj, z_threshold
                )

        # -------------------------------------------------------
        # 7) Expansión de transversos: componentes conexas sobre
        #    venoso genérico (7,8,9,10). Si un componente toca
        #    label 7 u 8 → todo el componente se convierte en 7/8.
        #    EXCEPCIÓN: si el componente toca el SSS fuera de la
        #    tórcula → es una cortical mal clasificada; se corrigen
        #    sus voxels 7/8 a 9/10 en lugar de expandir el transverso.
        # -------------------------------------------------------

        TORC_RADIUS_MM = 15.0
        dist_to_torc = np.linalg.norm(xyz_sss - torcular, axis=1)
        sss_mid_vox  = sss_pts_vox[dist_to_torc > TORC_RADIUS_MM]
        sss_mid_arr  = np.zeros(cleaned.shape, dtype=bool)
        if sss_mid_vox.shape[0] > 0:
            sss_mid_arr[sss_mid_vox[:, 0], sss_mid_vox[:, 1], sss_mid_vox[:, 2]] = True
        sss_mid_dilated = binary_dilation(sss_mid_arr, structure=np.ones((3, 3, 3)), iterations=2)

        generic_mask = np.isin(cleaned, [7, 8, 9, 10])
        comp_arr, n_comp = cc_label(generic_mask, structure=np.ones((3, 3, 3)))

        for comp_id in range(1, n_comp + 1):
            comp_mask = comp_arr == comp_id
            has_transv = np.any(comp_mask & np.isin(cleaned, [7, 8]))
            if not has_transv:
                continue

            pts_c = np.argwhere(comp_mask)
            xyz_c = vox2phys(pts_c, cta.origin, cta.spacing, cta.direction)

            # Si el componente toca SSS fuera de la tórcula → cortical mal clasificada:
            # convertir solo los voxels 7/8 del componente a 9/10
            if np.any(comp_mask & sss_mid_dilated):
                for idx, (i, j, k) in enumerate(pts_c):
                    if cleaned[i, j, k] in [7, 8]:
                        proj     = np.dot(xyz_c[idx] - global_centroid, main_dir)
                        plane    = planes[0] if proj <= med_proj else planes[1]
                        side_val = np.dot(xyz_c[idx] - plane["centroid"], plane["normal"])
                        cleaned[i, j, k] = 9 if side_val > 0 else 10
                continue

            # Si no toca SSS-medio → transverso real: convertir todo el componente a 7/8
            for idx, (i, j, k) in enumerate(pts_c):
                side_val = np.dot(xyz_c[idx] - torcular, torcular_plane["normal"])
                cleaned[i, j, k] = 7 if side_val > 0 else 8

        # -------------------------------------------------------
        # 8) Componentes de 7/8 que tocan el SSS en el medio (no tórcula)
        #    → son corticales mal clasificadas → convertir a 9/10
        # -------------------------------------------------------

        TORC_RADIUS_MM = 15.0

        dist_to_torc = np.linalg.norm(xyz_sss - torcular, axis=1)
        sss_mid_vox  = sss_pts_vox[dist_to_torc > TORC_RADIUS_MM]

        sss_mid_arr = np.zeros(cleaned.shape, dtype=bool)
        if sss_mid_vox.shape[0] > 0:
            sss_mid_arr[sss_mid_vox[:, 0], sss_mid_vox[:, 1], sss_mid_vox[:, 2]] = True
        sss_mid_dilated = binary_dilation(sss_mid_arr, structure=np.ones((3, 3, 3)), iterations=2)

        transv_mask = np.isin(cleaned, [7, 8])
        comp_tr, n_tr = cc_label(transv_mask, structure=np.ones((3, 3, 3)))

        for comp_id in range(1, n_tr + 1):
            comp_mask = comp_tr == comp_id
            # Si toca el medio del SSS (fuera de la tórcula) → cortical
            if not np.any(comp_mask & sss_mid_dilated):
                continue
            pts_c = np.argwhere(comp_mask)
            xyz_c = vox2phys(pts_c, cta.origin, cta.spacing, cta.direction)
            for idx, (i, j, k) in enumerate(pts_c):
                proj     = np.dot(xyz_c[idx] - global_centroid, main_dir)
                plane    = planes[0] if proj <= med_proj else planes[1]
                side_val = np.dot(xyz_c[idx] - plane["centroid"], plane["normal"])
                cleaned[i, j, k] = 9 if side_val > 0 else 10

    else:
        print("  AVISO: SSS con pocos voxels — no se aplica lateralidad por planos.")
        cleaned[cleaned == LBL_VEN_PENDING] = 7

    # -------------------------------------------------------
    # PASO FINAL: componentes 7/8 que tocan el SSS (en cleaned)
    # fuera de la tórcula → son corticales → convertir a 9/10
    # Usamos cleaned==6 (SSS expandido) en vez de top_arr,
    # porque el proximity fill ya acercó el SSS a la cortical.
    # -------------------------------------------------------
    if planes is not None and xyz_sss is not None:
        sss_clean = (cleaned == 6)

        # Excluir zona tórcula del SSS
        sss_all_vox = np.argwhere(sss_clean)
        if sss_all_vox.shape[0] > 0:
            xyz_sss_clean = vox2phys(sss_all_vox, cta.origin, cta.spacing, cta.direction)
            dist_torc = np.linalg.norm(xyz_sss_clean - torcular, axis=1)
            sss_mid_only = np.zeros(cleaned.shape, dtype=bool)
            mid_vox = sss_all_vox[dist_torc > TORC_RADIUS_MM]
            if mid_vox.shape[0] > 0:
                sss_mid_only[mid_vox[:, 0], mid_vox[:, 1], mid_vox[:, 2]] = True
            sss_mid_final = binary_dilation(sss_mid_only, structure=np.ones((3, 3, 3)), iterations=3)

            transv_final = np.isin(cleaned, [7, 8])
            comp_f, n_f = cc_label(transv_final, structure=np.ones((3, 3, 3)))

            # Tamaño del componente de transverso más grande (referencia)
            comp_f_sizes = [int((comp_f == cid).sum()) for cid in range(1, n_f + 1)]
            max_transv_size = max(comp_f_sizes) if comp_f_sizes else 1

            for comp_id in range(1, n_f + 1):
                comp_mask = comp_f == comp_id
                # Solo convertir componentes claramente pequeños respecto al transverso real
                # (los transversos reales son grandes aunque toquen el SSS)
                if comp_mask.sum() > max_transv_size * 0.3:
                    continue
                if not np.any(comp_mask & sss_mid_final):
                    continue
                pts_c = np.argwhere(comp_mask)
                xyz_c = vox2phys(pts_c, cta.origin, cta.spacing, cta.direction)
                for idx, (i, j, k) in enumerate(pts_c):
                    proj     = np.dot(xyz_c[idx] - global_centroid, main_dir)
                    plane    = planes[0] if proj <= med_proj else planes[1]
                    side_val = np.dot(xyz_c[idx] - plane["centroid"], plane["normal"])
                    cleaned[i, j, k] = 9 if side_val > 0 else 10

    # -------------------------------------------------------
    # PASO FINAL 2: fragmentos de transverso que tocan el
    # segmento grande del transverso CONTRARIO → absorber
    # con la lateralidad correcta (la del grande).
    # No se cuenta la unión con SSS: solo contacto directo
    # entre componentes 7/8.
    # -------------------------------------------------------
    transv_mask2 = np.isin(cleaned, [7, 8])
    comp_t2, n_t2 = cc_label(transv_mask2, structure=np.ones((3, 3, 3)))

    # Tamaño y label dominante de cada componente
    comp_sizes  = {}
    comp_labels = {}
    for cid in range(1, n_t2 + 1):
        mask = comp_t2 == cid
        comp_sizes[cid]  = int(mask.sum())
        vals = cleaned[mask]
        comp_labels[cid] = int(np.bincount(vals).argmax())  # 7 o 8

    # Para cada componente pequeño, ver si toca un componente más grande
    # de distinto label (excluyendo contacto a través del SSS)
    sss_mask_excl = binary_dilation((cleaned == 6), structure=np.ones((3, 3, 3)), iterations=1)

    for cid in range(1, n_t2 + 1):
        comp_mask = comp_t2 == cid
        comp_dil  = binary_dilation(comp_mask, structure=np.ones((3, 3, 3)), iterations=1)
        # Vecinos 7/8 que NO son este componente y NO pasan por SSS
        neighbors = comp_dil & transv_mask2 & ~comp_mask & ~sss_mask_excl

        touching_ids = np.unique(comp_t2[neighbors])
        touching_ids = touching_ids[touching_ids > 0]
        if len(touching_ids) == 0:
            continue

        # Componente vecino más grande
        biggest_neighbor = max(touching_ids, key=lambda x: comp_sizes[x])
        if comp_sizes[biggest_neighbor] <= comp_sizes[cid]:
            continue  # el vecino no es más grande, no absorber

        neighbor_label = comp_labels[biggest_neighbor]
        if neighbor_label == comp_labels[cid]:
            continue  # misma lateralidad, no hay nada que corregir

        # Absorber: asignar la lateralidad del vecino grande
        cleaned[comp_mask] = neighbor_label

    # -------------------------------------------------------
    # PASO FINAL 3: corticales (9/10) más cercanas a un
    # transverso que al SSS → convertir a transverso (7/8)
    # -------------------------------------------------------
    if planes is not None:
        cortical_mask = np.isin(cleaned, [9, 10])
        transv_mask3  = np.isin(cleaned, [7, 8])
        sss_mask3     = (cleaned == 6)

        # Distancia de cada voxel al SSS y al transverso más cercano
        dist_to_sss    = distance_transform_edt(~sss_mask3)
        dist_to_transv = distance_transform_edt(~transv_mask3)

        comp_cort, n_cort = cc_label(cortical_mask, structure=np.ones((3, 3, 3)))

        for cid in range(1, n_cort + 1):
            comp_mask = comp_cort == cid

            # Distancia mínima del componente al SSS y al transverso
            min_dist_sss    = dist_to_sss[comp_mask].min()
            min_dist_transv = dist_to_transv[comp_mask].min()

            if min_dist_transv >= min_dist_sss:
                continue  # está más cerca del SSS → es cortical, dejar como está

            # Más cerca del transverso → convertir a transverso con lateralidad
            pts_c = np.argwhere(comp_mask)
            xyz_c = vox2phys(pts_c, cta.origin, cta.spacing, cta.direction)
            for idx, (i, j, k) in enumerate(pts_c):
                side_val = np.dot(xyz_c[idx] - torcular, torcular_plane["normal"])
                cleaned[i, j, k] = 7 if side_val > 0 else 8

    # -------------------------------------------------------
    # Guardar NIfTI
    # -------------------------------------------------------
    final_img = ants.from_numpy(
        cleaned.astype(np.uint8),
        origin=cta.origin,
        spacing=cta.spacing,
        direction=cta.direction
    )
    ants.image_write(final_img, final_output)
    print("\n✅ Fusión guardada en:", final_output)
    print("Labels en resultado:", np.unique(cleaned))
    print("  1=VOG  2=STS  3=ICV  4=RBVR  6=SSS")
    print("  7=TransvSig-R  8=TransvSig-L  9=CorticalR  10=CorticalL")

    # -------------------------------------------------------
    # Guardar HTML con planos
    # -------------------------------------------------------
    if planes is not None and xyz_sss is not None:
        html_path = final_output.replace(".nii.gz", "_planes.html").replace(".nii", "_planes.html")
        render_html(cleaned, cta, planes, torcular_plane, torcular, z_threshold, xyz_sss, html_path)


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cta", help="Ruta a un archivo CTA individual (.nii o .nii.gz)")
    parser.add_argument("--input_folder", help="Carpeta con imágenes CTA (.nii.gz)")
    parser.add_argument("--output", required=True, help="Carpeta de salida")
    parser.add_argument("--dataset_artven", type=int, required=False)
    parser.add_argument("--dataset_topbrain", type=int, required=False)
    parser.add_argument("--skip_nnunet", action="store_true")
    args = parser.parse_args()

    # ----------------------------------------------------------
    # Modo carpeta
    # ----------------------------------------------------------
    if args.input_folder:
        output_folder = args.output
        os.makedirs(output_folder, exist_ok=True)
        cta_files = [f for f in os.listdir(args.input_folder) if f.endswith(".nii.gz") or f.endswith(".nii")]
        if not cta_files:
            raise FileNotFoundError(f"No se encontraron archivos .nii.gz en {args.input_folder}")

        for cta_file in cta_files:
            cta_path  = os.path.join(args.input_folder, cta_file)
            case_name = os.path.splitext(os.path.splitext(cta_file)[0])[0]
            case_out  = os.path.join(output_folder, case_name)
            work_dir  = os.path.join(case_out, "temp_work")
            os.makedirs(work_dir, exist_ok=True)

            if not args.skip_nnunet:
                print(f"\n=== NNUNET ARTERIA/VENA: {cta_file} ===")
                input_art    = prepare_input(cta_path, work_dir, "ARTVEN")
                pred_art_dir = os.path.join(work_dir, "Pred_ARTVEN")
                run_artven(input_art, pred_art_dir, args.dataset_artven)
                art_files = [f for f in os.listdir(pred_art_dir) if f.endswith(".nii.gz")]
                if not art_files:
                    raise FileNotFoundError(f"No .nii.gz found in {pred_art_dir}")
                art_output = os.path.join(work_dir, "CTA_artven.nii.gz")
                restore_space(cta_path, os.path.join(pred_art_dir, art_files[0]), art_output)

                print(f"\n=== NNUNET TOPBRAIN: {cta_file} ===")
                input_top    = prepare_input(cta_path, work_dir, "TOPBRAIN")
                pred_top_dir = os.path.join(work_dir, "Pred_TOPBRAIN")
                run_topbrain(input_top, pred_top_dir, args.dataset_topbrain)
                top_files = [f for f in os.listdir(pred_top_dir) if f.endswith(".nii.gz")]
                if not top_files:
                    raise FileNotFoundError(f"No .nii.gz found in {pred_top_dir}")
                top_output = os.path.join(work_dir, "CTA_topbrain.nii.gz")
                restore_space(cta_path, os.path.join(pred_top_dir, top_files[0]), top_output)
            else:
                print(f"\n⏩ SKIPPING NNUNET — usando segmentaciones existentes para {cta_file}")
                art_output = os.path.join(work_dir, "CTA_artven.nii.gz")
                top_output = os.path.join(work_dir, "CTA_topbrain.nii.gz")
                if not os.path.exists(art_output):
                    raise FileNotFoundError(f"Missing: {art_output}")
                if not os.path.exists(top_output):
                    raise FileNotFoundError(f"Missing: {top_output}")

            final_output = os.path.join(case_out, "CTA_final_hybrid.nii.gz")
            fuse_results(cta_path, art_output, top_output, final_output)
            print(f"\n=== PIPELINE COMPLETO: {cta_file} ===")

    # ----------------------------------------------------------
    # Modo archivo único
    # ----------------------------------------------------------
    elif args.cta:
        work_dir = os.path.join(args.output, "temp_work")
        os.makedirs(work_dir, exist_ok=True)

        if not args.skip_nnunet:
            print("\n=== NNUNET ARTERIA/VENA ===")
            input_art    = prepare_input(args.cta, work_dir, "ARTVEN")
            pred_art_dir = os.path.join(work_dir, "Pred_ARTVEN")
            run_artven(input_art, pred_art_dir, args.dataset_artven)
            art_files = [f for f in os.listdir(pred_art_dir) if f.endswith(".nii.gz")]
            if not art_files:
                raise FileNotFoundError(f"No .nii.gz found in {pred_art_dir}")
            art_output = os.path.join(work_dir, "CTA_artven.nii.gz")
            restore_space(args.cta, os.path.join(pred_art_dir, art_files[0]), art_output)

            print("\n=== NNUNET TOPBRAIN ===")
            input_top    = prepare_input(args.cta, work_dir, "TOPBRAIN")
            pred_top_dir = os.path.join(work_dir, "Pred_TOPBRAIN")
            run_topbrain(input_top, pred_top_dir, args.dataset_topbrain)
            top_files = [f for f in os.listdir(pred_top_dir) if f.endswith(".nii.gz")]
            if not top_files:
                raise FileNotFoundError(f"No .nii.gz found in {pred_top_dir}")
            top_output = os.path.join(work_dir, "CTA_topbrain.nii.gz")
            restore_space(args.cta, os.path.join(pred_top_dir, top_files[0]), top_output)
        else:
            print("\n⏩ SKIPPING NNUNET — usando segmentaciones existentes")
            art_output = os.path.join(work_dir, "CTA_artven.nii.gz")
            top_output = os.path.join(work_dir, "CTA_topbrain.nii.gz")
            if not os.path.exists(art_output):
                raise FileNotFoundError(f"Missing: {art_output}")
            if not os.path.exists(top_output):
                raise FileNotFoundError(f"Missing: {top_output}")

        final_output = os.path.join(args.output, "CTA_final_hybrid.nii.gz")
        fuse_results(args.cta, art_output, top_output, final_output)
        print("\n=== PIPELINE COMPLETO TERMINADO ===")

    else:
        print("Debe especificar --cta o --input_folder")
        exit(1)


if __name__ == "__main__":
    main()
