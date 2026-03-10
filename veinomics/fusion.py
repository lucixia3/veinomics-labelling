import numpy as np
import ants
from scipy.ndimage import (
    binary_dilation, distance_transform_edt,
    label as cc_label, binary_closing,
)

from .utils import vox2phys
from .laterality import sss_laterality_planes, classify_venous
from .vcm import detect_vcm
from .visualization import render_html


LBL_VOG  = 1
LBL_STS  = 2
LBL_ICV  = 3
LBL_RBVR = 4
LBL_LBVR = 5
LBL_SSS  = 6
LBL_PENDING = 7


def fuse_results(cta_path, artven_path, topbrain_path, final_output):
    print("\n=== FUSION ARTVEN + TOPBRAIN + SSS LATERALITY ===")

    cta     = ants.image_read(cta_path)
    art_arr = ants.image_read(artven_path).numpy()
    top_arr = ants.image_read(topbrain_path).numpy()

    ven_raw = (art_arr == 2)
    struct3 = np.ones((3, 3, 3))

    print("Topbrain labels:", np.unique(top_arr))

    expanded_top      = np.zeros_like(top_arr)
    icv_dilated       = binary_dilation(top_arr == LBL_ICV, structure=struct3)
    for lv in [LBL_VOG, LBL_STS, LBL_ICV, LBL_RBVR, LBL_LBVR, LBL_SSS]:
        mask = top_arr == lv
        if np.any(mask):
            expanded_top[binary_dilation(mask, structure=struct3)] = lv
    expanded_top[icv_dilated] = LBL_ICV

    final = np.zeros_like(top_arr)
    final[expanded_top > 0] = expanded_top[expanded_top > 0]

    remaining = ven_raw & (final == 0)
    if np.any(expanded_top > 0):
        dist, indices = distance_transform_edt(expanded_top == 0, return_indices=True)
        sx, sy, sz = indices
        for i, j, k in np.argwhere(remaining):
            if dist[i, j, k] <= 3:
                nl = expanded_top[sx[i, j, k], sy[i, j, k], sz[i, j, k]]
                if nl == LBL_VOG:
                    nl = LBL_PENDING
                final[i, j, k] = nl

    final[ven_raw & (final == 0)] = LBL_PENDING

    cleaned = np.zeros_like(final)
    for lv in np.unique(final):
        if lv == 0:
            continue
        mask   = binary_closing(final == lv, structure=struct3)
        comp, ncomp = cc_label(mask)
        for cid in range(1, ncomp + 1):
            region = comp == cid
            if region.sum() >= 200:
                cleaned[region] = lv

    cleaned[np.round(top_arr).astype(np.int32) == LBL_ICV] = LBL_ICV

    pend_mask = cleaned == LBL_PENDING
    if np.any(pend_mask):
        comp7, ncomp7 = cc_label(pend_mask, structure=struct3)
        for cid in range(1, ncomp7 + 1):
            cm = comp7 == cid
            if not np.any(cm):
                continue
            cdil = binary_dilation(cm, structure=struct3)
            if np.any(cdil & (cleaned == LBL_ICV)):
                cleaned[cm] = LBL_ICV
            elif np.any(cdil & (cleaned == LBL_VOG)) or np.any(cdil & (cleaned == LBL_RBVR)):
                cleaned[cm] = LBL_RBVR

    cleaned[cleaned == LBL_LBVR] = LBL_RBVR

    sss_pts_vox = np.argwhere(top_arr == LBL_SSS)
    planes = None
    xyz_sss = None

    if sss_pts_vox.shape[0] >= 6:
        xyz_sss = vox2phys(sss_pts_vox, cta.origin, cta.spacing, cta.direction)
        planes, torcular_plane, torcular, main_dir, global_centroid, med_proj, z_threshold = \
            sss_laterality_planes(xyz_sss)

        print(f"  SSS half-1 normal: {planes[0]['normal'].round(3)}")
        print(f"  SSS half-2 normal: {planes[1]['normal'].round(3)}")
        print(f"  Torcular: {torcular.round(1)}  Z-threshold: {z_threshold:.1f} mm")

        pts_vox = np.argwhere(cleaned == LBL_PENDING)
        if pts_vox.shape[0] > 0:
            xyz_pts = vox2phys(pts_vox, cta.origin, cta.spacing, cta.direction)
            for idx, (i, j, k) in enumerate(pts_vox):
                cleaned[i, j, k] = classify_venous(
                    xyz_pts[idx], planes, torcular_plane, torcular,
                    main_dir, global_centroid, med_proj, z_threshold,
                )

        TORC_RADIUS_MM = 15.0

        def _sss_mid_dilated(n_iter=2):
            dist_torc = np.linalg.norm(xyz_sss - torcular, axis=1)
            mid_vox   = sss_pts_vox[dist_torc > TORC_RADIUS_MM]
            arr = np.zeros(cleaned.shape, dtype=bool)
            if mid_vox.shape[0] > 0:
                arr[mid_vox[:, 0], mid_vox[:, 1], mid_vox[:, 2]] = True
            return binary_dilation(arr, structure=struct3, iterations=n_iter)

        sss_mid = _sss_mid_dilated(2)

        generic_mask  = np.isin(cleaned, [7, 8, 9, 10])
        comp_arr, n_c = cc_label(generic_mask, structure=struct3)
        for cid in range(1, n_c + 1):
            cm = comp_arr == cid
            if not np.any(cm & np.isin(cleaned, [7, 8])):
                continue
            pts_c = np.argwhere(cm)
            xyz_c = vox2phys(pts_c, cta.origin, cta.spacing, cta.direction)
            if np.any(cm & sss_mid):
                for idx, (i, j, k) in enumerate(pts_c):
                    if cleaned[i, j, k] in [7, 8]:
                        proj  = np.dot(xyz_c[idx] - global_centroid, main_dir)
                        pl    = planes[0] if proj <= med_proj else planes[1]
                        sv    = np.dot(xyz_c[idx] - pl["centroid"], pl["normal"])
                        cleaned[i, j, k] = 9 if sv > 0 else 10
            else:
                for idx, (i, j, k) in enumerate(pts_c):
                    sv = np.dot(xyz_c[idx] - torcular, torcular_plane["normal"])
                    cleaned[i, j, k] = 7 if sv > 0 else 8

        sss_mid2 = _sss_mid_dilated(2)
        transv_mask = np.isin(cleaned, [7, 8])
        comp_tr, n_tr = cc_label(transv_mask, structure=struct3)
        for cid in range(1, n_tr + 1):
            cm = comp_tr == cid
            if not np.any(cm & sss_mid2):
                continue
            pts_c = np.argwhere(cm)
            xyz_c = vox2phys(pts_c, cta.origin, cta.spacing, cta.direction)
            for idx, (i, j, k) in enumerate(pts_c):
                proj = np.dot(xyz_c[idx] - global_centroid, main_dir)
                pl   = planes[0] if proj <= med_proj else planes[1]
                sv   = np.dot(xyz_c[idx] - pl["centroid"], pl["normal"])
                cleaned[i, j, k] = 9 if sv > 0 else 10

        sss_clean_all = np.argwhere(cleaned == 6)
        if sss_clean_all.shape[0] > 0:
            xyz_sc = vox2phys(sss_clean_all, cta.origin, cta.spacing, cta.direction)
            dist_t = np.linalg.norm(xyz_sc - torcular, axis=1)
            sss_mid_only = np.zeros(cleaned.shape, dtype=bool)
            mv = sss_clean_all[dist_t > TORC_RADIUS_MM]
            if mv.shape[0] > 0:
                sss_mid_only[mv[:, 0], mv[:, 1], mv[:, 2]] = True
            sss_mid_final = binary_dilation(sss_mid_only, structure=struct3, iterations=3)

            tf_mask = np.isin(cleaned, [7, 8])
            comp_f, n_f = cc_label(tf_mask, structure=struct3)
            sizes_f = [int((comp_f == c).sum()) for c in range(1, n_f + 1)]
            max_sz  = max(sizes_f) if sizes_f else 1
            for cid in range(1, n_f + 1):
                cm = comp_f == cid
                if cm.sum() > max_sz * 0.3 or not np.any(cm & sss_mid_final):
                    continue
                pts_c = np.argwhere(cm)
                xyz_c = vox2phys(pts_c, cta.origin, cta.spacing, cta.direction)
                for idx, (i, j, k) in enumerate(pts_c):
                    proj = np.dot(xyz_c[idx] - global_centroid, main_dir)
                    pl   = planes[0] if proj <= med_proj else planes[1]
                    sv   = np.dot(xyz_c[idx] - pl["centroid"], pl["normal"])
                    cleaned[i, j, k] = 9 if sv > 0 else 10

        transv_m2 = np.isin(cleaned, [7, 8])
        comp_t2, n_t2 = cc_label(transv_m2, structure=struct3)
        comp_sizes  = {c: int((comp_t2 == c).sum()) for c in range(1, n_t2 + 1)}
        comp_labels = {c: int(np.bincount(cleaned[comp_t2 == c]).argmax()) for c in range(1, n_t2 + 1)}
        sss_excl = binary_dilation(cleaned == 6, structure=struct3)
        for cid in range(1, n_t2 + 1):
            cm    = comp_t2 == cid
            cdil  = binary_dilation(cm, structure=struct3)
            nbrs  = np.unique(comp_t2[cdil & transv_m2 & ~cm & ~sss_excl])
            nbrs  = nbrs[nbrs > 0]
            if not len(nbrs):
                continue
            big_nb = max(nbrs, key=lambda x: comp_sizes[x])
            if comp_sizes[big_nb] <= comp_sizes[cid]:
                continue
            if comp_labels[big_nb] == comp_labels[cid]:
                continue
            cleaned[cm] = comp_labels[big_nb]

        changed = True
        while changed:
            changed = False
            for my_lbl, other_lbl in [(7, 8), (8, 7)]:
                comp_mine,  n_mine  = cc_label(cleaned == my_lbl,  structure=struct3)
                comp_other, n_other = cc_label(cleaned == other_lbl, structure=struct3)
                if n_mine == 0 or n_other == 0:
                    continue
                sizes_mine  = {c: int((comp_mine  == c).sum()) for c in range(1, n_mine  + 1)}
                sizes_other = {c: int((comp_other == c).sum()) for c in range(1, n_other + 1)}
                main_mine  = max(sizes_mine,  key=lambda x: sizes_mine[x])
                main_other = max(sizes_other, key=lambda x: sizes_other[x])
                mask_main_other = comp_other == main_other
                for cid in range(1, n_mine + 1):
                    if cid == main_mine:
                        continue
                    cm   = comp_mine == cid
                    dil  = binary_dilation(cm, structure=struct3)
                    if np.any(dil & (comp_mine == main_mine)):
                        continue
                    if not np.any(dil & mask_main_other):
                        continue
                    cleaned[cm] = other_lbl
                    changed = True

        cort_mask3  = np.isin(cleaned, [9, 10])
        transv_m3   = np.isin(cleaned, [7, 8])
        dist_sss3   = distance_transform_edt(~(cleaned == 6))
        dist_transv = distance_transform_edt(~transv_m3)
        comp_c3, n_c3 = cc_label(cort_mask3, structure=struct3)
        for cid in range(1, n_c3 + 1):
            cm = comp_c3 == cid
            if dist_transv[cm].min() >= dist_sss3[cm].min():
                continue
            pts_c = np.argwhere(cm)
            xyz_c = vox2phys(pts_c, cta.origin, cta.spacing, cta.direction)
            for idx, (i, j, k) in enumerate(pts_c):
                sv = np.dot(xyz_c[idx] - torcular, torcular_plane["normal"])
                cleaned[i, j, k] = 7 if sv > 0 else 8

        cleaned = detect_vcm(cleaned, cta, top_arr, LBL_SSS)

    else:
        print("  WARNING: SSS has too few voxels — skipping laterality.")
        cleaned[cleaned == LBL_PENDING] = 7

    ants.image_write(
        ants.from_numpy(cleaned.astype(np.uint8),
                        origin=cta.origin, spacing=cta.spacing, direction=cta.direction),
        final_output,
    )
    print("\nSaved:", final_output)
    print("Labels:", np.unique(cleaned))
    print("  1=VOG  2=STS  3=ICV  4=RBVR  6=SSS")
    print("  7=TransvSig-R  8=TransvSig-L  9=CorticalR  10=CorticalL  11=VCM-R  12=VCM-L")

    if planes is not None and xyz_sss is not None:
        html_path = final_output.replace(".nii.gz", "_planes.html").replace(".nii", "_planes.html")
        render_html(cleaned, cta, planes, torcular_plane, torcular, z_threshold, xyz_sss, html_path)
