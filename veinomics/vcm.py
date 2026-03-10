import numpy as np
from scipy.ndimage import label as cc_label, distance_transform_edt
from skimage.morphology import skeletonize as _skeletonize


def detect_vcm(cleaned, cta, top_arr, LBL_SSS, SSS_PROX_MM=8.0):
    sss_mask    = (cleaned == 6) | (top_arr == LBL_SSS)
    dist_to_sss = distance_transform_edt(~sss_mask, sampling=list(cta.spacing))

    for cort_lbl, vcm_lbl, side in [(9, 11, "R"), (10, 12, "L")]:
        cort_mask = (cleaned == cort_lbl)
        if not np.any(cort_mask):
            continue

        skel     = _skeletonize(cort_mask)
        skel_pts = np.argwhere(skel)
        if skel_pts.shape[0] < 2:
            continue

        n_pts    = len(skel_pts)
        skel_idx = {(int(p[0]), int(p[1]), int(p[2])): i for i, p in enumerate(skel_pts)}

        adj = [[] for _ in range(n_pts)]
        for i, pt in enumerate(skel_pts):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        if di == dj == dk == 0:
                            continue
                        j = skel_idx.get((int(pt[0])+di, int(pt[1])+dj, int(pt[2])+dk))
                        if j is not None:
                            adj[i].append(j)

        degree   = [len(a) for a in adj]
        near_sss = np.array([
            dist_to_sss[int(p[0]), int(p[1]), int(p[2])] < SSS_PROX_MM
            for p in skel_pts
        ])

        d_sss  = np.full(n_pts, -1, dtype=np.int32)
        prev_s = np.full(n_pts, -1, dtype=np.int32)
        queue  = [i for i in range(n_pts) if near_sss[i]]
        for i in queue:
            d_sss[i] = 0
        h = 0
        while h < len(queue):
            u = queue[h]; h += 1
            for v in adj[u]:
                if d_sss[v] < 0:
                    d_sss[v]  = d_sss[u] + 1
                    prev_s[v] = u
                    queue.append(v)

        endpoints = [i for i, deg in enumerate(degree) if deg == 1 and d_sss[i] > 0]
        if not endpoints:
            continue

        vcm_tip   = max(endpoints, key=lambda i: d_sss[i])
        path_mask = np.zeros(cleaned.shape, dtype=bool)
        cur = vcm_tip
        while cur >= 0:
            pt = skel_pts[cur]
            path_mask[pt[0], pt[1], pt[2]] = True
            if near_sss[cur]:
                break
            cur = int(prev_s[cur])

        other_skel   = skel & ~path_mask
        dist_to_path = distance_transform_edt(~path_mask)
        if np.any(other_skel):
            vcm_mask = cort_mask & (dist_to_path < distance_transform_edt(~other_skel))
        else:
            vcm_mask = cort_mask.copy()

        comp_vcm, n_vcm = cc_label(vcm_mask, structure=np.ones((3, 3, 3)))
        if n_vcm > 0:
            biggest  = max(range(1, n_vcm + 1), key=lambda c: int((comp_vcm == c).sum()))
            vcm_mask = (comp_vcm == biggest)

        cleaned[vcm_mask] = vcm_lbl
        print(f"  VCM-{side}: ~{int(d_sss[vcm_tip]) * float(np.mean(cta.spacing)):.1f} mm from SSS")

    return cleaned
