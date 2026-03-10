import os
import subprocess
import numpy as np
import nibabel as nib


def run_command(cmd):
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError("nnU-Net inference failed")


def prepare_input(cta_path, work_dir, prefix):
    dst = os.path.join(work_dir, f"Inference_{prefix}")
    os.makedirs(dst, exist_ok=True)
    nib.save(nib.load(cta_path), os.path.join(dst, f"{prefix}_0000.nii.gz"))
    return dst


def restore_space(original_cta, prediction_path, output_path):
    orig = nib.load(original_cta)
    pred = nib.load(prediction_path)
    nib.save(nib.Nifti1Image(pred.get_fdata(), orig.affine, orig.header), output_path)


def vox2phys(ijk, origin, spacing, direction):
    return np.asarray(ijk, dtype=np.float64) * np.array(spacing) @ np.array(direction).T + np.array(origin)
