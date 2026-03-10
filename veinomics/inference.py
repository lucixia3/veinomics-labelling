import os
from .utils import run_command


def run_nnunet(inference_input, output_dir, dataset_id):
    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        f'nnUNetv2_predict '
        f'-i "{inference_input}" -o "{output_dir}" '
        f'-d {dataset_id} -c 3d_fullres '
        f'-p nnUNetResEncUNetLPlans '
        f'-chk checkpoint_best.pth -f all -device cuda'
    )
    run_command(cmd)
