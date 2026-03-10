import os
import argparse

os.environ["nnUNet_raw"]          = r"C:\nnunet\nnUNet_raw"
os.environ["nnUNet_preprocessed"] = r"C:\nnunet\nnUNet_preprocessed"
os.environ["nnUNet_results"]      = r"C:\nnunet\nnUNet_results"

from veinomics.utils import prepare_input, restore_space
from veinomics.inference import run_nnunet
from veinomics.fusion import fuse_results


def process_case(cta_path, case_out, dataset_artven, dataset_topbrain, skip_nnunet):
    work_dir = os.path.join(case_out, "temp_work")
    os.makedirs(work_dir, exist_ok=True)

    art_output = os.path.join(work_dir, "CTA_artven.nii.gz")
    top_output = os.path.join(work_dir, "CTA_topbrain.nii.gz")

    if not skip_nnunet:
        for prefix, output, dataset_id in [
            ("ARTVEN",   art_output, dataset_artven),
            ("TOPBRAIN", top_output, dataset_topbrain),
        ]:
            inp      = prepare_input(cta_path, work_dir, prefix)
            pred_dir = os.path.join(work_dir, f"Pred_{prefix}")
            run_nnunet(inp, pred_dir, dataset_id)
            preds = [f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")]
            if not preds:
                raise FileNotFoundError(f"No prediction found in {pred_dir}")
            restore_space(cta_path, os.path.join(pred_dir, preds[0]), output)
    else:
        for p in [art_output, top_output]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing: {p}")

    fuse_results(cta_path, art_output, top_output,
                 os.path.join(case_out, "CTA_final_hybrid.nii.gz"))


def main():
    parser = argparse.ArgumentParser(description="veinomics-label: cerebral venous segmentation pipeline")
    parser.add_argument("--cta",          help="Single CTA file (.nii or .nii.gz)")
    parser.add_argument("--input_folder", help="Folder with CTA files")
    parser.add_argument("--output",       required=True, help="Output folder")
    parser.add_argument("--dataset_artven",   type=int)
    parser.add_argument("--dataset_topbrain", type=int)
    parser.add_argument("--skip_nnunet",  action="store_true")
    args = parser.parse_args()

    if args.input_folder:
        os.makedirs(args.output, exist_ok=True)
        files = [f for f in os.listdir(args.input_folder)
                 if f.endswith(".nii.gz") or f.endswith(".nii")]
        if not files:
            raise FileNotFoundError(f"No NIfTI files found in {args.input_folder}")
        for f in files:
            case = os.path.splitext(os.path.splitext(f)[0])[0]
            process_case(
                os.path.join(args.input_folder, f),
                os.path.join(args.output, case),
                args.dataset_artven, args.dataset_topbrain, args.skip_nnunet,
            )
    elif args.cta:
        process_case(args.cta, args.output,
                     args.dataset_artven, args.dataset_topbrain, args.skip_nnunet)
    else:
        parser.error("Specify --cta or --input_folder")


if __name__ == "__main__":
    main()
