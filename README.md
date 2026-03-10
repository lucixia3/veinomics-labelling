# veinomics-labelling

**Automatic labeling of cerebral venous sinuses from CTA**

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![nnU-Net v2](https://img.shields.io/badge/nnU--Net-v2-orange)

---

## Overview

**veinomics-labelling** is a fully automated pipeline for the semantic segmentation and laterality labeling of the cerebral venous system from CT angiography (CTA). The pipeline delineates major dural venous sinuses — including the superior sagittal sinus, transverse-sigmoid complex, internal cerebral veins, vein of Galen, straight sinus, and cortical veins — and assigns each structure a lateralized label using an anatomically grounded geometric model derived from the SSS centerline. It additionally identifies the dominant cortical bridging vein (vena cerebri media, VCM) on each hemisphere via skeleton-based geodesic tracing.

---

## Label Scheme

| Label | Structure | Abbreviation |
|------:|-----------|--------------|
| 1 | Vein of Galen | VOG |
| 2 | Straight Sinus | STS |
| 3 | Internal Cerebral Veins | ICV |
| 4 | Rolandic Bridging Veins (bilateral) | RBVR |
| 6 | Superior Sagittal Sinus | SSS |
| 7 | Transverse–Sigmoid Sinus, Right | TransvSig-R |
| 8 | Transverse–Sigmoid Sinus, Left | TransvSig-L |
| 9 | Cortical Veins, Right | Cortical-R |
| 10 | Cortical Veins, Left | Cortical-L |
| 11 | Vena Cerebri Media, Right | VCM-R |
| 12 | Vena Cerebri Media, Left | VCM-L |

---

## Method

The pipeline operates in four stages:

**1. Dual nnU-Net inference.** 
Two independent 3D full-resolution nnU-Net v2 models (ResEncUNetL architecture) are applied to the CTA volume:
- *ArtVen*: binary artery/vein segmentation; the venous mask (label 2) seeds the fusion step.
- *TopBrain*: multi-class segmentation of named venous sinuses (SSS, STS, VOG, ICV, RBVR) trained on data with 5-fold cross-validation for 1000 epochs per fold.

**2. Label fusion and postprocessing.**
TopBrain labels are dilated and used to assign nearby venous voxels from the ArtVen mask. Pending voxels are resolved using proximity-based rules, merging components into the correct structures (e.g., ICV, RBVR).

**3. SSS geometry-based laterality.**
The SSS point cloud is split into two halves based on its principal axis. Venous voxels are classified relative to separation planes:
- Above the torcular Z-level → cortical vein (right/left by SSS half-plane).
- Below the torcular Z-level → transverse-sigmoid (right/left by torcular plane).

Iterative correction resolves topological inconsistencies.

**4. VCM detection.**
The cortical vein mask is skeletonized, and a BFS identifies the geodesic distance to the SSS. The VCM tip is defined, and the path to the SSS is traced, partitioning cortical voxels into VCM and other tributaries.

An interactive 3D HTML visualization is saved with each NIfTI output, displaying all labeled structures and the lateral separation planes.

---

## Installation

```bash
git clone https://github.com/lucixia3/veinomics-label.git
cd veinomics-label
pip install -r requirements.txt
```

nnU-Net v2 must be installed separately and the trained model weights placed in `C:\nnunet\nnUNet_results` (or update the paths at the top of `segment.py`). See the [nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet) for installation instructions.

---

## Usage

**Single CTA file:**

```bash
python segment.py \
    --cta /path/to/case.nii.gz \
    --output /path/to/output \
    --dataset_artven 101 \
    --dataset_topbrain 102
```

**Batch processing (folder of CTA files):**

```bash
python segment.py \
    --input_folder /path/to/cta_folder \
    --output /path/to/output_folder \
    --dataset_artven 101 \
    --dataset_topbrain 102
```

**Skip nnU-Net inference (use existing segmentations):**

```bash
python segment.py \
    --cta /path/to/case.nii.gz \
    --output /path/to/output \
    --skip_nnunet
```

When `--skip_nnunet` is used, `CTA_artven.nii.gz` and `CTA_topbrain.nii.gz` must already exist in `<output>/temp_work/`.

---

## Output

For each processed case the following files are written to the case output directory:

| File | Description |
|------|-------------|
| `CTA_final_hybrid.nii.gz` | Multi-label venous segmentation (labels 1–12, see table above) |
| `CTA_final_hybrid_planes.html` | Interactive 3D Plotly visualization with lateral separation planes |

Intermediate nnU-Net predictions are stored under `<output>/temp_work/`.

---

## Citation

If you use veinomics-label in your research, please cite the following works:

```bibtex
@article{veinomics2025,
  title   = {veinomics-label: Automatic Lateralized Segmentation of the Cerebral Venous System from CTA},
  author  = {},
  journal = {},
  year    = {2025},
}
```
For the TopBrain model (trained by me), please cite:

```bibtex
@misc{topbrain2025,
  title   = {TopBrain: A Deep Learning Framework for Segmentation of Cerebral Venous Structures from CTA},
  author  = {Your Name},
  year    = {2025},
  url     = {https://zenodo.org/records/16878417},
  note    = {Trained for 1000 epochs with 5-fold cross-validation}
}
```
And for the veins model, cite:
```bibtex
@article{ceballos2026,
  title   = {Robust automatic brain vessel segmentation in 3D CTA scans using dynamic 4D-CTA data},
  author  = {Alberto Mario Ceballos Arroyo, Shrikanth M. Yadav, Chu-Hsuan Lin, Jisoo Kim, Geoffrey S. Young, Huaizu Jiang, Lei Qin},
  journal = {arXiv preprint arXiv:2602.00391v1},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.00391v1},
  license = {CC BY 4.0}
}
```
---

## License

MIT License. See [LICENSE](LICENSE) for details.








