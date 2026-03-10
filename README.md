# veinomics-label

**Automatic labeling of cerebral venous sinuses from CTA**

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![nnU-Net v2](https://img.shields.io/badge/nnU--Net-v2-orange)

---

## Overview

**veinomics-label** is a fully automated pipeline for the semantic segmentation and laterality labeling of the cerebral venous system from CT angiography (CTA). The pipeline delineates major dural venous sinuses — including the superior sagittal sinus, transverse-sigmoid complex, internal cerebral veins, vein of Galen, straight sinus, and cortical veins — and assigns each structure a lateralized label using an anatomically grounded geometric model derived from the SSS centerline. It additionally identifies the dominant cortical bridging vein (vena cerebri media, VCM) on each hemisphere via skeleton-based geodesic tracing.

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
- *TopBrain*: multi-class segmentation of named venous sinuses (SSS, STS, VOG, ICV, transverse complex) in the superior cranial compartment.

**2. Label fusion and postprocessing.**
TopBrain labels are dilated one voxel and used to assign nearby venous voxels from the ArtVen mask. Residual venous voxels are classified as *pending*. A cascade of connected-component rules resolves pending voxels by proximity and topology: small isolated islands are discarded, components touching ICV are absorbed as ICV, and components touching RBVR or VOG are absorbed as RBVR.

**3. SSS geometry-based laterality.**
The SSS point cloud is decomposed by PCA into its principal axis. The sinus is split at its median projection into two halves; each half yields a sagittal separation plane via SVD. The inferior endpoint of the SSS (Torcular Herophili) anchors a third plane separating transverse-sigmoid structures. Every pending venous voxel is classified by its position relative to these planes:
- Above the torcular Z-level → cortical vein (right/left by SSS half-plane).
- Below the torcular Z-level → transverse-sigmoid (right/left by torcular plane).

Multiple iterative correction passes resolve topological inconsistencies: components contacting the SSS body outside the torcular region are reclassified as cortical; fragments attached exclusively to the contralateral transverse sinus are relabeled; cortical components closer to the transverse sinus than to the SSS are demoted to transverse.

**4. VCM detection.**
For each cortical hemisphere, the cortical vein mask is skeletonized and a multi-source BFS identifies the geodesic distance of every skeleton node to the SSS. The degree-1 endpoint (skeleton tip) farthest from the SSS defines the VCM tip. The skeleton path from tip to SSS is traced, and a Voronoi partition assigns the volumetric cortical voxels to VCM vs. other cortical tributaries.

An interactive 3D HTML visualization is saved alongside each NIfTI output, displaying all labeled structures and the laterality separation planes.

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

If you use veinomics-label in your research, please cite:

```
@article{veinomics2025,
  title   = {veinomics-label: Automatic Lateralized Segmentation of the Cerebral Venous System from CTA},
  author  = {},
  journal = {},
  year    = {2025},
}
```

*(Citation will be updated upon publication.)*

---

## License

MIT License. See [LICENSE](LICENSE) for details.
=======
Pipeline for automatic labeling of cerebral venous sinuses from CTA images, combining two nnU-Net models with rule-based post-processing.
=======
**Automatic labeling of cerebral venous sinuses from CTA**
>>>>>>> aee6980 (Restructure pipeline into veinomics package + VCM detection + updated README)

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![nnU-Net v2](https://img.shields.io/badge/nnU--Net-v2-orange)

---

## Overview

**veinomics-label** is a fully automated pipeline for the semantic segmentation and laterality labeling of the cerebral venous system from CT angiography (CTA). The pipeline delineates major dural venous sinuses — including the superior sagittal sinus, transverse-sigmoid complex, internal cerebral veins, vein of Galen, straight sinus, and cortical veins — and assigns each structure a lateralized label using an anatomically grounded geometric model derived from the SSS centerline. It additionally identifies the dominant cortical bridging vein (vena cerebri media, VCM) on each hemisphere via skeleton-based geodesic tracing.

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
- *TopBrain*: multi-class segmentation of named venous sinuses (SSS, STS, VOG, ICV, transverse complex) in the superior cranial compartment.

**2. Label fusion and postprocessing.**
TopBrain labels are dilated one voxel and used to assign nearby venous voxels from the ArtVen mask. Residual venous voxels are classified as *pending*. A cascade of connected-component rules resolves pending voxels by proximity and topology: small isolated islands are discarded, components touching ICV are absorbed as ICV, and components touching RBVR or VOG are absorbed as RBVR.

**3. SSS geometry-based laterality.**
The SSS point cloud is decomposed by PCA into its principal axis. The sinus is split at its median projection into two halves; each half yields a sagittal separation plane via SVD. The inferior endpoint of the SSS (Torcular Herophili) anchors a third plane separating transverse-sigmoid structures. Every pending venous voxel is classified by its position relative to these planes:
- Above the torcular Z-level → cortical vein (right/left by SSS half-plane).
- Below the torcular Z-level → transverse-sigmoid (right/left by torcular plane).

Multiple iterative correction passes resolve topological inconsistencies: components contacting the SSS body outside the torcular region are reclassified as cortical; fragments attached exclusively to the contralateral transverse sinus are relabeled; cortical components closer to the transverse sinus than to the SSS are demoted to transverse.

**4. VCM detection.**
For each cortical hemisphere, the cortical vein mask is skeletonized and a multi-source BFS identifies the geodesic distance of every skeleton node to the SSS. The degree-1 endpoint (skeleton tip) farthest from the SSS defines the VCM tip. The skeleton path from tip to SSS is traced, and a Voronoi partition assigns the volumetric cortical voxels to VCM vs. other cortical tributaries.

An interactive 3D HTML visualization is saved alongside each NIfTI output, displaying all labeled structures and the laterality separation planes.

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

<<<<<<< HEAD
For each case, the output folder contains:
- `CTA_final_hybrid.nii.gz` — labeled segmentation (labels 1–10)
- `CTA_final_hybrid_planes.html` — interactive 3D visualization with SSS planes and classification boundaries
>>>>>>> 616357c (Initial commit: veinomics-label pipeline)
=======
For each processed case the following files are written to the case output directory:

| File | Description |
|------|-------------|
| `CTA_final_hybrid.nii.gz` | Multi-label venous segmentation (labels 1–12, see table above) |
| `CTA_final_hybrid_planes.html` | Interactive 3D Plotly visualization with lateral separation planes |

Intermediate nnU-Net predictions are stored under `<output>/temp_work/`.

---

## Citation

If you use veinomics-label in your research, please cite:

```
@article{veinomics2025,
  title   = {veinomics-label: Automatic Lateralized Segmentation of the Cerebral Venous System from CTA},
  author  = {},
  journal = {},
  year    = {2025},
}
```

*(Citation will be updated upon publication.)*

---

## License

MIT License. See [LICENSE](LICENSE) for details.
>>>>>>> aee6980 (Restructure pipeline into veinomics package + VCM detection + updated README)

