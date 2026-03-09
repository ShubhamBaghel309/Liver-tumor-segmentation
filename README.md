# 3D Liver & Tumor Segmentation — Milestone 1

Automated segmentation of the **liver** and **liver tumors** in abdominal CT scans using 3D deep learning.  
This milestone covers the complete ML pipeline: data preprocessing, model training, evaluation, and inference — implemented as Jupyter Notebooks.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Pipeline Overview](#pipeline-overview)
5. [Data Preprocessing (`Datacleaning.ipynb`)](#data-preprocessing)
6. [3D U-Net (`UNet.ipynb`)](#3d-u-net)
7. [3D Attention U-Net (`AttentionU_net.ipynb`)](#3d-attention-u-net)
8. [Loss Function & Metrics](#loss-function--metrics)
9. [Training Configuration](#training-configuration)
10. [Inference & Post-Processing](#inference--post-processing)
11. [Requirements](#requirements)
12. [How to Run](#how-to-run)

---

## Project Overview

This project tackles **multi-class volumetric segmentation** of CT scans into three classes:

| Label | Class      |
|-------|------------|
| 0     | Background |
| 1     | Liver      |
| 2     | Tumor      |

**Milestone 1** goal: establish a working end-to-end pipeline from raw `.nii` CT volumes to trained segmentation models, evaluating both a standard 3D U-Net and a 3D Attention U-Net.

---

## Dataset

**LiTS — Liver Tumor Segmentation Challenge**

- Format: NIfTI (`.nii` / `.nii.gz`)
- Input volumes: named `volume-N.nii`
- Segmentation masks: named `segmentation-N.nii`
- Orientation: stored as `(H, W, D)`, transposed to `(D, H, W)` during preprocessing

Expected directory layout:

```
LiTS_small/
├── volumes/
│   ├── volume-0.nii
│   ├── volume-1.nii
│   └── ...
└── segmentations/
    ├── segmentation-0.nii
    ├── segmentation-1.nii
    └── ...
```

---

## Repository Structure

```
project/
├── Datacleaning.ipynb       # Data preprocessing and patch extraction
├── UNet.ipynb               # 3D U-Net — training, evaluation, inference
├── AttentionU_net.ipynb     # 3D Attention U-Net — training, evaluation, inference
├── X_train.npy              # Preprocessed training patches (generated)
├── Y_train.npy              # Corresponding training masks (generated)
├── X_val.npy                # Preprocessed validation patches (generated)
├── Y_val.npy                # Corresponding validation masks (generated)
├── best_model.pth           # Best 3D U-Net checkpoint (generated)
├── best_AttentionUnet.pth   # Best Attention U-Net checkpoint (generated)
└── final_results.txt        # Saved evaluation metrics (generated)
```

---

## Pipeline Overview

```
Raw CT volumes (.nii)
        │
        ▼
┌──────────────────────┐
│   Datacleaning.ipynb │
│  • HU Clipping       │
│  • Normalization     │
│  • Empty slice removal│
│  • 3D Patch Extraction│
│  • Train/Val Split   │
└──────────┬───────────┘
           │ X_train.npy / Y_train.npy
           │ X_val.npy   / Y_val.npy
           ▼
  ┌────────┴────────┐
  │                 │
  ▼                 ▼
UNet.ipynb    AttentionU_net.ipynb
  │                 │
  └────────┬────────┘
           ▼
  Sliding Window Inference
  + Post-Processing
  + Dice / IoU Evaluation
```

---

## Data Preprocessing

**Notebook:** `Datacleaning.ipynb`

### Steps

#### 1. Load NIfTI Volumes
Uses `nibabel` to load `.nii` / `.nii.gz` files and reads data as `float32` arrays.

#### 2. Orientation Fix
LiTS volumes are stored in `(H, W, D)` order. All volumes and masks are transposed to `(D, H, W)` for depth-first processing.

#### 3. HU Clipping
```
clip_hu(volume, min_hu=-200, max_hu=250)
```
Hounsfield Units (HU) are clipped to `[-200, 250]` to suppress irrelevant structures (bone, air) and focus on soft-tissue contrast.

#### 4. Z-Score Normalization
```
normalize(volume) → (volume - mean) / (std + 1e-8)
```
Per-volume mean-std normalization for stable training.

#### 5. Remove Empty Slices
Slices where the corresponding mask contains no annotations are discarded, reducing dataset size and training noise.

#### 6. Train / Validation Split
An 80/20 volume-level split is applied using `sklearn.model_selection.train_test_split` with `random_state=42`, ensuring no data leakage between train and validation sets.

#### 7. 3D Patch Extraction
```
patch_size = (16, 128, 128)   # D × H × W
stride     = (8, 64, 64)
```
A sliding window extracts overlapping 3D patches. Sampling is class-aware to handle severe class imbalance:

| Condition              | Action                     |
|------------------------|----------------------------|
| Patch contains tumor   | Always included            |
| Patch contains liver   | Always included            |
| Background-only patch  | Included with 5% probability |

#### 8. Save Patches
```python
np.save("X_train.npy", X_train)   # float32
np.save("Y_train.npy", Y_train)   # uint8
np.save("X_val.npy",   X_val)
np.save("Y_val.npy",   Y_val)
```

---

## 3D U-Net

**Notebook:** `UNet.ipynb`

### Architecture

The baseline model is a **3D U-Net** with 2 encoder levels and 2 decoder levels.

```
Input (1 × 16 × 128 × 128)
        │
   DoubleConv → 32 ch                 ← skip x1
        │
   MaxPool3d + DoubleConv → 64 ch     ← skip x2
        │
   MaxPool3d + DoubleConv → 128 ch    (bottleneck)
        │
   ConvTranspose3d + concat(x2) + DoubleConv → 64 ch
        │
   ConvTranspose3d + concat(x1) + DoubleConv → 32 ch
        │
   Conv3d(1×1×1) → 3 classes
```

**DoubleConv block:**
```
Conv3d → BatchNorm3d → ReLU → Conv3d → BatchNorm3d → ReLU
```

**Weight Initialization:** Kaiming Normal for all `Conv3d` layers; biases set to zero.

### Key Components

- `DoubleConv`: two consecutive 3×3×3 convolutions with BN + ReLU
- `Down`: `MaxPool3d(2)` followed by `DoubleConv`
- `Up`: `ConvTranspose3d(stride=2)` followed by skip concatenation and `DoubleConv`

---

## 3D Attention U-Net

**Notebook:** `AttentionU_net.ipynb`

### Architecture

The Attention U-Net extends the standard U-Net with **Attention Gates** on every skip connection. These gates learn to suppress irrelevant feature activations before they are concatenated in the decoder.

```
Input (1 × 16 × 128 × 128)
        │
   DoubleConv → 32 ch                     ← skip x1
        │
   MaxPool3d + DoubleConv → 64 ch         ← skip x2
        │
   MaxPool3d + DoubleConv → 128 ch        ← skip x3
        │
   MaxPool3d + DoubleConv → 256 ch        (bottleneck)
        │
   ConvTranspose3d + AttGate(x3) + DoubleConv → 128 ch
        │
   ConvTranspose3d + AttGate(x2) + DoubleConv → 64 ch
        │
   ConvTranspose3d + AttGate(x1) + DoubleConv → 32 ch
        │
   Conv3d(1×1×1) → 3 classes
```

The Attention U-Net is **one level deeper** than the standard U-Net (256-channel bottleneck vs 128), giving it greater representational capacity.

### Attention Gate

```python
class AttentionBlock(nn.Module):
    # Inputs:
    #   g — gating signal from decoder (coarser scale)
    #   x — skip connection from encoder (finer scale)
    #
    # Output:
    #   x * sigmoid( ReLU( W_g(g) + W_x(x) ) )
```

| Step | Operation |
|------|-----------|
| 1 | Project both `g` and `x` to intermediate channels `F_int = F_l // 2` via 1×1×1 convolution + BN |
| 2 | Add projections then apply ReLU |
| 3 | Project to a single channel, apply BN + Sigmoid to obtain attention map `ψ ∈ [0,1]` |
| 4 | Multiply the skip connection `x` element-wise by `ψ` |

This forces the model to focus on anatomically relevant regions (liver, tumors) and ignore surrounding tissue.

---

## Loss Function & Metrics

Both models use an identical training objective and evaluation protocol.

### Combined Loss

```
L_total = L_CE + L_Dice
```

**Weighted Cross-Entropy:**
```python
weight = [0.2, 0.3, 0.5]   # background, liver, tumor
```
Higher weight on the rare tumor class to counter class imbalance.

**Soft Dice Loss** (computed on foreground classes only, i.e., liver and tumor):
$$L_{\text{Dice}} = \frac{1}{C} \sum_{c=1}^{C} \left(1 - \frac{2 \sum p_c \cdot t_c + \varepsilon}{\sum p_c + \sum t_c + \varepsilon}\right)$$

where $p_c$ are softmax probabilities and $t_c$ is the one-hot target for class $c$.

### Evaluation Metrics

**Dice Score (per class):**
$$\text{Dice}_c = \frac{2 \cdot |P_c \cap T_c| + \varepsilon}{|P_c| + |T_c| + \varepsilon}$$

**IoU / Jaccard Score (per class):**
$$\text{IoU}_c = \frac{|P_c \cap T_c|}{|P_c \cup T_c|}$$

Reported per class (Background, Liver, Tumor) as well as **Mean Dice (Liver + Tumor)** as the primary model selection criterion.

---

## Training Configuration

| Hyperparameter         | Value                          |
|------------------------|--------------------------------|
| Patch size             | 16 × 128 × 128                 |
| Batch size             | 2                              |
| Optimizer              | Adam                           |
| Learning rate          | 1e-4                           |
| Weight decay           | 1e-5                           |
| LR scheduler           | ReduceLROnPlateau (patience=3, factor=0.5) |
| Epochs                 | 20                             |
| Best model criterion   | Highest Mean Dice (Liver + Tumor) on validation set |

### Data Augmentation (training only)

Applied on-the-fly per patch inside the `MedicalDataset.__getitem__` method:

| Augmentation        | Probability | Details                        |
|---------------------|-------------|--------------------------------|
| Horizontal flip     | 50%         | Axis W                         |
| Vertical flip       | 50%         | Axis H                         |
| 90° rotation        | 50%         | Random k ∈ {0,1,2,3}, HW plane |
| Intensity scaling   | 50%         | Scale ∈ [0.9, 1.1]             |
| Gaussian noise      | 50%         | σ = 0.02                       |
| Gamma correction    | 50%         | γ ∈ [0.8, 1.2], applied after re-normalization |

---

## Inference & Post-Processing

### Sliding Window Inference

Full CT volumes are segmented using an **overlapping sliding window** strategy that matches the training patch size, avoiding boundary artefacts via averaging:

```
patch_size = (16, 128, 128)
stride     = (8,  64,  64)
```

For each position, the model outputs softmax probabilities. Overlapping predictions are **accumulated and averaged**, then `argmax` is applied to produce the final segmentation map.

### Post-Processing Steps

Two post-processing strategies are applied to refine the raw predictions:

**1. Remove Small Regions**  
Connected components with fewer than 500 voxels are discarded to eliminate spurious noise predictions on both liver and tumor channels.

**2. Anatomical Constraint (UNet only — largest component)**  
- Keep only the **largest connected component** of the predicted liver (removes satellite false positives).  
- Enforce that **tumor voxels can only exist inside the liver** mask — any predicted tumor outside the liver is suppressed.  
- Rebuild the final segmentation: liver = 1, tumor = 2.

### Visualization

Side-by-side overlays are rendered for any CT slice containing annotated tissue using `matplotlib` and `cv2`:

- **Left**: raw CT slice (grayscale)  
- **Centre**: ground truth — green contours = liver, red contours = tumor  
- **Right**: model prediction — same colour scheme  

---

## Requirements

Install dependencies with:

```bash
pip install torch torchvision numpy nibabel scikit-learn scipy opencv-python matplotlib
```

| Package         | Purpose                                  |
|-----------------|------------------------------------------|
| `torch`         | Neural network framework (CUDA support)  |
| `numpy`         | Array operations                         |
| `nibabel`       | Load/save NIfTI medical images           |
| `scikit-learn`  | Train/validation split                   |
| `scipy`         | Connected component analysis             |
| `opencv-python` | Contour drawing for overlay visualisation |
| `matplotlib`    | Plotting and slice visualisation         |

A CUDA-capable GPU is strongly recommended for training at the 3D patch scale used in this project.

---

## How to Run

### Step 1 — Preprocess the Data

Open and run all cells in **`Datacleaning.ipynb`**:

- Set `image_dir` and `mask_dir` to point to your LiTS dataset.
- The notebook will produce `X_train.npy`, `Y_train.npy`, `X_val.npy`, `Y_val.npy` in the project root.

### Step 2 — Train the Standard 3D U-Net

Open and run all cells in **`UNet.ipynb`**:

- Loads the `.npy` patch arrays created in Step 1.
- Trains for 20 epochs, saving the best checkpoint as `best_model.pth`.
- Prints per-epoch Dice and IoU scores for all three classes.

### Step 3 — Train the 3D Attention U-Net

Open and run all cells in **`AttentionU_net.ipynb`**:

- Identical setup to the U-Net notebook.
- Best checkpoint saved as `best_AttentionUnet.pth`.

### Step 4 — Inference on a Full Volume

In either model notebook, update the path variables:

```python
img_path  = "path/to/volume-N.nii"
mask_path = "path/to/segmentation-N.nii"
```

Run the sliding window inference cells and visualisation cells to see predicted segmentation overlaid on CT slices.

---

## Notes

- All `.npy` files and `.pth` checkpoints are **generated artefacts** — they are not included in the repository and must be produced by running the notebooks in order.
- The `mmap_mode="r"` flag is used when loading `.npy` files in the model notebooks to keep memory usage low for large patch arrays.
- Class weights in the CrossEntropy loss `[0.2, 0.3, 0.5]` were chosen to reflect the relative rarity of tumors vs liver vs background in the LiTS dataset.
