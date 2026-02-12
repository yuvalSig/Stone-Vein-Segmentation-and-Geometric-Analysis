# Stone Vein Segmentation & Geometric Analysis Engine

 A deep-learning based system for high-resolution crack (vein) segmentation and geometric analysis on marble and stone surfaces.

🌐 Live Demo: https://yuvalsigavi.com/segmentation-engine/

---

## 📌 Project Overview

This project addresses the problem of detecting and analyzing thin, low-contrast cracks on stone and marble surfaces.

The system performs:
- Pixel-level crack segmentation
- Noise reduction and refinement
- Skeleton-based geometric analysis
- Width, orientation, and length estimation

The main industrial use case is quality control and comparison between:
- CAD / production templates
- Real manufactured slabs

---

##  Dataset

- Crack Dataset (Kaggle)
- Includes positive crack samples and negative samples
- Used exclusively for training and evaluation

Link:
https://www.kaggle.com/datasets/yatata1/crack-dataset

---

##  Model Architecture

### Initial Approach: DeepLabV3
- Detected crack centerlines
- Overestimated crack thickness
- Failed on thin structures
#### Input Image (Raw Stone Surface)
<img width="717" height="277" alt="1" src="https://github.com/user-attachments/assets/e41f660d-b574-4fc7-b459-4c0c655d2852" />


#### Ground Truth
<img width="721" height="276" alt="2" src="https://github.com/user-attachments/assets/39f368a2-9be0-4320-a942-ccd8ef51f392" />


#### DeepLabV3 Output (Overestimated Thickness)
<img width="718" height="252" alt="3" src="https://github.com/user-attachments/assets/c0af9114-5610-476e-985f-b4570a31e569" />


### Final Model: U-Net
- Preserves fine topology
- Improved thin-crack segmentation
- Better pixel accuracy

#### U-Net Output (Improved Thin Crack Detection)
> Evaluated using identical thresholding and post-processing parameters.
<img width="722" height="276" alt="4" src="https://github.com/user-attachments/assets/eed466dd-0ae8-413a-a110-0dd0e8a0b836" />

---

##  Training Strategy

### Loss Function
- Dice Loss
- Handles extreme class imbalance (<5% crack pixels)

### Optimization
- Adam optimizer
- Step-decay learning rate scheduler
- Prevents oscillatory convergence

---

##  Data Augmentation

Used to improve robustness:
- Rotation
- Flipping
- Brightness variation
- Contrast adjustment

Helps handle real-world lighting and orientation changes.

---

##  Hardware-Aware Deployment

Deployed on:
- AWS EC2 t3.small
- 2GB RAM
- CPU-only

Constraints:
- Full-image inference caused OOM
- Limited batch size

Solution:
- Sliding-window inference

---

##  Sliding Window Inference

High-resolution images are processed using:

1. Image tiling
2. Overlapping windows
3. Overlap averaging
4. Padding for border consistency

Benefits:
- Prevents memory overflow
- Reduces boundary artifacts
- Enables large-image processing

---

##  Post-Processing

Applied on probability maps:

### Hysteresis Thresholding
- High threshold → strong pixels
- Low threshold → connected weak pixels

### Morphological Refinement
- Noise removal
- Area filtering (50–80 px)

Produces clean binary crack masks.

---

##  Geometric Analysis

After segmentation:

### Skeletonization
- Zhang-Suen algorithm
- Produces 1-pixel-wide centerline

### Orientation Estimation
- PCA on skeleton coordinates

### Width Estimation
- Distance Transform
- Local width = 2 × distance

Outputs:
- Crack angle
- Width profile
- Length estimation

---

##  Evaluation Metrics

Pixel-level validation using:
- Precision
- Recall
- F1-score

Compared against ground-truth masks.

---

##  Limitations & Future Work

### Current Limitations
- Reduced performance on dark surfaces
- CPU-only inference (30–50 sec per image)

### Planned Improvements
- GPU deployment
- Dataset balancing
- Real-time inference
- Improved low-light robustness

---



