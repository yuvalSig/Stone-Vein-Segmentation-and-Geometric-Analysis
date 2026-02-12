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

### Final Model: U-Net
- Preserves fine topology
- Improved thin-crack segmentation
- Better pixel accuracy

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



