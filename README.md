# 🐄 Indian Cattle Breed Identification using YOLOv8x + EfficientNetV2-S  
### A Two-Stage Computer Vision Pipeline for Fine-Grained Breed Classification

This project presents a two-stage deep learning pipeline for **automatic identification of Indian indigenous cattle breeds** from real-world farm images.  
It integrates:

1. **YOLOv8x (COCO-pretrained)** for cattle region extraction  
2. **EfficientNetV2-S** for fine-grained breed classification  

Despite severe dataset issues and minimal high-quality crops, the classifier achieves:

- **Top-1 Accuracy:** ~78%  
- **Top-5 Accuracy:** ~98%  
- **Macro F1-Score:** ~0.78  

---

## 📌 1. Introduction

Breed identification plays a critical role in:

- Livestock productivity  
- Feeding and breeding optimization  
- Genetic conservation  
- Health monitoring  

However, India’s bovine datasets suffer from:

- Mixed cattle and buffalo species  
- Non-standardized backgrounds  
- High intra-class variation  
- Very similar-looking breeds  
- Partial-body images  
- Class imbalance  

This project addresses these issues through a **two-stage workflow**:

### 1️⃣ YOLOv8x (COCO pretrained)  
Used only as a detector to crop cattle from raw images.

### 2️⃣ EfficientNetV2-S  
Fine-tuned for multi-class Indian cattle breed classification.

This is one of the few experimental works focusing on **Indian indigenous breeds**.

---

## 🎯 2. Experimental Objective

The objective was to build a robust end-to-end pipeline that can:

1. Detect cattle in real-world images  
2. Generate consistent YOLO-based crops  
3. Classify breeds using EfficientNetV2-S  
4. Handle non-ideal conditions  
5. Check feasibility for field deployment  

### 🐄 Not Attempted: Body Condition Score (BCS)  
BCS requires side-view datasets + pose estimation + BCS labels.  
Since these were not available, BCS evaluation was not performed.

---

## 🗂 3. Dataset Description

### 3.1 Source  
Kaggle: **Indian Bovine Breeds Dataset**  
Contains both cattle and buffalo images.

### 3.2 Raw Dataset Problems
- Mixed buffalo and cow species  
- Class imbalance  
- Partial-body visibility  
- Real-world farm backgrounds  
- High inter-breed similarity  
- Inconsistent viewpoints  

### 3.3 Final Dataset After Cleaning
- Buffalo classes removed  
- 16 cattle breeds retained  
- YOLO crops generated using `yolov8x.pt`  
- Crops manually verified  

### Final Image Split
| Split | Count |
|-------|-------|
| Train | 2176 |
| Validation | 625 |
| Test | 318 |

Stored in: `final_class_distribution.csv`

---

## 🔧 4. Methodology

### 4.1 Stage 1 — YOLOv8x Detection

YOLOv8x COCO was used **without retraining**:

```python
from ultralytics import YOLO
detector = YOLO("yolov8x.pt")
```

### Key Observations

- COCO has only **one generic cow class**  
- Indian breeds are visually different  
- Approx. **0.5% detection success rate** on the raw dataset  
- Frequent failure patterns:
  - Head-only crops  
  - Missed detections  
  - False positives  

Still, enough clean crops were collected for training.

#### Figure 1: Example YOLO Crops  
(from `cow_crops/`)

#### Figure 2: Detection Success vs Failure  
(from `figure2_fallback.png`)

---

### 4.2 Stage 2 — EfficientNetV2-S Classification

#### Training Configuration
- Image size: `256x256`
- Optimizer: `Adam`
- Loss: `SparseCategoricalCrossentropy`
- Batch size: `16–32`

#### Two-Phase Training Strategy

**Phase 1 — Feature Adaptation (12 epochs)**  
- Higher LR  
- Adapt pretrained weights  

**Phase 2 — Fine-Tuning (15 epochs)**  
- Lower LR  
- Reduce overfitting  
- Improve stability  

Model saved as:  
`efficientnet_v2s_final.keras`

---

# 5. Results & Evaluation

## 5.1 Overall Performance

| Metric | Score |
|--------|--------|
| Top-1 Accuracy | **~78%** |
| Top-5 Accuracy | **98.1%** |
| Macro F1-Score | **~0.78** |
| Test Accuracy | **0.75786** |

### Interpretation
- Solid generalization  
- Excellent fine-grained learning  
- High robustness to cluttered backgrounds and partial views  

---

## 5.2 Training Curves

Available in `/results`:

- **EfficientNet Training Accuracy (Reconstructed).png**
- **EfficientNet Training Loss (Reconstructed).png**

---

## 5.3 Confusion Matrix

Two matrices included:

- `Confusion Matrix.png`  
- `Normalized Confusion Matrix.png`  

### Major Confusions
- Hallikar ↔ Bargur  
- Ongole ↔ Deoni  
- Jersey ↔ Brown Swiss  

Reasons:
- Visual similarity  
- Partial-body crops  
- Imbalanced dataset  

---

## 5.4 Per-Class Performance

Files:  
- `per_class_accuracy.csv`  
- `classification_report.txt`

### Strongest Classes:
- **Banni — 100%**
- **Holstein — 100%**
- Toda: ~85%
- Brown Swiss: ~83%
- Sahiwal: ~81%

### Weakest Classes:
- Ongole: ~58%
- Jersey: ~62%
- Bargur: ~66%

---

## 5.5 Qualitative Prediction Grid

Image:  
`Prediction_Grid (4x6).png`

Shows correct, incorrect, borderline, and difficult samples.

---

# 6. System Effectiveness

The system successfully achieved:

- High Top-1 and Top-5 performance  
- Robust breed identification from noisy images  
- Fine-grained morphological feature extraction  
- Strong real-world applicability  

A strong prototype for livestock monitoring.

---

# 7. Limitations

### Detection Stage
- COCO YOLO cannot detect Indian cattle reliably  
- Low detection success  
- Many head-only or wrong-animal crops  

### Classification Stage
- Very similar breeds  
- Partial-body visibility  
- Class imbalance  

### BCS Stage
- No BCS labels  
- No side-profile images  
- Would require pose estimation  

---

# 8. Future Scope

- Train YOLO on Indian cattle dataset  
- Add pose estimation (HRNet, MediaPipe)  
- Use segmentation for cleaner crops  
- Collect BCS-labelled side-view images  
- Upgrade classifier to Swin-ViT / CoAtNet  
- Add Test-Time Augmentation (TTA)  
- Explore metric learning / Siamese networks  

---

# 9. Conclusion

A two-stage CV system was developed using:

- **YOLOv8x** for cattle detection  
- **EfficientNetV2-S** for breed classification  

Despite dataset quality issues, it achieved:

- **~78% Top-1 Accuracy**  
- **98% Top-5 Accuracy**  
- **~0.78 Macro F1-Score**

This forms a strong foundation for:

- Automated livestock ID  
- Future BCS prediction  
- Real-world smart farming AI solutions  

---

# 📁 Project Structure

```
cattle-detection-classification/
│── script.py
│── script1.py
│── efficientnet_v2s_final.keras
│── classification_report.txt
│── per_class_accuracy.csv
│── final_class_distribution.csv
│── Cow_Experimentation_Arnav Saxena.docx
│
├── results/
│   ├── Confusion Matrix.png
│   ├── Normalized Confusion Matrix.png
│   ├── EfficientNet Training Accuracy (Reconstructed).png
│   ├── EfficientNet Training Loss (Reconstructed).png
│   ├── Prediction_Grid (4x6).png
│   ├── figure1_cropped.png
│   ├── figure2_fallback.png
│
├── classify_data/
├── cow_crops/
└── README.md
```

---

# 📬 Contact  

**Arnav Saxena**  
🔗 LinkedIn: https://www.linkedin.com/in/arnav-saxena-a9a217367  
📧 Email: **arnav12saxena@gmail.com**
