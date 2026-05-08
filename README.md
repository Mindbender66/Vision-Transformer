# 🧠 Vision Transformer for Brain Disease Classification on MRI Images

> A complete **Vision Transformer (ViT)** implementation built entirely from scratch using **TensorFlow/Keras** for multi-class brain disease classification from MRI scans.  
> This project integrates advanced transformer concepts like **Shifted Patch Tokenization (SPT)** and **Locality Self-Attention (LSA)** for improved performance on medical imaging datasets.

---

## 🩺 Problem Statement

Brain diseases such as **Atrophy**, **Ischemia**, and **White Matter Disease (WMD)** are difficult to diagnose accurately due to subtle visual differences in MRI scans. Manual analysis requires expert radiologists and significant time.

This project automates the classification process using a custom-built Vision Transformer architecture capable of learning both:

- **Global relationships** between image regions
- **Local spatial patterns** critical for medical imaging

---

## 🏷️ Classes

| Class | Description |
|-------|-------------|
| `atrofi` | Brain tissue shrinkage associated with neurodegenerative diseases |
| `iskemi` | Reduced blood supply to brain tissue (ischemia/stroke) |
| `normal` | Healthy MRI scans with no visible abnormalities |
| `WMD` | White Matter Disease causing lesions in white matter regions |

---

## 📦 Dataset

- **Type:** Brain MRI Images
- **Format:** `.jpg`
- **Classes:** 4
- **Image Size:** Loaded at `150×150`, resized internally to `72×72`
- **Split:** 90% Training / 10% Testing
- **Loading Method:** OpenCV (`cv2`) + `glob`

### 📂 Dataset Structure

```bash
train/
├── atrofi/
├── iskemi/
├── normal/
└── WMD/
```

### 🏷️ Label Encoding

```python
labels = {
    'atrofi': 0,
    'iskemi': 1,
    'normal': 2,
    'WMD': 3
}
```

---

## 🧠 Model Architecture

This project implements a custom Vision Transformer pipeline consisting of:

```text
Input MRI Image (72 × 72 × 3)
        │
        ▼
Shifted Patch Tokenization (SPT)
        │
        ▼
Patch Encoder + Positional Embeddings
        │
        ▼
8 × Transformer Encoder Blocks
   ├── Locality Self-Attention (LSA)
   ├── Layer Normalization
   ├── MLP Block (GELU)
   └── Residual Connections
        │
        ▼
Flatten
        │
        ▼
MLP Classification Head
        │
        ▼
Dense(4) → Disease Prediction
```

---

## 🚀 Key Innovations

### 🔹 Shifted Patch Tokenization (SPT)

Unlike standard ViTs, this model creates **4 diagonally shifted versions** of every image before extracting patches.

✅ Benefits:
- Captures neighboring pixel context
- Improves local feature learning
- Better performance on small medical datasets

---

### 🔹 Locality Self-Attention (LSA)

Custom attention mechanism where:
- Self-attention of a patch to itself is masked
- Attention is forced across neighboring patches
- Includes a trainable temperature parameter `τ`

✅ Benefits:
- Prevents attention collapse
- Improves cross-patch learning
- Enhances locality awareness

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 72 × 72 |
| Patch Size | 6 |
| Number of Patches | 144 |
| Projection Dimension | 64 |
| Transformer Layers | 8 |
| Attention Heads | 4 |
| Batch Size | 256 |
| Epochs | 60 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |

---

## 🔄 Data Augmentation

Applied only to the training dataset:

| Augmentation | Value |
|--------------|------|
| Horizontal Flip | ✅ |
| Random Rotation | ±2% |
| Random Zoom | ±20% |
| Normalization | ✅ |
| Resize | 72×72 |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | **80.28%** |
| Precision | 67.54% |
| Recall | 80.28% |
| F1 Score | 73.36% |
| AUC-ROC | 39.28% |

> Achieved using a fully custom Vision Transformer trained from scratch without transfer learning.

---

## 📈 Training Strategy

### 🔥 WarmUp Cosine Learning Rate Scheduler

The model uses a custom scheduler with:
1. **Linear Warm-Up**
2. **Cosine Annealing Decay**

Benefits:
- Stabilizes transformer training
- Prevents early gradient instability
- Improves convergence on small datasets

---

## 🧪 Evaluation Metrics

The following metrics are calculated:

```python
accuracy_score()
precision_score()
recall_score()
f1_score()
roc_auc_score()
```

---

## 📊 Visualizations Included

| Visualization | Purpose |
|---------------|---------|
| Model Architecture Plot | Layer-wise ViT structure |
| Accuracy Curve | Training performance |
| Loss Curve | Model convergence |
| Attention Flow | Transformer attention behavior |
| Patch Extraction Visualization | SPT patch generation |

---

## 🧰 Tech Stack

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | Deep learning framework |
| TensorFlow Addons | AdamW optimizer |
| NumPy | Numerical operations |
| OpenCV (`cv2`) | MRI image loading |
| scikit-learn | Metrics & train-test split |
| matplotlib | Visualization |
| tqdm | Progress tracking |
| visualkeras | Model visualization |

---

## 🗂️ Project Structure

```bash
VisionTransformer/
│
├── Vision Transformer.py      # Main training script
├── README.md                  # Project documentation
├── vit_classifier.png         # Model architecture image
├── model1.png                 # Visualized model structure
└── dataset/
    ├── atrofi/
    ├── iskemi/
    ├── normal/
    └── WMD/
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone <your-repository-link>
cd VisionTransformer
```

---

### 2️⃣ Install Dependencies

```bash
pip install tensorflow tensorflow-addons numpy matplotlib scikit-learn opencv-python tqdm visualkeras ann_visualizer
```

---

### 3️⃣ Prepare Dataset

Organize dataset folders as:

```bash
train/
├── atrofi/
├── iskemi/
├── normal/
└── WMD/
```

Update the dataset path in the script:

```python
dataset_path = '/content/drive/MyDrive/train'
```

---

### 4️⃣ Run the Project

```bash
python "Vision Transformer.py"
```

---

## 🔍 Single Image Prediction

```python
image = cv2.imread("sample.jpg")
image = cv2.resize(image, (72,72))

prediction = model.predict(image)
predicted_class = np.argmax(prediction)
```

---

## 💡 Key Design Decisions

- Built **entirely from scratch** without pretrained models
- Used **SPT + LSA** to improve locality learning
- Chose **AdamW** for transformer optimization
- Applied **GELU activation** instead of ReLU
- Used **SparseCategoricalCrossentropy(from_logits=True)** for numerical stability

---

## 🔬 Future Improvements

- [ ] Add pretrained ViT transfer learning
- [ ] Implement GradCAM attention visualization
- [ ] Add Streamlit/Gradio deployment
- [ ] Export model to ONNX/TFLite
- [ ] Improve minority-class AUC score
- [ ] Add k-fold cross validation

---

## 📄 Research Reference

**Paper:**  
*Deep Learning Architecture with Shifted Patch Tokenization and Transformers for Accurate Brain Disease Classification on MR Images*  
Published in **IEEE Sensors Journal**

---

## ⚠️ Disclaimer

> This project is intended for **educational and research purposes only**.  
> It is **not** a clinical diagnostic tool and should not replace professional medical evaluation.

---

## 🙋 Author

**Valmiki**

---

## 📄 License

This project is open source and available under the **MIT License**.
