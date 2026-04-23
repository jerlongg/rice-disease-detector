# Rice Disease Detector

A deep learning system for automatic rice leaf disease detection using transfer learning. Detects 6 conditions (5 diseases + healthy) with high accuracy and provides interpretable predictions via Grad-CAM++ visualizations.

**Models:** ResNet-50 (high accuracy) & MobileNetV3-Large (lightweight, fast)  
**Dataset:** 13,250 labeled rice leaf images across 6 classes  
**Deployment:** Flask web app with dual-model support, runs on localhost or cloud

---

## 🎯 Quick Start

### Prerequisites
- Python 3.9+
- GPU recommended (CUDA 11.8+)

### Installation

```bash
# Clone or navigate to project
cd rice-disease-detector

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
cd app
python app.py
```

**Access:** Open your browser → `http://localhost:5000`

---

## 📊 Key Features

✅ **Dual-Model System**
- **ResNet-50**: 25.6M params, ~95% accuracy, high reliability
- **MobileNetV3**: 5.4M params, ~94% accuracy, 5× faster inference

✅ **Interpretability**
- Grad-CAM++ heatmaps show which leaf regions triggered the prediction
- Helps farmers/agronomists understand and verify AI decisions

✅ **User-Friendly Web Interface**
- Upload leaf images (JPG/PNG)
- Switch between models mid-session
- See confidence scores and per-class probabilities
- Visual heatmap overlay on predictions

✅ **Actionable Guidance**
- Disease description and symptoms
- Recommended treatment options
- Severity levels (none/medium/high)
- Type (Bacterial/Fungal/Viral/Insect)

✅ **Production-Ready**
- Model weights in optimized `.safetensors` format
- GPU-accelerated inference (~200-400ms per image)
- Error handling and validation
- API endpoints (JSON responses)

---

## 🌾 Disease Classes

| Class | Type | Severity | Key Symptom |
|-------|------|----------|-------------|
| **Bacterial Blight** | Bacterial | High | Yellow-white leaf stripes |
| **Leaf Blast** | Fungal | High | Diamond-shaped gray lesions |
| **Brown Spot** | Fungal | Medium | Circular brown spots with yellow halos |
| **Tungro Virus** | Viral | High | Yellow-orange discoloration, stunted growth |
| **Healthy Leaf** | None | None | No visible symptoms |
| **Rice Hispa** | Insect | Medium | White streaks parallel to veins |

---

## 📁 Project Structure

```
rice-disease-detector/
├── README.md                         
├── requirements.txt                 
│
├── app/                                
│   ├── app.py                         
│   ├── model.py                      
│   ├── inference.py                
│   └── templates/
│       └── index.html                
│
├── models/                             
│   ├── resnet50_rice.safetensors      # ResNet-50 checkpoint
│   └── mobilenetv3_rice.safetensors   # MobileNetV3 checkpoint
│
├── notebooks/                          
│   ├── 01_eda.ipynb                   
│   ├── 02_split.ipynb                 
│   ├── 03_train_rice_resnet50.ipynb   
│   ├── 04_evaluate_rice_resnet50.ipynb
│   ├── 05_train_mobilenetv3.ipynb     
│   ├── 06_evaluate_rice_mobilenetv3.ipynb 
│   ├── resnet_architecture.md         
│   └── mobilenetv3_architecture.md    
│
├── data/
│   ├── raw/                           
│   ├── processed/
│   │   ├── train/                   
│   │   ├── val/                     
│   │   └── test/                     
│   └── samples/                      
│
└── outputs/    # Results & visualizations
```

---

## 🚀 How to Use the Web App

### Basic Workflow

1. **Open App** → `http://localhost:5000`
2. **Select Model** → Choose ResNet-50 (default) or MobileNetV3
3. **Upload Image** → Click upload or drag-and-drop a rice leaf photo
4. **View Prediction**
   - Top prediction with confidence (0–100%)
   - Disease type and severity
   - Recommended actions
   - All class probabilities
   - Grad-CAM++ heatmap overlay

### API Endpoints

#### `/predict` (POST)
Upload image and get prediction.

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@leaf.jpg" \
  -F "model=resnet50"
```

**Response:**
```json
{
  "model": "resnet50",
  "class": "Blast",
  "label": "Leaf Blast",
  "type": "Fungal",
  "confidence": 94.23,
  "description": "Diamond-shaped lesions with gray centers and brown borders...",
  "action": "Apply fungicides (tricyclazole)...",
  "severity": "high",
  "low_conf": false,
  "original_b64": "iVBORw0KGgoAAAANS...",  # Base64 image
  "heatmap_b64": "iVBORw0KGgoAAAANS...",   # Base64 heatmap
  "all_scores": [
    {"class": "Blast", "label": "Leaf Blast", "score": 0.9423},
    {"class": "Brown_Spot", "label": "Brown Spot", "score": 0.0342},
    ...
  ]
}
```

#### `/models` (GET)
Get available models.

```bash
curl http://localhost:5000/models
```

Response:
```json
{
  "models": ["resnet50", "mobilenetv3"],
  "default": "resnet50"
}
```

#### `/health` (GET)
Check server status.

```bash
curl http://localhost:5000/health
```

---

## 📈 Model Performance

**Test Set Evaluation (2,463 images)**

| Metric | ResNet-50 | MobileNetV3 |
|--------|-----------|-------------|
| **Accuracy** | 99.96% | 99.92% |
| **Precision** | 99.96% avg | 99.92% avg |
| **Recall** | 99.97% avg | 99.93% avg |
| **F1-Score** | 0.9997 avg | 0.9993 avg |
| **Parameters** | 25.6M | 3.1M |
| **Inference (GPU)** | ~350ms | ~70ms |
| **Inference (CPU)** | ~2.5s | ~500ms |

Both models achieve near-perfect performance across all 6 disease classes. MobileNetV3 provides exceptional accuracy with 8.2× fewer parameters—ideal for constrained environments.

---

## 🔧 Training Your Own Model

To retrain models on custom data:

```bash
# Prepare data
cd notebooks
jupyter notebook 02_split.ipynb  # Organize train/val/test

# Train ResNet-50
jupyter notebook 03_train_rice_resnet50.ipynb

# Evaluate
jupyter notebook 04_evaluate_rice_resnet50.ipynb

# Train MobileNetV3
jupyter notebook 05_train_mobilenetv3.ipynb
jupyter notebook 06_evaluate_rice_mobilenetv3.ipynb
```

**Key configs in notebooks:**
- `IMG_SIZE = 224` — Input image size
- `BATCH_SIZE = 32` — Batch size
- `PHASE1_EPOCHS = 10` — Head-only training
- `PHASE2_EPOCHS = 20` — Fine-tuning
- `PHASE1_LR = 1e-3`, `PHASE2_LR = 1e-4` — Learning rates

---

## 📝 Dataset

**Total Images:** 13,250 across 3 splits

| Class | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Bacterial_Blight | 1,838 | 399 | 415 | 2,652 |
| Blast | 1,302 | 286 | 272 | 1,860 |
| Brown_Spot | 2,345 | 503 | 503 | 3,351 |
| Tungro | 699 | 165 | 162 | 1,026 |
| Healthy_Rice_Leaf | 914 | 196 | 196 | 1,306 |
| Rice_Hispa | 2,137 | 458 | 460 | 3,055 |
| **TOTAL** | **9,235** | **2,007** | **2,008** | **13,250** |

**Augmentation:** Horizontal flips, rotations (±15°), color jitter, random crops, shadows, JPEG compression, solid background patches.


---

## 📚 Additional Resources

- **Detailed Methodology:** See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
- **Architecture Details:** [ResNet-50 Explanation](notebooks/resnet_architecture.md) | [MobileNetV3 Explanation](notebooks/mobilenetv3_architecture.md)
- **Training Notebooks:** See `notebooks/` folder
- **Evaluation Metrics:** Run evaluation notebooks in `notebooks/`

---

## 📄 License

This project is provided as-is for educational and agricultural research purposes.

---

##  Development Notes

- **Framework:** PyTorch 2.1+
- **Model Format:** SafeTensors (faster loading, safer serialization)
- **Preprocessing:** OpenCV (cv2) for image ops, NumPy for math
- **Visualization:** Grad-CAM++ (class activation mapping)
- **Web:** Flask (lightweight, no complex dependencies)
- **GPU:** CUDA 11.8+ via PyTorch

**Key design decisions:**
- Transfer learning (ImageNet pretraining) for data efficiency
- Two-phase training (freeze backbone → fine-tune) for stability
- Weighted sampling during training to handle class imbalance
- Label smoothing (0.1) to reduce overconfidence on OOD inputs
- Dropout + batch norm for regularization

---

## 🎓 Citation

If you use this project, please reference:

```bibtex
@software{rice_disease_detector_2026,
  title={Rice Disease Detector: Deep Learning System for Automated Leaf Disease Classification},
  author={Jerlong},
  year={2026},
  url={https://github.com/jerlongg/rice-disease-detector}
}
```

---

