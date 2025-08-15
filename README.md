# COMPASS  
**Cross-modal Object Mapping and Precise Automated Scene Segmentation**  
[![GitHub](https://img.shields.io/badge/Code-View_on_GitHub-black?logo=github)](https://github.com/SANKALP9TRIPATHI/COMPASS/blob/main/compass%20.ipynb)

---

## 📌 Abstract
The **COMPASS** model is a **dense scene localization framework** that operates on natural language queries, enabling fine-grained mapping of target objects and regions within complex, multi-object environments.  
It bridges the gap between **visual perception** and **linguistic understanding** by integrating joint vision–language embeddings to pinpoint queried objects with **pixel-level** or **bounding-box precision**.  

Unlike traditional object detection pipelines limited to predefined categories, COMPASS is **open-vocabulary**, capable of responding to **arbitrary text prompts**, including unseen object names, descriptive phrases, and contextual relations.  

**Key Applications:**
- 🛡 Surveillance analytics – Detect suspicious activity from descriptions.
- 🤖 Autonomous navigation – Robotic localization via spoken instructions.
- 🕶 Augmented reality – Highlight described objects in live video.
- 📊 Contextual visual analytics – Dynamically filter scene content by queries.

By combining **cross-modal attention** with **multi-scale feature extraction**, COMPASS achieves **custom-IoU score of 0.64** even in cluttered and occluded scenes.

---

## 📂 Dataset
**Dataset Used:** [COCO 2017](https://cocodataset.org/#home)  
- **118,000** training images  
- **5,000** validation images  
- Bounding boxes, segmentation masks, and captions for **80 object categories**  

### Why COCO 2017?
- **Rich Annotations:** Supports both visual and textual modalities.  
- **Complex Scenes:** Cluttered and diverse backgrounds for robust localization.  
- **Multi-Object Context:** Many interacting objects per image.  
- **Benchmark Standard:** Enables fair performance comparison.  

### Data Preprocessing in COMPASS
- Resizing + normalization using ImageNet mean/std.  
- Tokenization & embedding for text queries.  
- Filtering for valid bounding boxes.  
- Augmentations: horizontal flip & color jitter.

---

## 🏗 Architecture Overview
COMPASS tackles **three key challenges**:  
- **Semantic richness:** Handling complex, compositional queries.  
- **Scale variation:** Multi-scale feature representation.  
- **Open-vocabulary adaptability:** No retraining needed for unseen objects.

### 1. Visual Feature Backbone
- **Vision Transformer (ViT)** pre-trained on large image–text datasets.  
- **Feature Pyramid Network (FPN)** for multi-scale fusion.  

### 2. Text Encoding Module
- **CLIP-based Transformer text encoder** with token-level embeddings.  

### 3. Cross-Modal Fusion Layer
- **Multi-Head Cross-Attention (MHCA)** for bidirectional alignment.  

### 4. Localization Head
- **Bounding Box Branch:** DETR-style regression, anchor-free.  
- **Segmentation Branch:** Pixel-wise prediction via upsampling + skip connections.  

### 5. Loss Functions
- **CIoU Loss:** Bounding box regression.  
- **Cross-Entropy Loss:** Classification confidence.  
- **Binary Cross-Entropy:** Segmentation refinement.  
- **Contrastive Loss:** Semantic–visual alignment.

---

## 📊 Previous Approaches & Limitations
| Approach | Limitation |
|----------|------------|
| **Object Detection + Caption Matching** | Fails on unseen/unlabeled objects. |
| **Region Proposal + Language Grounding** | Computationally heavy, misses small objects. |
| **Transformer-based Grounding Models (MDETR, GroundingDINO)** | Only bounding boxes, no fine-grained segmentation. |

**COMPASS** improves by combining **detection + pixel-level segmentation** in a single pipeline.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SANKALP9TRIPATHI/COMPASS.git
cd COMPASS


python3 -m venv compass_env
source compass_env/bin/activate  # On Linux/Mac
compass_env\Scripts\activate     # On Windows


pip install -r requirements.txt


jupyter notebook "compass .ipynb"


#inside the Single Image Prediction cell:-
image_path = "sample.jpg"
query = "the person in a red shirt"

