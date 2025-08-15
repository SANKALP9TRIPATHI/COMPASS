# COMPASS
The COMPASS (Cross-modal Object Mapping and Precise Automated Scene Segmentation) model is a dense scene localization framework that operates on natural language queries, enabling fine-grained mapping of target objects and regions within complex, multi-object environments. It bridges the gap between visual perception and linguistic understanding, integrating joint vision–language embeddings to pinpoint queried objects with pixel-level or bounding-box precision. 
Unlike traditional object detection pipelines, which are constrained to a predefined set of object categories (e.g., COCO’s 80 classes), COMPASS is open-vocabulary by design. This allows it to respond to arbitrary text prompts, including unseen object names, descriptive phrases, and contextual relations making it adaptable to diverse and dynamic domains 

🌟 Key Features
Open-Vocabulary Detection: Handles arbitrary text queries without being limited to predefined object categories

Cross-Modal Attention: Advanced fusion mechanism between visual and textual features

Multi-Scale Processing: Captures both fine-grained details and global scene context

Real-Time Inference: Optimized for practical deployment scenarios

Dense Scene Understanding: Excels in cluttered, occluded, or densely populated environments

Custom IoU Score: Achieves 0.64 IoU on complex localization tasks

🚀 Applications
Surveillance Analytics: Detect suspicious activities through natural language descriptions

Autonomous Navigation: Robotic localization based on spoken instructions

Augmented Reality: Real-time object highlighting in live video feeds

Contextual Visual Analytics: Dynamic scene content filtering via text queries

🛠️ Installation
Clone the repository:

bash
git clone https://github.com/your-username/compass-scene-localization.git
cd compass-scene-localization
Create a virtual environment:

bash
python -m venv compass_env
source compass_env/bin/activate  # On Windows: compass_env\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Download COCO 2017 dataset:

bash
# Create data directory
mkdir data
cd data

# Download COCO 2017 (training and validation sets)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract files
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
🎯 Quick Start
Basic Usage
python
import torch
from PIL import Image
from transformers import CLIPProcessor
from compass_model import EnhancedSceneLocalizationModel, COCOSceneLocalizationInference

# Load pre-trained model
model = EnhancedSceneLocalizationModel(num_classes=80)
checkpoint = torch.load("model_outputs/coco_scene_localization_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

# Initialize inference system
inference = COCOSceneLocalizationInference(model)

# Load image and make prediction
image = Image.open("path/to/your/image.jpg")
query = "person with red shirt"

# Get prediction
bbox, confidence, attention_weights = inference.predict(image, query)
print(f"Bounding box: {bbox}")
print(f"Confidence: {confidence:.3f}")

# Visualize results
inference.visualize_prediction(image, query, bbox, confidence)


