# PlantGuard-AI

Deep learning plant disease detection system using PyTorch and transfer learning. Classifies 38 crop diseases across 14 plant species with 95% accuracy on the PlantVillage dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-95.02%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Project Overview

PlantGuard-AI is a computer vision system that helps farmers and agricultural professionals identify plant diseases from leaf images. Using PyTorch and ResNet18 transfer learning, the model achieves **95.02% test accuracy** on 8,147 images across 38 disease classes.

**Key Features:**
- **Transfer Learning** with ResNet18 (ImageNet pre-trained weights)
- **Custom Data Augmentation** pipeline for robust predictions
- **Production-Ready** training loop with checkpointing and learning rate scheduling
- **Comprehensive Evaluation** with confusion matrices and per-class metrics
- **Automated Visualizations** for model performance analysis

## 📊 Model Performance

- **Test Accuracy**: 95.02%
- **Precision**: 95.75%
- **Recall**: 95.02%
- **F1-Score**: 95.01%
- **Training Time**: ~2 hours (10 epochs on CPU)
- **Dataset Size**: 54,305 images across 38 classes

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/SNMiguel/PlantGuard-AI.git
cd PlantGuard-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Download PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
2. Extract to `data/raw/` (should contain 38 disease class folders)

### Train the Model
```bash
# Quick training (5 epochs)
python main.py --data_dir data/raw --epochs 5 --batch_size 32

# Full training (10 epochs, recommended)
python main.py --data_dir data/raw --epochs 10 --batch_size 32 --verbose

# Advanced training with frozen backbone
python main.py --data_dir data/raw --epochs 15 --freeze_backbone
```

### Make Predictions
```bash
# Predict on a single image
python predict.py --image_path path/to/leaf_image.jpg --model_path models/saved/resnet18_best.pth
```

## 📁 Project Structure
```
PlantGuard-AI/
├── data/
│   └── raw/                     # PlantVillage dataset (38 disease folders)
├── models/
│   ├── resnet_model.py          # ResNet18 implementation with transfer learning
│   └── saved/                   # Trained model checkpoints (.pth files)
├── utils/
│   ├── data_loader.py           # Dataset loading and splitting
│   ├── transforms.py            # Data augmentation pipeline
│   ├── metrics.py               # Evaluation metrics
│   └── visualizer.py            # Plotting and visualization
├── results/
│   └── visualizations/          # Training curves, confusion matrix, predictions
├── train.py                     # Training script
├── main.py                      # Main pipeline orchestrator
├── predict.py                   # Single image prediction
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

## 🌱 Supported Plant Diseases

The model classifies **38 disease classes** across **14 crop species**:

**Crops**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

**Example Diseases**:
- Apple: Scab, Black rot, Cedar apple rust
- Tomato: Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus
- Potato: Early blight, Late blight
- Grape: Black rot, Esca, Leaf blight
- And 23 more disease + healthy classes...

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: ResNet18 (pre-trained on ImageNet)
- **Transfer Learning**: Fine-tuned final layer for 38-class classification
- **Input Size**: 224x224 RGB images
- **Total Parameters**: 11.2M (19K trainable when backbone frozen)

### Data Augmentation
- Random horizontal/vertical flips
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformations
- ImageNet normalization

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau
- **Batch Size**: 32
- **Train/Val/Test Split**: 70/15/15

## 📈 Results

### Training Curves
*Training and validation loss/accuracy over 10 epochs*

### Confusion Matrix
*38x38 heatmap showing per-class predictions*

### Sample Predictions
*Visual examples of correct and incorrect classifications*

(Check `results/visualizations/` for generated plots)

## 🔧 Customization

### Use Different Model Architecture
```python
# In main.py, change model parameter
python main.py --model resnet34  # Options: resnet18, resnet34, resnet50
```

### Adjust Training Parameters
```python
python main.py --epochs 20 --batch_size 64 --freeze_backbone
```

### Add Custom Disease Classes
1. Add new disease folders to `data/raw/`
2. Model will automatically detect and train on new classes

## 📝 Key Learnings

- **Transfer learning** significantly reduces training time and data requirements
- **Data augmentation** is critical for model generalization on agricultural images
- **ResNet18** provides excellent balance between accuracy and computational efficiency
- **Proper train/val/test splitting** prevents overfitting on time-series-like datasets
- **Checkpointing** enables resuming training and deploying best models

## 🎓 Skills Demonstrated

- Deep learning with PyTorch
- Transfer learning and fine-tuning
- Computer vision for image classification
- Custom data augmentation pipelines
- Production ML workflows (training loops, checkpointing, evaluation)
- Large-scale dataset handling (54K+ images)
- Model performance analysis and visualization

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features or improvements
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Miguel Shema Ngabonziza**
- LinkedIn: [linkedin.com/in/migztech](https://linkedin.com/in/migztech)
- GitHub: [github.com/SNMiguel](https://github.com/SNMiguel)
- Portfolio: [migztech.vercel.app](https://migztech.vercel.app)

## 🙏 Acknowledgments

- PlantVillage dataset provided by Penn State University
- ResNet architecture from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Built as part of technical skill development for AI/ML engineering roles
- Inspired by the need for accessible agricultural technology in developing regions

## 🔮 Future Enhancements

- [ ] Deploy as REST API with FastAPI
- [ ] Build Gradio web interface for live predictions
- [ ] Add explainability with Grad-CAM visualizations
- [ ] Implement mobile app for field use
- [ ] Expand to additional crop species
- [ ] Add disease severity classification
- [ ] Multi-language support for global accessibility

---

⭐ **If you found this project helpful, please consider giving it a star!**
