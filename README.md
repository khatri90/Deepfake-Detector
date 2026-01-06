# DeepFake Face Detector

> A Deep Learning Solution for Detecting AI-Generated Synthetic Faces

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**DeepFake Face Detector** is a state-of-the-art deep learning system that distinguishes between authentic human photographs and AI-generated synthetic faces (from GANs like StyleGAN, Midjourney, DALL-E, etc.).

As AI-generated imagery becomes increasingly sophisticated, the ability to detect synthetic content is critical for:
- **Media Authentication** - Verify the authenticity of images in journalism
- **Identity Verification** - Prevent fraud in KYC/onboarding processes
- **Social Media Integrity** - Combat fake profiles and misinformation
- **Digital Forensics** - Support investigations involving manipulated media

## Key Features

- **Transfer Learning** - Fine-tuned EfficientNet-B0/ResNet50 pretrained on ImageNet
- **High Accuracy** - Optimized for detecting subtle GAN artifacts
- **Production Ready** - Modular codebase with configuration management
- **Comprehensive Pipeline** - End-to-end preprocessing, training, and evaluation

## Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Image                          │
│                    (224 x 224 x 3)                      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              EfficientNet-B0 Backbone                   │
│              (Pretrained on ImageNet)                   │
│                                                         │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│   │  Conv   │──▶│  MBConv │──▶│  MBConv │──▶ ...       │
│   │  Stem   │   │ Blocks  │   │ Blocks  │              │
│   └─────────┘   └─────────┘   └─────────┘              │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               Global Average Pooling                    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                 Classification Head                     │
│                                                         │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│   │ Dropout  │──▶│  Dense   │──▶│ Softmax  │           │
│   │  (0.2)   │   │    2     │   │          │           │
│   └──────────┘   └──────────┘   └──────────┘           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Output: [Real, AI-Generated]               │
└─────────────────────────────────────────────────────────┘
```

## Dataset

| Category | Count | Description |
|----------|-------|-------------|
| **Real Faces** | 5,000 | Authentic photographs from CelebA dataset |
| **AI-Generated** | 4,630 | Synthetic faces from StyleGAN |
| **Total** | 9,630 | Balanced binary classification |

### Data Split
- **Training**: 70% (6,741 images)
- **Validation**: 15% (1,445 images)
- **Testing**: 15% (1,444 images)

> **Note**: Images are not included in this repository due to size. See [Dataset Setup](#dataset-setup) below.

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/deepfake-face-detector.git
cd deepfake-face-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Download the dataset from [Kaggle/Google Drive/etc.]
2. Extract images into the project directory:

```
deepfake-face-detector/
├── Real Images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── AI-Generated Images/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── ...
```

## Project Structure

```
deepfake-face-detector/
│
├── config.py              # Configuration & hyperparameters
├── dataset.py             # Custom PyTorch Dataset class
├── preprocessing.py       # Data transforms & DataLoader setup
├── utils.py               # Helper functions & utilities
├── requirements.txt       # Python dependencies
│
├── Real Images/           # Real face images (not in repo)
├── AI-Generated Images/   # Synthetic face images (not in repo)
│
└── output/                # Generated during training
    ├── models/            # Saved model checkpoints
    └── logs/              # Training logs & metrics
```

## Usage

### 1. Verify Data Pipeline

```bash
python preprocessing.py
```

### 2. Train Model

```bash
python train.py
```

### 3. Evaluate Model

```bash
python evaluate.py --checkpoint output/models/best_model.pth
```

### 4. Inference on Single Image

```bash
python predict.py --image path/to/image.jpg
```

## Configuration

All hyperparameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | 224 | Input image dimensions |
| `BATCH_SIZE` | 32 | Training batch size |
| `LEARNING_RATE` | 1e-4 | Initial learning rate |
| `NUM_EPOCHS` | 30 | Maximum training epochs |
| `MODEL_NAME` | efficientnet_b0 | Backbone architecture |

## Data Augmentation

Training images undergo real-time augmentation:

- Random Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation)
- Random Erasing (p=0.1)

## Technical Details

### Why Transfer Learning?

1. **Limited Data**: ~10K images is insufficient for training from scratch
2. **Feature Reuse**: ImageNet features (edges, textures) transfer well to faces
3. **Faster Convergence**: Pretrained weights provide strong initialization
4. **Better Generalization**: Reduces overfitting on small datasets

### Detection Approach

The model learns to identify subtle artifacts in AI-generated faces:
- Unnatural skin textures
- Inconsistent lighting/shadows
- Asymmetric facial features
- Background anomalies
- Frequency domain artifacts

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | TBD |
| **Precision** | TBD |
| **Recall** | TBD |
| **F1-Score** | TBD |
| **AUC-ROC** | TBD |

## Future Enhancements

- [ ] Add frequency domain (FFT) feature analysis
- [ ] Implement attention visualization (Grad-CAM)
- [ ] Support for video deepfake detection
- [ ] REST API for deployment
- [ ] ONNX export for edge deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [StyleGAN](https://github.com/NVlabs/stylegan)
- [PyTorch](https://pytorch.org/)
- [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)

---

**Disclaimer**: This tool is intended for research and educational purposes. Always respect privacy and legal considerations when analyzing facial images.
