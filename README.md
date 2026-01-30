# Polylingo: Unicode Character ML Pipeline

A machine learning toolkit for analyzing and generating Unicode characters across 52 world writing systems. Includes a dataset generator, classifier, variational autoencoder (VAE), and diffusion model.

## Features

- **Dataset Generator**: Creates 64x64 PNG images for ~17,000 Unicode characters using Google's Noto fonts
- **ResNet Classifier**: Classifies characters into their script/language family
- **VAE**: Learns a latent space representation of characters for exploration and interpolation
- **Diffusion Model**: Generates new, convincing character shapes with class-conditional control

## Quick Start

```bash
# Clone the repo
git clone https://github.com/biostatisticalmachinelearning/polylingo
cd polylingo

# Setup (auto-detects GPU)
make setup

# Generate the dataset (~17k images)
make data

# Run a quick test of all models (2-3 epochs each)
make test
```

## Setup

### Prerequisites

- Python 3.10+
- ~500MB disk space for fonts and images
- GPU recommended (NVIDIA CUDA or Apple MPS)

### Installation

**Auto-detect GPU (recommended):**
```bash
make setup
```

**NVIDIA GPU (CUDA):**
```bash
make setup-cuda
```

**Apple Silicon (MPS):**
```bash
make setup-mps
```

**CPU only:**
```bash
make setup-cpu
```

### Manual Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# For NVIDIA GPU, install CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Dataset Generation

Generate character images for 52 scripts:

```bash
make data
# or
python generate_unicode_dataset.py
```

This downloads Noto fonts and generates 64x64 PNG images:

```
data/unicode_chars/
├── latin/          # 526 images (A-Z, a-z, accented chars, etc.)
├── greek/          # 368 images
├── cyrillic/       # 297 images
├── arabic/         # 243 images
├── han_cjk/        # 8,000 images (Chinese characters)
├── hiragana/       # 91 images
├── hangul/         # 2,350 images (Korean syllables)
├── devanagari/     # 94 images (Hindi, Sanskrit)
├── ... (44 more scripts)
└── metadata.json   # Character names and codepoints
```

## Training

### 1. ResNet Classifier

Train a classifier to identify which script a character belongs to:

```bash
make train-classifier
# or
python scripts/train_classifier.py --epochs 50 --batch-size 128
```

### 2. Variational Autoencoder (VAE)

Train a VAE to learn a latent space representation:

```bash
make train-vae
# or
python scripts/train_vae.py --epochs 100 --latent-dim 64 --beta 1.0
```

Options:
- `--latent-dim`: Size of latent space (default: 64)
- `--beta`: KL divergence weight, higher = more disentangled (default: 1.0)
- `--conditional`: Train a class-conditional VAE

### 3. Diffusion Model

Train a diffusion model to generate new characters:

```bash
make train-diffusion
# or
python scripts/train_diffusion.py --epochs 200 --batch-size 64
```

Options:
- `--timesteps`: Number of diffusion steps (default: 1000)
- `--cfg-scale`: Classifier-free guidance scale (default: 3.0)
- `--beta-schedule`: Noise schedule, "cosine" or "linear"

**Note**: Diffusion models need more training time. For experimentation:
```bash
make train-diffusion-quick  # 50 epochs
```

## Sampling

### Diffusion Sampling

```bash
# Generate random samples (DDIM for speed)
python scripts/sample_diffusion.py --num-samples 64 --use-ddim

# Generate specific scripts
python scripts/sample_diffusion.py --class-label 0 --num-samples 16 --use-ddim

# With higher guidance
python scripts/sample_diffusion.py --cfg-scale 5.0 --use-ddim
```

## Project Structure

```
polylingo/
├── polylingo/                   # Main package
│   ├── __init__.py             # Package exports
│   ├── data/                   # Data loading and transforms
│   │   ├── dataset.py          # UnicodeDataset class
│   │   ├── transforms.py       # Image transforms
│   │   ├── balancing.py        # Class weight computation
│   │   └── loaders.py          # Data loader factory
│   ├── models/                 # Neural network architectures
│   │   ├── classifier.py       # ResNet classifier
│   │   ├── vae.py              # Variational Autoencoder
│   │   ├── unet.py             # U-Net for diffusion
│   │   └── diffusion.py        # Gaussian diffusion process
│   ├── configs/                # Configuration dataclasses
│   │   ├── base.py             # Base configuration
│   │   ├── classifier.py       # Classifier config
│   │   ├── vae.py              # VAE config
│   │   └── diffusion.py        # Diffusion config
│   ├── trainers/               # Training loops
│   │   ├── base.py             # Base trainer
│   │   ├── classifier.py       # Classifier trainer
│   │   ├── vae.py              # VAE trainer
│   │   └── diffusion.py        # Diffusion trainer
│   └── utils/                  # Utilities
│       ├── device.py           # Device detection
│       ├── ema.py              # Exponential moving average
│       ├── early_stopping.py   # Early stopping
│       └── schedulers.py       # LR schedulers
├── scripts/                    # Training/sampling scripts
│   ├── train_classifier.py
│   ├── train_vae.py
│   ├── train_diffusion.py
│   └── sample_diffusion.py
├── generate_unicode_dataset.py # Dataset generation
├── requirements.txt
├── setup.py
├── Makefile
└── README.md
```

## Class Balancing

The dataset is highly imbalanced:
- **Largest**: Han CJK (8,000 characters)
- **Smallest**: Tagalog (17 characters)

All models use **weighted random sampling** to ensure minority scripts are well-represented.

## GPU Memory Requirements

| Model | Batch Size | ~VRAM |
|-------|------------|-------|
| Classifier | 128 | 2-3 GB |
| VAE | 128 | 2-3 GB |
| Diffusion | 64 | 4-6 GB |
| Diffusion | 32 | 2-3 GB |

Reduce batch size if you run out of memory:
```bash
python scripts/train_diffusion.py --batch-size 32
```

## Tips for Better Results

### VAE
- Use `--beta 0.5` for better reconstructions (less disentangled)
- Use `--beta 4.0` for more disentangled latent space
- Try `--latent-dim 128` for more expressive representations

### Diffusion
- Train for at least 100 epochs for decent results
- Use `--cfg-scale 5.0` or higher for more distinct class features
- DDIM with 50 steps is ~20x faster than full DDPM sampling

## License

MIT
