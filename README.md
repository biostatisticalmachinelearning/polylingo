# Unicode Character ML Pipeline

A machine learning pipeline for generating and analyzing Unicode characters across 52 world writing systems. Includes a dataset generator, classifier, variational autoencoder (VAE), and diffusion model.

## Features

- **Dataset Generator**: Creates 64x64 PNG images for ~17,000 Unicode characters using Google's Noto fonts
- **ResNet Classifier**: Classifies characters into their script/language family
- **VAE**: Learns a latent space representation of characters for exploration and interpolation
- **Diffusion Model**: Generates new, convincing character shapes with class-conditional control

## Quick Start

```bash
# Clone the repo
git clone <repo-url>
cd autoencoding-scripts

# Setup (auto-detects GPU)
make setup

# Generate the dataset (~17k images, takes a few minutes)
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
output/
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
python train.py --epochs 50 --batch-size 128
```

Test predictions:
```bash
python predict.py output/latin/0041.png output/hiragana/3042.png
```

### 2. Variational Autoencoder (VAE)

Train a VAE to learn a latent space representation:

```bash
make train-vae
# or
python train_vae.py --epochs 100 --latent-dim 64 --beta 1.0
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
python train_diffusion.py --epochs 200 --batch-size 64
```

Options:
- `--timesteps`: Number of diffusion steps (default: 1000)
- `--cfg-scale`: Classifier-free guidance scale (default: 3.0)
- `--beta-schedule`: Noise schedule, "cosine" or "linear"

**Note**: Diffusion models need more training time. For experimentation, start with:
```bash
make train-diffusion-quick  # 50 epochs
```

## Sampling & Exploration

### VAE Exploration

```bash
# Generate random samples from latent space
python explore_latent.py sample --num 64 --output samples.png

# Interpolate between two characters
python explore_latent.py interpolate \
    --char1 output/latin/0041.png \
    --char2 output/latin/005A.png \
    --steps 11 \
    --output A_to_Z.png

# Find visually similar characters
python explore_latent.py neighbors --char output/latin/0041.png --k 10

# Visualize latent space with t-SNE
python explore_latent.py visualize --method tsne --output tsne.png

# Traverse individual latent dimensions
python explore_latent.py traverse --dim 0 --output dim0.png
python explore_latent.py traverse --output traversals/  # All dimensions
```

### Diffusion Sampling

```bash
# Generate random samples (DDIM for speed)
python sample_diffusion.py --num 64 --ddim --output samples.png

# Generate specific scripts
python sample_diffusion.py --class latin --num 16 --ddim
python sample_diffusion.py --class hiragana --num 16 --cfg-scale 5.0

# Generate one sample per script
python sample_diffusion.py --all-classes --ddim --output all_scripts.png

# List available scripts
python sample_diffusion.py --list-classes
```

## Class Balancing

The dataset is highly imbalanced:
- **Largest**: Han CJK (8,000 characters)
- **Smallest**: Tagalog (17 characters)

All models use **weighted random sampling** to ensure minority scripts are well-represented:

```
Class distribution and sampling weights:
  han_cjk                  :  6400 samples, weight=0.275
  hangul                   :  1880 samples, weight=0.507
  ...
  tagalog                  :    14 samples, weight=5.872
```

## Project Structure

```
autoencoding-scripts/
├── generate_unicode_dataset.py  # Dataset generation
├── train.py                     # Classifier training
├── predict.py                   # Classifier inference
├── train_vae.py                 # VAE training
├── explore_latent.py            # VAE latent space tools
├── train_diffusion.py           # Diffusion training
├── sample_diffusion.py          # Diffusion sampling
├── ml/                          # Classifier module
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   └── trainer.py
├── vae/                         # VAE module
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   └── trainer.py
├── diffusion/                   # Diffusion module
│   ├── config.py
│   ├── dataset.py
│   ├── diffusion.py
│   ├── model.py
│   └── trainer.py
├── requirements.txt
├── Makefile
└── README.md
```

## GPU Memory Requirements

| Model | Batch Size | ~VRAM |
|-------|------------|-------|
| Classifier | 128 | 2-3 GB |
| VAE | 128 | 2-3 GB |
| Diffusion | 64 | 4-6 GB |
| Diffusion | 32 | 2-3 GB |

Reduce batch size if you run out of memory:
```bash
python train_diffusion.py --batch-size 32
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
