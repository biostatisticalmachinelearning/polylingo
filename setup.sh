#!/bin/bash
# Setup script for Unicode Character ML Pipeline

set -e

echo "=========================================="
echo "Unicode Character ML Pipeline Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for GPU
echo ""
echo "Checking GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'MPS available: {torch.backends.mps.is_available()}')
"

# CUDA-specific setup hint
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo ""
    echo "CUDA detected! You're good to go."
else
    echo ""
    echo "No CUDA detected. If you have an NVIDIA GPU, install CUDA version:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the environment:  source venv/bin/activate"
echo "  2. Generate the dataset:      python generate_unicode_dataset.py"
echo "  3. Train a model:             python train_vae.py --epochs 100"
echo ""
echo "Or use make commands:  make help"
