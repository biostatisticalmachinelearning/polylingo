.PHONY: setup setup-cuda setup-mps data train-classifier train-vae train-diffusion sample clean help

# Default target
help:
	@echo "Unicode Character ML Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create venv and install dependencies (auto-detect GPU)"
	@echo "  make setup-cuda     - Setup for NVIDIA GPU (CUDA)"
	@echo "  make setup-mps      - Setup for Apple Silicon (MPS)"
	@echo "  make setup-cpu      - Setup for CPU only"
	@echo ""
	@echo "Data:"
	@echo "  make data           - Download fonts and generate character images"
	@echo ""
	@echo "Training:"
	@echo "  make train-classifier         - Train ResNet classifier (50 epochs)"
	@echo "  make train-vae                - Train VAE (100 epochs)"
	@echo "  make train-diffusion          - Train diffusion model (200 epochs)"
	@echo "  make train-diffusion-quick    - Train diffusion model (50 epochs, for testing)"
	@echo "  make train-all                - Train all models"
	@echo ""
	@echo "Sampling & Exploration:"
	@echo "  make sample-vae               - Generate VAE samples and reconstructions"
	@echo "  make sample-diffusion         - Generate diffusion samples"
	@echo "  make explore-latent           - Explore VAE latent space"
	@echo ""
	@echo "Utilities:"
	@echo "  make test           - Quick test of all pipelines (2-3 epochs each)"
	@echo "  make clean          - Remove generated files and checkpoints"

# Python and venv
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

# Setup targets
setup: $(VENV)/bin/activate
	@echo "Setup complete! Activate with: source venv/bin/activate"

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Detecting GPU..."
	@$(PY) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

setup-cuda: $(VENV)/bin/activate
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	@echo "CUDA setup complete!"

setup-mps: $(VENV)/bin/activate
	@echo "MPS (Apple Silicon) uses default PyTorch installation"
	@echo "Setup complete!"

setup-cpu: $(VENV)/bin/activate
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@echo "CPU-only setup complete!"

# Data generation
data: $(VENV)/bin/activate
	$(PY) generate_unicode_dataset.py

# Training targets
train-classifier: $(VENV)/bin/activate
	$(PY) train.py --epochs 50 --batch-size 128

train-vae: $(VENV)/bin/activate
	$(PY) train_vae.py --epochs 100 --batch-size 128 --latent-dim 64

train-diffusion: $(VENV)/bin/activate
	$(PY) train_diffusion.py --epochs 200 --batch-size 64 --sample-interval 20

train-diffusion-quick: $(VENV)/bin/activate
	$(PY) train_diffusion.py --epochs 50 --batch-size 64 --sample-interval 10

train-all: train-classifier train-vae train-diffusion

# Sampling and exploration
sample-vae: $(VENV)/bin/activate
	$(PY) explore_latent.py sample --num 64 --output vae_samples.png
	$(PY) explore_latent.py interpolate \
		--char1 output/latin/0041.png \
		--char2 output/latin/005A.png \
		--output vae_interpolation.png
	@echo "Samples saved to vae_samples.png and vae_interpolation.png"

sample-diffusion: $(VENV)/bin/activate
	$(PY) sample_diffusion.py --num 64 --ddim --output diffusion_samples.png
	$(PY) sample_diffusion.py --all-classes --ddim --cfg-scale 3.0 --output diffusion_all_classes.png
	@echo "Samples saved to diffusion_samples.png and diffusion_all_classes.png"

explore-latent: $(VENV)/bin/activate
	mkdir -p latent_exploration
	$(PY) explore_latent.py visualize --method tsne --output latent_exploration/tsne.png
	$(PY) explore_latent.py stats --output-dir latent_exploration
	$(PY) explore_latent.py traverse --output latent_exploration/traversals
	@echo "Exploration results saved to latent_exploration/"

# Quick test (2-3 epochs each)
test: $(VENV)/bin/activate
	@echo "Testing classifier..."
	$(PY) train.py --epochs 2 --batch-size 64
	@echo ""
	@echo "Testing VAE..."
	$(PY) train_vae.py --epochs 3 --batch-size 64 --latent-dim 32
	@echo ""
	@echo "Testing diffusion..."
	$(PY) train_diffusion.py --epochs 2 --batch-size 32 --sample-interval 1
	@echo ""
	@echo "All tests passed!"

# Cleanup
clean:
	rm -rf checkpoints/ checkpoints_vae/ checkpoints_diffusion/
	rm -rf latent_traversals/ latent_exploration/
	rm -f *.png
	rm -rf __pycache__ */__pycache__
	@echo "Cleaned up generated files"

clean-all: clean
	rm -rf $(VENV) output/ fonts/
	@echo "Cleaned everything including venv, output, and fonts"
