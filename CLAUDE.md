# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Git Commit Policy
**NEVER automatically add or commit changes.** Only stage and commit files when explicitly requested by the user. This applies to all situations - no exceptions.

## Project Overview

Underwater image enhancement ML pipeline that automates manual GoPro RAW editing for Seattle Aquarium ROV surveys. Replaces time-intensive manual Adobe Lightroom editing with trained deep learning models.

## Environment Setup

```bash
python3.10 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt  # Full dev (training + inference)
pip install -r requirements_gui.txt  # GUI-only
```

## Common Commands

### Training (All-in-One)
```bash
huggingface-cli login  # One-time setup
python training/setup_and_train.py  # Downloads dataset, prepares, trains

# Resume from checkpoint
python training/train.py --resume checkpoints/checkpoint_epoch_10.pth

# Monitor progress
tensorboard --logdir logs
```

### Inference
```bash
python inference/inference.py input.jpg --checkpoint checkpoints/best_model.pth
python inference/inference.py /path/to/images --checkpoint checkpoints/best_model.pth --compare
python inference/inference.py input.jpg --checkpoint checkpoints/best_model.pth --full-size
```

### GPR Preprocessing
```bash
python preprocessing/preprocess_images.py /path/to/gpr/files --output-dir processed
```

### Testing
```bash
pytest tests/test_inference.py -v          # Inference tests (run in CI)
pytest tests/test_preprocessing.py -v      # Preprocessing tests
python training/test_dataset_scripts.py    # Dataset prep script tests
pytest tests/test_inference.py -v -k test_load_checkpoint  # Single test by name
```
CI ([.github/workflows/test-inference.yml](.github/workflows/test-inference.yml)) runs `pytest tests/test_inference.py` on Ubuntu/macOS/Windows with Python 3.10/3.11 (CPU-only torch). Tests use tiny synthetic images; fixtures live in [tests/conftest.py](tests/conftest.py).

### GUI Development
```bash
python gui/app.py  # Development mode
pyinstaller gui/pyinstaller.spec --clean --noconfirm  # Build executable
./dist/UnderwaterEnhancer --smoke-test  # Test build
```

### Cleanup
```bash
python training/cleanup_training.py --dry-run  # Preview
python training/cleanup_training.py --force  # Execute
```

## High-Level Architecture

### Model Architectures

Model selected via `model:` in [setup_and_train_config.yaml](setup_and_train_config.yaml) (default: `ss_uie`):

| Model (`model:` value) | File | Params | Use Case |
|-------|------|--------|----------|
| `unet` | [src/models/unet_autoencoder.py](src/models/unet_autoencoder.py) | ~31M | Faster training, good baseline |
| `ushape_transformer` | [src/models/ushape_transformer.py](src/models/ushape_transformer.py) | ~50M | Better quality, slower training |
| `ss_uie` | [src/models/ss_uie.py](src/models/ss_uie.py) | — | Default. State-Space UIE (Mamba + FFT). **Requires CUDA + `mamba-ssm`** |
| `3d_lut` | [src/models/lut_3d.py](src/models/lut_3d.py) | <1M | Image-adaptive 3D LUT. Per-pixel colour transform; preserves fine texture by construction. Best for mostly-global (Lightroom-style) edits |

[src/models/attention_unet.py](src/models/attention_unet.py) also exists as an additional variant. Inference auto-detects architecture from checkpoint contents — see [inference/inference.py](inference/inference.py).

**U-Net**: 5-level encoder (64→128→256→512→1024 channels), symmetric decoder with skip connections. Combined L1+MSE loss (80%/20%).

**U-shape Transformer**: CMSFFT (Cross-scale Multi-scale Fusion FFT) + SGFMT (Spatial-Gated Feed-forward Modulation Transformer) attention mechanisms.

**SS-UIE**: State-Space model with Mamba blocks + FFT attention. GPU-only (needs `pip install mamba-ssm causal-conv1d timm einops`). Reference implementation in [lib/SS-UIE/](lib/SS-UIE/); setup notes in [training/SS_UIE_SETUP.md](training/SS_UIE_SETUP.md).

**3D LUT** ([Zeng et al., ECCV 2020](https://arxiv.org/abs/2009.14468)): a small CNN reads a downsampled thumbnail to predict per-image weights that fuse N learnable basis 3D LUTs; the colour transform is applied per-pixel to the full-resolution image (pure-PyTorch trilinear, CPU/MPS/CUDA). Because the full-res image never passes through a downsampling bottleneck, fine texture is preserved by construction. Trained by default with the **composite** loss; inference runs at native resolution. Design notes: [docs/3DLUT_PLAN.md](docs/3DLUT_PLAN.md).

### Loss Functions

Selected via `loss:` in config / `--loss` (default `auto`, which picks per model): `combined` (L1+MSE, in [training/train.py](training/train.py)), `ss_uie` (paper multi-term loss), or `composite` ([src/losses/composite_loss.py](src/losses/composite_loss.py)). The **composite** loss = L1 + (1−MS-SSIM) + Focal Frequency Loss (+ optional LPIPS) and targets the high-frequency texture that pixel losses over-smooth; it is self-contained (no extra required deps) except LPIPS, which needs `pip install lpips` and is off by default. Term weights are tunable via `--composite-weights L1 MSSSIM FFL LPIPS`.

### Data Pipeline

```
GPR Files → gpr_tools → DNG → TIFF (4606×4030) → Training patches (512×512)
```

1. **GPR Processing** ([preprocessing/](preprocessing/)): gpr_tools converts GPR→DNG, center crops to 4606×4030, saves as TIFF
2. **Dataset Prep** ([dataset_prep/](dataset_prep/)): Pairs raw/enhanced images, creates 80/20 train/val splits
3. **Training** ([training/train.py](training/train.py)): Early stopping, LR scheduling, saves best model by validation loss
4. **Inference** ([inference/inference.py](inference/inference.py)): Tiled processing for arbitrary sizes, batch support

### GUI Application

- **Framework**: customtkinter (native desktop, Tkinter-based)
- **Packaging**: PyInstaller (bundles PyTorch; see [BUILD_README.md](BUILD_README.md))
- **Entry point**: [gui/app.py](gui/app.py)
- **Core logic**: [src/gui/image_processor.py](src/gui/image_processor.py), [src/gui/main_window.py](src/gui/main_window.py)
- **GPR support**: [src/converters/gpr_converter.py](src/converters/gpr_converter.py) + bundled `binaries/<platform>/gpr_tools`

Build docs: [BUILD_README.md](BUILD_README.md) | macOS security: [gui/docs/MACOS_APP_INSTALLATION.md](gui/docs/MACOS_APP_INSTALLATION.md)

### Key Configuration

Training config in [setup_and_train_config.yaml](setup_and_train_config.yaml). `setup_and_train.py` reads this file and drives download → prepare → crop → train; `steps:` flags skip already-completed stages. Notable keys:
- `training.model`: `unet`, `ushape_transformer`, `ss_uie`, or `3d_lut`
- `training.loss`: `auto` (per-model default), `combined`, `composite`, or `ss_uie`
- `training.lut_dim` / `training.lut_num`: 3D LUT grid size and number of basis LUTs (only used by `3d_lut`)
- `training.batch_size`: small by default (3) — tune to GPU memory
- `training.image_size`: 512 (training patch/crop size)
- `training.epochs`, `training.learning_rate`, `training.early_stopping` (patience; 0 disables)
- `training.random_crop` / `crops_per_image`: extract N random crops per source image at native resolution instead of resizing
- Memory knobs for large models: `amp` (FP16), `gradient_checkpointing`, `optimizer_8bit` (needs bitsandbytes), `compile` (torch.compile)

## External Dependencies

- **GPR Tools**: `./build_scripts/compile_gpr_tools.sh` (or `install_gpr_tools.sh`) compiles from https://github.com/keenanjohnson/gpr_tools
- **Hardware**: CUDA GPU recommended for training (~50x faster), CPU fine for inference. `ss_uie` requires CUDA (mamba-ssm).

## Resources

- Pre-trained models: https://huggingface.co/Seattle-Aquarium
- Dataset: https://huggingface.co/datasets/Seattle-Aquarium/Seattle_Aquarium_benthic_imagery
- Parent project: https://github.com/Seattle-Aquarium/CCR_development