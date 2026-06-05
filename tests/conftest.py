"""
Pytest fixtures for inference testing
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ushape_transformer import UShapeTransformer


# Test dimensions - must match inference.py defaults for compatibility
TEST_IMAGE_SIZE = 64  # Small test images for speed
TEST_UNET_BASE_FEATURES = 64  # Must match inference.py hardcoded default
TEST_USHAPE_IMG_DIM = 256  # Minimum required by U-Shape Transformer architecture


@pytest.fixture(scope="session")
def test_image_path(tmp_path_factory):
    """Create a small test image for inference testing"""
    tmp_dir = tmp_path_factory.mktemp("test_images")
    image_path = tmp_dir / "test_input.jpg"

    # Create a simple test image (64x64 RGB)
    img_array = np.random.randint(0, 255, (TEST_IMAGE_SIZE, TEST_IMAGE_SIZE, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(image_path)

    return str(image_path)


@pytest.fixture(scope="session")
def test_image_dir(tmp_path_factory, test_image_path):
    """Return directory containing test images"""
    return str(Path(test_image_path).parent)


@pytest.fixture(scope="session")
def large_test_image_path(tmp_path_factory):
    """Create a large, non-square test image that exceeds the 2048px tiling
    threshold on both axes. Used to verify resolution-independent models (3D LUT)
    process the whole frame in one pass instead of tiling."""
    tmp_dir = tmp_path_factory.mktemp("large_test_images")
    image_path = tmp_dir / "large_test_input.jpg"

    # Non-square and > 2048 on both axes (so width != height catches axis swaps).
    img_array = np.random.randint(0, 255, (2100, 2400, 3), dtype=np.uint8)
    Image.fromarray(img_array).save(image_path)

    return str(image_path)


def create_unet_checkpoint(path: Path, base_features: int = TEST_UNET_BASE_FEATURES):
    """Create a minimal UNet checkpoint for testing

    Note: inference.py uses ColabUNetAutoencoder when base_features=64,
    which has a slightly different architecture (with bias in conv layers).
    """
    # Import the Colab-compatible model that inference.py uses
    from inference.inference import ColabUNetAutoencoder

    model = ColabUNetAutoencoder(n_channels=3, n_classes=3)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 1,
        'val_loss': 0.1,
        'model_config': {
            'n_channels': 3,
            'n_classes': 3,
            'base_features': base_features,
            'bilinear': False,
            'image_size': TEST_IMAGE_SIZE,
        }
    }

    torch.save(checkpoint, path)
    return path


def create_ushape_checkpoint(path: Path, img_dim: int = TEST_USHAPE_IMG_DIM, legacy_keys: bool = False):
    """
    Create a minimal U-Shape Transformer checkpoint for testing

    Args:
        path: Where to save the checkpoint
        img_dim: Image dimension (must be divisible by patch_dim=16)
        legacy_keys: If True, use old 'transformer.net' naming instead of 'transformer.layers'
    """
    model = UShapeTransformer(
        img_dim=img_dim,
        patch_dim=16,
        embedding_dim=64,  # Smaller than default 512
        num_channels=3,
        num_heads=4,  # Smaller than default 8
        num_layers=2,  # Smaller than default 4
        hidden_dim=32,  # Smaller than default 256
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        in_ch=3,
        out_ch=3,
        return_single=True,
    )

    state_dict = model.state_dict()

    # Simulate legacy checkpoint format if requested
    if legacy_keys:
        state_dict = {
            key.replace('transformer.layers.', 'transformer.net.'): value
            for key, value in state_dict.items()
        }

    checkpoint = {
        'model_state_dict': state_dict,
        'epoch': 1,
        'val_loss': 0.1,
        'model_config': {
            'n_channels': 3,
            'n_classes': 3,
            'image_size': img_dim,
            'patch_dim': 16,
            'embedding_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'hidden_dim': 32,
        }
    }

    torch.save(checkpoint, path)
    return path


def create_lut3d_checkpoint(path: Path, lut_dim: int = 9, lut_num: int = 3):
    """Create a minimal 3D LUT checkpoint for testing.

    Uses a small lut_dim for speed; the model is resolution-independent so the
    LUT grid size is unrelated to the test image size.
    """
    from src.models.lut_3d import LUT3D

    model = LUT3D(n_channels=3, n_luts=lut_num, lut_dim=lut_dim)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 1,
        'val_loss': 0.1,
        'model_config': {
            'n_channels': 3,
            'n_classes': 3,
            'image_size': TEST_IMAGE_SIZE,
            'model': '3d_lut',
            'lut_dim': lut_dim,
            'lut_num': lut_num,
        }
    }

    torch.save(checkpoint, path)
    return path


@pytest.fixture(scope="session")
def unet_checkpoint_path(tmp_path_factory):
    """Create a UNet checkpoint for testing"""
    tmp_dir = tmp_path_factory.mktemp("checkpoints")
    checkpoint_path = tmp_dir / "unet_test.pth"
    return create_unet_checkpoint(checkpoint_path)


@pytest.fixture(scope="session")
def lut3d_checkpoint_path(tmp_path_factory):
    """Create a 3D LUT checkpoint for testing"""
    tmp_dir = tmp_path_factory.mktemp("checkpoints")
    checkpoint_path = tmp_dir / "lut3d_test.pth"
    return create_lut3d_checkpoint(checkpoint_path)


@pytest.fixture(scope="session")
def ushape_checkpoint_path(tmp_path_factory):
    """Create a U-Shape Transformer checkpoint for testing (current format)"""
    tmp_dir = tmp_path_factory.mktemp("checkpoints")
    checkpoint_path = tmp_dir / "ushape_test.pth"
    return create_ushape_checkpoint(checkpoint_path, legacy_keys=False)


@pytest.fixture(scope="session")
def ushape_legacy_checkpoint_path(tmp_path_factory):
    """Create a U-Shape Transformer checkpoint with legacy 'transformer.net' keys"""
    tmp_dir = tmp_path_factory.mktemp("checkpoints")
    checkpoint_path = tmp_dir / "ushape_legacy_test.pth"
    return create_ushape_checkpoint(checkpoint_path, legacy_keys=True)


@pytest.fixture(scope="session")
def compiled_unet_checkpoint_path(tmp_path_factory):
    """Create a UNet checkpoint with torch.compile() prefix in keys"""
    from inference.inference import ColabUNetAutoencoder

    tmp_dir = tmp_path_factory.mktemp("checkpoints")
    checkpoint_path = tmp_dir / "unet_compiled_test.pth"

    model = ColabUNetAutoencoder(n_channels=3, n_classes=3)

    # Simulate torch.compile() state dict with '_orig_mod.' prefix
    state_dict = {
        f'_orig_mod.{key}': value
        for key, value in model.state_dict().items()
    }

    checkpoint = {
        'model_state_dict': state_dict,
        'epoch': 1,
        'val_loss': 0.1,
        'model_config': {
            'n_channels': 3,
            'n_classes': 3,
            'base_features': TEST_UNET_BASE_FEATURES,
            'bilinear': False,
            'image_size': TEST_IMAGE_SIZE,
        }
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
