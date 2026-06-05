#!/usr/bin/env python3
"""
Inference script for underwater image enhancement
Run trained model on new images
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import logging
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet_autoencoder import UNetAutoencoder, LightweightUNet
from src.models.attention_unet import AttentionUNet
from src.models.ushape_transformer import UShapeTransformer
from src.models.ss_uie import SSUIEModel, is_ss_uie_available, get_ss_uie_unavailable_reason
from src.models.lut_3d import LUT3D

# Colab-compatible model blocks (matching exact architecture from notebook)
class ColabDoubleConv(torch.nn.Module):
    """Double Convolution Block matching Colab notebook (with bias)"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ColabDown(torch.nn.Module):
    """Downscaling with maxpool then double conv - Colab version"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            ColabDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ColabUp(torch.nn.Module):
    """Upscaling then double conv - Colab version"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ColabDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ColabUNetAutoencoder(torch.nn.Module):
    """Colab-compatible U-Net matching the exact architecture from the notebook"""
    def __init__(self, n_channels=3, n_classes=3):
        super(ColabUNetAutoencoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder - matching Colab notebook exactly
        self.inc = ColabDoubleConv(n_channels, 64)
        self.down1 = ColabDown(64, 128)
        self.down2 = ColabDown(128, 256)
        self.down3 = ColabDown(256, 512)
        self.down4 = ColabDown(512, 1024)
        
        # Decoder - matching Colab notebook exactly
        self.up1 = ColabUp(1024, 512)
        self.up2 = ColabUp(512, 256)
        self.up3 = ColabUp(256, 128)
        self.up4 = ColabUp(128, 64)
        self.outc = torch.nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return torch.sigmoid(logits)

# Import blocks from local model for non-Colab models
from src.models.unet_autoencoder import DoubleConv, Down, Up

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Inferencer:
    """Inference class for underwater image enhancement"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None):
        """Initialize inferencer with trained model"""

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Detect model type from checkpoint
        self.detected_model_type = self._detect_model_type(checkpoint)

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Handle Colab checkpoint format - create default config
            self.config = self._create_default_config(checkpoint)
        
        # Setup device (CUDA > MPS > CPU) - matching train.py
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using device: {self.device}")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info(f"Using device: {self.device} (Apple Silicon)")
        else:
            self.device = torch.device('cpu')
            logger.info(f"Using device: {self.device}")
        
        self.setup_model()

        # Handle torch.compile() models that have '_orig_mod.' prefix in state dict keys
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            logger.info("Detected torch.compile() model, stripping '_orig_mod.' prefix from keys")
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}

        # Handle legacy U-Shape Transformer checkpoints that used 'transformer.net' instead of 'transformer.layers'
        if any(key.startswith('transformer.net.') for key in state_dict.keys()):
            logger.info("Detected legacy checkpoint, remapping 'transformer.net' to 'transformer.layers'")
            state_dict = {key.replace('transformer.net.', 'transformer.layers.'): value for key, value in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
        
        self.setup_transforms()

    def _detect_model_type(self, checkpoint):
        """Detect model type from checkpoint state dict keys"""
        state_dict = checkpoint.get('model_state_dict', {})

        # Strip '_orig_mod.' prefix if present (from torch.compile())
        keys = [key.replace('_orig_mod.', '') for key in state_dict.keys()]

        # Check for SS-UIE specific keys (wrapped or unwrapped)
        if any(key.startswith('model.dense_memory') or key.startswith('dense_memory') for key in keys):
            logger.info("Detected SS-UIE model from checkpoint")
            return 'SSUIEModel'

        # Check for image-adaptive 3D LUT specific keys
        if any(key == 'luts' or key.startswith('weight_head') for key in keys):
            logger.info("Detected image-adaptive 3D LUT model from checkpoint")
            return 'LUT3D'

        # Check for U-Shape Transformer specific keys
        if any(key.startswith('mtc.') for key in keys):
            logger.info("Detected U-Shape Transformer model from checkpoint")
            return 'UShapeTransformer'

        # Check for Attention U-Net specific keys
        if any('attention' in key.lower() for key in keys):
            logger.info("Detected Attention U-Net model from checkpoint")
            return 'AttentionUNet'

        # Default to standard U-Net
        logger.info("Detected standard U-Net model from checkpoint")
        return 'UNetAutoencoder'

    @staticmethod
    def _lut_shape_from_checkpoint(checkpoint):
        """Derive (lut_dim, lut_num) from a 3D LUT checkpoint's 'luts' tensor.

        Falls back to the training defaults (33, 3) if the tensor is absent.
        """
        state_dict = checkpoint.get('model_state_dict', {})
        for key, val in state_dict.items():
            if key.replace('_orig_mod.', '') == 'luts':
                # luts shape: [n_luts, 3, dim, dim, dim]
                return int(val.shape[2]), int(val.shape[0])
        return 33, 3

    def _create_default_config(self, checkpoint):
        """Create default config for Colab-trained models"""
        # Check if checkpoint has model_config (from final save in Colab)
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            config = {
                'model': {
                    'type': self.detected_model_type,
                    'n_channels': model_config.get('n_channels', 3),
                    'n_classes': model_config.get('n_classes', 3),
                },
                'data': {
                    'image_size': [model_config.get('image_size', 256), model_config.get('image_size', 256)]
                },
                'inference': {
                    'resize_inference': True  # Default: resize to training size for faster inference
                }
            }

            # Add model-specific parameters
            if self.detected_model_type == 'UShapeTransformer':
                config['model'].update({
                    'img_dim': model_config.get('image_size', 256),
                    'patch_dim': model_config.get('patch_dim', 16),
                    'embedding_dim': model_config.get('embedding_dim', 512),
                    'num_heads': model_config.get('num_heads', 8),
                    'num_layers': model_config.get('num_layers', 4),
                    'hidden_dim': model_config.get('hidden_dim', 256),
                    'return_single': True  # Always return single output for inference
                })
            elif self.detected_model_type == 'SSUIEModel':
                # SS-UIE uses H, W from ss_uie_H/W or falls back to image_size
                img_size = model_config.get('ss_uie_H', model_config.get('image_size', 256))
                config['model'].update({
                    'H': img_size,
                    'W': model_config.get('ss_uie_W', img_size),
                    'channels': 16,
                    'num_memblock': 4,  # Paper default
                    'num_resblock': 4,  # Paper default
                })
            elif self.detected_model_type == 'LUT3D':
                lut_dim, lut_num = self._lut_shape_from_checkpoint(checkpoint)
                config['model'].update({
                    'lut_dim': model_config.get('lut_dim', lut_dim),
                    'lut_num': model_config.get('lut_num', lut_num),
                })
                # 3D LUT is resolution-independent and per-pixel: process at native
                # resolution to preserve fine texture (the point of this model).
                config['inference']['resize_inference'] = False
            else:
                config['model'].update({
                    'base_features': 64,
                    'bilinear': False
                })

            return config
        else:
            # Fallback config for basic Colab checkpoints
            # Check for training_config which older checkpoints may have
            training_config = checkpoint.get('training_config', {})
            image_size = training_config.get('image_size', 256)

            config = {
                'model': {
                    'type': self.detected_model_type,
                    'n_channels': 3,
                    'n_classes': 3,
                },
                'data': {
                    'image_size': [image_size, image_size]
                },
                'inference': {
                    'resize_inference': True
                }
            }

            # Add model-specific parameters
            if self.detected_model_type == 'UShapeTransformer':
                config['model'].update({
                    'img_dim': image_size,
                    'patch_dim': 16,
                    'embedding_dim': 512,
                    'num_heads': 8,
                    'num_layers': 4,
                    'hidden_dim': 256,
                    'return_single': True
                })
            elif self.detected_model_type == 'SSUIEModel':
                config['model'].update({
                    'H': image_size,
                    'W': image_size,
                    'channels': 16,
                    'num_memblock': 4,  # Paper default
                    'num_resblock': 4,  # Paper default
                })
            elif self.detected_model_type == 'LUT3D':
                lut_dim, lut_num = self._lut_shape_from_checkpoint(checkpoint)
                config['model'].update({'lut_dim': lut_dim, 'lut_num': lut_num})
                config['inference']['resize_inference'] = False
            else:
                config['model'].update({
                    'base_features': 64,
                    'bilinear': False
                })

            return config
    
    def setup_model(self):
        """Initialize model architecture"""
        model_type = self.config['model']['type']
        model_params = {
            'n_channels': self.config['model']['n_channels'],
            'n_classes': self.config['model']['n_classes']
        }

        if model_type == 'UShapeTransformer':
            # U-Shape Transformer uses different parameter names
            model_params = {
                'img_dim': self.config['model'].get('img_dim', 256),
                'patch_dim': self.config['model'].get('patch_dim', 16),
                'embedding_dim': self.config['model'].get('embedding_dim', 512),
                'num_channels': self.config['model']['n_channels'],
                'num_heads': self.config['model'].get('num_heads', 8),
                'num_layers': self.config['model'].get('num_layers', 4),
                'hidden_dim': self.config['model'].get('hidden_dim', 256),
                'in_ch': self.config['model']['n_channels'],
                'out_ch': self.config['model']['n_classes'],
                'return_single': self.config['model'].get('return_single', True)
            }
            self.model = UShapeTransformer(**model_params)
        elif model_type == 'LightweightUNet':
            model_params['base_features'] = self.config['model']['base_features']
            self.model = LightweightUNet(**model_params)
        elif model_type == 'UNetAutoencoder':
            # Check if this is a Colab checkpoint (without base_features parameter)
            if 'base_features' not in self.config['model'] or self.config['model']['base_features'] == 64:
                # Use Colab-compatible model for exact architecture match
                self.model = ColabUNetAutoencoder(**model_params)
            else:
                model_params['base_features'] = self.config['model']['base_features']
                model_params['bilinear'] = self.config['model'].get('bilinear', False)
                self.model = UNetAutoencoder(**model_params)
        elif model_type == 'AttentionUNet':
            model_params['base_features'] = self.config['model']['base_features']
            self.model = AttentionUNet(**model_params)
        elif model_type == 'SSUIEModel':
            if not is_ss_uie_available():
                reason = get_ss_uie_unavailable_reason()
                raise ImportError(f"Cannot load SS-UIE model: {reason}")
            model_params = {
                'in_channels': self.config['model']['n_channels'],
                'channels': self.config['model'].get('channels', 16),
                'num_memblock': self.config['model'].get('num_memblock', 4),  # Paper default
                'num_resblock': self.config['model'].get('num_resblock', 4),  # Paper default
                'drop_rate': 0.0,
                'H': self.config['model'].get('H', 256),
                'W': self.config['model'].get('W', 256),
            }
            self.model = SSUIEModel(**model_params)
        elif model_type == 'LUT3D':
            self.model = LUT3D(
                n_channels=self.config['model']['n_channels'],
                n_luts=self.config['model'].get('lut_num', 3),
                lut_dim=self.config['model'].get('lut_dim', 33),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model = self.model.to(self.device)
    
    def setup_transforms(self):
        """Setup image transformations"""
        if self.config.get('inference', {}).get('resize_inference', False):
            # Use inference_size if available, otherwise fall back to data image_size
            inference_config = self.config.get('inference', {})
            if 'inference_size' in inference_config:
                image_size = tuple(inference_config['inference_size'])
            else:
                image_size = tuple(self.config['data']['image_size'])

            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
            self.inverse_transform = transforms.Compose([
                transforms.ToPILImage()
            ])
        else:
            self.transform = transforms.ToTensor()
            self.inverse_transform = transforms.ToPILImage()
    
    def pad_to_multiple(self, tensor, multiple=32):
        """Pad tensor to be divisible by multiple"""
        _, _, h, w = tensor.shape
        
        # Calculate padding needed
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        # Pad to make divisible by multiple
        if pad_h > 0 or pad_w > 0:
            # Use reflection padding to avoid artifacts
            padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
            tensor = torch.nn.functional.pad(tensor, padding, mode='reflect')
        
        return tensor, (pad_h, pad_w)

    def process_image_tiled(self, image_path: Path, tile_size=None, overlap=128, progress_callback=None):
        """Process large image using tiling with seamless blending to avoid artifacts

        Args:
            image_path: Path to input image
            tile_size: Size of each tile (default: 1024 for U-Net, img_dim for U-Shape Transformer)
            overlap: Overlap between tiles (used for blending)
            progress_callback: Optional callback(message) for progress updates
        """
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        width, height = original_size

        # For U-Shape Transformer and SS-UIE, use the model's resolution as tile size
        model_type = self.config['model']['type']
        if tile_size is None:
            if model_type == 'UShapeTransformer':
                tile_size = self.config['model'].get('img_dim', 256)
                # Use smaller overlap for smaller tiles
                overlap = min(overlap, tile_size // 4)
                logger.info(f"U-Shape Transformer: using tile_size={tile_size} (model img_dim)")
            elif model_type == 'SSUIEModel':
                # SS-UIE requires exact H, W resolution for FFT weights
                tile_size = self.config['model'].get('H', 256)
                # Use larger overlap for SS-UIE due to global receptive field (Mamba + FFT)
                # This helps blend tile boundaries where global context discontinuities occur
                overlap = max(overlap, tile_size // 3)  # ~170px for 512px tiles
                logger.info(f"SS-UIE: using tile_size={tile_size} (model H/W) with {overlap}px overlap")
            else:
                tile_size = 1024

        logger.info(f"Processing {width}x{height} image with {tile_size}x{tile_size} tiles and {overlap}px overlap")
        if progress_callback:
            progress_callback(f"Processing {width}x{height} image with {tile_size}x{tile_size} tiles")

        # Create accumulation buffers for weighted blending
        output_array = np.zeros((height, width, 3), dtype=np.float32)
        weight_array = np.zeros((height, width), dtype=np.float32)

        # Calculate total tiles for progress tracking
        tiles_x = (width + tile_size - overlap - 1) // (tile_size - overlap)
        tiles_y = (height + tile_size - overlap - 1) // (tile_size - overlap)
        total_tiles = tiles_x * tiles_y
        current_tile = 0

        # Process tiles
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                current_tile += 1
                # Define tile boundaries
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)

                tile_width = x_end - x
                tile_height = y_end - y

                # Extract tile
                tile = img.crop((x, y, x_end, y_end))
                tile_original_size = tile.size

                # For resolution-dependent models (U-Shape Transformer, SS-UIE), resize tile to exact model size
                # SS-UIE has learnable FFT filters (K) that are resolution-specific
                # U-Shape Transformer has fixed positional embeddings
                needs_exact_resolution = model_type in ('UShapeTransformer', 'SSUIEModel')
                tile_size_mismatch = (tile_width != tile_size or tile_height != tile_size)

                if needs_exact_resolution and tile_size_mismatch:
                    # Resize to exact tile_size for model processing
                    tile_resized = tile.resize((tile_size, tile_size), Image.LANCZOS)
                    input_tensor = self.transform(tile_resized).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        output_tensor = self.model(input_tensor)

                    output_tensor = output_tensor.squeeze(0).cpu()
                    enhanced_tile_resized = self.inverse_transform(output_tensor)

                    # Resize back to original tile size
                    enhanced_tile = enhanced_tile_resized.resize(tile_original_size, Image.LANCZOS)
                elif needs_exact_resolution:
                    # Full-size tile for resolution-dependent models (no padding needed)
                    input_tensor = self.transform(tile).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        output_tensor = self.model(input_tensor)

                    output_tensor = output_tensor.squeeze(0).cpu()
                    enhanced_tile = self.inverse_transform(output_tensor)
                else:
                    # Standard processing with padding for U-Net models
                    input_tensor = self.transform(tile).unsqueeze(0).to(self.device)
                    padded_tensor, (pad_h, pad_w) = self.pad_to_multiple(input_tensor, multiple=32)

                    with torch.no_grad():
                        output_tensor = self.model(padded_tensor)

                    # Remove padding
                    if pad_h > 0 or pad_w > 0:
                        _, _, h, w = input_tensor.shape
                        output_tensor = output_tensor[:, :, :h, :w]

                    output_tensor = output_tensor.squeeze(0).cpu()
                    enhanced_tile = self.inverse_transform(output_tensor)

                # Convert to numpy array for blending
                enhanced_array = np.array(enhanced_tile, dtype=np.float32)

                # Create blend mask for this tile
                # Determine which edges need blending
                blend_left = overlap if x > 0 else 0
                blend_right = overlap if x_end < width else 0
                blend_top = overlap if y > 0 else 0
                blend_bottom = overlap if y_end < height else 0

                # Create the blend mask
                mask = np.ones((tile_height, tile_width), dtype=np.float32)

                # Apply gradients to edges (handle small tiles properly)
                if blend_left > 0:
                    actual_blend_left = min(blend_left, tile_width)
                    gradient = np.linspace(0, 1, actual_blend_left)
                    mask[:, :actual_blend_left] = np.minimum(mask[:, :actual_blend_left], gradient)

                if blend_right > 0:
                    actual_blend_right = min(blend_right, tile_width)
                    gradient = np.linspace(1, 0, actual_blend_right)
                    mask[:, -actual_blend_right:] = np.minimum(mask[:, -actual_blend_right:], gradient)

                if blend_top > 0:
                    actual_blend_top = min(blend_top, tile_height)
                    gradient = np.linspace(0, 1, actual_blend_top).reshape(-1, 1)
                    mask[:actual_blend_top, :] = np.minimum(mask[:actual_blend_top, :], gradient)

                if blend_bottom > 0:
                    actual_blend_bottom = min(blend_bottom, tile_height)
                    gradient = np.linspace(1, 0, actual_blend_bottom).reshape(-1, 1)
                    mask[-actual_blend_bottom:, :] = np.minimum(mask[-actual_blend_bottom:, :], gradient)

                # Apply weighted blending (broadcast mask across RGB channels)
                output_array[y:y_end, x:x_end] += enhanced_array * mask[:, :, np.newaxis]
                weight_array[y:y_end, x:x_end] += mask

                logger.info(f"Processed tile ({x}, {y}) to ({x_end}, {y_end})")
                if progress_callback:
                    progress_callback(f"Processed tile {current_tile}/{total_tiles} ({x}, {y}) to ({x_end}, {y_end})")

        # Normalize by weights to get final image
        # Avoid division by zero (shouldn't happen, but be safe)
        weight_array = np.maximum(weight_array, 1e-8)
        output_array /= weight_array[:, :, np.newaxis]

        # Convert back to PIL Image
        output_array = np.clip(output_array, 0, 255).astype(np.uint8)
        output_img = Image.fromarray(output_array, 'RGB')

        return output_img

    def process_image(self, image_path: Path, output_path: Path = None, progress_callback=None):
        """Process a single image

        Args:
            image_path: Path to input image
            output_path: Optional output path for saving
            progress_callback: Optional callback(message) for progress updates
        """
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        width, height = original_size

        # Check model type and inference settings
        model_type = self.config['model']['type']
        resize_inference = self.config.get('inference', {}).get('resize_inference', False)

        # 3D LUT: a single global, per-pixel colour transform that is resolution-
        # independent. Always process the whole image in ONE forward pass -- never
        # tile and never resize. Tiling would run the weight-predictor on each tile
        # separately, applying a *different* LUT to each region (colour seams); and
        # resizing would apply the LUT at low resolution then upscale, blurring away
        # the fine texture this model exists to preserve. No padding is needed:
        # grid_sample handles any H x W and the predictor resizes to 256 internally.
        if model_type == 'LUT3D':
            logger.info(f"Processing {width}x{height} image whole at native resolution (3D LUT)")
            if progress_callback:
                progress_callback(f"Processing {width}x{height} image at native resolution")
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output_tensor = self.model(input_tensor)
            output_img = self.inverse_transform(output_tensor.squeeze(0).cpu())

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_img.save(output_path, quality=95)
                logger.info(f"Saved enhanced image to {output_path}")
            return output_img

        # Determine tile size threshold based on model type
        if model_type == 'UShapeTransformer':
            # For U-Shape Transformer, always use tiling to maintain max resolution
            # Use smaller threshold since tiles will be model's img_dim (typically 256)
            tile_threshold = self.config['model'].get('img_dim', 256)
        elif model_type == 'SSUIEModel':
            # For SS-UIE, use model's H (or W) as tile threshold
            tile_threshold = self.config['model'].get('H', 256)
        else:
            # For U-Net models, use tiling for large images
            tile_threshold = 2048

        # Use tiling for images larger than threshold (unless resize_inference is set)
        if not resize_inference and (width > tile_threshold or height > tile_threshold):
            logger.info(f"Processing {width}x{height} image with tiled approach for maximum resolution")
            output_img = self.process_image_tiled(image_path, progress_callback=progress_callback)
        elif resize_inference:
            # User explicitly requested resize mode - process at training resolution
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_tensor = self.model(input_tensor)

            output_tensor = output_tensor.squeeze(0).cpu()
            output_img = self.inverse_transform(output_tensor)

            # Resize back to original size
            output_img = output_img.resize(original_size, Image.LANCZOS)
        else:
            # Small images - process at original resolution with padding (U-Net only)
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Pad to make dimensions compatible with model
            padded_tensor, (pad_h, pad_w) = self.pad_to_multiple(input_tensor, multiple=32)

            with torch.no_grad():
                output_tensor = self.model(padded_tensor)

            # Remove padding from output
            if pad_h > 0 or pad_w > 0:
                _, _, h, w = input_tensor.shape
                output_tensor = output_tensor[:, :, :h, :w]

            output_tensor = output_tensor.squeeze(0).cpu()
            output_img = self.inverse_transform(output_tensor)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_img.save(output_path, quality=95)
            logger.info(f"Saved enhanced image to {output_path}")
        
        return output_img
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         extensions: list = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']):
        """Process all images in a directory"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        for img_path in tqdm(image_files, desc="Processing images"):
            output_path = output_dir / f"{img_path.stem}_enhanced{img_path.suffix}"
            self.process_image(img_path, output_path)
    
    def compare_images(self, image_path: Path, save_comparison: bool = True):
        """Create side-by-side comparison of original and enhanced"""
        original = Image.open(image_path).convert('RGB')
        enhanced = self.process_image(image_path)
        
        width, height = original.size
        comparison = Image.new('RGB', (width * 2, height))
        comparison.paste(original, (0, 0))
        comparison.paste(enhanced, (width, 0))
        
        if save_comparison:
            output_path = image_path.parent / f"{image_path.stem}_comparison.jpg"
            comparison.save(output_path, quality=95)
            logger.info(f"Saved comparison to {output_path}")
        
        return comparison


def main():
    parser = argparse.ArgumentParser(description="Run inference on underwater images")
    parser.add_argument('input', type=str, help='Input image or directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (optional for Colab checkpoints)')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory or file')
    parser.add_argument('--compare', action='store_true',
                        help='Create side-by-side comparison')
    parser.add_argument('--full-size', action='store_true',
                        help='Process at original resolution (4606x4030) instead of resizing')
    
    args = parser.parse_args()
    
    # Override config for full-size processing if requested
    config_override = None
    if args.full_size:
        config_override = {'inference': {'resize_inference': False}}
    
    inferencer = Inferencer(args.checkpoint, args.config)
    
    # Apply full-size override if requested
    if config_override:
        inferencer.config.update(config_override)
        inferencer.setup_transforms()  # Refresh transforms
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        if args.compare:
            inferencer.compare_images(input_path)
        else:
            if output_path.suffix:
                inferencer.process_image(input_path, output_path)
            else:
                output_file = output_path / f"{input_path.stem}_enhanced{input_path.suffix}"
                inferencer.process_image(input_path, output_file)
    elif input_path.is_dir():
        inferencer.process_directory(input_path, output_path)
    else:
        logger.error(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()