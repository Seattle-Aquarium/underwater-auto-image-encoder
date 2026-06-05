#!/usr/bin/env python3
"""
Underwater Image Enhancement Training Script
Standalone version matching the Jupyter notebook workflow exactly
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from PIL import Image

# Optional 8-bit optimizer support
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.ushape_transformer import UShapeTransformer
from src.models.ss_uie import SSUIEModel, is_ss_uie_available
from src.models.lut_3d import LUT3D
from src.losses import CompositeLoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnderwaterDataset(Dataset):
    """Dataset that supports flexible image sizes and random cropping"""

    def __init__(self, input_dir, target_dir, file_indices, image_size=512, augment=False,
                 use_random_crop=False, crops_per_image=1):
        """
        Args:
            image_size: Target size. Can be:
                       - int: Square images (e.g., 512 -> 512×512)
                       - tuple: (width, height)
                       - None: Use original full size (no resize)
            use_random_crop: If True, extract random crops instead of resizing.
                            This preserves native resolution detail.
            crops_per_image: Number of random crops per image (only used with use_random_crop=True).
                            Higher values = more training data from each image.
        """
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        self.augment = augment
        self.use_random_crop = use_random_crop
        self.crops_per_image = crops_per_image if use_random_crop else 1

        # Get all files with various image extensions
        input_extensions = ('.tiff', '.tif', '.TIFF', '.TIF')
        target_extensions = ('.tiff', '.tif', '.TIFF', '.TIF', '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG')

        all_input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(input_extensions)])
        all_target_files = sorted([f for f in os.listdir(target_dir) if f.endswith(target_extensions)])

        # Filter indices to valid range
        valid_indices = [i for i in file_indices if i < min(len(all_input_files), len(all_target_files))]

        # Select files based on indices
        self.input_files = [all_input_files[i] for i in valid_indices]
        self.target_files = [all_target_files[i] for i in valid_indices]

        # Convert image_size to tuple
        if isinstance(image_size, int):
            self.crop_size = (image_size, image_size)
        else:
            self.crop_size = image_size if image_size else (512, 512)

        # Define transforms based on mode
        if use_random_crop:
            # Random crop mode: no resize, extract crops at native resolution
            self.transform = transforms.ToTensor()
        elif image_size is not None:
            # Resize mode (original behavior)
            transform_list = [
                transforms.Resize(self.crop_size),
                transforms.ToTensor(),
            ]
            self.transform = transforms.Compose(transform_list)
        else:
            # Full-size: no resizing
            self.transform = transforms.ToTensor()

    def __len__(self):
        # Total samples = images × crops_per_image
        return len(self.input_files) * self.crops_per_image

    def __getitem__(self, idx):
        # Map index to file and crop number
        file_idx = idx // self.crops_per_image
        crop_idx = idx % self.crops_per_image

        # Load images
        input_path = self.input_dir / self.input_files[file_idx]
        target_path = self.target_dir / self.target_files[file_idx]

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')

        if self.use_random_crop:
            # Extract random crop at same position from both images
            w, h = input_img.size
            crop_w, crop_h = self.crop_size

            # Random position (use idx-based seed for reproducibility within epoch)
            # but different across epochs due to DataLoader shuffling
            max_x = max(0, w - crop_w)
            max_y = max(0, h - crop_h)

            # Use a seed based on crop_idx to get different crops from same image
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            x = torch.randint(0, max_x + 1, (1,)).item() if max_x > 0 else 0
            y = torch.randint(0, max_y + 1, (1,)).item() if max_y > 0 else 0

            # Crop both images at same position
            input_img = input_img.crop((x, y, x + crop_w, y + crop_h))
            target_img = target_img.crop((x, y, x + crop_w, y + crop_h))

            # Apply augmentation if enabled
            if self.augment:
                # Random horizontal flip
                if torch.rand(1).item() > 0.5:
                    input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)
                    target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)
                # Random vertical flip
                if torch.rand(1).item() > 0.5:
                    input_img = input_img.transpose(Image.FLIP_TOP_BOTTOM)
                    target_img = target_img.transpose(Image.FLIP_TOP_BOTTOM)

            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        else:
            # Original resize behavior
            if self.augment:
                # Apply same random transform to both images
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                aug_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ])
                torch.manual_seed(seed)
                input_img = aug_transform(input_img)
                torch.manual_seed(seed)
                target_img = aug_transform(target_img)

            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img


class DoubleConv(nn.Module):
    """Double Convolution Block"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetAutoencoder(nn.Module):
    """U-Net based autoencoder for image enhancement"""
    def __init__(self, n_channels=3, n_classes=3):
        super(UNetAutoencoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

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


class CombinedLoss(nn.Module):
    """Combined loss function for image enhancement"""
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        mse = self.mse_loss(pred, target)
        return self.alpha * l1 + self.beta * mse


class SSUIECombinedLoss(nn.Module):
    """SS-UIE paper loss function: SSIM + L1 + LAB + LCH + FDL

    From the paper (train.py line 242):
    loss_final = loss_ssim*10 + loss_RGB*10 + loss_lch + loss_lab*0.0001 + fdl_loss*10000
    """
    def __init__(self):
        super().__init__()
        # Add SS-UIE library paths for imports
        # Order matters: SS-UIE root must come first so 'from utils.X import Y' works
        ss_uie_path = Path(__file__).parent.parent / "lib" / "SS-UIE"
        ss_uie_ssim_path = ss_uie_path / "pytorch-ssim-loss"

        # Add SS-UIE root first (for 'utils.ptcolor' etc), then ssim path
        for path in [str(ss_uie_path), str(ss_uie_ssim_path)]:
            if path not in sys.path:
                sys.path.insert(0, path)

        # Import SS-UIE loss components
        from pytorch_ssim import SSIM
        from utils.LAB import lab_Loss
        from utils.LCH import lch_Loss
        from utils.FDL import FDL

        # Initialize loss components (matching paper settings)
        self.ssim_loss = SSIM().cuda() if torch.cuda.is_available() else SSIM()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.lab_loss = lab_Loss().cuda() if torch.cuda.is_available() else lab_Loss()
        self.lch_loss = lch_Loss().cuda() if torch.cuda.is_available() else lch_Loss()
        self.fdl_loss = FDL(
            loss_weight=1.0,
            alpha=2.0,
            patch_factor=4,
            ave_spectrum=True,
            log_matrix=True,
            batch_matrix=True
        ).cuda() if torch.cuda.is_available() else FDL(
            loss_weight=1.0, alpha=2.0, patch_factor=4,
            ave_spectrum=True, log_matrix=True, batch_matrix=True
        )

        # Paper weights
        self.ssim_weight = 10.0
        self.l1_weight = 10.0
        self.lch_weight = 1.0
        self.lab_weight = 0.0001
        self.fdl_weight = 10000.0

        # Range penalty weight - penalizes outputs outside [0,1]
        # This is NOT in the original paper but necessary since we removed sigmoid
        self.range_weight = 100.0

    def forward(self, pred, target):
        # SSIM loss (1 - SSIM, so lower is better)
        loss_ssim = 1 - self.ssim_loss(pred, target)

        # L1 loss (normalized by image size as in paper)
        h, w = pred.shape[2], pred.shape[3]
        loss_l1 = self.l1_loss(pred, target).sum() / (h * w)

        # Color space losses
        loss_lch = self.lch_loss(pred, target)
        loss_lab = self.lab_loss(pred, target)

        # Frequency domain loss
        loss_fdl = self.fdl_loss(pred, target)

        # Range penalty: penalize values outside [0,1]
        # Uses ReLU to only penalize out-of-range values
        below_zero = torch.relu(-pred)  # positive where pred < 0
        above_one = torch.relu(pred - 1)  # positive where pred > 1
        loss_range = (below_zero.mean() + above_one.mean())

        # Combined loss (paper formula + range penalty)
        total_loss = (
            self.ssim_weight * loss_ssim +
            self.l1_weight * loss_l1 +
            self.lch_weight * loss_lch +
            self.lab_weight * loss_lab +
            self.fdl_weight * loss_fdl +
            self.range_weight * loss_range
        )

        return total_loss


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def load_split(split_file, num_files):
    """Load train/validation split from file"""
    if split_file.exists():
        try:
            # Try to load as JSON first
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            train_indices = split_data['train']
            val_indices = split_data['validation']
        except json.JSONDecodeError:
            # Parse as plain text with comma-separated indices
            with open(split_file, 'r') as f:
                lines = f.readlines()

            train_indices = []
            val_indices = []

            for line in lines:
                line = line.strip()
                if 'Training' in line or 'training' in line:
                    continue
                elif 'Validation' in line or 'validation' in line:
                    continue
                elif line and not line.startswith('#'):
                    indices = [int(x.strip()) for x in line.split(',') if x.strip().isdigit()]
                    if not train_indices:
                        train_indices.extend(indices)
                    else:
                        val_indices.extend(indices)

        logger.info(f"Loaded split: {len(train_indices)} train, {len(val_indices)} validation")
        return train_indices, val_indices
    else:
        # Create 80/20 split
        logger.info("No split file found, creating 80/20 train/validation split")
        indices = np.random.permutation(num_files)
        split_point = int(0.8 * num_files)
        train_indices = indices[:split_point].tolist()
        val_indices = indices[split_point:].tolist()
        logger.info(f"Created split: {len(train_indices)} train, {len(val_indices)} validation")
        return train_indices, val_indices


def get_model_regularization(model):
    """Return a model's self-regularization term (e.g. 3D LUT smoothness/monotonicity).

    Returns 0.0 for models without one. Unwraps torch.compile's ``_orig_mod``.
    """
    target = getattr(model, '_orig_mod', model)
    reg_fn = getattr(target, 'regularization_loss', None)
    return reg_fn() if callable(reg_fn) else 0.0


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False):
    """Train for one epoch with optional mixed precision"""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = len(dataloader)

    with tqdm(dataloader, desc="Training") as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if use_amp and scaler is not None:
                # Mixed precision forward pass
                with autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) + get_model_regularization(model)

                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets) + get_model_regularization(model)
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            # Calculate PSNR in FP32 for accuracy
            with torch.no_grad():
                batch_psnr = calculate_psnr(outputs.float(), targets.float()).item()

            total_loss += batch_loss
            total_psnr += batch_psnr

            pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'PSNR': f'{batch_psnr:.2f} dB'
            })

    return total_loss / num_batches, total_psnr / num_batches


def validate_epoch(model, dataloader, criterion, device, use_amp=False):
    """Validate for one epoch with optional mixed precision"""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                if use_amp:
                    with autocast(device_type=device.type, dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                batch_loss = loss.item()
                # Calculate PSNR in FP32 for accuracy
                batch_psnr = calculate_psnr(outputs.float(), targets.float()).item()

                total_loss += batch_loss
                total_psnr += batch_psnr

                pbar.set_postfix({
                    'Loss': f'{batch_loss:.4f}',
                    'PSNR': f'{batch_psnr:.2f} dB'
                })

    return total_loss / num_batches, total_psnr / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train Underwater Image Enhancement Model')

    # Data arguments
    parser.add_argument('--input-dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--target-dir', type=str, required=True, help='Directory with target images')
    parser.add_argument('--split-file', type=str, default=None, help='Train/val split file (optional)')

    # Training arguments
    parser.add_argument('--image-size', type=int, default=1024,
                       help='Training image size (use 0 for full-size, default: 1024)')
    parser.add_argument('--random-crop', action='store_true',
                       help='Use random crops at native resolution instead of resizing. '
                            'Preserves detail and gives more diverse training data.')
    parser.add_argument('--crops-per-image', type=int, default=8,
                       help='Number of random crops per image when --random-crop is enabled (default: 8)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers (default: 4)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for models')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    # Loss configuration
    parser.add_argument('--l1-weight', type=float, default=0.8, help='L1 loss weight (default: 0.8)')
    parser.add_argument('--mse-weight', type=float, default=0.2, help='MSE loss weight (default: 0.2)')
    parser.add_argument('--loss', type=str, default='auto',
                       choices=['auto', 'combined', 'composite', 'ss_uie'],
                       help="Loss function. 'auto' (default) picks per model: ss_uie->ss_uie, "
                            "3d_lut->composite, others->combined. 'composite' = L1 + MS-SSIM + "
                            "Focal Frequency (+ optional LPIPS) for texture preservation.")
    parser.add_argument('--composite-weights', type=float, nargs=4,
                       metavar=('L1', 'MSSSIM', 'FFL', 'LPIPS'),
                       default=[1.0, 1.0, 1.0, 0.0],
                       help='Weights for the composite loss terms: L1 MS-SSIM FFL LPIPS '
                            '(default: 1.0 1.0 1.0 0.0; LPIPS 0 disables it and avoids the lpips dep)')

    # Model selection
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'ushape_transformer', 'ss_uie', '3d_lut'],
                       help='Model architecture (default: unet)')

    # 3D LUT configuration
    parser.add_argument('--lut-dim', type=int, default=33,
                       help='3D LUT grid resolution per axis (default: 33)')
    parser.add_argument('--lut-num', type=int, default=3,
                       help='Number of basis 3D LUTs to fuse (default: 3)')

    # Early stopping
    parser.add_argument('--early-stopping', type=int, default=15,
                       help='Early stopping patience - stop if val loss does not improve for N epochs (default: 15, 0 to disable)')

    # Mixed precision training
    parser.add_argument('--amp', action='store_true',
                       help='Enable automatic mixed precision (FP16) training for reduced memory usage')

    # Gradient checkpointing
    parser.add_argument('--gradient-checkpointing', action='store_true',
                       help='Enable gradient checkpointing to reduce memory usage (trades compute for memory)')

    # 8-bit optimizer
    parser.add_argument('--optimizer-8bit', action='store_true',
                       help='Use 8-bit Adam optimizer (requires bitsandbytes) for ~1.5-2GB memory savings')

    # torch.compile
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile for potential speedups and memory optimization (PyTorch 2.0+)')

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using device: {device}")
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info(f"Using device: {device} (Apple Silicon)")
        # MPS doesn't provide direct memory query, but M1/M2/M3 typically have unified memory
        logger.info("Note: MPS uses unified memory shared with system RAM")
    else:
        device = torch.device('cpu')
        logger.info(f"Using device: {device}")
        logger.warning("No GPU acceleration available. Training will be slow.")

    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)

    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.tiff', '.tif', '.TIFF', '.TIF'))])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith(('.tiff', '.tif', '.TIFF', '.TIF', '.png', '.PNG', '.jpg', '.JPG'))])

    logger.info(f"Found {len(input_files)} input images")
    logger.info(f"Found {len(target_files)} target images")

    num_files = min(len(input_files), len(target_files))

    # Load split
    split_file = Path(args.split_file) if args.split_file else input_dir.parent / 'split.txt'
    train_indices, val_indices = load_split(split_file, num_files)

    # Create datasets
    image_size = None if args.image_size == 0 else args.image_size
    use_random_crop = args.random_crop
    crops_per_image = args.crops_per_image

    if use_random_crop:
        resolution_str = f"{image_size}×{image_size} random crops (native resolution)"
        logger.info(f"Training mode: Random crops at native resolution")
        logger.info(f"Crop size: {image_size}×{image_size}")
        logger.info(f"Crops per image: {crops_per_image}")
    else:
        resolution_str = "Full-size" if image_size is None else f"{image_size}×{image_size}"
        logger.info(f"Training resolution: {resolution_str}")

    logger.info(f"Batch size: {args.batch_size}")

    train_dataset = UnderwaterDataset(input_dir, target_dir, train_indices,
                                     image_size=image_size, augment=True,
                                     use_random_crop=use_random_crop,
                                     crops_per_image=crops_per_image)
    val_dataset = UnderwaterDataset(input_dir, target_dir, val_indices,
                                   image_size=image_size, augment=False,
                                   use_random_crop=use_random_crop,
                                   crops_per_image=1)  # Only 1 crop for validation

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    # Disable pin_memory for MPS (not supported yet)
    use_pin_memory = device.type == 'cuda'

    # persistent_workers/prefetch_factor are only valid with worker processes
    # (num_workers > 0); they error if passed when loading on the main process.
    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs = {'persistent_workers': True, 'prefetch_factor': 4}

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        **worker_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        **worker_kwargs
    )

    # Test memory with one batch
    logger.info("Testing memory with sample batch...")
    try:
        sample_batch = next(iter(train_loader))
        sample_inputs = sample_batch[0].to(device)
        logger.info(f"Batch shape: {sample_inputs.shape}")

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"Memory for data batch: {memory_used:.2f} GB")
    except Exception as e:
        logger.error(f"Memory test failed: {e}")
        logger.error("Batch size may be too large for available GPU memory")
        return

    # Create model based on selection
    use_gradient_checkpointing = args.gradient_checkpointing

    if args.model == 'ushape_transformer':
        # U-shape Transformer requires specific image size (must be divisible by 16)
        model_img_size = image_size if image_size is not None else 256
        if model_img_size % 16 != 0:
            logger.warning(f"Image size {model_img_size} not divisible by 16, adjusting to {(model_img_size // 16) * 16}")
            model_img_size = (model_img_size // 16) * 16

        model = UShapeTransformer(
            img_dim=model_img_size,
            in_ch=3,
            out_ch=3,
            return_single=True,  # Return single output for non-GAN training
            gradient_checkpointing=use_gradient_checkpointing
        ).to(device)
        logger.info(f"Using U-shape Transformer model (img_dim={model_img_size})")
        if use_gradient_checkpointing:
            logger.info("Gradient checkpointing: ENABLED (memory-efficient mode)")
    elif args.model == 'ss_uie':
        # SS-UIE requires specific H, W at construction time
        if not is_ss_uie_available():
            logger.error("SS-UIE model requires mamba-ssm package (CUDA only).")
            logger.error("Install with: pip install mamba-ssm causal-conv1d timm einops")
            return

        model_img_size = image_size if image_size is not None else 256
        model = SSUIEModel(
            in_channels=3,
            channels=16,
            num_memblock=4,  # Paper default
            num_resblock=4,  # Paper default
            drop_rate=0.0,
            H=model_img_size,
            W=model_img_size
        ).to(device)
        logger.info(f"Using SS-UIE model (H={model_img_size}, W={model_img_size})")
        if use_gradient_checkpointing:
            logger.warning("Gradient checkpointing requested but not yet implemented for SS-UIE model")
    elif args.model == '3d_lut':
        # Image-adaptive 3D LUT: resolution-independent, texture-preserving.
        model = LUT3D(n_channels=3, n_luts=args.lut_num, lut_dim=args.lut_dim).to(device)
        logger.info(f"Using image-adaptive 3D LUT model (lut_dim={args.lut_dim}, n_luts={args.lut_num})")
        if use_gradient_checkpointing:
            logger.warning("Gradient checkpointing requested but not supported for 3D LUT model")
    else:
        model = UNetAutoencoder(n_channels=3, n_classes=3).to(device)
        logger.info("Using U-Net Autoencoder model")
        if use_gradient_checkpointing:
            logger.warning("Gradient checkpointing requested but not supported for U-Net model")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1e6:.2f} MB (FP32)")

    # Loss and optimizer.
    # 'auto' resolves to a per-model default; an explicit --loss overrides it.
    is_ss_uie = args.model == 'ss_uie'
    loss_choice = args.loss
    if loss_choice == 'auto':
        if args.model == 'ss_uie':
            loss_choice = 'ss_uie'
        elif args.model == '3d_lut':
            loss_choice = 'composite'
        else:
            loss_choice = 'combined'

    if loss_choice == 'ss_uie':
        criterion = SSUIECombinedLoss()
        logger.info("Using SS-UIE paper loss: SSIM*10 + L1*10 + LCH + LAB*0.0001 + FDL*10000")
    elif loss_choice == 'composite':
        w_l1, w_mssim, w_ffl, w_lpips = args.composite_weights
        criterion = CompositeLoss(w_l1=w_l1, w_mssim=w_mssim, w_ffl=w_ffl, w_lpips=w_lpips).to(device)
        logger.info(f"Using composite texture loss: L1*{w_l1} + (1-MS-SSIM)*{w_mssim} "
                    f"+ FFL*{w_ffl} + LPIPS*{w_lpips}")
    else:
        criterion = CombinedLoss(alpha=args.l1_weight, beta=args.mse_weight)
        logger.info(f"Using combined loss: L1*{args.l1_weight} + MSE*{args.mse_weight}")

    # Select optimizer - SS-UIE uses different betas per paper
    use_8bit_optimizer = args.optimizer_8bit

    # SS-UIE paper uses beta1=0.5 for faster adaptation (common in image restoration)
    adam_betas = (0.5, 0.999) if is_ss_uie else (0.9, 0.999)

    if use_8bit_optimizer:
        if HAS_BITSANDBYTES:
            optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr, betas=adam_betas)
            logger.info(f"Using 8-bit Adam optimizer (bitsandbytes) with betas={adam_betas}")
        else:
            logger.warning("8-bit optimizer requested but bitsandbytes not installed. Falling back to standard Adam.")
            logger.warning("Install with: pip install bitsandbytes")
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=adam_betas)
            use_8bit_optimizer = False
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=adam_betas)
        if is_ss_uie:
            logger.info(f"Using SS-UIE paper optimizer settings: Adam with betas={adam_betas}")

    # SS-UIE paper uses StepLR scheduler; other models use ReduceLROnPlateau
    if is_ss_uie:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.8)
        logger.info("Using SS-UIE paper scheduler: StepLR(step_size=400, gamma=0.8)")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # torch.compile for PyTorch 2.0+ (deferred until after checkpoint loading)
    use_compile = args.compile
    compile_mode = None
    if use_compile:
        if hasattr(torch, 'compile'):
            # Use 'default' mode when gradient checkpointing is enabled
            # 'reduce-overhead' uses CUDA graphs which conflict with checkpointing's control flow
            if use_gradient_checkpointing:
                compile_mode = 'default'
                logger.info("Using torch.compile with 'default' mode (gradient checkpointing enabled)")
            else:
                compile_mode = 'reduce-overhead'
                logger.info("Using torch.compile with 'reduce-overhead' mode")
        else:
            logger.warning("torch.compile requested but not available (requires PyTorch 2.0+)")
            use_compile = False

    # Mixed precision setup (supports CUDA and MPS)
    use_amp = args.amp and device.type in ('cuda', 'mps')
    scaler = GradScaler(device.type) if use_amp else None

    if args.amp and device.type not in ('cuda', 'mps'):
        logger.warning("AMP requested but not using CUDA or MPS - disabling mixed precision")
    if use_amp:
        logger.info("Mixed precision (AMP) training: ENABLED")
        logger.info(f"Model size with FP16: ~{total_params * 2 / 1e6:.2f} MB")
    else:
        logger.info("Mixed precision (AMP) training: DISABLED")

    # Training state
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []
    patience_counter = 0

    # Resume from checkpoint
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            train_psnrs = checkpoint.get('train_psnrs', [])
            val_psnrs = checkpoint.get('val_psnrs', [])
            patience_counter = checkpoint.get('patience_counter', 0)
            # Restore scaler state if available and using AMP
            if use_amp and scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("Restored AMP scaler state from checkpoint")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")

    # Apply torch.compile after checkpoint loading to avoid state_dict key mismatches
    if use_compile and compile_mode is not None:
        logger.info("Compiling model (this may take a moment on first forward pass)...")
        model = torch.compile(model, mode=compile_mode)
        logger.info("Model compiled successfully")

    # Training loop
    logger.info(f"\nStarting training for {args.epochs} epochs")
    logger.info("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 80)

        # Train
        train_loss, train_psnr = train_epoch(model, train_loader, criterion, optimizer, device,
                                              scaler=scaler, use_amp=use_amp)
        train_losses.append(train_loss)
        train_psnrs.append(train_psnr)

        # Validate
        val_loss, val_psnr = validate_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)

        # Update learning rate
        # StepLR doesn't take metrics, ReduceLROnPlateau does
        if is_ss_uie:
            scheduler.step()
        else:
            scheduler.step(val_loss)

        logger.info(f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f} dB")
        logger.info(f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f} dB")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_psnrs': train_psnrs,
            'val_psnrs': val_psnrs,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'training_config': {
                'resolution': resolution_str,
                'batch_size': args.batch_size,
                'image_size': image_size if image_size is not None else 4606,
                'model': args.model,
                'amp': use_amp,
                'gradient_checkpointing': use_gradient_checkpointing,
                'optimizer_8bit': use_8bit_optimizer,
                'compile': use_compile,
            }
        }
        # Save scaler state for AMP
        if use_amp and scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        # Save latest checkpoint
        latest_checkpoint = checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_checkpoint)

        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = output_dir / 'best_model.pth'

            # Add model config for inference
            checkpoint['model_config'] = {
                'n_channels': 3,
                'n_classes': 3,
                'image_size': image_size if image_size is not None else 4606,
                'model': args.model,
            }
            # Add SS-UIE specific params if applicable
            if args.model == 'ss_uie':
                checkpoint['model_config']['ss_uie_H'] = model_img_size
                checkpoint['model_config']['ss_uie_W'] = model_img_size
            # Add 3D LUT specific params if applicable
            if args.model == '3d_lut':
                checkpoint['model_config']['lut_dim'] = args.lut_dim
                checkpoint['model_config']['lut_num'] = args.lut_num

            torch.save(checkpoint, best_model_path)
            logger.info(f"✓ Saved best model: {best_model_path}")
        else:
            patience_counter += 1
            if args.early_stopping > 0:
                logger.info(f"No improvement in val loss ({patience_counter}/{args.early_stopping})")
                if patience_counter >= args.early_stopping:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            periodic_checkpoint = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, periodic_checkpoint)
            logger.info(f"✓ Saved periodic checkpoint: {periodic_checkpoint}")

    # Save final model
    final_model_path = output_dir / 'final_model.pth'
    final_model_config = {
        'n_channels': 3,
        'n_classes': 3,
        'image_size': image_size if image_size is not None else 4606,
        'model': args.model,
    }
    # Add SS-UIE specific params if applicable
    if args.model == 'ss_uie':
        final_model_config['ss_uie_H'] = model_img_size
        final_model_config['ss_uie_W'] = model_img_size
    # Add 3D LUT specific params if applicable
    if args.model == '3d_lut':
        final_model_config['lut_dim'] = args.lut_dim
        final_model_config['lut_num'] = args.lut_num
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': final_model_config,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_psnrs': train_psnrs,
            'val_psnrs': val_psnrs,
        },
        'training_config': {
            'resolution': resolution_str,
            'batch_size': args.batch_size,
            'epochs': len(train_losses),
            'model': args.model,
            'amp': use_amp,
            'gradient_checkpointing': use_gradient_checkpointing,
            'optimizer_8bit': use_8bit_optimizer,
            'compile': use_compile,
        }
    }
    torch.save(final_checkpoint, final_model_path)

    logger.info("\n" + "=" * 80)
    logger.info("✓ Training complete!")
    logger.info(f"✓ Best model saved to: {output_dir / 'best_model.pth'}")
    logger.info(f"✓ Final model saved to: {final_model_path}")
    logger.info(f"✓ Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
