#!/usr/bin/env python3
"""
All-in-one script for downloading, preparing, cropping, and training.

Perfect for ephemeral VMs where you need to run the full workflow frequently.
Handles authentication, downloads from HuggingFace, prepares the dataset,
crops images to standard dimensions, and starts training.

The script is idempotent - it will skip steps that are already completed,
so you can safely re-run it if interrupted.
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description, check=True):
    """Run a shell command and log the output."""
    logger.info(f"{'='*80}")
    logger.info(f"Step: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*80}")

    try:
        result = subprocess.run(
            cmd,
            check=check,
            text=True,
            capture_output=False  # Show output in real-time
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if check:
            raise
        return False


def check_huggingface_auth():
    """Check if HuggingFace authentication is available."""
    # Check for token in environment
    if os.environ.get("HF_TOKEN"):
        logger.info("✓ HuggingFace token found in environment (HF_TOKEN)")
        return True

    # Check for saved token
    try:
        from huggingface_hub import get_token
        token = get_token()
        if token:
            logger.info("✓ HuggingFace token found (from huggingface-cli login)")
            return True
    except ImportError:
        pass
    except Exception:
        pass

    logger.warning("⚠️  No HuggingFace authentication found")
    logger.warning("You may hit rate limits during download")
    logger.warning("To authenticate, run: huggingface-cli login")
    return False


def load_config(config_path: str = "setup_and_train_config.yaml"):
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.warning("Using default values. Create setup_and_train_config.yaml to customize.")
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        logger.warning("Falling back to default values")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="All-in-one script: Download, prepare, crop, and train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (loads from setup_and_train_config.yaml)
  python training/setup_and_train.py

  # Use custom config file
  python training/setup_and_train.py --config my_config.yaml

  # Override specific settings via command line
  python training/setup_and_train.py --batch-size 4 --epochs 100

  # Skip download if already exists
  python training/setup_and_train.py --skip-download

  # Process only specific sets
  python training/setup_and_train.py --sets set01 set02

  # Force output cropping (rarely needed)
  python training/setup_and_train.py --crop-output
        """
    )

    # Config file argument
    parser.add_argument(
        '--config',
        type=str,
        default='setup_and_train_config.yaml',
        help='Path to configuration file (default: setup_and_train_config.yaml)'
    )

    # Dataset arguments (defaults will be loaded from config)
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=None,
        help='Directory to download/store raw dataset'
    )

    parser.add_argument(
        '--working-dir',
        type=str,
        default=None,
        help='Working directory for prepared dataset'
    )

    parser.add_argument(
        '--repo-id',
        type=str,
        default=None,
        help='HuggingFace repository ID'
    )

    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace API token (optional, can also use HF_TOKEN env var)'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=None,
        help='Maximum number of retry attempts for HuggingFace rate limits (default: 20)'
    )

    parser.add_argument(
        '--retry-delay',
        type=int,
        default=None,
        help='Initial delay in seconds before first retry (default: 310 - HF rate limit is 5 min)'
    )

    parser.add_argument(
        '--sets',
        nargs='+',
        default=None,
        help='Specific sets to process (e.g., set01 set02). If not specified, processes all sets'
    )

    # Cropping arguments
    parser.add_argument(
        '--crop-width',
        type=int,
        default=None,
        help='Crop width for input images'
    )

    parser.add_argument(
        '--crop-height',
        type=int,
        default=None,
        help='Crop height for input images'
    )

    parser.add_argument(
        '--crop-output',
        action='store_true',
        help='Crop output/target images (by default, output cropping is skipped since HuggingFace outputs are already consistent)'
    )

    # Training arguments
    parser.add_argument(
        '--image-size',
        type=int,
        default=None,
        help='Training image size'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='DataLoader workers for parallel image decoding'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for trained models'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Checkpoint directory'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=['unet', 'ushape_transformer', 'ss_uie', '3d_lut'],
        help='Model architecture: unet, ushape_transformer, ss_uie, or 3d_lut'
    )

    parser.add_argument(
        '--loss',
        type=str,
        default=None,
        choices=['auto', 'combined', 'composite', 'ss_uie'],
        help="Loss function (default: auto, picks per model). 'composite' = "
             "L1 + MS-SSIM + Focal Frequency (+ optional LPIPS) for texture preservation."
    )

    parser.add_argument(
        '--lut-dim',
        type=int,
        default=None,
        help='3D LUT grid resolution per axis (default: 33, only used for --model 3d_lut)'
    )

    parser.add_argument(
        '--lut-num',
        type=int,
        default=None,
        help='Number of basis 3D LUTs to fuse (default: 3, only used for --model 3d_lut)'
    )

    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable automatic mixed precision (FP16) training for reduced memory usage'
    )

    parser.add_argument(
        '--gradient-checkpointing',
        action='store_true',
        help='Enable gradient checkpointing to reduce memory usage (trades compute for memory)'
    )

    parser.add_argument(
        '--optimizer-8bit',
        action='store_true',
        help='Use 8-bit Adam optimizer (requires bitsandbytes) for ~1.5-2GB memory savings'
    )

    parser.add_argument(
        '--compile',
        action='store_true',
        help='Use torch.compile for potential speedups and memory optimization (PyTorch 2.0+)'
    )

    parser.add_argument(
        '--early-stopping',
        type=int,
        default=None,
        help='Early stopping patience - stop if val loss does not improve for N epochs (default: 15, 0 to disable)'
    )

    parser.add_argument(
        '--random-crop',
        action='store_true',
        help='Use random crops at native resolution instead of resizing (preserves detail)'
    )

    parser.add_argument(
        '--crops-per-image',
        type=int,
        default=None,
        help='Number of random crops per image when --random-crop is enabled (default: 8)'
    )

    # Step control
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download step (use if dataset already downloaded)'
    )

    parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip preparation step (use if already prepared)'
    )

    parser.add_argument(
        '--skip-crop',
        action='store_true',
        help='Skip cropping step (use if already cropped)'
    )

    parser.add_argument(
        '--prepare-only',
        action='store_true',
        help='Only download, prepare, and crop - do not start training'
    )

    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Use symlinks instead of copying files during preparation (saves disk space)'
    )

    parser.add_argument(
        '--split-ratio',
        type=float,
        default=None,
        help='Train/validation split ratio'
    )

    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)

    # Apply config defaults, then override with CLI arguments
    # Download config
    download_config = config.get('download', {})
    dataset_dir = args.dataset_dir or download_config.get('dataset_dir', 'dataset_raw')
    repo_id = args.repo_id or download_config.get('repo_id', 'Seattle-Aquarium/Seattle_Aquarium_benthic_imagery')
    # Note: hf_token is ONLY from CLI or env var, never from config file (security)
    hf_token = args.hf_token

    # Preparation config
    prep_config = config.get('preparation', {})
    working_dir = args.working_dir or prep_config.get('working_dir', 'training_dataset')
    sets = args.sets or prep_config.get('sets')
    symlink = args.symlink or prep_config.get('symlink', False)
    split_ratio = args.split_ratio or prep_config.get('split_ratio', 0.8)

    # Cropping config
    crop_config = config.get('cropping', {})
    crop_width = args.crop_width or crop_config.get('input', {}).get('width', 4606)
    crop_height = args.crop_height or crop_config.get('input', {}).get('height', 4030)
    crop_output = args.crop_output or crop_config.get('output', {}).get('enabled', False)

    # Training config
    training_config = config.get('training', {})
    image_size = args.image_size or training_config.get('image_size', 1024)
    batch_size = args.batch_size or training_config.get('batch_size', 8)
    epochs = args.epochs or training_config.get('epochs', 50)
    num_workers = args.num_workers if args.num_workers is not None else training_config.get('num_workers', 4)
    lr = args.lr or training_config.get('learning_rate', 1e-4)
    output_dir = args.output_dir or training_config.get('output_dir', 'output')
    checkpoint_dir = args.checkpoint_dir or training_config.get('checkpoint_dir', 'checkpoints')
    resume = args.resume or training_config.get('resume')
    model = args.model or training_config.get('model', 'unet')
    loss = args.loss or training_config.get('loss', 'auto')
    lut_dim = args.lut_dim or training_config.get('lut_dim', 33)
    lut_num = args.lut_num or training_config.get('lut_num', 3)
    # Handle amp: CLI flag takes precedence (when True), then config, default False
    use_amp = args.amp or training_config.get('amp', False)
    # Handle gradient_checkpointing: CLI flag takes precedence (when True), then config, default False
    use_gradient_checkpointing = args.gradient_checkpointing or training_config.get('gradient_checkpointing', False)
    # Handle optimizer_8bit: CLI flag takes precedence (when True), then config, default False
    use_optimizer_8bit = args.optimizer_8bit or training_config.get('optimizer_8bit', False)
    # Handle compile: CLI flag takes precedence (when True), then config, default False
    use_compile = args.compile or training_config.get('compile', False)
    # Handle random_crop: CLI flag takes precedence (when True), then config, default False
    use_random_crop = args.random_crop or training_config.get('random_crop', False)
    # Crops per image (only used when random_crop is enabled)
    crops_per_image = args.crops_per_image or training_config.get('crops_per_image', 8)
    # Early stopping patience
    early_stopping = args.early_stopping if args.early_stopping is not None else training_config.get('early_stopping', 15)

    # Step control
    steps_config = config.get('steps', {})
    skip_download = args.skip_download or steps_config.get('skip_download', False)
    skip_prepare = args.skip_prepare or steps_config.get('skip_prepare', False)
    skip_crop = args.skip_crop or steps_config.get('skip_crop', False)
    prepare_only = args.prepare_only or steps_config.get('prepare_only', False)

    # Convert to Path objects
    dataset_dir = Path(dataset_dir)
    working_dir = Path(working_dir)

    logger.info("="*80)
    logger.info("UNDERWATER IMAGE ENHANCEMENT - AUTOMATED TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Dataset directory: {dataset_dir.absolute()}")
    logger.info(f"Working directory: {working_dir.absolute()}")
    logger.info(f"Repository: {repo_id}")
    logger.info("="*80)

    # Step 1: Check HuggingFace authentication
    if not skip_download:
        check_huggingface_auth()

    # Step 2: Download dataset
    if not skip_download:
        logger.info("\n📥 STEP 1/4: Downloading dataset from HuggingFace...")

        cmd = [
            sys.executable, str(Path(__file__).parent.parent / 'dataset_prep' / 'download_dataset.py'),
            '--repo-id', repo_id,
            '--output', str(dataset_dir)
        ]

        if hf_token:
            cmd.extend(['--token', hf_token])

        # Add retry parameters if specified
        max_retries = args.max_retries or download_config.get('max_retries')
        retry_delay = args.retry_delay or download_config.get('retry_delay')

        if max_retries is not None:
            cmd.extend(['--max-retries', str(max_retries)])

        if retry_delay is not None:
            cmd.extend(['--retry-delay', str(retry_delay)])

        run_command(cmd, "Download dataset from HuggingFace")
    else:
        logger.info("\n⏭️  STEP 1/4: Skipping download (--skip-download specified)")
        if not dataset_dir.exists():
            logger.error(f"Dataset directory does not exist: {dataset_dir}")
            logger.error("Remove --skip-download or ensure the dataset is already downloaded")
            return 1

    # Step 3: Prepare dataset (organize sets into input/target structure)
    if not skip_prepare:
        logger.info("\n🗂️  STEP 2/4: Preparing dataset (organizing sets)...")

        # Determine source directories
        if sets:
            # Use specific sets
            source_dirs = [str(dataset_dir / set_name) for set_name in sets]
            logger.info(f"Processing specific sets: {', '.join(sets)}")
        else:
            # Process all sets
            source_dirs = [str(dataset_dir)]
            logger.info("Processing all sets")

        cmd = [
            sys.executable, str(Path(__file__).parent.parent / 'dataset_prep' / 'prepare_huggingface_dataset.py'),
            *source_dirs,
            '--output', str(working_dir),
            '--split-ratio', str(split_ratio)
        ]

        if symlink:
            cmd.append('--symlink')

        run_command(cmd, "Prepare dataset structure")
    else:
        logger.info("\n⏭️  STEP 2/4: Skipping preparation (--skip-prepare specified)")
        if not working_dir.exists():
            logger.error(f"Working directory does not exist: {working_dir}")
            logger.error("Remove --skip-prepare or ensure the dataset is already prepared")
            return 1

    # Step 4: Crop images
    if not skip_crop:
        logger.info("\n✂️  STEP 3/4: Cropping images to standard dimensions...")

        input_dir = working_dir / "input"
        target_dir = working_dir / "target"
        input_cropped_dir = working_dir / "input_cropped"
        target_cropped_dir = working_dir / "target_cropped"

        # Crop input images
        logger.info(f"Cropping input images to {crop_width}×{crop_height}...")
        cmd = [
            sys.executable, str(Path(__file__).parent.parent / 'dataset_prep' / 'crop_tiff.py'),
            str(input_dir),
            '--output-dir', str(input_cropped_dir),
            '--width', str(crop_width),
            '--height', str(crop_height)
        ]
        run_command(cmd, "Crop input images")

        # Crop target images (only if explicitly requested)
        if crop_output:
            logger.info(f"Cropping target images to {crop_width}×{crop_height}...")
            cmd = [
                sys.executable, str(Path(__file__).parent.parent / 'dataset_prep' / 'crop_tiff.py'),
                str(target_dir),
                '--output-dir', str(target_cropped_dir),
                '--width', str(crop_width),
                '--height', str(crop_height),
                '--preserve-format'
            ]
            run_command(cmd, "Crop target images")
        else:
            logger.info("Skipping target image cropping (HuggingFace outputs are already consistent)")
            # Just symlink or copy target directory
            if symlink and not target_cropped_dir.exists():
                target_cropped_dir.symlink_to(target_dir.absolute())
            elif not target_cropped_dir.exists():
                import shutil
                shutil.copytree(target_dir, target_cropped_dir)
    else:
        logger.info("\n⏭️  STEP 3/4: Skipping cropping (--skip-crop specified)")
        input_cropped_dir = working_dir / "input_cropped"
        target_cropped_dir = working_dir / "target_cropped"

        if not input_cropped_dir.exists() or not target_cropped_dir.exists():
            logger.error("Cropped directories do not exist")
            logger.error("Remove --skip-crop or ensure images are already cropped")
            return 1

    # Stop here if prepare-only mode
    if prepare_only:
        logger.info("\n✓ Dataset preparation complete!")
        logger.info("="*80)
        logger.info("Dataset is ready for training. To start training, run:")
        logger.info(f"  python training/train.py \\")
        logger.info(f"    --input-dir {input_cropped_dir} \\")
        logger.info(f"    --target-dir {target_cropped_dir} \\")
        logger.info(f"    --image-size {image_size} \\")
        logger.info(f"    --batch-size {batch_size} \\")
        logger.info(f"    --epochs {epochs} \\")
        logger.info(f"    --model {model}")
        logger.info("="*80)
        return 0

    # Step 5: Train model
    logger.info("\n🚀 STEP 4/4: Starting training...")

    cmd = [
        sys.executable, str(Path(__file__).parent / 'train.py'),
        '--input-dir', str(input_cropped_dir),
        '--target-dir', str(target_cropped_dir),
        '--image-size', str(image_size),
        '--batch-size', str(batch_size),
        '--epochs', str(epochs),
        '--num-workers', str(num_workers),
        '--lr', str(lr),
        '--output-dir', output_dir,
        '--checkpoint-dir', checkpoint_dir,
        '--model', model
    ]

    if loss and loss != 'auto':
        cmd.extend(['--loss', loss])

    if model == '3d_lut':
        cmd.extend(['--lut-dim', str(lut_dim), '--lut-num', str(lut_num)])

    if resume:
        cmd.extend(['--resume', resume])

    if use_amp:
        cmd.append('--amp')

    if use_gradient_checkpointing:
        cmd.append('--gradient-checkpointing')

    if use_optimizer_8bit:
        cmd.append('--optimizer-8bit')

    if use_compile:
        cmd.append('--compile')

    if use_random_crop:
        cmd.append('--random-crop')
        cmd.extend(['--crops-per-image', str(crops_per_image)])

    cmd.extend(['--early-stopping', str(early_stopping)])

    run_command(cmd, "Train underwater enhancement model")

    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Best model saved to: {output_dir}/best_model.pth")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}/")
    logger.info("\nTo run inference:")
    logger.info(f"  python inference.py input.jpg --checkpoint {output_dir}/best_model.pth")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        logger.info("You can resume by re-running the same command")
        logger.info("Already completed steps will be skipped automatically")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
