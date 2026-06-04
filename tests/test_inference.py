"""
Tests for inference script - ensures model loading and inference work correctly
for all model types and checkpoint formats.
"""

import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.inference import Inferencer


class TestModelDetection:
    """Test that model type detection works correctly"""

    def test_detect_unet_model(self, unet_checkpoint_path):
        """UNet checkpoints should be detected as UNetAutoencoder"""
        checkpoint = torch.load(unet_checkpoint_path, map_location='cpu')
        inferencer = Inferencer.__new__(Inferencer)
        model_type = inferencer._detect_model_type(checkpoint)
        assert model_type == 'UNetAutoencoder'

    def test_detect_ushape_model(self, ushape_checkpoint_path):
        """U-Shape Transformer checkpoints should be detected by 'mtc.' keys"""
        checkpoint = torch.load(ushape_checkpoint_path, map_location='cpu')
        inferencer = Inferencer.__new__(Inferencer)
        model_type = inferencer._detect_model_type(checkpoint)
        assert model_type == 'UShapeTransformer'

    def test_detect_ushape_legacy_model(self, ushape_legacy_checkpoint_path):
        """Legacy U-Shape checkpoints should still be detected correctly"""
        checkpoint = torch.load(ushape_legacy_checkpoint_path, map_location='cpu')
        inferencer = Inferencer.__new__(Inferencer)
        model_type = inferencer._detect_model_type(checkpoint)
        assert model_type == 'UShapeTransformer'

    def test_detect_lut3d_model(self, lut3d_checkpoint_path):
        """3D LUT checkpoints should be detected by their 'luts' key"""
        checkpoint = torch.load(lut3d_checkpoint_path, map_location='cpu')
        inferencer = Inferencer.__new__(Inferencer)
        model_type = inferencer._detect_model_type(checkpoint)
        assert model_type == 'LUT3D'


class TestLUT3DInference:
    """Tests for 3D LUT model inference"""

    def test_load_lut3d_checkpoint(self, lut3d_checkpoint_path):
        """3D LUT checkpoint should load successfully with correct LUT shape"""
        inferencer = Inferencer(str(lut3d_checkpoint_path))
        assert inferencer.model is not None
        assert inferencer.detected_model_type == 'LUT3D'
        assert inferencer.config['model']['lut_dim'] == 9
        assert inferencer.config['model']['lut_num'] == 3

    def test_lut3d_inference_single_image(self, lut3d_checkpoint_path, test_image_path):
        """3D LUT should process a single image successfully"""
        inferencer = Inferencer(str(lut3d_checkpoint_path))
        result = inferencer.process_image(test_image_path)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'

    def test_lut3d_output_shape_matches_input(self, lut3d_checkpoint_path, test_image_path):
        """3D LUT processes at native resolution: output size == input size"""
        original = Image.open(test_image_path)
        inferencer = Inferencer(str(lut3d_checkpoint_path))
        result = inferencer.process_image(test_image_path)
        assert result.size == original.size


class TestUNetInference:
    """Tests for UNet model inference"""

    def test_load_unet_checkpoint(self, unet_checkpoint_path):
        """UNet checkpoint should load successfully"""
        inferencer = Inferencer(str(unet_checkpoint_path))
        assert inferencer.model is not None
        assert inferencer.detected_model_type == 'UNetAutoencoder'

    def test_unet_inference_single_image(self, unet_checkpoint_path, test_image_path):
        """UNet should process a single image successfully"""
        inferencer = Inferencer(str(unet_checkpoint_path))

        # Load and process image
        result = inferencer.process_image(test_image_path)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'
        assert result.size[0] > 0 and result.size[1] > 0

    def test_unet_compiled_checkpoint(self, compiled_unet_checkpoint_path, test_image_path):
        """UNet checkpoint with torch.compile() prefix should load and work"""
        inferencer = Inferencer(str(compiled_unet_checkpoint_path))
        result = inferencer.process_image(test_image_path)

        assert result is not None
        assert isinstance(result, Image.Image)


class TestUShapeTransformerInference:
    """Tests for U-Shape Transformer model inference"""

    def test_load_ushape_checkpoint(self, ushape_checkpoint_path):
        """U-Shape Transformer checkpoint should load successfully"""
        inferencer = Inferencer(str(ushape_checkpoint_path))
        assert inferencer.model is not None
        assert inferencer.detected_model_type == 'UShapeTransformer'

    def test_ushape_inference_single_image(self, ushape_checkpoint_path, test_image_path):
        """U-Shape Transformer should process a single image successfully"""
        inferencer = Inferencer(str(ushape_checkpoint_path))

        result = inferencer.process_image(test_image_path)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGB'

    def test_ushape_legacy_checkpoint(self, ushape_legacy_checkpoint_path, test_image_path):
        """Legacy U-Shape checkpoint with 'transformer.net' keys should load and work"""
        inferencer = Inferencer(str(ushape_legacy_checkpoint_path))

        assert inferencer.model is not None
        assert inferencer.detected_model_type == 'UShapeTransformer'

        result = inferencer.process_image(test_image_path)
        assert result is not None
        assert isinstance(result, Image.Image)


class TestInferenceCLI:
    """Test the inference CLI script"""

    def test_cli_unet_inference(self, unet_checkpoint_path, test_image_path, tmp_path):
        """CLI should successfully run inference with UNet model"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "inference/inference.py",
                test_image_path,
                "--checkpoint", str(unet_checkpoint_path),
                "--output", str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Check output file was created
        output_files = list(output_dir.glob("*_enhanced.*"))
        assert len(output_files) == 1, f"Expected 1 output file, found {len(output_files)}"

    def test_cli_ushape_inference(self, ushape_checkpoint_path, test_image_path, tmp_path):
        """CLI should successfully run inference with U-Shape Transformer model"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "inference/inference.py",
                test_image_path,
                "--checkpoint", str(ushape_checkpoint_path),
                "--output", str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        output_files = list(output_dir.glob("*_enhanced.*"))
        assert len(output_files) == 1

    def test_cli_ushape_legacy_inference(self, ushape_legacy_checkpoint_path, test_image_path, tmp_path):
        """CLI should successfully run inference with legacy U-Shape checkpoint"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "inference/inference.py",
                test_image_path,
                "--checkpoint", str(ushape_legacy_checkpoint_path),
                "--output", str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        output_files = list(output_dir.glob("*_enhanced.*"))
        assert len(output_files) == 1

    def test_cli_batch_inference(self, unet_checkpoint_path, test_image_dir, tmp_path):
        """CLI should successfully process a directory of images"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                "inference/inference.py",
                test_image_dir,
                "--checkpoint", str(unet_checkpoint_path),
                "--output", str(output_dir),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Verify at least one output file was created
        output_files = list(output_dir.glob("*_enhanced.*"))
        assert len(output_files) >= 1, "No output files were created"


class TestOutputConsistency:
    """Test that model outputs are consistent and valid"""

    def test_unet_output_shape_matches_input(self, unet_checkpoint_path, test_image_path):
        """UNet output should have same spatial dimensions as input"""
        inferencer = Inferencer(str(unet_checkpoint_path))

        # Get input image size
        input_img = Image.open(test_image_path)
        input_size = input_img.size  # (width, height)

        result = inferencer.process_image(test_image_path)

        # Result is PIL Image with .size = (width, height)
        assert result.size[0] == input_size[0], "Width mismatch"
        assert result.size[1] == input_size[1], "Height mismatch"

    def test_ushape_output_shape_matches_input(self, ushape_checkpoint_path, test_image_path):
        """U-Shape output should have same spatial dimensions as input"""
        inferencer = Inferencer(str(ushape_checkpoint_path))

        input_img = Image.open(test_image_path)
        input_size = input_img.size

        result = inferencer.process_image(test_image_path)

        assert result.size[0] == input_size[0], "Width mismatch"
        assert result.size[1] == input_size[1], "Height mismatch"

    def test_output_is_valid_image(self, unet_checkpoint_path, test_image_path, tmp_path):
        """Output should be saveable as a valid image"""
        inferencer = Inferencer(str(unet_checkpoint_path))
        result = inferencer.process_image(test_image_path)

        # Save and reload to verify it's a valid image
        output_path = tmp_path / "output.png"
        result.save(output_path)

        reloaded = Image.open(output_path)
        assert reloaded.size == result.size
        assert reloaded.mode == 'RGB'
