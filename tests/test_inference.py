"""
Tests for inference script - ensures model loading and inference work correctly
for all model types and checkpoint formats.
"""

import subprocess
import sys
from pathlib import Path
from unittest import mock

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

    def test_lut3d_large_image_processed_whole_not_tiled(
        self, lut3d_checkpoint_path, large_test_image_path):
        """A large (>2048px) image must be processed in one pass, not tiled.

        Tiling would run the weight-predictor per tile and apply a different LUT to
        each region, breaking the model's single global colour transform. Guard
        against a regression that routes 3D LUT through process_image_tiled.
        """
        original = Image.open(large_test_image_path)
        assert max(original.size) > 2048  # fixture must exercise the tiling threshold

        inferencer = Inferencer(str(lut3d_checkpoint_path))
        with mock.patch.object(
                inferencer, 'process_image_tiled',
                side_effect=AssertionError('3D LUT must not be tiled')) as tiled:
            result = inferencer.process_image(large_test_image_path)

        tiled.assert_not_called()
        # Native resolution preserved exactly (and no width/height axis swap).
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


class TestJpegSaveOptions:
    """Tests for JPEG save-option plumbing (quality/subsampling/optimize/progressive)"""

    def _spy_on_save(self, monkeypatch):
        """Patch PIL's Image.save to record the params of the last save call."""
        captured = {}
        original_save = Image.Image.save

        def spy(self, fp, *args, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return original_save(self, fp, *args, **kwargs)

        monkeypatch.setattr(Image.Image, "save", spy)
        return captured

    def test_save_options_passed_through_to_pil(self, unet_checkpoint_path, test_image_path, tmp_path, monkeypatch):
        """All JPEG options should reach the underlying PIL save() call"""
        captured = self._spy_on_save(monkeypatch)
        inferencer = Inferencer(str(unet_checkpoint_path))
        opts = {"quality": 80, "subsampling": 2, "optimize": True, "progressive": True}
        inferencer.process_image(test_image_path, tmp_path / "out.jpg", save_options=opts)

        assert captured.get("quality") == 80
        assert captured.get("subsampling") == 2
        assert captured.get("optimize") is True
        assert captured.get("progressive") is True

    def test_default_jpeg_quality_is_95(self, unet_checkpoint_path, test_image_path, tmp_path, monkeypatch):
        """With no save_options, JPEG output should default to quality=95 (prior behavior)"""
        captured = self._spy_on_save(monkeypatch)
        inferencer = Inferencer(str(unet_checkpoint_path))
        inferencer.process_image(test_image_path, tmp_path / "out.jpg")

        assert captured.get("quality") == 95

    def test_jpeg_options_not_applied_to_tiff(self, unet_checkpoint_path, test_image_path, tmp_path, monkeypatch):
        """JPEG-only params must not be passed when the output is TIFF"""
        captured = self._spy_on_save(monkeypatch)
        inferencer = Inferencer(str(unet_checkpoint_path))
        inferencer.process_image(test_image_path, tmp_path / "out.tiff",
                                 save_options={"quality": 50, "progressive": True})

        assert "quality" not in captured
        assert "progressive" not in captured

    def test_quality_affects_output_size(self, unet_checkpoint_path, test_image_path, tmp_path):
        """Lower quality should yield a smaller file than higher quality"""
        inferencer = Inferencer(str(unet_checkpoint_path))
        low = tmp_path / "low.jpg"
        high = tmp_path / "high.jpg"
        inferencer.process_image(test_image_path, low, save_options={"quality": 15})
        inferencer.process_image(test_image_path, high, save_options={"quality": 95})

        assert low.stat().st_size < high.stat().st_size

    def test_progressive_flag_encoded(self, unet_checkpoint_path, test_image_path, tmp_path):
        """The progressive option should produce a progressive JPEG"""
        inferencer = Inferencer(str(unet_checkpoint_path))
        out = tmp_path / "prog.jpg"
        inferencer.process_image(test_image_path, out, save_options={"quality": 90, "progressive": True})

        assert "progression" in Image.open(out).info


class TestExifPreservation:
    """Tests that source EXIF metadata is carried into enhanced outputs"""

    MODEL_TAG = 0x0110  # EXIF 'Model' tag

    def _jpeg_with_exif(self, path, model="GoPro TEST"):
        exif = Image.Exif()
        exif[self.MODEL_TAG] = model
        Image.effect_noise((64, 64), 50).convert("RGB").save(path, exif=exif.tobytes(), quality=95)

    def test_exif_preserved_in_jpeg_output(self, unet_checkpoint_path, tmp_path):
        src = tmp_path / "src.jpg"
        self._jpeg_with_exif(src)
        inferencer = Inferencer(str(unet_checkpoint_path))
        out = tmp_path / "out.jpg"
        inferencer.process_image(str(src), out)

        assert Image.open(out).getexif().get(self.MODEL_TAG) == "GoPro TEST"

    def test_exif_preserved_in_tiff_output(self, unet_checkpoint_path, tmp_path):
        src = tmp_path / "src.jpg"
        self._jpeg_with_exif(src)
        inferencer = Inferencer(str(unet_checkpoint_path))
        out = tmp_path / "out.tiff"
        inferencer.process_image(str(src), out)

        assert Image.open(out).getexif().get(self.MODEL_TAG) == "GoPro TEST"

    def test_exif_coexists_with_save_options(self, unet_checkpoint_path, tmp_path):
        src = tmp_path / "src.jpg"
        self._jpeg_with_exif(src)
        inferencer = Inferencer(str(unet_checkpoint_path))
        out = tmp_path / "out.jpg"
        inferencer.process_image(str(src), out, save_options={"quality": 80, "progressive": True})

        result = Image.open(out)
        assert result.getexif().get(self.MODEL_TAG) == "GoPro TEST"
        assert "progression" in result.info

    def test_explicit_exif_arg_overrides_source(self, unet_checkpoint_path, tmp_path):
        """An explicit exif= (e.g. salvaged from a GPR's DNG) is used over source EXIF"""
        src = tmp_path / "src.jpg"
        self._jpeg_with_exif(src, model="SOURCE")
        override = Image.Exif()
        override[self.MODEL_TAG] = "FROM_DNG"
        inferencer = Inferencer(str(unet_checkpoint_path))
        out = tmp_path / "out.jpg"
        inferencer.process_image(str(src), out, exif=override.tobytes())

        assert Image.open(out).getexif().get(self.MODEL_TAG) == "FROM_DNG"

    def test_source_without_exif_saves_cleanly(self, unet_checkpoint_path, test_image_path, tmp_path):
        """A source with no EXIF should save without error"""
        inferencer = Inferencer(str(unet_checkpoint_path))
        out = tmp_path / "out.jpg"
        inferencer.process_image(test_image_path, out)

        assert out.exists()

    def test_malformed_exif_falls_back_to_metadata_free_save(self, unet_checkpoint_path, tmp_path):
        """A malformed EXIF block should not fail the export"""
        inferencer = Inferencer(str(unet_checkpoint_path))
        out = tmp_path / "out.jpg"
        inferencer._save_image(Image.new("RGB", (16, 16)), out, save_options=None, exif=b"not-valid-exif")

        assert out.exists()
