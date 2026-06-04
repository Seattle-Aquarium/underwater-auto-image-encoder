"""
Unit tests for the image-adaptive 3D LUT model and the texture-preserving
composite loss.

These run on CPU (CI-friendly): tiny tensors, no GPU required.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lut_3d import LUT3D
from src.losses import CompositeLoss, FocalFrequencyLoss, ms_ssim


class TestLUT3DModel:
    """Core behaviour of the LUT3D model."""

    def test_forward_shape_and_range(self):
        """Output must match input spatial shape and stay in [0, 1]."""
        model = LUT3D(lut_dim=9)
        x = torch.rand(2, 3, 32, 48)
        out = model(x)
        assert out.shape == x.shape
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_resolution_independent(self):
        """The same model must accept arbitrary resolutions (per-pixel LUT)."""
        model = LUT3D(lut_dim=9)
        for h, w in [(16, 16), (33, 21), (64, 80)]:
            out = model(torch.rand(1, 3, h, w))
            assert out.shape == (1, 3, h, w)

    def test_identity_lut_preserves_image(self):
        """Applying the identity LUT must return the input (texture-safe core).

        This is the key property: a 3D LUT can only remap colours, never blur.
        Trilinear interpolation of the identity grid is exact.
        """
        dim = 17
        identity = LUT3D._identity_lut(dim).unsqueeze(0)  # [1, 3, D, D, D]
        img = torch.rand(1, 3, 40, 24)
        out = LUT3D.apply_lut(identity, img)
        assert torch.allclose(out, img, atol=1e-4)

    def test_untrained_model_is_approximately_identity(self):
        """With identity-initialised basis LUT + zeroed others, the fresh model
        starts close to a no-op, which trains stably."""
        torch.manual_seed(0)
        model = LUT3D(lut_dim=33).eval()
        img = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            out = model(img)
        # Predictor weights start near [1, 0, 0]; output should be close to input.
        assert torch.allclose(out, img, atol=0.05)

    def test_gradients_flow(self):
        """Both the LUT parameters and the predictor must receive gradients."""
        model = LUT3D(lut_dim=9)
        img = torch.rand(2, 3, 24, 24)
        target = torch.rand(2, 3, 24, 24)
        loss = torch.nn.functional.l1_loss(model(img), target) + model.regularization_loss()
        loss.backward()
        assert model.luts.grad is not None
        assert model.luts.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.predictor.parameters())

    def test_regularization_is_scalar_nonnegative(self):
        model = LUT3D(lut_dim=9)
        reg = model.regularization_loss()
        assert reg.ndim == 0
        assert reg.item() >= 0.0


class TestCompositeLoss:
    """The texture-preserving composite loss."""

    def test_runs_and_is_scalar(self):
        loss_fn = CompositeLoss()
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_zero_when_identical(self):
        """Identical images should give ~zero composite loss."""
        loss_fn = CompositeLoss()
        x = torch.rand(1, 3, 64, 64)
        loss = loss_fn(x, x.clone())
        assert loss.item() < 1e-4

    def test_backward(self):
        loss_fn = CompositeLoss()
        pred = torch.rand(2, 3, 32, 32, requires_grad=True)
        target = torch.rand(2, 3, 32, 32)
        loss_fn(pred, target).backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()

    def test_handles_small_images(self):
        """MS-SSIM must adapt its scale count so tiny images don't crash."""
        loss_fn = CompositeLoss()
        loss = loss_fn(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))
        assert torch.isfinite(loss)


class TestLossPrimitives:
    def test_ms_ssim_identical_is_one(self):
        x = torch.rand(1, 3, 64, 64)
        assert ms_ssim(x, x.clone()).item() == pytest.approx(1.0, abs=1e-3)

    def test_ffl_identical_is_zero(self):
        ffl = FocalFrequencyLoss(ave_spectrum=True, log_matrix=True, batch_matrix=True)
        x = torch.rand(2, 3, 32, 32)
        assert ffl(x, x.clone()).item() < 1e-6

    def test_ffl_nonneg_for_different(self):
        ffl = FocalFrequencyLoss()
        loss = ffl(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32))
        assert loss.item() >= 0.0
