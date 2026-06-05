"""
Image-Adaptive 3D LUT for Underwater Image Enhancement

Based on: Zeng et al., "Learning Image-Adaptive 3D Lookup Tables for High
Performance Photo Enhancement in Real-Time" (ECCV 2020 / TPAMI 2020).
Paper:  https://arxiv.org/abs/2009.14468
Code:   https://github.com/HuiZeng/Image-Adaptive-3DLUT

Why this model for our task
---------------------------
Our editors apply mostly-GLOBAL Lightroom adjustments (white balance, exposure,
curves, dehaze, HSL) -- i.e. a colour-to-colour mapping applied identically to
every pixel. A 3D LUT *is* exactly that function.

A small CNN looks only at a DOWN-SAMPLED copy of the image to predict per-image
weights that fuse N learnable basis LUTs. The fused colour transform is then
applied PER-PIXEL to the FULL-RESOLUTION image via trilinear interpolation. The
full-resolution image never passes through a downsampling/upsampling bottleneck,
so fine-scale texture/edges are preserved *by construction* -- the model can only
remap colours, it cannot blur.

This implementation uses a pure-PyTorch trilinear lookup (via F.grid_sample) so it
runs on CPU / MPS / CUDA without a compiled CUDA extension, matching this repo's
cross-platform inference tests.

I/O convention (matches the other models in this repo):
    forward(x): x is [B, 3, H, W] float in [0, 1]; returns [B, 3, H, W] in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LUT3D(nn.Module):
    """Image-adaptive 3D LUT enhancement model.

    Args:
        n_channels (int): Number of image channels (default: 3, RGB).
        n_luts (int): Number of learnable basis 3D LUTs to fuse (default: 3).
            LUT 0 is initialised to the identity mapping; the rest start at zero.
        lut_dim (int): Resolution of each LUT grid axis (default: 33).
        predictor_size (int): Side length the input is resized to before the
            weight-predictor CNN sees it (default: 256). The full-resolution image
            is never downsampled -- only this thumbnail is.
        tv_weight (float): Weight for the LUT smoothness (total-variation) penalty.
        mn_weight (float): Weight for the LUT monotonicity penalty.
    """

    def __init__(self, n_channels=3, n_luts=3, lut_dim=33, predictor_size=256,
                 tv_weight=1e-4, mn_weight=10.0):
        super().__init__()
        if n_channels != 3:
            raise ValueError("LUT3D currently supports 3-channel (RGB) images only")

        self.n_channels = n_channels
        self.n_luts = n_luts
        self.lut_dim = lut_dim
        self.predictor_size = predictor_size
        self.tv_weight = tv_weight
        self.mn_weight = mn_weight

        # Learnable basis LUTs: [n_luts, 3, D, D, D].
        # The grid axes correspond to (r, g, b) input coordinates; the channel
        # dim holds the mapped (r', g', b') output colour.
        luts = torch.zeros(n_luts, 3, lut_dim, lut_dim, lut_dim)
        luts[0] = self._identity_lut(lut_dim)  # first basis LUT = identity
        self.luts = nn.Parameter(luts)

        # Weight predictor CNN. Runs only on the down-sampled thumbnail and outputs
        # one fusion weight per basis LUT.
        self.predictor = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d(1),
        )
        self.weight_head = nn.Linear(128, n_luts)
        # Initialise so the model starts as ~identity: bias puts weight 1 on the
        # identity LUT and 0 on the others. The head weights use a tiny random
        # init (not zeros) so the predictor receives gradients from the first
        # step while the starting transform stays an approximate no-op.
        nn.init.normal_(self.weight_head.weight, std=1e-4)
        with torch.no_grad():
            bias = torch.zeros(n_luts)
            bias[0] = 1.0
            self.weight_head.bias.copy_(bias)

    @staticmethod
    def _identity_lut(dim):
        """Build an identity 3D LUT of shape [3, dim, dim, dim].

        Entry [c, i, j, k] returns the c-th input coordinate, so the LUT maps
        (r, g, b) -> (r, g, b). Trilinear interpolation of this grid is exact
        (it is linear), so the identity LUT is a true no-op.
        """
        coords = torch.linspace(0.0, 1.0, dim)
        r = coords.view(dim, 1, 1).expand(dim, dim, dim)
        g = coords.view(1, dim, 1).expand(dim, dim, dim)
        b = coords.view(1, 1, dim).expand(dim, dim, dim)
        return torch.stack([r, g, b], dim=0)

    @staticmethod
    def apply_lut(lut, img):
        """Apply per-image 3D LUTs to a batch of images via trilinear lookup.

        Args:
            lut: [B, 3, D, D, D] fused LUT per image (axes = r, g, b).
            img: [B, 3, H, W] in [0, 1].
        Returns:
            [B, 3, H, W] mapped image.
        """
        b_size = img.shape[0]
        # grid_sample expects coords in (x, y, z) order mapping to the volume's
        # (W, H, D) axes. Our LUT axes are (r=D, g=H, b=W), so x<-blue, y<-green,
        # z<-red. Coordinates are rescaled from [0, 1] to [-1, 1].
        r = img[:, 0]
        g = img[:, 1]
        b = img[:, 2]
        grid = torch.stack([b, g, r], dim=-1)            # [B, H, W, 3]
        grid = grid * 2.0 - 1.0
        grid = grid.unsqueeze(1)                          # [B, 1, H, W, 3]
        out = F.grid_sample(lut, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        # out: [B, 3, 1, H, W] -> [B, 3, H, W]
        return out.squeeze(2)

    def _predict_weights(self, x):
        thumb = F.interpolate(x, size=(self.predictor_size, self.predictor_size),
                              mode='bilinear', align_corners=False)
        feat = self.predictor(thumb).flatten(1)          # [B, 128]
        return self.weight_head(feat)                     # [B, n_luts]

    def forward(self, x):
        weights = self._predict_weights(x)               # [B, n_luts]
        # Fuse basis LUTs per image: [B, 3, D, D, D].
        fused = torch.einsum('bn,ncxyz->bcxyz', weights, self.luts)
        out = self.apply_lut(fused, x)
        return out.clamp(0.0, 1.0)

    def regularization_loss(self):
        """Smoothness (TV) + monotonicity penalties on the basis LUTs.

        These keep the learned colour transforms well-behaved (no banding) and
        monotonic, as in Zeng et al. Returns a scalar tensor; add it to the data
        loss during training.
        """
        lut = self.luts                                   # [n, 3, D, D, D]
        dr = lut[:, :, 1:, :, :] - lut[:, :, :-1, :, :]
        dg = lut[:, :, :, 1:, :] - lut[:, :, :, :-1, :]
        db = lut[:, :, :, :, 1:] - lut[:, :, :, :, :-1]
        tv = (dr ** 2).mean() + (dg ** 2).mean() + (db ** 2).mean()
        # Monotonicity: each output channel should not decrease along its own
        # input axis (channel 0/r along axis r, etc.).
        mn = (torch.relu(-dr[:, 0]).mean()
              + torch.relu(-dg[:, 1]).mean()
              + torch.relu(-db[:, 2]).mean())
        return self.tv_weight * tv + self.mn_weight * mn
