"""
Texture-preserving composite loss for image enhancement.

Motivation
----------
Pixel L1/MSE losses provably over-smooth: they minimise error by predicting the
blurry average of plausible textures, and neural nets are spectrally biased toward
low frequencies. This composite loss adds two terms that directly target the
high-frequency texture pixel losses discard:

  * MS-SSIM   -- preserves contrast/structure in high-frequency regions
                 (Zhao et al., IEEE TCI 2017).
  * Focal Frequency Loss -- adaptively focuses on hard-to-synthesise frequency
                 components, countering spectral bias (Jiang et al., ICCV 2021).
  * LPIPS     -- optional learned perceptual term (off by default; over-weighting
                 it can hallucinate texture, which is risky for downstream
                 scientific classification).

All implementations are self-contained pure-PyTorch (no SS-UIE lib dependency,
no extra required packages) so they run on CPU in CI. Images are expected as
[B, C, H, W] float in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Standard MS-SSIM scale weights (Wang et al. 2003).
_MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim(x, y, window, window_size, channels, data_range=1.0):
    """Return (mean SSIM, mean contrast-structure) for a single scale."""
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    cs = (2 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)
    ssim_map = ((2 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)) * cs
    return ssim_map.mean(), cs.mean()


def ms_ssim(x, y, window_size=11, sigma=1.5, data_range=1.0, weights=_MS_SSIM_WEIGHTS):
    """Multi-scale SSIM in [0, 1]. Adapts the number of scales to image size so it
    is safe on small inputs (e.g. CI fixtures)."""
    channels = x.shape[1]
    min_side = min(x.shape[-2], x.shape[-1])
    # Each extra scale halves the image; need >= window_size at the coarsest scale.
    max_levels = 1
    while max_levels < len(weights) and (min_side >> max_levels) >= window_size:
        max_levels += 1
    w = torch.tensor(weights[:max_levels], device=x.device, dtype=x.dtype)
    w = w / w.sum()

    window = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    mcs = []
    ssim_val = None
    for i in range(max_levels):
        ssim_val, cs = _ssim(x, y, window, window_size, channels, data_range)
        if i < max_levels - 1:
            mcs.append(torch.relu(cs))
            x = F.avg_pool2d(x, kernel_size=2)
            y = F.avg_pool2d(y, kernel_size=2)
    mcs.append(torch.relu(ssim_val))
    mcs = torch.stack(mcs)
    return torch.prod(mcs ** w)


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss (Jiang et al., ICCV 2021).

    Penalises the discrepancy between the 2D DFTs of prediction and target,
    adaptively up-weighting frequency components that are hard to match.
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1,
                 ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def _tensor2freq(self, x):
        pf = self.patch_factor
        b, c, h, w = x.shape
        if h % pf != 0 or w % pf != 0:
            pf = 1  # fall back to whole-image FFT for non-divisible sizes
        ph, pw = h // pf, w // pf
        patches = []
        for i in range(pf):
            for j in range(pf):
                patches.append(x[:, :, i * ph:(i + 1) * ph, j * pw:(j + 1) * pw])
        y = torch.stack(patches, dim=1)                  # [B, P, C, ph, pw]
        freq = torch.fft.fft2(y, norm='ortho')
        return torch.stack([freq.real, freq.imag], dim=-1)  # [B, P, C, ph, pw, 2]

    def forward(self, pred, target):
        pred_freq = self._tensor2freq(pred)
        target_freq = self._tensor2freq(target.detach())

        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, dim=0, keepdim=True)
            target_freq = torch.mean(target_freq, dim=0, keepdim=True)

        # Spectral weight matrix from the prediction/target distance.
        matrix = (pred_freq - target_freq) ** 2
        matrix = torch.sqrt(matrix[..., 0] + matrix[..., 1]) ** self.alpha
        if self.log_matrix:
            matrix = torch.log(matrix + 1.0)
        if self.batch_matrix:
            matrix = matrix / (matrix.max() + 1e-12)
        else:
            denom = matrix.amax(dim=(-2, -1), keepdim=True) + 1e-12
            matrix = matrix / denom
        matrix = matrix.clamp(0.0, 1.0).detach()

        freq_dist = (pred_freq - target_freq) ** 2
        freq_dist = freq_dist[..., 0] + freq_dist[..., 1]
        return self.loss_weight * (matrix * freq_dist).mean()


class CompositeLoss(nn.Module):
    """Texture-preserving composite loss: L1 + MS-SSIM + Focal Frequency (+ LPIPS).

    Args:
        w_l1: weight on L1 (colour/luminance anchor).
        w_mssim: weight on (1 - MS-SSIM) (structure / high-frequency contrast).
        w_ffl: weight on Focal Frequency Loss (high-frequency recovery).
        w_lpips: weight on LPIPS perceptual loss (0 = disabled; requires the
            `lpips` package and carries hallucination risk if large).
        lpips_net: backbone for LPIPS ('alex' or 'vgg').

    Note: these weights operate on terms with different natural scales and should
    be tuned against the downstream metric (CoralNet/Toolbox accuracy), not PSNR.
    """

    def __init__(self, w_l1=1.0, w_mssim=1.0, w_ffl=1.0, w_lpips=0.0,
                 lpips_net='alex', ffl_kwargs=None):
        super().__init__()
        self.w_l1 = w_l1
        self.w_mssim = w_mssim
        self.w_ffl = w_ffl
        self.w_lpips = w_lpips
        self.l1 = nn.L1Loss()
        ffl_kwargs = ffl_kwargs or dict(alpha=1.0, patch_factor=1,
                                        ave_spectrum=True, log_matrix=True,
                                        batch_matrix=True)
        self.ffl = FocalFrequencyLoss(**ffl_kwargs)

        self.lpips = None
        if w_lpips > 0:
            try:
                import lpips  # noqa: F401
                self.lpips = lpips.LPIPS(net=lpips_net)
                for p in self.lpips.parameters():
                    p.requires_grad_(False)
            except ImportError as e:
                raise ImportError(
                    "w_lpips > 0 requires the 'lpips' package. "
                    "Install with: pip install lpips"
                ) from e

    def forward(self, pred, target):
        loss = self.w_l1 * self.l1(pred, target)
        if self.w_mssim > 0:
            loss = loss + self.w_mssim * (1.0 - ms_ssim(pred, target))
        if self.w_ffl > 0:
            loss = loss + self.w_ffl * self.ffl(pred, target)
        if self.lpips is not None:
            # LPIPS expects inputs in [-1, 1].
            loss = loss + self.w_lpips * self.lpips(
                pred * 2 - 1, target * 2 - 1).mean()
        return loss
