# Plan: Image-Adaptive 3D LUT for Underwater Enhancement

**Status:** Proposal for review
**Goal:** Beat the current U-Net on *fine-scale texture fidelity* while matching or
exceeding its color fidelity, for the supervised task of replicating expert Lightroom
edits (mostly-global edits) of GoPro RAW benthic imagery.

---

## 1. Why a 3D LUT (the core idea)

Your editors apply **mostly-global** color/tone adjustments (white balance, exposure,
curves, dehaze, HSL). That is, mathematically, a **color-to-color mapping** applied
identically to every pixel regardless of position. A 3D LUT *is* exactly that function.

The decisive property for your texture problem:

> A small CNN looks only at a **downsampled** copy of the image to predict the LUT.
> The learned color transform is then applied **per-pixel to the full-resolution
> image** via trilinear interpolation. The full-res image never passes through a
> downsampling/upsampling bottleneck.

**Consequence:** spatial high-frequency structure (texture, edges, sand grain, coral
detail) is *geometrically untouched* — the model can only remap colors, it cannot blur.
This is why it is the top-ranked architecture to close your texture gap: for a global
edit it **cannot lose texture by construction.**

Reference: Zeng et al., *"Learning Image-Adaptive 3D Lookup Tables for High Performance
Photo Enhancement in Real-Time"* (ECCV 2020 / TPAMI 2020).
Paper: https://arxiv.org/abs/2009.14468 · Code: https://github.com/HuiZeng/Image-Adaptive-3DLUT

It was benchmarked on **MIT-Adobe FiveK** (expert retouchers A–E) — the direct analog
of your expert Lightroom targets — and outperformed prior photo-enhancement SOTA on
PSNR/SSIM/color-difference by a large margin, at <600K params and <2ms for 4K images.

### How it works (architecture)

```
                 ┌─────────────────────────────────────────┐
 full-res input ─┤                                          │
   [B,3,H,W]     │   (used only as the image being mapped)  │
                 └───────────────┬──────────────────────────┘
                                 │
   downsample to 256×256         │ apply learned LUT per-pixel
        │                        │ (trilinear interp, full res)
        ▼                        ▼
   small CNN  ──►  weights w1..wN  ──►  T = Σ wi · LUT_i  ──►  output [B,3,H,W]
 (classifier)     (content-adaptive)     (fused 3D LUT)
                                              ▲
                          N learnable basis 3D LUTs (e.g. 33×33×33×3)
```

- **N basis LUTs** (typically 3): one is initialized to identity, others learned.
- **Weight predictor**: a tiny CNN on the 256×256 thumbnail → N scalar weights.
- **Fused LUT** `T = Σ wᵢ·LUTᵢ` is applied to the full-res input by trilinear lookup.
- **Regularization** (from the paper): smoothness + monotonicity on the LUTs to avoid
  banding and keep transforms well-behaved. We will port these.

### Honest limitation

A pure 3D LUT applies a **global** color transform — it cannot do spatially-varying
local edits (masks, dodge/burn, graduated filters). You confirmed edits are *mostly
global*, so this is the right tool. If a meaningful minority of edits turn out to be
local, the fallback is a **bilateral-grid + LUT** or **HDRNet** variant (see §7).

---

## 2. How this fits the existing codebase

The repo already supports multiple architectures selected by a config/CLI string, with
checkpoint-based architecture auto-detection at inference. We slot in cleanly.

| Concern | Existing mechanism | What we add |
|---|---|---|
| Model selection | `--model` arg in `training/train.py` (`unet`/`ushape_transformer`/`ss_uie`), dispatched by `if/elif` (~L655–696) | Add `'3d_lut'` choice + an `elif` branch |
| Model code | `src/models/*.py`, each a `nn.Module` taking `[B,3,H,W]` in `[0,1]` and returning same | New `src/models/lut_3d.py` |
| Loss | `CombinedLoss` (L1+MSE 80/20) and `SSUIECombinedLoss`, selected ~L703–708 | New `CompositeLoss` + a `--loss` selector |
| Dataset | `UnderwaterDataset` returns paired `[3,H,W]` in `[0,1]`, supports `random_crop`/`crops_per_image`/`image_size` | **No change** |
| Checkpoint | `model_state_dict` + `model_config`/`training_config` dicts incl. `model` string | Store `model: '3d_lut'` + LUT params (dim, n_luts) in `model_config` |
| Inference | `_detect_model_type()` sniffs state_dict keys; `setup_model()` rebuilds | Add a key-sniff branch returning `'LUT3D'` + instantiation |

**I/O convention to honor:** input/target are `[B,3,H,W]` float in `[0,1]`, no
normalization. Output must be `[0,1]`. (Match `UNetAutoencoder`'s sigmoid-style range;
the LUT output is naturally bounded if LUT entries are clamped to `[0,1]`.)

---

## 3. Implementation steps

### Step 1 — Add the model: `src/models/lut_3d.py`
- Class `LUT3D(nn.Module)`:
  - `__init__(self, n_channels=3, n_luts=3, lut_dim=33, backbone='small_cnn')`
  - Learnable parameter: `self.luts` of shape `[n_luts, 3, lut_dim, lut_dim, lut_dim]`,
    with `luts[0]` initialized to the identity mapping.
  - Weight predictor: small CNN (5 conv blocks + global pool + FC → `n_luts`), runs on
    the input downsampled to 256×256 (use `F.interpolate`).
  - `forward(x)`: predict weights → fuse LUT → apply via trilinear interpolation to
    full-res `x` → clamp to `[0,1]` → return `[B,3,H,W]`.
- **Trilinear apply**: port the official CUDA op from the Zeng repo *or* implement a
  pure-PyTorch `F.grid_sample`-based 3D lookup (slower, fully portable, CPU/MPS-safe —
  preferred for this repo's cross-platform CI). Decide in §6.
- Port the **TV/smoothness + monotonicity regularizers** as a method returning a scalar
  (added to the loss in Step 2).

### Step 2 — Add the composite loss (also reusable by U-Net; see §4)
In `training/train.py`, add `CompositeLoss(nn.Module)`:
```
loss = w_l1   * L1(pred, tgt)
     + w_mssim * (1 - MS_SSIM(pred, tgt))      # structure / high-freq contrast
     + w_ffl   * FocalFrequencyLoss(pred, tgt) # high-frequency recovery
     + w_lpips * LPIPS(pred, tgt)              # optional perceptual (small weight)
     + lut_reg                                  # only for 3D LUT (smoothness+monotonic)
```
- Reuse the **MS-SSIM** and **FFL** implementations the SS-UIE path already pulls in
  (`SSUIECombinedLoss` already uses SSIM + a frequency-domain loss — confirm and reuse
  those modules rather than adding deps).
- LPIPS via `lpips` package (optional; guard import like `bitsandbytes`/`mamba`).
- Add CLI/config `--loss {combined,ss_uie,composite}` and weight args with sane
  defaults; select around L703–708.

### Step 3 — Register the model (training)
- Add `'3d_lut'` to the `--model` `choices` (~L520).
- Add an `elif args.model == '3d_lut':` branch (~after L696) instantiating `LUT3D(...)`.
- For 3D LUT, default `--loss composite` (LUT works best with structure+freq losses).
- Optimizer: the paper uses a **higher LR for the LUT entries** than the CNN. Start
  simple (single Adam, lr=1e-4 as now); add param groups only if convergence is slow.

### Step 4 — Checkpoint metadata
- Ensure `checkpoint['model_config']['model'] = '3d_lut'` and add `lut_dim`, `n_luts`.
  The training loop already writes `model_config`/`training_config` — extend the dicts
  in the LUT branch (mirror how `ss_uie` adds `ss_uie_H/W`).

### Step 5 — Inference auto-detection
- In `inference/inference.py` `_detect_model_type()`, add a branch: if state_dict keys
  contain `luts` (or `weight_predictor.`), return `'LUT3D'`.
- In `setup_model()`, add a `LUT3D` instantiation reading `lut_dim`/`n_luts` from config.
- Tiled/full-size inference: the LUT is resolution-independent and texture-safe, so
  `--full-size` is ideal here (no tiling artifacts; the transform is per-pixel global).

### Step 6 — Tests
- Add a tiny synthetic test in `tests/test_inference.py` style: build `LUT3D`, run a
  3×64×64 random tensor, assert output shape `[1,3,64,64]` and range `[0,1]`.
- Identity check: with only the identity LUT active, output ≈ input (validates the
  trilinear apply). This is a strong correctness guard.

---

## 4. Recommended experiment sequence

Run these **in parallel** — they're independent and both cheap:

1. **Tier 1 (loss-only baseline):** retrain the *existing U-Net* with `--loss composite`
   (MS-SSIM + L1 + Focal Frequency Loss, optional LPIPS). Tells you how much of the gap
   is "just the objective." Lowest effort, no new architecture.
2. **Tier 2 (the bet):** train `LUT3D` with `--loss composite`. Expected to preserve
   texture perfectly (global color map) *and* improve color fidelity.

**Evaluate on your real metric** — CoralNet/Toolbox downstream classification accuracy
— not PSNR/SSIM. None of the source papers used your metric; texture gains here are
inferred from mechanism, so they must be confirmed empirically on Seattle Aquarium data.
Watch specifically for the failure modes in §5.

---

## 5. Risks & how we mitigate

| Risk | Mitigation |
|---|---|
| Pure LUT too global; misses any local edits | Confirmed edits are mostly-global; fallback to bilateral-grid LUT / HDRNet (§7) |
| LUT banding / posterization | Smoothness + monotonicity regularizers (from paper); use `lut_dim=33` |
| Trilinear CUDA op not portable to CPU/MPS CI | Implement pure-PyTorch `grid_sample` fallback (§6) |
| LPIPS/adversarial **hallucinating** texture (bad for a scientific classifier) | Keep LPIPS weight small or omit; **no GAN by default**; validate on CoralNet, not just perception |
| Color gamut clipping at LUT edges | Clamp LUT entries and output to `[0,1]`; identity-init the first LUT |

---

## 6. Open implementation decision: trilinear lookup

- **Option A — official CUDA extension** (from Zeng repo): fastest, but needs
  compilation and is CUDA-only, conflicting with this repo's CPU/MPS/Windows CI.
- **Option B — pure PyTorch via `F.grid_sample`** (3D): portable, no build step, runs in
  CI and on Mac/CPU for inference. Slower, but you stated speed is *not* a concern.
- **Recommendation: Option B.** Aligns with the repo's cross-platform inference tests
  and avoids a build dependency. Revisit A only if training throughput becomes painful.

---

## 7. Fallbacks if the global LUT underfits

In priority order (all preserve texture by avoiding the U-Net bottleneck):

1. **Bilateral-grid + 3D LUT** (Kim & Cho, ECCV 2024) — adds *local* spatial awareness
   via bilateral-grid slicing. https://github.com/WontaeaeKim/LUTwithBGrid
2. **HDRNet** (Gharbi et al., SIGGRAPH 2017) — affine transforms in bilateral space,
   edge-preserving slicing to full res. https://arxiv.org/abs/1707.02880
3. **CSRNet** (ECCV 2020) — ~37K-param per-pixel 1×1-conv MLP; even simpler than a LUT,
   global-only. https://arxiv.org/abs/2104.06279

---

## 8. Does this replace the papers you already linked?

**Short answer: yes, the 3D-LUT/color-transform family is a better fit for your specific
goal (texture fidelity on mostly-global edits) than every paper on your original list.**
Reasoning per paper:

| Your linked paper | Verdict for *texture-on-global-edits* | Why |
|---|---|---|
| **U-shape Transformer** (2021) | Already tested — color-good, texture-worse | Encoder/decoder bottleneck; not texture-preserving by design |
| **SS-UIE / Adaptive Dual-domain** (AAAI 2025) | Marginal for *your* goal | Strong model, but CUDA-only (mamba) and still a bottleneck arch; its FFT loss is the transferable idea — we already adopt frequency loss in §3 |
| **MMLE** (2022) | No | MATLAB, traditional (non-learned); won't match expert edits |
| **Ucolor** (2021) | No | Requires extra computed inputs (transmission/depth maps) — you ruled this out |
| **Semi-UIR** (CVPR 2023) | No | Semi-supervised for unpaired data; you *have* paired data — supervised is simpler and stronger |
| **IACC** (2024) | No (for texture) | Targets mixed natural/artificial *illumination correction*, not texture fidelity |
| **DM_underwater** (diffusion, 2023) | No | Reproducibility nightmare (per its issues); diffusion risks **hallucinating** texture — dangerous for a scientific classifier |

**Caveat on "better":** the 3D-LUT advantage is *mechanistic* (it cannot blur a global
edit) and validated on the analogous FiveK retouching task — but no source measured
texture parity with human edits on data like yours. So it is the **highest-probability**
candidate, to be **confirmed empirically** via Tier 1 vs Tier 2 on CoralNet accuracy.

The single most transferable idea *from* those papers — the **frequency-domain loss** in
SS-UIE/U-shape — we are keeping (Focal Frequency Loss in the composite loss), so you get
that benefit regardless of architecture.

---

## 9. Effort estimate

| Task | Rough effort |
|---|---|
| `lut_3d.py` (model + pure-PyTorch trilinear + regularizers) | ~1 day |
| `CompositeLoss` + `--loss` wiring (reusing existing SSIM/FFL) | ~0.5 day |
| train.py registration + checkpoint metadata | ~0.5 day |
| inference detection + tests | ~0.5 day |
| Train both runs + CoralNet eval | GPU/eval time |

---

## 10. Decision checklist before coding

- [ ] Confirm edits are mostly-global (done — confirmed).
- [ ] Confirm we evaluate on CoralNet/Toolbox accuracy, not just PSNR/SSIM.
- [ ] Confirm pure-PyTorch trilinear (Option B) is acceptable vs CUDA op.
- [ ] Confirm whether to include LPIPS in the composite loss (hallucination tradeoff).
- [ ] Confirm we run Tier 1 (U-Net + composite loss) as the control alongside Tier 2.
