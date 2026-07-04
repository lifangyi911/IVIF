# 2D Gaussian Splatting for IVIF: Research Strategy

Date: 2026-07-01

This document turns the first-round IVIF literature review into research decisions for a stable Q1 journal paper. It is intentionally written as a working strategy, not a final paper outline.

## 1. What the Literature Says We Should Not Do

The recent IVIF literature is already crowded in several directions:

1. A generic "better fusion backbone" is weak. Recent papers already cover CNN, Transformer, BiFormer, linear attention, GRU/memory units, Mamba-like long-dependence ideas, SAM/CLIP/DINO priors, diffusion, and LUT deployment.
2. A generic "semantic-aware fusion" story is crowded. Local 2025-2026 papers include SpTFuse, SETFusion, SPGFusion, BSPFusion, SGDFuse, SDSFusion, HCLFuse, and DCEvo, all tying fusion to semantic/task information.
3. A generic "degraded scene fusion" story is also occupied. SDSFusion handles degraded scenes with semantic awareness; Deno-IF handles noisy fusion; REFusion targets smoke interference; Text-IF and related prompt methods target degradation-aware fusion.
4. A generic "fast fusion" story has a strong top-tier competitor: LUT-Fuse, which directly targets extremely fast IVIF via learnable lookup tables.
5. A pure "continuous representation for IVIF" story already has INRFuse, which uses INR/SIREN coordinate MLPs and claims cross-resolution fusion and super-resolution reconstruction.

Therefore, our paper should not be framed as:

- "We introduce 2D GS to IVIF" only.
- "We design a new attention/CNN/Mamba backbone."
- "We use GS because GS is newer than INR."
- "We improve standard IVIF metrics on TNO/RoadScene/MSRS/M3FD" only.

The story must be driven by an IVIF-specific unresolved problem where GS properties are necessary.

## 2. Most Promising Problem Definition

Recommended problem:

**Cross-resolution and arbitrary-scale infrared-visible image fusion with efficient region-level continuous rendering.**

Why this is the best fit:

- IVIF sensors can naturally differ in resolution, field of view, noise, and imaging quality.
- TPAMI 2023 explicitly discusses VIF for images of different resolutions as a future direction.
- INRFuse confirms that cross-resolution continuous IVIF is a plausible research problem, but it uses dense coordinate-wise MLP queries.
- GSPan/GSASR technical assets directly support arbitrary-scale rendering and scale-decoupled inference.
- This framing makes the paper look IVIF-driven rather than "GSPan copied to another task."

Proposed one-sentence gap:

> Existing IVIF methods mostly decode fused images on a fixed pixel grid; even recent continuous INR-based IVIF relies on dense point-wise coordinate queries. This leaves cross-resolution fusion and high-resolution arbitrary-scale rendering inefficient and weakly structured. IVIF needs a continuous region-level representation that can explicitly model thermal targets and visible textures while rendering efficiently at arbitrary output grids.

## 3. Why 2D Gaussian Splatting Is Necessary

2D GS should be justified by IVIF-specific signal structure:

| IVIF requirement | Why fixed-grid / INR is insufficient | Why 2D GS helps |
|---|---|---|
| Thermal targets have spatial extent, not isolated pixels | Pixel decoding and point-query INR do not explicitly model target regions | Gaussian center, scale, and alpha can represent salient thermal regions as spatial primitives |
| Visible images contain oriented textures and edges | Generic feature decoders may blur or over-inject texture | Anisotropic covariance and correlation can align with edges, roads, contours, and building structures |
| Cross-resolution fusion needs continuous output | Fixed-grid networks require resize/decoder changes; INR queries every output coordinate | GS predicts primitives once, then rasterizes to arbitrary grids |
| High-resolution inference needs efficiency | Diffusion/Transformer/INR can be expensive at large output sizes | Rasterization is parallel and can decouple primitive estimation from rendering resolution |
| Fusion decisions should be interpretable | Black-box feature fusion makes modal selection hard to inspect | Primitive alpha/modality weights can be visualized as IR/VIS contribution maps |
| Degraded visible or noisy IR should not dominate | Hand-crafted losses may preserve bad information | Primitive quality gates can suppress low-confidence/noisy details locally |

This is the core distinction from GSPan:

- GSPan solves PAN-MS spatial-spectral enhancement.
- Our IVIF method should solve IR-VIS **thermal-texture selection under cross-resolution rendering**.
- The primitive attributes must encode modality-specific IVIF roles, not just generic residual coefficients.

## 4. Recommended Method Concept

Working name:

**GIVFuse: Gaussian Primitive Representation for Cross-Resolution Infrared-Visible Image Fusion**

Core pipeline:

1. **Dual-stream modality encoder**
   - IR stream extracts thermal saliency and low-frequency target support.
   - VIS stream extracts texture, gradient, and scene structure.
   - Use a compact CNN + window cross-attention encoder. Do not make the backbone the main novelty.

2. **Primitive-level modality interaction**
   - Learn content primitives and detail primitives.
   - Each primitive predicts:
     - center `mu=(x,y)`
     - scale `sigma_x, sigma_y`
     - orientation/correlation `rho`
     - opacity/confidence `alpha`
     - modality weights `w_ir, w_vis` or separate coefficients for thermal/detail branches
     - fused intensity/residual coefficient
   - The key IVIF module should be a **thermal-texture primitive selector**, not a generic attention block.

3. **Continuous Gaussian rendering**
   - Render a fused residual/detail field or fused luminance field at requested output resolution.
   - Prefer residual rendering:
     - base = visible luminance or adaptive base from source images
     - rendered GS field = thermal/detail enhancement residual
     - output = base + rendered residual
   - Residual rendering is more stable and closer to the GSPan asset, but the explanation must be IVIF-specific.

4. **Cross-resolution training/inference**
   - Train with aligned or synthetically cross-resolution pairs.
   - At inference, estimate primitives at source or reduced resolution and render to arbitrary target grid.
   - Add a scale-decoupled inference mode similar in spirit to GSPan SDAI, but rename and justify it for IVIF high-resolution deployment.

5. **Loss design**
   - Avoid relying only on handcrafted intensity + gradient losses.
   - Minimum loss set:
     - thermal saliency preservation: fused output should preserve IR target intensity/contrast in salient regions
     - visible texture preservation: fused output should preserve VIS gradients where visible quality is reliable
     - structural consistency: SSIM/edge consistency with both modalities
     - cross-resolution consistency: rendered outputs at different scales should be mutually consistent after downsampling
     - primitive regularization: avoid degenerate giant/small Gaussians, encourage sparse/meaningful alpha
   - Stronger option:
     - cross-reconstruction loss inspired by FreeFusion: fused/latent features should reconstruct or predict complementary modality cues, reducing pure hand-crafted loss dependence.

## 5. Backbone Decision

Recommendation:

**Do not switch the core to Mamba for v1. Keep a dual-stream CNN + local/window attention design and focus novelty on primitive generation/rendering.**

Rationale:

- Transformer/semantic attention is saturated but still acceptable as infrastructure.
- Mamba/GRU/memory is already represented by MemoryFusion and related long-dependence works; switching to Mamba creates another crowded backbone story and increases implementation risk.
- CNN-only may be efficient but may weaken cross-modal/long-range argument unless paired with strong primitive interaction.
- GSPan's existing dual-stream interaction can be reused structurally, but module names and feature roles must be rewritten:
  - PAN high-frequency stream -> VIS texture/detail stream
  - MS spectral stream -> IR saliency/thermal stream plus visible base
  - spectral coefficient vector -> modality/fusion coefficient vector
  - residual HRMS field -> fused thermal-texture residual field

Practical architecture choice:

- Encoder: lightweight CNN residual blocks for local features.
- Interaction: 1-2 window cross-attention blocks for IR-VIS primitive conditioning.
- Primitive self-attention: keep if useful, because neighboring primitives should coordinate.
- Decoder: MLP heads for Gaussian attributes.
- Renderer: reuse/adapt current CUDA 2D GS rendering, but output channel and alpha semantics need redesign.

## 6. Candidate Research Stories

| Story | Strength | Weakness | Recommendation |
|---|---|---|---|
| A. Cross-resolution/arbitrary-scale IVIF | Best fit to GS; clear contrast with fixed-grid and INRFuse; strong link to GSPan assets without looking copied | Need construct convincing cross-resolution protocol | Primary story |
| B. Degradation-aware primitive fusion | Matches IVIF reality; GS scale/alpha can suppress noise/smoke/low-light artifacts | Strong competitors: SDSFusion, Deno-IF, REFusion, Text-IF | Use as secondary experiment, not main claim |
| C. Misalignment-compatible IVIF | GS coordinates/covariance plausibly absorb local shifts; important future trend | Harder to prove; registration-fusion literature is complex | Optional robustness study |
| D. Semantic-guided GS fusion | SAM/CLIP can guide primitives | Very crowded; risk of being another SAM-fusion paper | Avoid as main story; optional analysis only |
| E. Extreme real-time IVIF | GS rasterization is efficient | LUT-Fuse is very strong; GS may not beat LUT on mobile | Frame as high-resolution scalable, not mobile fastest |

Recommended final combination:

**A as the main problem + B/C as practical stress tests + primitive visualization as interpretability evidence.**

## 7. Baseline Selection Strategy

Do not compare against every method in the field. Use a tiered baseline set.

### Must-have baselines

| Type | Methods | Purpose |
|---|---|---|
| Classic deep | DenseFuse, RFN-Nest, U2Fusion | Show continuity with standard IVIF baselines |
| Efficient/CNN | IFCNN or SDNet, PIAFusion | Cover fast/general and illumination-aware methods |
| Transformer/decomposition | SwinFusion, CDDFuse | Cover modern Transformer/decomposition baseline |
| Task-aware | SeAFusion or TarDAL or MetaFusion | Cover downstream-aware methods |
| Continuous | INRFuse | Closest neighbor for arbitrary/cross-resolution claim |

### Conditional baselines

| If claiming... | Add... |
|---|---|
| degradation/noise | Deno-IF, SDSFusion, REFusion/Text-IF if code available |
| diffusion quality | DDFM, Diff-IF, SGDFuse/HCLFuse if reproducible |
| efficiency | LUT-Fuse, SDNet, IFCNN |
| misalignment | UMFusion, MURF, SuperFusion, ReCoNet |
| semantic segmentation | SeAFusion, SegMiF, PSFusion |

### What to report

- Standard fusion metrics: EN, MI, SF, AG, SD, CC, SCD, VIF, QAB/F, SSIM.
- Efficiency: runtime, FPS, params, FLOPs, memory; include output resolution scaling curves.
- Cross-resolution: train/test scale generalization, non-integer scale tests, consistency across scales.
- Downstream: at least one of object detection on M3FD or semantic segmentation on MFNet/FMB.

## 8. Experiments That Make the Paper Distinctive

### Experiment 1: Standard IVIF benchmark

Purpose: show the method is not worse than mainstream IVIF.

Datasets: TNO, RoadScene, MSRS, M3FD.

Baselines: DenseFuse, RFN-Nest, U2Fusion, PIAFusion, SwinFusion, CDDFuse, SeAFusion/TarDAL, INRFuse.

### Experiment 2: Cross-resolution IVIF

Purpose: directly support the main problem.

Protocol:

- Create synthetic cross-resolution pairs by downsampling IR or VIS by scale factors such as 0.5, 0.75, 1.25 mismatch, then compare output at target resolution.
- Test non-integer target scales, e.g. 1.2, 1.5, 2.1, 3.0.
- Compare:
  - resize-then-fuse baselines
  - fixed-grid fusion models
  - INRFuse coordinate-query method
  - our GS rendering method

Key metrics:

- standard fusion metrics at target resolution
- cross-scale consistency after downsampling
- runtime vs target pixels

### Experiment 3: Scale-decoupled high-resolution inference

Purpose: show GS is not only a representation trick but enables a useful deployment mode.

Protocol:

- Estimate primitives at reduced resolution.
- Render fused result at full or higher resolution.
- Vary estimation scale and rendering scale independently.
- Report quality-speed-memory tradeoff.

Expected result:

- Full estimation gives best quality.
- Reduced estimation gives much faster inference with acceptable quality.
- This is the IVIF version of scale-decoupled rendering, but should be described as high-resolution multimodal fusion deployment.

### Experiment 4: Degradation stress test

Purpose: make the method feel grounded in IVIF reality.

Conditions:

- VIS low-light or fog/smoke simulation.
- IR Gaussian/Poisson noise.
- Mixed quality: one modality reliable in one region, unreliable in another.

Evidence to show:

- quality-aware primitive alpha suppresses degraded information.
- thermal targets remain salient.
- visible textures are preserved only where reliable.

### Experiment 5: Slight misalignment stress test

Purpose: optional but valuable if results are good.

Protocol:

- Apply small translations, local elastic deformation, or affine jitter to one modality.
- Compare ghosting artifacts and task metrics.

Caution:

- Do not claim full registration unless explicitly predicting deformation fields.
- Frame as "misalignment tolerance" or "small offset robustness."

### Experiment 6: Primitive visualization

Purpose: prove interpretability and IVIF specificity.

Show:

- Gaussian ellipses over IR target regions and VIS texture regions.
- `w_ir/w_vis` or alpha maps.
- Scale/orientation maps.
- Ablation: remove modality weights, fix sigma/rho, remove residual rendering, remove cross-resolution consistency.

## 9. Innovation Claims to Use Carefully

Safe claims:

- "A continuous region-level Gaussian primitive representation for IVIF."
- "A primitive-level thermal-texture selection mechanism."
- "Efficient arbitrary-grid rendering for cross-resolution IVIF."
- "Scale-decoupled primitive estimation and rendering for high-resolution IVIF."
- "Explicit primitive visualization of modality contribution."

Claims needing formal novelty audit:

- "First 2D Gaussian Splatting method for IVIF."
- "First arbitrary-scale IVIF method."
- "Solves misalignment."
- "Real-time fastest IVIF."

Avoid:

- "GS has never been used in image fusion" unless thoroughly verified.
- "The method is semantic-aware" as the main claim.
- "The backbone is novel" unless there is a truly new architectural principle.

## 10. Preliminary Paper Skeleton

1. Introduction
   - IVIF goal: thermal targets + visible textures.
   - Current issue: fixed-grid fusion and task/semantic/degradation trends.
   - Gap: cross-resolution/high-resolution arbitrary rendering remains underexplored; INR starts this direction but uses inefficient point queries and lacks explicit region structure.
   - Solution: Gaussian primitive fusion.

2. Related Work
   - Deep IVIF: AE/CNN/GAN/Transformer/Diffusion/task-aware.
   - Continuous and cross-resolution image fusion: INRFuse, INR/LIIF-like works.
   - Gaussian Splatting in image representation/SR and why IVIF differs.

3. Method
   - Problem formulation for cross-resolution IVIF.
   - Dual-stream thermal-texture encoder.
   - Gaussian primitive generator with modality-aware attributes.
   - Continuous rendering and scale-decoupled inference.
   - Losses and training.

4. Experiments
   - Standard IVIF.
   - Cross-resolution/arbitrary-scale.
   - High-resolution efficiency.
   - Degradation/misalignment robustness.
   - Ablation and visualization.

5. Conclusion

## 11. Immediate Next Technical Step

Before implementing, do a small feasibility prototype:

1. Pick one dataset first, ideally RoadScene or MSRS.
2. Convert GSPan input/output:
   - `lr_u` -> visible luminance / visible feature input
   - `pan` -> infrared image or thermal saliency branch
   - output channels -> 1 fused luminance channel first
3. Use existing GS batch renderer for single-channel output.
4. Train a minimal residual-rendering version with basic unsupervised loss.
5. Verify:
   - does GS rendering produce stable fused images?
   - do Gaussian positions/alpha correlate with IR targets or VIS edges?
   - is cross-resolution rendering plausible before adding complex modules?

Only after this prototype works should we decide whether to add SAM/task heads/degradation modules.

