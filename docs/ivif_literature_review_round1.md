# IVIF Literature Review Round 1

Date: 2026-07-01

This note summarizes the first pass over the local folders `综述/` and `2025-2026 papers/`, plus targeted online checks. The goal is to understand IVIF before designing a 2D Gaussian Splatting based paper that does not look like a simple second migration from GSPan.

## 1. Field Snapshot

Infrared-visible image fusion (IVIF) aims to preserve thermal saliency from infrared images and texture/scene detail from visible images in one fused image. The field has shifted from hand-designed fusion rules to deep learning, and more recently toward task-adaptive, semantic-aware, degradation-aware, efficient, and registration-compatible fusion.

Three survey papers are the most useful entry points:

| Role | Paper | Main value for us |
|---|---|---|
| Classical survey | Ma et al., "Infrared and visible image fusion methods and applications: A survey", Information Fusion 2019 | Establishes classical categories: multi-scale transform, sparse representation, neural networks, subspace, saliency, hybrid methods; also discusses registration and metrics. |
| Deep-learning survey | Zhang & Demiris, "Visible and Infrared Image Fusion Using Deep Learning", TPAMI 2023 | Gives deep-learning taxonomy: AE, CNN, GAN, Transformer, supervised/unsupervised, application-oriented VIF, datasets and metrics. Strong warning: no true GT and no well-established benchmark. |
| Latest broad survey | Liu et al., "Infrared and Visible Image Fusion: From Data Compatibility to Task Adaption", TPAMI 2025 | Reframes the field around data compatibility, perception/task adaptation, registration, attack robustness, and efficiency. This is the most important survey for finding current gaps. |

Key field facts:

- There is no real ground-truth fused image, so most IVIF methods depend on unsupervised losses, pseudo labels, task labels, or reconstruction constraints.
- Visual metrics alone are unstable; different metrics can favor different methods, and older/traditional methods can still be competitive on VIFB.
- Recent high-level IVIF papers increasingly justify fusion by downstream detection/segmentation, not only by EN/MI/SSIM/QAB/F.
- Real deployment issues are now explicit research topics: misalignment, low-light/smoke/noise/fog, attacks, high-resolution efficiency, and low-power inference.
- Continuous representation exists in IVIF via INRFuse, but current INR-style work is point-query MLP based and weak in efficiency/region-level structure modeling.

## 2. Classic Baselines Researchers Often Cite

These methods should be considered when building the baseline set. The final experimental set should be smaller and reproducible, but the related work needs to cover these families.

| Family | Representative methods | Why they matter |
|---|---|---|
| Traditional strong baselines | GFF, CBF, LatLRR, LP-SR, NSCT-SR | Surveys still show traditional methods can be competitive on some metrics; useful for reviewer trust. |
| AE / encoder-decoder | DenseFuse, NestFuse, RFN-Nest | DenseFuse is a frequent early deep baseline; RFN-Nest is a common end-to-end residual fusion reference. |
| CNN / general fusion | IFCNN, U2Fusion, SDNet, PIAFusion | U2Fusion and IFCNN are common unified/general fusion baselines; SDNet is real-time; PIAFusion handles illumination-aware fusion. |
| GAN | FusionGAN, DDcGAN, GANMcC | Important deep-learning milestones for target preservation and modality-specific discrimination. |
| Transformer | SwinFusion, YDTR, CDDFuse | SwinFusion introduced long-range cross-domain fusion; CDDFuse is a very common recent CVPR 2023 baseline. |
| Diffusion / generative | DDFM, Diff-IF, HCLFuse, SGDFuse | High-quality generation is a hot direction, but computational cost is a weakness. |
| Task-oriented | TarDAL, MetaFusion, SeAFusion, SegMiF, PSFusion | These define current task-adaptive IVIF: detection/segmentation guidance and semantic-feature interaction. |
| Registration-compatible | UMFusion, MURF, SuperFusion, ReCoNet, RFVIF, SemLA, IMF | Important if we claim real-world robustness or misalignment handling. |
| Vision-language / prompts | Text-IF, FILM/Prompt-like methods, SPGFusion | Recent papers use text, CLIP/DINO/SAM semantic priors; useful competitors if we discuss semantic guidance. |
| Efficiency | SDNet, IFCNN, LUT-Fuse | Any practical/high-resolution claim must compare runtime/params/FLOPs, not only image metrics. |

Common datasets:

- Pure fusion/test: TNO, RoadScene, VIFB.
- Driving/detection/low-light: LLVIP, M3FD, MSRS.
- Semantic segmentation: MFNet, FMB.
- Misalignment/registration: often built from RoadScene/M3FD with synthetic deformation or uses specialized registration-fusion settings.

Common metrics:

- Pixel/information/structure: EN, MI, SF, AG, SD, CC, SCD, VIF, QAB/F, SSIM.
- Task metrics: mAP/precision/recall for detection; mIoU/Acc for segmentation.
- Efficiency: runtime, FPS, FLOPs, parameter count, memory.
- For our future paper, the metric story should be tied to the claimed problem; otherwise the method risks becoming another metric-chasing IVIF network.

## 3. 2025-2026 Papers in Local Folder, Grouped

| Category | Local paper | Core idea | Relevance to 2D GS idea |
|---|---|---|---|
| Feature disentanglement / commonality | D3Fuse, Pattern Recognition 2025 | Adds scene commonality as a third-dimensional feature, with contrastive learning and modality features. | Shows reviewers value modality-common vs modality-specific separation. GS primitives can be decomposed into thermal-target and texture-detail primitives. |
| Adaptive/task-aware fusion | ACFNet, Pattern Recognition 2025 | Adaptive cross-fusion and gating for downstream object detection. | If we use GS, adaptive modality weights per primitive are more defensible than global fusion weights. |
| SAM/semantic Transformer | SpTFuse, Pattern Recognition 2025 | SAM semantic prior branch with multi-level collaborative Transformer. | Strong semantic prior competitor; GS needs either avoid this lane or use semantic anchors efficiently. |
| Full-scale/multiscale fidelity | VFFusion, Pattern Recognition 2025 | BiFormer multi-scale encoder, full-scale interaction, visual-fidelity branch. | Multiscale feature preservation is central; GS scale/covariance can provide a more explicit scale representation. |
| Memory/sequence architecture | MemoryFusion, Pattern Recognition 2026 | GRU-like memory units retain shallow/deep features and reduce redundancy. | Long-term dependency is already being attacked; do not make "better feature memory" the main novelty. |
| Semantic Transformer | SETFusion, Pattern Recognition 2026 | Pyramid and multi-scale semantic Transformer, visual information fidelity loss. | Semantic modeling is crowded; GS should not rely only on "semantic transformer" novelty. |
| Multi-modal/multi-scale compensation | KBS 2026 cross-compensation paper | Multi-scale cross-compensation using Transformer. | Again confirms multi-scale compensation is saturated. |
| Linear attention efficiency | LACT-Fusion, KBS 2026 | Linear attention guided cross-modal learning. | Efficient attention is a competitor if we keep a heavy attention backbone. |
| Hierarchical semantic collaboration | EAAI 2026 semantic collaboration paper | Hierarchical semantic Transformer/collaboration. | Another sign that semantic collaboration alone is not enough. |
| Manifold representation | GrFormer, Information Fusion 2026 | Models modality data/subspaces on Grassmann manifold. | Useful contrast: geometric representation is accepted, but theirs is feature-manifold, not continuous spatial primitive. |
| Pretrained semantic priors | SPGFusion, Information Fusion 2026 | Uses CLIP/DINO and semantic priors. | Avoid competing directly with large pretrained semantic-prior papers unless using SAM/CLIP only as auxiliary. |
| Bottom-up semantics | BSPFusion, Information Fusion 2026 | Bottom-up semantically progressive fusion. | Reinforces semantic-progressive trend. |
| Cross reconstruction / no hand-crafted fusion loss | FreeFusion, TPAMI 2025 | Cross-reconstruction learning to align domains without hand-crafted fusion losses; dynamic interaction with segmentation features. | Very important: if our method still depends on hand-crafted intensity/gradient losses only, it will look outdated. |
| Continuous representation | INRFuse, IEEE CYBER 2025 | INR/SIREN-style coordinate MLP for IVIF; supports different resolutions and SR by dense coordinate queries. | Closest direct neighbor. Our GS story must emphasize region-level primitives, efficient rasterization, explicit geometry, and scale-decoupled inference. |
| Evolutionary task-loss balancing | DCEvo, CVPR 2025 | Evolutionary search balances fusion and downstream task losses; cross-dimensional embedding links task and fusion features. | Indicates loss balancing is a recognized pain point. Could inspire automatic loss weights, but not our main route. |
| Diffusion + SAM | SGDFuse, Information Fusion 2026 | SAM-guided diffusion; treats IVIF as semantic-guided generation, addresses semantic blindness/artifacts. | Strong generative competitor; weakness is speed and complexity, where GS may win. |
| Noisy/degraded fusion | Deno-IF, NeurIPS 2025 | Unsupervised noisy visible/IR fusion with convolutional low-rank priors and joint denoising/fusion. | Degradation robustness is a promising experiment axis for GS: primitive scale/alpha can suppress noisy detail. |
| Cognitive/generative | HCLFuse, NeurIPS 2025 | Human cognitive laws, variational bottleneck, diffusion with physical guidance. | Shows "interpretability / modal selection" matters. GS parameters can be visualized more directly. |
| Smoke/degraded scenes | REFusion, TMM 2026 | Dual-stage fusion for smoke interference, cross-source context association, differential information reinforcement. | Application-specific degraded scenarios are publishable; GS can target smoke/fog/noise/misalignment if technically justified. |
| Degraded scenes + semantics | SDSFusion, TIP 2025 | Unified enhancement, fusion, and semantic task interaction for low-light/extreme weather. | Strong degraded-scene baseline; our method needs a different mechanism, e.g. continuous primitive rendering and scale-decoupled efficiency. |
| Extreme efficiency | LUT-Fuse, ICCV 2025 | Distills fusion network to learnable lookup tables; extremely fast, mobile-friendly. | Efficiency is already a top-tier topic. GS can target high-resolution scalable rendering, not necessarily mobile LUT speed. |

## 4. Current Research Saturation

Crowded directions:

- "Design a better Transformer/CNN fusion backbone."
- "Use SAM/CLIP/DINO semantic priors" without a new task setting.
- "Use diffusion for higher visual quality" without efficiency mitigation.
- "Manually combine intensity/gradient/SSIM losses" as the central contribution.
- "Improve multi-scale feature interaction" as a generic module story.

Still open or less saturated:

- Cross-resolution or arbitrary-resolution IVIF, especially with efficient high-resolution rendering.
- Region-level continuous fusion representation beyond pixel-wise INR.
- Scale-decoupled inference for large fused images.
- Primitive-level interpretability: modality weights, thermal saliency, texture orientation, and uncertainty/noise suppression.
- Fusion under imperfect real-world data: resolution mismatch, small spatial misalignment, noise, low-light/smoke/fog.
- Efficiency-quality tradeoff against INR and diffusion under high output resolution.

## 5. Why 2D Gaussian Splatting May Be Necessary

The strongest starting story is not "GS is new in IVIF". It should be:

> Existing IVIF methods mostly formulate fusion as fixed-grid pixel reconstruction or feature decoding. This makes them weak for cross-resolution fusion, large-scene scalable rendering, and explicit region-level modal selection. INR-based IVIF begins to address continuous resolution but relies on dense coordinate-wise MLP queries, making high-resolution rendering expensive and offering limited explicit structure. 2D Gaussian primitives provide a continuous, region-level, anisotropic, and efficiently rasterized representation that naturally matches IVIF's need to combine thermal targets and visible textures across scales.

Specific GS advantages:

- Center `mu`: can adapt to thermal targets, visible edges, or salient multimodal structures.
- Scale `sigma_x/sigma_y`: can model target extent, texture granularity, and denoising/smoothing strength.
- Correlation/orientation `rho`: can align elongated structures such as roads, pedestrians, vehicles, edges, and building contours.
- Alpha/modality weight: can encode primitive-level confidence or IR/VIS contribution.
- Color/intensity coefficient: can render fused luminance or residual detail.
- Rasterization: avoids dense per-pixel MLP queries and can support high-resolution or scale-decoupled rendering.

Main reviewer risk:

- If the method only replaces INR MLP with GS rasterization, it is incremental.
- The IVIF-specific design must be primitive-level modal selection: e.g., thermal target primitives, visible texture primitives, quality-aware alpha, or scale/misalignment-aware covariance.

## 6. Preliminary Method Direction

Recommended paper angle:

**Continuous Gaussian Primitive Fusion for Cross-Resolution Infrared-Visible Image Fusion**

Possible technical formulation:

- Inputs: IR image and VIS image, initially aligned but optionally with different resolutions.
- Backbone: dual-stream lightweight encoder; visible branch emphasizes gradients/textures, IR branch emphasizes saliency/thermal targets.
- Primitive generator: predicts a set of Gaussian primitives, each with geometry and modality coefficients.
- Rendering: renders either (a) a fused image directly, or preferably (b) a fused residual/detail field added to a base luminance image.
- Losses: avoid relying only on hand-crafted fusion losses; combine reconstruction/self-consistency, gradient/detail preservation, thermal saliency preservation, and possibly cross-reconstruction or task-aware losses.
- Inference: support arbitrary output grid and scale-decoupled primitive estimation/rendering.

Backbone judgment:

- Do not make the backbone the main novelty. The field is saturated with Transformer/SAM/Mamba-like backbones.
- Reusing the current GSPan attention framework is acceptable as a starting implementation, but it must be renamed/reworked around IVIF semantics:
  - GSPan: PAN structural guidance + MS spectral context.
  - IVIF-GS: IR thermal saliency + VIS texture/detail, with primitive-level quality/modality gating.
- A hybrid CNN + window/cross attention encoder is a conservative first version. Mamba is not necessary unless we need long-range low-cost sequence modeling, and it would compete with MemoryFusion/long-dependence papers.

## 7. Experiment Ideas That Fit IVIF Reality

Minimum standard experiments:

- Datasets: TNO, RoadScene, MSRS, M3FD; add LLVIP/MFNet/FMB if doing detection/segmentation.
- Baselines: DenseFuse, RFN-Nest, U2Fusion, SDNet, PIAFusion, SwinFusion, CDDFuse, SeAFusion/TarDAL/MetaFusion, DDFM/Diff-IF, plus INRFuse if reproducible.
- Metrics: EN, MI, SF, AG, SD, CC, SCD, VIF, QAB/F, SSIM; plus runtime/FLOPs/params.
- Downstream: object detection on M3FD or segmentation on MFNet/FMB if positioning as task-adaptive.

Experiments that can make the GS story distinctive:

- Cross-resolution IVIF: downsample one modality or request fused output at multiple scales; compare fixed-grid models, interpolation, INRFuse, and GS rendering.
- Arbitrary-scale rendering: train at one/few scales and test at unseen non-integer output scales.
- Scale-decoupled inference: estimate primitives at lower resolution and render at target resolution; report quality-speed tradeoff.
- High-resolution stress test: crop/stitched large scenes; compare memory/runtime with INR/diffusion/Transformer baselines.
- Degradation robustness: noise, low light, fog/smoke; inspect whether primitive scale/alpha suppresses degraded visible textures while keeping IR targets.
- Small misalignment stress test: synthetic translation/deformation; test whether anisotropic covariance and local primitive support reduce ghosting.
- Primitive visualization: overlay centers/ellipses/alpha/modality weights on IR/VIS/fused images to show thermal target and visible edge selection.

## 8. Current Recommendation

For a stable Q1 journal target, the safest and most differentiated route is:

1. Main problem: cross-resolution/arbitrary-scale IVIF, not generic IVIF.
2. Main method: continuous region-level Gaussian primitive representation with primitive-level modality selection.
3. Main contrast: fixed-grid IVIF and INR-based continuous IVIF.
4. Main evidence: arbitrary scale, large-resolution efficiency, visual metrics, and one downstream task.
5. Optional extension: degraded/noisy or slight misalignment stress tests as practical scenarios.

Working title options:

- GS-Fuse: Continuous Gaussian Primitive Representation for Cross-Resolution Infrared and Visible Image Fusion
- GIVFuse: Gaussian Implicit-Explicit Primitive Fusion for Arbitrary-Scale Infrared-Visible Imaging
- GaussianIVF: Region-Level Continuous Representation for Efficient Infrared-Visible Image Fusion

## 9. Sources Used

Local PDFs:

- `综述/1-s2.0-S1566253517307972-main.pdf`
- `综述/Visible_and_Infrared_Image_Fusion_Using_Deep_Learning (1).pdf`
- `综述/Infrared_and_Visible_Image_Fusion_From_Data_Compatibility_to_Task_Adaption.pdf`
- all PDFs under `2025-2026 papers/`

Key DOI/source links for checking:

- Ma et al. 2019 survey: https://doi.org/10.1016/j.inffus.2018.02.004
- Zhang & Demiris 2023 TPAMI survey: https://doi.org/10.1109/TPAMI.2023.3261282
- Liu et al. 2025 TPAMI survey: https://doi.org/10.1109/TPAMI.2024.3521416
- DenseFuse: https://doi.org/10.1109/TIP.2018.2887342
- FusionGAN: https://doi.org/10.1016/j.inffus.2018.09.004
- U2Fusion: https://doi.org/10.1109/TPAMI.2020.3012548
- RFN-Nest: https://doi.org/10.1016/j.inffus.2021.02.023
- VIFB benchmark: https://openaccess.thecvf.com/content_CVPRW_2020/html/w50/Zhang_VIFB_A_Visible_and_Infrared_Image_Fusion_Benchmark_CVPRW_2020_paper.html
- GSASR: https://arxiv.org/abs/2501.06838

Open checks performed:

- Targeted searches for "Gaussian Splatting infrared visible image fusion", "2D Gaussian Splatting image fusion infrared", and related terms did not reveal an obvious existing 2D-GS IVIF paper in the first-pass search. This is not a final novelty proof; a formal novelty audit should still be done before claiming "first".
