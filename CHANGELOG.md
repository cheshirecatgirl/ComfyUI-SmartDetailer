# 1.0.0

- Masked detailing with native sampling, inpaint conditioning and tiled VAE processing.
- Separate denoising and composite feathering, with Differential Diffusion support.
- Crop-aligned ControlNet hints and sampling schedules.
- Class, size and confidence-based region selection; mask expansion and erosion at detail or source resolution.
- Edge-aware crop context, aspect-preserving minimum sampling dimensions, and growth padding resolved at the final sampling scale.
- Sequential model and LoRA passes with accumulated region exports.
- Local multi-model YOLO detection, shared model paths, SAM and Florence2 bridges.
- Per-class encoded conditioning with default prompt fallback.
- Soft sampling masks in every mode, early metadata and previous-crop validation, class filtering before PyTorch detector limits, detector cache invalidation and compact mask allocation.
- Compatibility CI, executable API examples and benchmark reporting.
