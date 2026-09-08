# Verification

Smart Detailer 1.0.0, checked on 2026-09-07.

## Environment

- ComfyUI release `v0.34.0` (`12d5279438bfefc058a269eae805ceab6047777f`) and development commit `41db8f4fa1587d139e412a57b9b69394e3b13f95`; Python tests and live HTTP checks run against both. The installed frontend package was 1.51.9.
- The earlier development checkout `82db4037ce78bfc0c5c65a4b83ef9ca2a01e85aa` also reported version 0.34.0; it was not the release tag.
- Python 3.12.13, PyTorch 2.8.0+cpu and torchvision 0.23.0+cpu.
- Ultralytics 8.4.142 for the detector inference check.

## Results

Core tests and 40 ComfyUI integration/regression tests pass. Tests import ComfyUI's actual implementations and schemas. Codec and detector fixtures isolate specific boundaries; no test downloads a checkpoint.

The native sampling test runs three accumulated passes: a small SD1.5-style model, a separate SDXL-style model, then a pass combining native ControlNet, Differential Diffusion and tiled VAE processing. These use actual ComfyUI neural networks with deterministic, untrained weights. Outputs remain finite, pixels outside the composite mask remain unchanged, and the original model and ControlNet are not mutated. This verifies execution and model handoff, not image quality.

Coverage includes:

- Native masked, inpaint and erase encoders; tiled encoding/decoding and preservation of soft noise masks.
- Separate denoising/composite feather, existing denoising callbacks and rejection of known image-specific model patches that require per-crop rebuilding.
- Crop-aligned ControlNet hints, frame mapping, schedule validation, disabled strength and preservation of negative hint values.
- Erosion at image/crop boundaries, fully erased masks, region limits, area-ratio filters, confidence/spatial ordering and stable seeds.
- Source-pixel mask growth and composite feather across sampling resolutions, checked with a constant-output sampler.
- Class-specific conditioning through filtered passes or one-pass overrides, immutable mapping/region accumulation, duplicate-label errors and fallback for unmatched labels.
- Invalid confidence and unsupported conditioning in later regions fail before the first sampler call.
- Changed or missing weight files invalidate detector fingerprints. Only NMS survivors are rasterized; compact segmentation masks reproduce full-mask resizing and union. A 6×9 crop retains exactly 216 bytes of float32 tensor storage.
- Generated benchmark workflow links and node IDs match the real schemas. Pixel metrics identify outside-mask changes. A live cached workflow is marked ineligible for timing.
- Same-label NMS, adjustable IoU, multi-detector duplicates, empty frames, union metadata and detector offload on failure.
- ComfyUI's YAML loader and folder registry: shared Ultralytics paths, aliases, duplicate roots, distinct same-named files, uppercase ONNX extensions and local fallback paths.
- Crop extraction and PNG saving, including tight bounds, alpha/mask output, workflow metadata and output-directory traversal rejection.

Real `yolo11n.pt` inference on Ultralytics' bus image at 479×637 followed by a blank frame produced detections `[5, 0]`, masks shaped `[5, 637, 479]` and five correctly assigned region records: one bus and four persons. Weights were on CPU after inference. This checks preprocessing and frame ownership; specialist detector accuracy depends on the supplied weights.

A live ComfyUI server exposed all ten nodes and the new controls through `/object_info`. A detector entry registered through an external YAML file appeared in the dropdown. A queued Mask Fusion workflow completed and saved a PNG.

## Run

From this repository in ComfyUI's Python environment:

```sh
python tests/run_tests.py
COMFYUI_PATH=/absolute/path/to/ComfyUI python -m unittest discover -s tests -p 'test_*.py' -v
```

PowerShell:

```powershell
$env:COMFYUI_PATH = 'C:\ComfyUI'
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Reviewed implementations

The following source revisions informed the design. Smart Detailer calls ComfyUI's native APIs; no extension implementation was copied into this project.

| Source | Relevant behavior and decision |
|---|---|
| [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack/tree/429d0159ad429e64d2b3916e6e7be9c22d025c3c/modules/impact) | Differential Diffusion, denoising-mask feather, crop-aligned controls and tiled VAE support informed the corresponding native integrations. Detection remains separate; per-class conditioning supports filtered passes and external encoded overrides. |
| [ADetailer](https://github.com/Bing-su/adetailer/tree/3a599f5d4607d8f9d8b9fc5a15526197418dae1a/adetailer) | Erosion, relative-size filters and confidence ordering informed region selection. Smart Detailer's ratios use thresholded mask area. |
| [UniversalDetailer](https://github.com/ekakit/ComfyUI-UniversalDetailer/tree/be04a3fdeabbdb2e3bfa02d2e349c6080089e68c) | Reviewed sampling and model handling. Smart Detailer uses native KSampler and surfaces sampling failures. |
| [ComfyUI Inpaint Nodes](https://github.com/Acly/comfyui-inpaint-nodes/tree/d4a318f00fffbd269418057f869e9bc912832229) | Denoising and compositing masks serve different purposes. Image-specific Fooocus patches require crop-specific features and are rejected here. |
| [Inpaint Crop and Stitch](https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch/tree/606e2b4fd83fc44e1f0b403e1f076501db8c3749) | Reviewed crop context, resize alignment and stitching. Smart Detailer keeps mask geometry in source-image pixels and preserves pixels outside the composite mask. |
| [Ultimate SD Upscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale/blob/a5547db9e1d07d3318bb21e9e9c474f4c1e9c8df/crop_model_patch.py) | DiffSynth and Z-Image patches hold image-specific state. Those patches need dedicated crop adapters; this version rejects them. |

## Compatibility and benchmark maintenance

The compatibility workflow checks the minimum supported release (0.34.0), the dynamically resolved latest release and the upstream default branch on Linux, plus the minimum release on Windows. It runs on pushes, pull requests, manual dispatch and weekly schedules. Action revisions are pinned; jobs use read-only repository permissions. Future upstream failures are reported rather than silently skipped. Local tests do not establish that hosted Windows jobs have passed.

ComfyUI's [V3 documentation](https://docs.comfy.org/custom-nodes/v3_migration) specifies `fingerprint_inputs` for file-dependent caches and notes that `latest` is under development. The numbered `v0_0_2` adapter in the tested source also re-exports `latest` and declares `STABLE = False`, so switching imports alone would not freeze the API. The supported minimum is declared using the documented [`requires-comfyui` field](https://docs.comfy.org/registry/specifications). Registry publication requires a separately registered publisher and has not been performed.

[Benchmark instructions](examples/README.md) provide a trained-checkpoint workflow generator, API runner and outside-mask pixel checks. Reports retain the submitted graph, its hash, runtime information, output records and observed timing. Cached targets are explicitly ineligible. Trained-model comparisons and peak VRAM measurements remain outstanding; no benchmark results are synthesized.

## Limits

- No trained diffusion/inpaint checkpoint, character LoRA, CUDA GPU, ONNX provider or SAM checkpoint was available for quality or performance validation.
- SAM and Florence bridges were checked against upstream source and structured fixtures; full model inference was not run.
- Stability Matrix support was checked at its ComfyUI path-registration contract. The Windows application itself was not run.
- The browser refused the local preview with `ERR_BLOCKED_BY_CLIENT`. Backend startup, schemas, HTTP execution and output files were verified; the canvas was not visually inspected.
- Existing full-image ControlNet, GLIGEN and rectangular area conditioning are rejected. Use the detailer's ControlNet inputs for per-crop control. Known image-specific model patches and video VAEs remain unsupported.
