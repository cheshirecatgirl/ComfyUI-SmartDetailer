# Smart Detailer

ComfyUI nodes for detailing masked image regions with the model, VAE and conditioning in your workflow. Version 1.0.0.

## Install

Place this repository in `ComfyUI/custom_nodes/ComfyUI-SmartDetailer` and restart ComfyUI. Keep one installation. Requires ComfyUI 0.34.0 or newer with its V3 node API; the tested runtime is listed in [verification.md](verification.md).

Core detailing uses ComfyUI's dependencies. For YOLO detection, install `ultralytics` in the same Python environment. ONNX detectors also require a suitable ONNX runtime. Detector weights are supplied separately.

## Basic workflow

1. Generate or load an image at the resolution you want to edit.
2. Supply individual region masks from a detector, segmenter or mask editor.
3. Connect IMAGE, MASK, MODEL, VAE and positive/negative CONDITIONING to **Smart Detailer**.
4. Connect its IMAGE output to Preview Image or Save Image. Connect `regions` to **Smart Detailer Save** to export separate crops.

Use conditioning encoded with the CLIP and LoRAs matching the supplied model. `denoise=0` preserves the image and records the regions without sampling.

## Models, prompts and multiple passes

Each pass has its own model, conditioning, VAE and sampling settings. To detail faces and hands differently, use two passes with `label_filter=face` and `label_filter=hand`, and supply their respective conditioning. Filters accept comma-separated exact labels and ignore case; an empty filter selects all labels.

Connect both outputs between passes:

| Output from pass A | Input to pass B |
|---|---|
| `image` | `image` |
| `regions` | `previous_regions` |

The last pass returns the final image and accumulated region records. Stored detailed crops are refreshed from that image; earlier inputs remain unchanged. This works for any number of sequential passes without merge nodes. **Smart Detailer Merge** combines separately collected region records; it does not composite images.

When classes share the same model, **Smart Detailer Conditioning** assigns encoded positive/negative conditioning to comma-separated labels. Chain entries through `previous`, then connect the final `conditioning` output to `conditioning_overrides` on the detailer. Matching is exact and case-insensitive. An override replaces both conditioning inputs for that class; include any shared prompt content when encoding it. Unmatched labels use the detailer's normal conditioning. Duplicate assignments raise an error. Use separate detailer passes when the model or sampling settings differ.

Region labels come from `region_metadata` or the optional `region_labels` string. Filtering keeps original mask indices and seeds. A pass with no matching regions returns the image unchanged and retains previous records.

## Sampling and masks

| Setting | Behavior |
|---|---|
| `sampling_mode=masked` | Encodes the original crop and applies a latent noise mask. Default for ordinary image checkpoints. |
| `sampling_mode=inpaint` | Builds native inpaint conditioning for each crop. Requires a compatible inpaint model. |
| `sampling_mode=erase` | Replaces masked source pixels before encoding through VAE Encode for Inpainting. |
| `mask_grow` | Radius in input-image pixels. Positive expands the edit mask; negative shrinks it. Fully erased masks are skipped. |
| `noise_mask_feather` | Softens the denoising mask and enables native Differential Diffusion. Default 0. An existing model denoising-mask function is preserved. |
| `feather` | Softens the final composite mask independently of denoising. |
| `vae_mode=tiled` | Forces native tiled encoding and decoding. `auto` uses ComfyUI's normal processing and memory fallback. |

Both feather controls use input-image pixels, with Gaussian sigma equal to half the value. Changing sampling resolution keeps those widths anchored to the final image. The denoising mask and final blend mask remain separate.

`guide_size` targets the longest dimension of the detected region. `context_factor` adds surrounding image, and `max_size` caps the sampling crop. Crop dimensions are aligned for the supplied VAE. `upscale_only` prevents voluntary downscaling; `max_size` can still require it.

The defaults are 1024 guide size, 1536 maximum size, 1.6 context factor and 0.32 denoise. Adjust these to the model and region. Increasing denoise permits more reconstruction and can change identity or anatomy.

`min_region_area` filters by mask pixels. `min_region_ratio` and `max_region_ratio` filter by thresholded mask area divided by image area. `region_order` supports size, confidence, input order, left-to-right and center-first. `max_regions` limits the whole pass across the image batch. Seeds depend on original image/mask indices, so changing processing order does not reassign them.

## ControlNet

Connect a loaded `CONTROL_NET` to `control_net` and a prepared hint to `control_image`. The hint must match the input image's dimensions and have either one shared frame or one frame per image. Each region gets a matching cropped and resized hint through native ControlNet Apply Advanced. `control_strength`, `control_start` and `control_end` control its schedule; strength 0 disables it.

Choose a ControlNet compatible with the sampling model and prepare the hint with its appropriate preprocessor. Hint values are preserved, including negative values used by some inpainting preprocessors.

Connect text conditioning before any full-image ControlNet or inpaint encoder. Existing full-image ControlNet, GLIGEN and rectangular area conditioning are rejected. Spatial conditioning masks are cropped and resized automatically.

Known image-specific Fooocus, DiffSynth and Z-Image control patches also require per-crop setup and are rejected. Ordinary LoRA patches and a supplied Differential Diffusion model are retained. Video VAEs are outside this extension's image workflow.

## Detection and segmentation

| Source | Wiring |
|---|---|
| YOLO Detector | `masks` to `mask`; `metadata` to `region_metadata`. Leave the detailer's `bboxes` unconnected. |
| Native SAM individual masks across multiple images | Connect both matching `masks` and `bboxes`; nested box counts restore frame ownership. |
| Florence2 grounding | Convert JSON to boxes, refine them with SAM, then supply masks. Reuse labels only if they still match the mask order and count. |
| Manual masks | Supply MASK directly; use `region_labels` when class filtering is needed. |

The detailer derives crop bounds from masks. Its optional `bboxes` input supplies frame mapping and fallback metadata. Native BOUNDING_BOX values are box dictionaries, potentially nested by frame.

Without metadata or nested boxes, automatic mapping accepts one image with many region masks, matching image/mask counts, or one mask shared across images. Other nonempty counts require explicit mapping. Batched detector instances should always carry their matching metadata or boxes. Zero masks return an unchanged image.

### YOLO models

The detector loads local Ultralytics-compatible `.pt` and `.onnx` files. It searches:

- `models/ultralytics/bbox` and `models/ultralytics/segm`
- `models/detectors` and `models/yolo`
- ComfyUI-registered paths under `ultralytics`, `ultralytics_bbox`, `ultralytics_segm`, `detectors` and `yolo`

Stability Matrix's registered shared paths are supported. For a manually managed `extra_model_paths.yaml`:

```yaml
smart_detailer:
  base_path: D:/AI/Models
  ultralytics: Ultralytics
```

The Ultralytics folder can contain `bbox/` and `segm/`. Restart ComfyUI after changing path configuration. `model_path_override` accepts an explicit file path. Duplicate resolved files are listed once; different files sharing a relative name get separate numbered entries.

`additional_models` accepts dropdown names, one per line. Models run sequentially. Confidence-ranked, same-label NMS removes overlapping duplicates per image using `iou`; different labels are retained. `max_detections` applies to the combined result per image. PyTorch detector weights return to CPU after inference; predictor sessions are released for both PyTorch and ONNX. Weight-file path, size and modification/change times participate in ComfyUI cache invalidation.

Box-only candidates are rasterized after NMS. Segmentation candidates retain compact nonzero bounds until selected; the final dense MASK batch is allocated once. Large numbers of retained regions still require memory proportional to region count and image area.

Box-only models produce rectangular masks. Segmentation models can return silhouettes. Individual mode emits one mask per detection; union mode combines detections per frame. Use individual mode when separate regions or class-specific prompts are needed.

## Region tools

| Node | Use |
|---|---|
| Smart Detailer Conditioning | Assign encoded conditioning to labels within one model pass. |
| Smart Detailer Extract | Select a region by index; return its crop, mask, crop-relative box and metadata. |
| Smart Detailer Save | Export selected crops as PNGs, with optional transparency and mask files. |
| Smart Detailer Merge | Combine region records; optional `final_image` refreshes their detailed crops. |
| Smart Detailer Info | Return a JSON summary for text display or another consumer. |
| Mask to Bounding Boxes | Return one box per nonempty mask entry. |
| Florence2 JSON to Bounding Boxes | Convert grounding results while retaining labels and frame boundaries. |
| Mask Fusion | Union, intersection, subtraction or XOR, pairwise by default. `collapse` combines each input batch first. |

Extract and Save support `tight` and `context` crops. Enable `store_source_crops` on passes that need before/after comparisons; the default saves memory. The source variant preserves that pass's input. Requesting an unavailable source crop raises an error.

`SMART_REGIONS` stores versioned records with labels, confidence, image/mask indices, bounds, seeds, settings and CPU crop tensors. A manual mask's default confidence of 1.0 is a placeholder, not a detector measurement. Saved PNGs support ComfyUI workflow metadata and normal output-directory validation.

See [examples and benchmarks](examples/README.md) for runnable API workflows and comparative testing. See [verification.md](verification.md) for tests, limitations and implementation references. This extension uses the MIT license; dependencies and model weights retain their own licenses.
