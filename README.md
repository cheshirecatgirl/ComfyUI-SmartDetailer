# Smart Detailer

Detail masked regions in ComfyUI using the model, VAE and conditioning already in your workflow. Detection is separate: feed it masks from YOLO, SAM, Florence2 or a mask editor. The node returns the finished image and a record of each region.

## Install

Clone this repository into `ComfyUI/custom_nodes/ComfyUI-SmartDetailer` and restart ComfyUI. It requires ComfyUI 0.34.0 or newer. Core detailing needs no packages beyond ComfyUI. For the optional YOLO Detector, install `ultralytics` in ComfyUI's Python environment; ONNX weights also need an ONNX runtime. Supply your own detector weights.

## Basic workflow

1. Connect IMAGE, MASK, MODEL, VAE, and positive and negative CONDITIONING to **Smart Detailer**.
2. Connect its `image` output to Preview Image or Save Image. Connect `regions` to **Smart Detailer Save** for separate crops.

Use conditioning encoded with the CLIP and LoRAs that match the model. `denoise=0` records regions without changing the image or running the sampler.

For separate face and hand models, chain two detailers: connect the first `image` to the second `image`, and the first `regions` to the second `previous_regions`. Set `label_filter` on each pass and supply conditioning from that pass's model. The last pass holds the finished image and all region records. **Smart Detailer Merge** is for records from separate branches; it does not combine images.

If classes share a model, use **Smart Detailer Conditioning** to assign encoded positive and negative conditioning to their labels. Chain its `previous` input for more labels and connect the result to `conditioning_overrides`. Unmatched labels use the detailer's regular conditioning. An override replaces both prompts for its class, so include any shared prompt text in it. Labels match exactly, ignoring case.

## Masks and sampling

`guide_size` aims at the longest side of the detected region. `context_factor` adds surrounding image; `max_size` limits the sampling crop. Crops near an image edge shift inward to keep context, and narrow regions gain context rather than being stretched to the sampler's minimum size. An image too narrow for that raises an error. `upscale_only` avoids voluntary downscaling, though the `max_size` limit can still require it.

- `sampling_mode=masked` uses a latent noise mask with a regular image model. `inpaint` builds conditioning for a compatible inpaint model. `erase` clears masked pixels before VAE encoding.
- `mask_grow` expands or shrinks the edit area. `mask_grow_space=detail` measures its radius at sampling resolution; `source` measures it in input-image pixels. Crops expand as needed to keep growth and feather inside the available image.
- `feather` softens the final blend. `noise_mask_feather` softens the sampling mask and enables ComfyUI's Differential Diffusion if the model does not already provide a denoising-mask function. Both feather widths use input-image pixels.
- `vae_mode=tiled` forces tiled encoding and decoding. Leave it on `auto` unless you need tiled processing.

Defaults are 1024 for `guide_size`, 1536 for `max_size`, 1.6 for `context_factor`, and 0.32 for `denoise`. Adjust them for your model and region. More denoise allows bigger changes, including changes to identity or anatomy. Increasing the sampling size alone does not guarantee better detail.

Use `min_region_area` or the image-relative `min_region_ratio` and `max_region_ratio` to reject unwanted masks. `region_order` controls processing order; `max_regions` caps work across the image batch. Seeds stay tied to their original image and mask indices even if you change the order.

For per-region ControlNet, connect `control_net` and a prepared `control_image` at the input image's resolution. The hint may contain one shared frame or one per image; each region receives the matching crop. Connect text conditioning before full-image ControlNet or inpaint conditioning. Full-image ControlNet, GLIGEN, area conditioning and image-specific Fooocus, DiffSynth or Z-Image patches cannot be reused on resized crops and are rejected. Spatial conditioning masks are cropped for each region.

## Detection

| Source | Connect to Smart Detailer |
|---|---|
| YOLO Detector | `masks` → `mask`; `metadata` → `region_metadata`. Leave `bboxes` empty. |
| SAM individual masks across image batches | Connect matching `mask` and nested `bboxes` to preserve frame ownership. |
| Florence2 | Convert grounded JSON to boxes, refine with SAM, then connect the masks. Reuse labels only when their order still matches. |
| Manual masks | Connect `mask`; set `region_labels` if you want class filtering. |

Crop bounds come from the mask. A single mask can be shared across an image batch; one mask per frame also works. For several regions across several frames, supply detector metadata or matching per-frame boxes. An empty mask batch leaves the image unchanged.

**YOLO Detector** reads local `.pt` and `.onnx` weights from ComfyUI's registered `ultralytics`, `ultralytics_bbox`, `ultralytics_segm`, `detectors` or `yolo` paths, including shared paths set by Stability Matrix. It also checks `models/ultralytics/bbox`, `models/ultralytics/segm`, `models/detectors` and `models/yolo`. To add a separate model folder through `extra_model_paths.yaml`:

```yaml
smart_detailer:
  base_path: D:/AI/Models
  ultralytics: Ultralytics
```

Use `model_path_override` for a specific file. `additional_models` accepts detector names from the dropdown, one per line. The node removes overlapping same-label detections across models using `iou`; other labels remain separate. `class_filter` accepts exact names or numeric IDs. PyTorch weights filter classes before YOLO's per-model detection limit; ONNX results filter afterward. Raise `max_detections` for dense scenes, since each model applies that limit before cross-model suppression.

Box-only models return rectangular masks; segmentation models can return silhouettes. Keep `individual_masks` enabled for separate regions or class-specific prompts. Disable it to combine detections into one mask per frame.

## Region tools

| Node | Use |
|---|---|
| Smart Detailer Extract | Get one region's image, mask, box and metadata. |
| Smart Detailer Save | Save separate PNG crops, with optional transparency or mask files. |
| Smart Detailer Merge | Collect records from separate passes or branches. |
| Smart Detailer Info | Show a JSON summary of region records. |
| Mask to Bounding Boxes | Turn each mask into a box for SAM. |
| Florence2 JSON to Bounding Boxes | Convert grounding results and keep labels and frame mapping. |
| Mask Fusion | Combine masks by union, intersection, subtraction or XOR. |

Extract and Save offer `tight` and `context` crops. Enable `store_source_crops` on a pass if you want its before and after crops; leaving it off uses less memory. Connect `final_image` to Save or Merge when later work should appear in the exported detailed crops.

See [workflows and benchmarks](examples/README.md) for runnable examples, and [verification](verification.md) for test results, implementation references and limits. This project is MIT licensed; installed dependencies and model weights have their own licences.
