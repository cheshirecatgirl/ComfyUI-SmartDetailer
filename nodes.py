from __future__ import annotations

import json
import logging
import math
from collections import OrderedDict
from threading import RLock
from PIL import Image
from pathlib import Path
from typing import Any

import torch

import nodes as comfy_nodes
import comfy.model_management
import comfy.samplers
import folder_paths
from comfy_api.latest import ComfyExtension, io, ui
from typing_extensions import override

from .regions import (
    add_region,
    crop_region_tensor,
    get_region,
    label_list,
    merge_regions,
    new_regions,
    parse_metadata,
    public_region_metadata,
    region_confidence,
    region_label,
    refresh_regions,
    safe_filename_component,
    select_regions,
    summary_json,
)
from .region_ops import (
    boxes_from_masks,
    combine_masks,
    detail_target_size,
    expand_bbox,
    gaussian_blur_mask,
    grow_mask,
    mask_bbox,
    masks_for_images,
    normalize_florence_payload,
    normalize_mask,
    resize_image_nhwc,
    resize_mask,
)

from .sampling import sample_crop, crop_conditioning, sampling_alignment, prepare_detail_model, validate_conditioning

log = logging.getLogger("SmartDetailer")
SmartRegions = io.Custom("SMART_REGIONS")
SmartConditioning = io.Custom("SMART_CONDITIONING")
FlorenceJSON = io.Custom("JSON")

_YOLO_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_YOLO_CACHE_LIMIT = 2
_YOLO_LOCK = RLock()


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _detector_model_map() -> dict[str, str]:
    base = Path(folder_paths.models_dir)
    categories = {
        "ultralytics_bbox": "ultralytics/bbox",
        "ultralytics_segm": "ultralytics/segm",
        "ultralytics": "ultralytics",
        "detectors": "detectors",
        "yolo": "yolo",
    }
    roots = []
    for category, prefix in categories.items():
        if category in folder_paths.folder_names_and_paths:
            roots.extend((Path(p), prefix) for p in folder_paths.get_folder_paths(category))
    roots.extend((base / prefix, prefix) for prefix in
                 ("ultralytics/bbox", "ultralytics/segm", "detectors", "yolo"))
    found: dict[str, str] = {}
    seen = set()
    for root, prefix in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*"), key=lambda p: str(p).casefold()):
            if not path.is_file() or path.suffix.lower() not in {".pt", ".onnx"}:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            name = f"{prefix}/{path.relative_to(root).as_posix()}"
            display, ordinal = name, 2
            while display in found:
                display = f"{name} [{ordinal}]"
                ordinal += 1
            found[display] = str(resolved)
    return dict(sorted(found.items(), key=lambda kv: kv[0].lower()))


def _resolve_detector_path(model_name: str, model_path_override: str) -> str:
    if model_path_override and model_path_override.strip():
        path = Path(model_path_override.strip()).expanduser()
    else:
        model_map = _detector_model_map()
        if model_name not in model_map:
            raise FileNotFoundError(
                "Detector model not found. Put .pt/.onnx under models/ultralytics/bbox, "
                "models/ultralytics/segm, models/detectors, or models/yolo; configure shared paths "
                "in extra_model_paths.yaml; or use model_path_override."
            )
        path = Path(model_map[model_name])
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return str(path)


def _detector_file_identity(path: str):
    stat = Path(path).stat()
    return (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _detector_paths(model_name, model_path_override="", additional_models=""):
    paths = [_resolve_detector_path(model_name, model_path_override)]
    paths.extend(_resolve_detector_path(name.strip(), "") for name in additional_models.splitlines() if name.strip())
    return list(dict.fromkeys(str(Path(p).resolve()) for p in paths))


def _get_yolo_model(path: str):
    key = _detector_file_identity(path)
    if key in _YOLO_CACHE:
        model = _YOLO_CACHE.pop(key)
        _YOLO_CACHE[key] = model
        return model
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "YOLO Detector is optional and requires the `ultralytics` Python package. "
            "The rest of Smart Detailer has no Ultralytics dependency."
        ) from e
    model = YOLO(path)
    _YOLO_CACHE[key] = model
    while len(_YOLO_CACHE) > _YOLO_CACHE_LIMIT:
        _YOLO_CACHE.popitem(last=False)
    return model


def _pack_detector_mask(mask):
    """Retain only nonzero pixels; clone so a crop cannot retain the full batch."""
    mask = normalize_mask(mask)
    nonzero = mask[0] > 0
    ys = torch.where(nonzero.any(dim=1))[0]
    xs = torch.where(nonzero.any(dim=0))[0]
    if xs.numel() == 0:
        return (None, None, tuple(mask.shape[-2:]))
    x1, y1, x2, y2 = int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1
    return (mask[:, y1:y2, x1:x2].cpu().clone(), (x1, y1, x2, y2), tuple(mask.shape[-2:]))


def _write_detector_mask(destination, candidate):
    box, packed, _ = candidate
    height, width = destination.shape[-2:]
    if packed is None:
        x1, y1 = math.floor(box['x']), math.floor(box['y'])
        x2, y2 = math.ceil(box['x'] + box['width']), math.ceil(box['y'] + box['height'])
        destination[:, y1:y2, x1:x2] = 1
        return
    pixels, bounds, shape = packed
    if pixels is None:
        return
    x1, y1, x2, y2 = bounds
    if shape == (height, width):
        target = destination[:, y1:y2, x1:x2]
        torch.maximum(target, pixels, out=target)
    else:
        original = torch.zeros((1, *shape))
        original[:, y1:y2, x1:x2] = pixels
        torch.maximum(destination, resize_mask(original, height, width), out=destination)


def _conditioning_map(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or value.get('type') != 'SMART_CONDITIONING' or value.get('version') != 1:
        raise ValueError('Expected SMART_CONDITIONING version 1')
    entries = value.get('entries')
    if not isinstance(entries, dict) or not all(isinstance(k, str) and k and isinstance(v, tuple) and len(v) == 2 for k, v in entries.items()):
        raise ValueError('Invalid SMART_CONDITIONING entries')
    return entries


class SmartDetailerConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='SmartDetailerConditioning', display_name='Smart Detailer Conditioning', category='Smart Detailer',
            description='Assign encoded positive and negative conditioning to region labels. Chain entries for additional classes.',
            inputs=[io.String.Input('labels', default='face'), io.Conditioning.Input('positive'),
                    io.Conditioning.Input('negative'), SmartConditioning.Input('previous', optional=True)],
            outputs=[SmartConditioning.Output('conditioning')])

    @classmethod
    def execute(cls, labels, positive, negative, previous=None):
        wanted = {label.casefold() for label in label_list(labels)}
        if not wanted:
            raise ValueError('Supply at least one region label')
        entries = dict(_conditioning_map(previous))
        duplicate = wanted.intersection(entries)
        if duplicate:
            raise ValueError('Conditioning already assigned for: ' + ', '.join(sorted(duplicate)))
        entries.update({label: (positive, negative) for label in wanted})
        return io.NodeOutput({'type': 'SMART_CONDITIONING', 'version': 1, 'entries': entries})


class SmartDetailer(io.ComfyNode):
    """Mask-first detailer that preserves every processed region in SMART_REGIONS."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SmartDetailer",
            display_name="Smart Detailer",
            category="Smart Detailer",
            search_aliases=["detailer", "face detailer", "mask detailer", "pony", "illustrious", "inpaint detailer"],
            description=(
                "Details masked regions using the supplied model, VAE and conditioning. "
                "Returns the composite image and region crops, masks, boxes and metadata."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=200),
                io.Float.Input("cfg", default=5.0, min=0.0, max=100.0, step=0.1),
                io.Float.Input("denoise", default=0.32, min=0.0, max=1.0, step=0.01),
                io.Int.Input("guide_size", default=1024, min=128, max=4096, step=64,
                             tooltip="Target longest dimension of the detected object before max_size limits the context crop."),
                io.Int.Input("max_size", default=1536, min=256, max=4096, step=64),
                io.Float.Input("context_factor", default=1.6, min=1.0, max=4.0, step=0.05,
                               tooltip="Context around the target region. A larger crop gives the model more surrounding image."),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="dpmpp_2m_sde", advanced=True),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="karras", advanced=True),
                io.Int.Input("mask_grow", default=4, min=-128, max=128, advanced=True,
                             tooltip="Radius in input IMAGE pixels. Positive expands the mask; negative shrinks it."),
                io.Int.Input("feather", default=8, min=0, max=128, advanced=True,
                             tooltip="Blend softness in input IMAGE pixels (Gaussian sigma = feather / 2). Applied when compositing the restored crop."),
                io.Float.Input("mask_threshold", default=0.35, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Int.Input("min_region_area", default=64, min=1, max=16_777_216, advanced=True),
                io.Int.Input("max_regions", default=16, min=1, max=256, advanced=True),
                io.Combo.Input("region_order", options=["large_first", "small_first", "input_order", "confidence", "left_to_right", "center_first"], default="large_first", advanced=True),
                io.Combo.Input("mask_batch_mode", options=["auto", "regions_for_first_image", "one_per_image"], default="auto", advanced=True),
                io.Boolean.Input("upscale_only", default=True, advanced=True),
                io.String.Input("region_labels", default="", optional=True, advanced=True,
                                tooltip="Optional comma-separated labels matching the mask batch, e.g. face,left_hand,right_hand."),
                io.String.Input("region_metadata", default="", optional=True, multiline=True, advanced=True,
                                tooltip="Optional JSON list matching the mask batch. YOLO Detector metadata can be connected here."),
                io.Boolean.Input("store_source_crops", default=False, advanced=True,
                                 tooltip="Retain pre-detail crops for Extract's source variant. Off saves RAM; enable on passes where before/after comparisons are needed."),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                     tooltip="Matching SAM3 bboxes preserve frame mapping for individual masks in image batches."),
                SmartRegions.Input("previous_regions", optional=True,
                                  tooltip="Accumulate previous passes and refresh their crops from this pass's final image."),
                io.Combo.Input("sampling_mode", options=["masked", "inpaint", "erase"], default="masked", optional=True, advanced=True,
                               tooltip="masked preserves source latents; inpaint builds crop conditioning for an inpaint model; erase removes masked pixels before encoding."),
                io.String.Input("label_filter", default="", optional=True, advanced=True,
                                tooltip="Exact comma-separated region labels, case-insensitive. Empty details all labels. Use a separate pass and matching conditioning for each target class."),
                io.Int.Input("noise_mask_feather", default=0, min=0, max=128, optional=True, advanced=True,
                             tooltip="Denoising-mask softness in input IMAGE pixels. Above zero enables native Differential Diffusion unless the model already has a denoising-mask function. Does not change the final blend mask."),
                io.Float.Input("min_region_ratio", default=0.0, min=0.0, max=1.0, step=0.001, optional=True, advanced=True,
                               tooltip="Minimum fraction of image pixels covered by a region's thresholded mask."),
                io.Float.Input("max_region_ratio", default=1.0, min=0.0, max=1.0, step=0.001, optional=True, advanced=True,
                               tooltip="Maximum fraction of image pixels covered by a region's thresholded mask."),
                io.Combo.Input("vae_mode", options=["auto", "tiled"], default="auto", optional=True, advanced=True,
                               tooltip="auto uses ComfyUI's normal VAE processing and memory fallback; tiled forces tiled encode and decode."),
                io.ControlNet.Input("control_net", optional=True),
                io.Image.Input("control_image", optional=True,
                               tooltip="Prepared ControlNet hint aligned to the input IMAGE, at the same dimensions. One shared hint or one per image. Cropped and resized for each region."),
                io.Float.Input("control_strength", default=1.0, min=0.0, max=10.0, step=0.05, optional=True, advanced=True),
                io.Float.Input("control_start", default=0.0, min=0.0, max=1.0, step=0.01, optional=True, advanced=True),
                io.Float.Input("control_end", default=1.0, min=0.0, max=1.0, step=0.01, optional=True, advanced=True),
                SmartConditioning.Input("conditioning_overrides", optional=True,
                                        tooltip="Per-label encoded conditioning. Unmatched regions use this node's positive and negative inputs."),
            ],
            outputs=[
                io.Image.Output("image"),
                SmartRegions.Output("regions"),
            ],
        )

    @classmethod
    def execute(
        cls, image, mask, model, positive, negative, vae, seed, steps, cfg, denoise, guide_size, max_size,
        context_factor, sampler_name, scheduler, mask_grow, feather, mask_threshold, min_region_area,
        max_regions, region_order, mask_batch_mode, upscale_only, region_labels="", region_metadata="",
        store_source_crops=False, bboxes=None, previous_regions=None, sampling_mode="masked", label_filter="",
        noise_mask_feather=0, min_region_ratio=0.0, max_region_ratio=1.0, vae_mode="auto",
        control_net=None, control_image=None, control_strength=1.0, control_start=0.0, control_end=1.0,
        conditioning_overrides=None,
    ) -> io.NodeOutput:
        if image.ndim != 4 or min(image.shape[:3]) < 1 or image.shape[-1] not in (3, 4):
            raise ValueError(f"IMAGE must be [N,H,W,C], got {tuple(image.shape)}")
        if not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError("IMAGE must contain finite floating-point pixels")
        if sampling_mode not in {"masked", "inpaint", "erase"}:
            raise ValueError("Invalid sampling_mode")
        if not 0 <= denoise <= 1 or steps < 1 or max_regions < 1 or context_factor < 1:
            raise ValueError("Invalid sampling or region limits")
        if not 0 <= min_region_ratio <= max_region_ratio <= 1:
            raise ValueError("Region ratios must satisfy 0 <= min <= max <= 1")
        if not -128 <= mask_grow <= 128 or not 0 <= noise_mask_feather <= 128 or not 0 <= feather <= 128:
            raise ValueError("Invalid mask radius")
        if vae_mode not in {"auto", "tiled"}:
            raise ValueError("Invalid vae_mode")
        if (control_net is None) != (control_image is None):
            raise ValueError("Connect both control_net and its prepared control_image")
        if not 0 <= control_strength <= 10 or not 0 <= control_start < control_end <= 1:
            raise ValueError("ControlNet needs nonnegative strength and 0 <= start < end <= 1")
        output_image = image.clone()
        source_image = image.detach().cpu() if store_source_crops else None
        masks = normalize_mask(mask)
        batch, height, width, _ = output_image.shape
        if control_image is not None:
            if (control_image.ndim != 4 or control_image.shape[0] not in (1, batch)
                    or control_image.shape[1:3] != (height, width) or control_image.shape[-1] not in (3, 4)
                    or not control_image.is_floating_point() or not torch.isfinite(control_image).all()):
                raise ValueError("control_image must be finite RGB/RGBA, match IMAGE dimensions, and have batch 1 or IMAGE batch")
        if masks.shape[-2:] != (height, width):
            masks = resize_mask(masks, height, width)

        supplied_metadata = parse_metadata(region_metadata)
        supplied_labels = label_list(region_labels)
        assignments = masks_for_images(masks, batch, mask_batch_mode, supplied_metadata, bboxes)
        if not supplied_metadata and bboxes is not None:
            flat = ([bboxes] if isinstance(bboxes, dict) else
                    [box for frame in bboxes for box in frame] if bboxes and isinstance(bboxes[0], list) else bboxes)
            if len(flat) != masks.shape[0]:
                raise ValueError("bboxes must match the individual mask count")
            supplied_metadata = [dict(box) for box in flat]
        wanted_labels = {label.casefold() for label in label_list(label_filter)}
        overrides = _conditioning_map(conditioning_overrides)
        jobs = []
        for image_idx, mask_idx in assignments:
            if wanted_labels and region_label(mask_idx, supplied_labels, supplied_metadata).casefold() not in wanted_labels:
                continue
            bbox = mask_bbox(masks[mask_idx:mask_idx + 1], mask_threshold)
            if bbox is None:
                continue
            x1, y1, x2, y2, area = bbox
            if area < min_region_area or not min_region_ratio <= area / (height * width) <= max_region_ratio:
                continue
            jobs.append({"image_idx": image_idx, "mask_idx": mask_idx, "bbox": (x1, y1, x2, y2), "area": area,
                         "confidence": region_confidence(mask_idx, supplied_metadata)})

        if region_order == "large_first":
            jobs.sort(key=lambda j: j["area"], reverse=True)
        elif region_order == "small_first":
            jobs.sort(key=lambda j: j["area"])
        elif region_order == "confidence":
            jobs.sort(key=lambda j: j["confidence"], reverse=True)
        elif region_order == "left_to_right":
            jobs.sort(key=lambda j: (j["image_idx"], j["bbox"][0], j["bbox"][1]))
        elif region_order == "center_first":
            jobs.sort(key=lambda j: ((j["bbox"][0] + j["bbox"][2] - width) / width) ** 2
                      + ((j["bbox"][1] + j["bbox"][3] - height) / height) ** 2)
        elif region_order != "input_order":
            raise ValueError("Invalid region_order")
        if len(jobs) > max_regions:
            log.warning("Pass limited to %s of %s eligible regions", max_regions, len(jobs))
        if denoise > 0:
            checked = set()
            for job in jobs:
                label = region_label(job['mask_idx'], supplied_labels, supplied_metadata).casefold()
                for conditioning in overrides.get(label, (positive, negative)):
                    if id(conditioning) not in checked:
                        validate_conditioning(conditioning, batch)
                        checked.add(id(conditioning))

        regions = new_regions(output_image.shape)
        if previous_regions is not None:
            # Validate before any sampling; never leave an expensive pass ending in a merge error.
            merge_regions([previous_regions, regions])
        alignment = sampling_alignment(vae) if denoise > 0 else 8
        sampling_model = prepare_detail_model(model, noise_mask_feather > 0) if jobs and denoise > 0 else model

        for job in jobs:
            comfy.model_management.throw_exception_if_processing_interrupted()
            image_idx = job["image_idx"]
            mask_idx = job["mask_idx"]
            region_mask = masks[mask_idx:mask_idx + 1].to(output_image.device)
            x1, y1, x2, y2 = job["bbox"]
            bbox_w, bbox_h = x2 - x1, y2 - y1
            cx1, cy1, cx2, cy2 = expand_bbox(job["bbox"], width, height, context_factor)
            margin = max(0, int(mask_grow)) + math.ceil(max(float(feather), float(noise_mask_feather)) * 1.5)
            cx1, cy1 = max(0, min(cx1, x1 - margin)), max(0, min(cy1, y1 - margin))
            cx2, cy2 = min(width, max(cx2, x2 + margin)), min(height, max(cy2, y2 + margin))
            crop_w, crop_h = cx2 - cx1, cy2 - cy1

            target_w, target_h = detail_target_size(crop_w, crop_h, bbox_w, bbox_h, guide_size, max_size, upscale_only)
            cap = (int(max_size) // alignment) * alignment
            if cap < alignment:
                raise ValueError("max_size is smaller than the VAE alignment")
            target_w = min(cap, max(alignment, round(target_w / alignment) * alignment))
            target_h = min(cap, max(alignment, round(target_h / alignment) * alignment))
            current_crop = output_image[image_idx:image_idx + 1, cy1:cy2, cx1:cx2, :3]
            original_mask_crop = region_mask[:, cy1:cy2, cx1:cx2].clamp(0.0, 1.0)
            hard_edit_mask = grow_mask(original_mask_crop, mask_grow).clamp(0.0, 1.0)
            if not torch.any(hard_edit_mask > 0):
                continue
            if len(regions["regions"]) >= max_regions:
                break
            paste_mask = gaussian_blur_mask(hard_edit_mask, feather)

            detail_image = resize_image_nhwc(current_crop, target_h, target_w)
            detail_mask = resize_mask(hard_edit_mask, target_h, target_w).clamp(0.0, 1.0)
            detail_noise_mask = (resize_mask(gaussian_blur_mask(hard_edit_mask, noise_mask_feather), target_h, target_w)
                                 if noise_mask_feather > 0 else None)
            region_seed = (int(seed) + image_idx * max(1, masks.shape[0]) + mask_idx) & 0xFFFFFFFFFFFFFFFF
            if float(denoise) > 0.0:
                crop = (cx1, cy1, cx2, cy2)
                label = region_label(mask_idx, supplied_labels, supplied_metadata).casefold()
                region_positive, region_negative = overrides.get(label, (positive, negative))
                pos = crop_conditioning(region_positive, image_idx, batch, crop, (target_h, target_w), (height, width))
                neg = crop_conditioning(region_negative, image_idx, batch, crop, (target_h, target_w), (height, width))
                if control_net is not None and control_strength > 0:
                    frame = image_idx if control_image.shape[0] > 1 else 0
                    # Hints can contain negative sentinels used by inpaint ControlNets.
                    hint = torch.nn.functional.interpolate(
                        control_image[frame:frame + 1, cy1:cy2, cx1:cx2, :3].movedim(-1, 1).float(),
                        size=(target_h, target_w), mode="bilinear", align_corners=False).movedim(1, -1)
                    pos, neg = comfy_nodes.ControlNetApplyAdvanced().apply_controlnet(
                        pos, neg, control_net, hint, float(control_strength), float(control_start), float(control_end), vae=vae)
                decoded = sample_crop(comfy_nodes, sampling_model, vae, detail_image, detail_mask, pos, neg,
                                      region_seed, int(steps), float(cfg), sampler_name, scheduler,
                                      float(denoise), sampling_mode, detail_noise_mask, vae_mode)
                restored = resize_image_nhwc(decoded, crop_h, crop_w).to(output_image)
                alpha = paste_mask.to(output_image).unsqueeze(-1)
                blended = current_crop * (1.0 - alpha) + restored * alpha
                # Preserve the original tensor exactly outside the feathered edit support.
                output_image[image_idx:image_idx + 1, cy1:cy2, cx1:cx2, :3] = torch.where(
                    alpha > 0, blended.clamp(0, 1), current_crop)

            inherited = supplied_metadata[mask_idx] if mask_idx < len(supplied_metadata) else {}
            region = {
                "region_id": len(regions["regions"]),
                "image_index": image_idx,
                "mask_index": mask_idx,
                "label": region_label(mask_idx, supplied_labels, supplied_metadata),
                "confidence": job["confidence"],
                "conditioning_override": region_label(mask_idx, supplied_labels, supplied_metadata).casefold() in overrides,
                "bbox": [x1, y1, x2, y2],
                "crop": [cx1, cy1, cx2, cy2],
                "area": job["area"],
                "detail_size": [target_w, target_h],
                "seed": region_seed,
                "mask_grow": int(mask_grow),
                "feather": int(feather),
                "noise_mask_feather": int(noise_mask_feather),
                "sampling_mode": sampling_mode,
                "vae_mode": vae_mode,
                "control": ({"strength": float(control_strength), "start": float(control_start), "end": float(control_end)}
                            if control_net is not None and control_strength > 0 else None),
                "metadata": inherited,
                # Keep region payloads CPU-side so retaining SMART_REGIONS does not pin unnecessary VRAM.
                "mask_crop": original_mask_crop.detach().float().cpu().clone(),
            }
            if source_image is not None:
                region["source_crop"] = source_image[image_idx:image_idx + 1, cy1:cy2, cx1:cx2, :3].clone()
            add_region(regions, region)

        # Snapshot crops only after all passes so coarse regions include later fine edits (e.g. eyes inside a face).
        for region in regions["regions"]:
            image_idx = int(region["image_index"])
            cx1, cy1, cx2, cy2 = map(int, region["crop"])
            region["detailed_crop"] = output_image[image_idx:image_idx + 1, cy1:cy2, cx1:cx2, :3].detach().float().cpu().clone()

        if previous_regions is not None:
            regions = refresh_regions(merge_regions([previous_regions, regions]), output_image)
        return io.NodeOutput(output_image, regions)


class SmartDetailerExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SmartDetailerExtract",
            display_name="Smart Detailer Extract",
            category="Smart Detailer",
            description="Extract one region from SMART_REGIONS without rerunning detection or detailing.",
            inputs=[
                SmartRegions.Input("regions"),
                io.Int.Input("index", default=0, min=-1024, max=1024),
                io.Combo.Input("variant", options=["detailed", "source"], default="detailed"),
                io.Combo.Input("crop_mode", options=["tight", "context"], default="tight"),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.BoundingBox.Output("bbox"),
                io.String.Output("metadata"),
            ],
        )

    @classmethod
    def execute(cls, regions, index, variant, crop_mode) -> io.NodeOutput:
        region = get_region(regions, index)
        image, mask = crop_region_tensor(region, variant=variant, crop_mode=crop_mode)
        if crop_mode == "tight":
            h, w = image.shape[1:3]
            bx, by, bw, bh = 0.0, 0.0, float(w), float(h)
        else:
            x1, y1, x2, y2 = map(float, region["bbox"])
            cx1, cy1, _, _ = map(float, region["crop"])
            bx, by, bw, bh = x1 - cx1, y1 - cy1, x2 - x1, y2 - y1
        bbox = [{"x": bx, "y": by, "width": bw, "height": bh, "score": float(region.get("confidence", 1.0))}]
        return io.NodeOutput(image, mask, bbox, _safe_json(public_region_metadata(region)))


class SmartDetailerSave(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SmartDetailerSave",
            display_name="Smart Detailer Save",
            category="Smart Detailer",
            description="Save selected region crops as separate PNG files at their native sizes.",
            inputs=[
                SmartRegions.Input("regions"),
                io.String.Input("filename_prefix", default="smart_detailer"),
                io.Combo.Input("crop_mode", options=["tight", "context"], default="tight"),
                io.Boolean.Input("transparent_background", default=False,
                                 tooltip="When enabled, save RGBA using the stored segmentation mask as alpha."),
                io.Boolean.Input("save_masks", default=False),
                io.String.Input("label_filter", default="", optional=True,
                                tooltip="Optional comma-separated exact labels. Blank saves all regions."),
                io.Image.Input("final_image", optional=True, advanced=True,
                               tooltip="Optional later/final composite. When connected, region crops are refreshed from it before saving."),
            ],
            outputs=[],
            is_output_node=True,
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, regions, filename_prefix, crop_mode, transparent_background, save_masks, label_filter="", final_image=None) -> io.NodeOutput:
        if final_image is not None:
            regions = refresh_regions(regions, final_image)
        regions = select_regions(regions, label_filter)
        results = []
        for ordinal, region in enumerate(regions):
            image, mask = crop_region_tensor(region, variant="detailed", crop_mode=crop_mode)
            label = safe_filename_component(region.get("label", "region"))
            rid = int(region.get("region_id", ordinal))
            frame = int(region.get("image_index", 0))
            stem = f"{filename_prefix}/{label}_f{frame:04d}_r{rid:03d}"
            if transparent_background:
                alpha = mask.clamp(0.0, 1.0).unsqueeze(-1)
                to_save = torch.cat([image[..., :3], alpha], dim=-1)
            else:
                to_save = image[..., :3]
            saved = ui.ImageSaveHelper.get_save_images_ui(to_save, filename_prefix=stem, cls=cls)
            results.extend(saved.results)
            if save_masks:
                mask_image = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
                saved_mask = ui.ImageSaveHelper.get_save_images_ui(mask_image, filename_prefix=stem + "_mask", cls=cls)
                results.extend(saved_mask.results)
        return io.NodeOutput(ui={"images": results})


class SmartDetailerMerge(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SmartDetailerMerge",
            display_name="Smart Detailer Merge",
            category="Smart Detailer",
            description="Combine region records from separate detail passes (for example face, hands, feet, or different character-LoRA branches) without duplicating their tensors.",
            inputs=[
                SmartRegions.Input("regions_a"),
                SmartRegions.Input("regions_b", optional=True),
                SmartRegions.Input("regions_c", optional=True),
                SmartRegions.Input("regions_d", optional=True),
                io.Image.Input("final_image", optional=True, advanced=True,
                               tooltip="Optional final composite used to refresh all stored region crops after merging."),
            ],
            outputs=[SmartRegions.Output("regions")],
        )

    @classmethod
    def execute(cls, regions_a, regions_b=None, regions_c=None, regions_d=None, final_image=None) -> io.NodeOutput:
        merged = merge_regions([regions_a, regions_b, regions_c, regions_d])
        if final_image is not None:
            merged = refresh_regions(merged, final_image)
        return io.NodeOutput(merged)


class YOLODetector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        models = list(_detector_model_map().keys()) or ["<use model_path_override>"]
        return io.Schema(
            node_id="SmartDetailerYOLODetector",
            display_name="YOLO Detector",
            category="Smart Detailer/Detection",
            search_aliases=["deepghs", "anime face detector", "hand detector", "foot detector", "ultralytics"],
            description=(
                "Runs local Ultralytics-compatible detectors and returns boxes, masks and JSON metadata. "
                "Box-only models produce rectangular masks; use a segmenter to refine them."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("model_name", options=models),
                io.String.Input("model_path_override", default="", optional=True, advanced=True),
                io.Float.Input("confidence", default=0.30, min=0.0, max=1.0, step=0.01),
                io.Float.Input("iou", default=0.50, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Overlap threshold passed to YOLO and used for same-label NMS across all detector outputs per image."),
                io.Int.Input("imgsz", default=640, min=128, max=4096, step=32, advanced=True),
                io.Int.Input("max_detections", default=32, min=1, max=512, advanced=True),
                io.String.Input("class_filter", default="", optional=True, advanced=True),
                io.Boolean.Input("individual_masks", default=True, advanced=True),
                io.String.Input("additional_models", default="", optional=True, multiline=True, advanced=True,
                                tooltip="Additional detector names from the model dropdown, one per line. Runs sequentially; exact duplicate paths are removed."),
            ],
            outputs=[io.BoundingBox.Output("bboxes"), io.Mask.Output("masks"), io.String.Output("metadata")],
        )

    @classmethod
    def fingerprint_inputs(cls, model_name, model_path_override="", additional_models="", **kwargs):
        return tuple(_detector_file_identity(path) for path in _detector_paths(model_name, model_path_override, additional_models))

    @classmethod
    def execute(cls, image, model_name, confidence, iou, imgsz, max_detections, class_filter="", individual_masks=True, model_path_override="", additional_models="") -> io.NodeOutput:
        if image.ndim != 4 or min(image.shape[:3]) < 1 or image.shape[-1] < 3 or not torch.isfinite(image).all():
            raise ValueError("YOLO requires a finite, nonempty IMAGE batch")
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1 or max_detections < 1 or imgsz < 1:
            raise ValueError('Invalid detector thresholds or limits')
        paths = _detector_paths(model_name, model_path_override, additional_models)
        batch, height, width, _ = image.shape
        wanted = {x.strip().lower() for x in class_filter.split(",") if x.strip()}
        detections = [[] for _ in range(batch)]
        # PIL inputs let Ultralytics letterbox arbitrary dimensions and return original-image coordinates.
        source = [Image.fromarray((im[..., :3].detach().float().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8")) for im in image]
        device = str(comfy.model_management.get_torch_device())
        with _YOLO_LOCK:
            for path in paths:
                comfy.model_management.throw_exception_if_processing_interrupted()
                yolo = _get_yolo_model(path)
                try:
                    for frame_i, pixels in enumerate(source):
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        result = yolo.predict(source=pixels, conf=float(confidence), iou=float(iou),
                                              imgsz=int(imgsz), max_det=int(max_detections), retina_masks=True,
                                              device=device, verbose=False)[0]
                        boxes = result.boxes
                        if boxes is None:
                            raise ValueError("Detector checkpoint must perform detection or segmentation, not classification")
                        seg = None if result.masks is None else result.masks.data
                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = boxes.xyxy[i].detach().cpu().tolist()
                            score, cid = float(boxes.conf[i]), int(boxes.cls[i])
                            label = str(result.names[cid])
                            if wanted and label.lower() not in wanted and str(cid) not in wanted:
                                continue
                            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)):
                                raise ValueError("Detector returned non-finite coordinates or confidence")
                            x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                            y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                            if x2 <= x1 or y2 <= y1:
                                continue
                            packed = None if seg is None else _pack_detector_mask(seg[i:i+1])
                            meta = {"image_index": frame_i, "frame": frame_i, "detection": i,
                                    "label": label, "class_id": cid, "confidence": score,
                                    "bbox": [x1, y1, x2, y2], "model": Path(path).name,
                                    "segmentation_model": seg is not None}
                            box = {"x": x1, "y": y1, "width": x2-x1, "height": y2-y1, "score": score, "label": label}
                            detections[frame_i].append((box, packed, meta))
                        del seg, result, boxes
                finally:
                    # Cache weights in host memory; do not pin detector VRAM during diffusion.
                    try:
                        if Path(path).suffix.lower() == ".pt":
                            yolo.to("cpu")
                    finally:
                        # Also release ONNX provider sessions rather than retaining device memory.
                        yolo.predictor = None

        selected = []
        for frame_i, found in enumerate(detections):
            kept = []
            for candidate in sorted(found, key=lambda d: d[2]["confidence"], reverse=True):
                if any(old[2]["label"].lower() == candidate[2]["label"].lower() and
                       _box_iou(old[2]["bbox"], candidate[2]["bbox"]) > iou for old in kept):
                    continue
                kept.append(candidate)
                if len(kept) >= max_detections:
                    break
            selected.append(kept)
        del detections, found
        count = sum(len(kept) for kept in selected) if individual_masks else batch
        masks_output = torch.zeros((count, height, width))
        frame_boxes, metadata = [], []
        for frame_i, kept in enumerate(selected):
            frame_boxes.append([d[0] for d in kept])
            if individual_masks:
                for candidate in kept:
                    meta = candidate[2]
                    index = len(metadata)
                    meta["mask_index"] = index
                    _write_detector_mask(masks_output[index:index + 1], candidate)
                    metadata.append(meta)
            else:
                for candidate in kept:
                    _write_detector_mask(masks_output[frame_i:frame_i + 1], candidate)
                metadata.append({"image_index": frame_i, "mask_index": frame_i, "label": "union",
                                 "detections": [d[2] for d in kept]})
        return io.NodeOutput(frame_boxes[0] if batch == 1 else frame_boxes, masks_output, _safe_json(metadata))


def _box_iou(a, b):
    iw, ih = max(0, min(a[2], b[2])-max(a[0], b[0])), max(0, min(a[3], b[3])-max(a[1], b[1]))
    intersection = iw * ih
    return intersection / max(1e-12, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


class MaskToBoundingBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SmartDetailerMaskToBoundingBoxes",
            display_name="Mask to Bounding Boxes",
            category="Smart Detailer/Detection",
            description="Convert each MASK batch element to native ComfyUI BoundingBox, useful for specialist masks → SAM3 refinement.",
            inputs=[
                io.Mask.Input("mask"),
                io.Float.Input("threshold", default=0.35, min=0.0, max=1.0, step=0.01),
                io.Int.Input("min_area", default=16, min=1, max=16_777_216),
                io.Int.Input("max_boxes", default=64, min=1, max=256),
            ],
            outputs=[io.BoundingBox.Output("bboxes"), io.String.Output("metadata")],
        )

    @classmethod
    def execute(cls, mask, threshold, min_area, max_boxes) -> io.NodeOutput:
        boxes = boxes_from_masks(mask, threshold, min_area, max_boxes)
        native = [{k: v for k, v in b.items() if k in ("x", "y", "width", "height", "score")} for b in boxes]
        return io.NodeOutput(native, _safe_json(boxes))


class Florence2JSONToBoundingBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SmartDetailerFlorence2JSONToBoundingBoxes",
            display_name="Florence2 JSON to Bounding Boxes",
            category="Smart Detailer/Detection",
            description="Convert Florence2 grounding results to native bounding boxes and JSON metadata.",
            inputs=[FlorenceJSON.Input("data")],
            outputs=[io.BoundingBox.Output("bboxes"), io.String.Output("metadata")],
        )

    @classmethod
    def execute(cls, data) -> io.NodeOutput:
        boxes = normalize_florence_payload(data)
        if boxes and isinstance(boxes[0], list):
            metadata = [{**box, "image_index": frame} for frame, entries in enumerate(boxes) for box in entries]
        else:
            metadata = boxes
        return io.NodeOutput(boxes, _safe_json(metadata))


class MaskFusion(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SmartDetailerMaskFusion",
            display_name="Mask Fusion",
            category="Smart Detailer/Detection",
            description="Combine masks by union, intersection, subtraction or exclusive-or.",
            inputs=[
                io.Mask.Input("mask_a"),
                io.Mask.Input("mask_b", optional=True),
                io.Mask.Input("mask_c", optional=True),
                io.Mask.Input("mask_d", optional=True),
                io.Combo.Input("operation", options=["union", "intersection", "subtract", "xor"], default="union"),
                io.Combo.Input("batch_mode", options=["pairwise", "collapse"], default="pairwise", optional=True,
                               tooltip="pairwise combines matching masks; collapse first unions the masks within each input."),
            ],
            outputs=[io.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, mask_a, mask_b=None, mask_c=None, mask_d=None, operation="union", batch_mode="pairwise") -> io.NodeOutput:
        return io.NodeOutput(combine_masks([mask_a, mask_b, mask_c, mask_d], operation, batch_mode=batch_mode))


class SmartDetailerInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SmartDetailerInfo",
            display_name="Smart Detailer Info",
            category="Smart Detailer",
            description="Return a tensor-free JSON summary of SMART_REGIONS for inspection, automation, or dataset tooling.",
            inputs=[SmartRegions.Input("regions")],
            outputs=[io.String.Output("json")],
        )

    @classmethod
    def execute(cls, regions) -> io.NodeOutput:
        return io.NodeOutput(summary_json(regions))


class SmartDetailerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SmartDetailer,
            SmartDetailerConditioning,
            SmartDetailerExtract,
            SmartDetailerSave,
            SmartDetailerMerge,
            YOLODetector,
            MaskToBoundingBoxes,
            Florence2JSONToBoundingBoxes,
            MaskFusion,
            SmartDetailerInfo,
        ]


async def comfy_entrypoint() -> SmartDetailerExtension:
    return SmartDetailerExtension()
