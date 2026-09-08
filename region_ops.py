from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.detach().float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[1] == 1:
            mask = mask[:, 0]
        else:
            raise ValueError("MASK must have a single channel; convert RGB images to MASK first")
    if mask.ndim != 3:
        raise ValueError(f"Expected MASK [N,H,W], got {tuple(mask.shape)}")
    if not torch.isfinite(mask).all():
        raise ValueError("MASK contains NaN or infinity")
    return mask.clamp(0.0, 1.0)


def resize_mask(mask: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    mask = normalize_mask(mask)
    if mask.shape[0] == 0:
        return mask.new_empty((0, height, width))
    kwargs = {"size": (height, width), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(mask.unsqueeze(1), **kwargs)[:, 0]


def resize_image_nhwc(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE [N,H,W,C], got {tuple(image.shape)}")
    x = image.movedim(-1, 1)
    x = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    return x.movedim(1, -1).clamp(0.0, 1.0)


def grow_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    mask = normalize_mask(mask)
    if pixels == 0:
        return mask
    radius = abs(int(pixels))
    value = mask if pixels > 0 else -mask
    value = value.unsqueeze(1)
    if pixels < 0:
        value = F.pad(value, (radius, radius, radius, radius), value=0)
    result = F.max_pool2d(value, kernel_size=radius * 2 + 1, stride=1, padding=radius if pixels > 0 else 0)[:, 0]
    return result if pixels > 0 else -result


def gaussian_blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    mask = normalize_mask(mask)
    if radius <= 0:
        return mask
    sigma = max(radius / 2.0, 0.5)
    half = max(1, int(math.ceil(sigma * 3.0)))
    if mask.shape[-2] > 1 and mask.shape[-1] > 1:
        half = min(half, mask.shape[-2] - 1, mask.shape[-1] - 1)
    xs = torch.arange(-half, half + 1, dtype=mask.dtype, device=mask.device)
    kernel = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    x = mask.unsqueeze(1)
    x = F.pad(x, (half, half, 0, 0), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, half, half), mode="replicate")
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x[:, 0].clamp(0.0, 1.0)


def mask_bbox(mask: torch.Tensor, threshold: float = 0.5):
    values = normalize_mask(mask)[0]
    m = (values >= threshold) & (values > 0)
    ys, xs = torch.where(m)
    if xs.numel() == 0:
        return None
    x1 = int(xs.min().item())
    x2 = int(xs.max().item()) + 1
    y1 = int(ys.min().item())
    y2 = int(ys.max().item()) + 1
    return x1, y1, x2, y2, int(m.sum().item())


def expand_bbox(bbox, image_w: int, image_h: int, context_factor: float):
    x1, y1, x2, y2 = bbox[:4]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nw, nh = bw * max(1.0, context_factor), bh * max(1.0, context_factor)
    return (
        max(0, int(math.floor(cx - nw / 2.0))),
        max(0, int(math.floor(cy - nh / 2.0))),
        min(image_w, int(math.ceil(cx + nw / 2.0))),
        min(image_h, int(math.ceil(cy + nh / 2.0))),
    )


def round_to_multiple(value: int, multiple: int = 8, minimum: int = 64) -> int:
    return max(minimum, int(round(value / multiple) * multiple))


def detail_target_size(crop_w: int, crop_h: int, bbox_w: int, bbox_h: int,
                       guide_size: int, max_size: int, upscale_only: bool = True):
    obj_long = max(1, bbox_w, bbox_h)
    scale = guide_size / float(obj_long)
    if upscale_only:
        scale = max(1.0, scale)
    if max(crop_w, crop_h) * scale > max_size:
        scale = max_size / float(max(crop_w, crop_h))
    tw = round_to_multiple(max(64, int(round(crop_w * scale))), 8)
    th = round_to_multiple(max(64, int(round(crop_h * scale))), 8)
    if max(tw, th) > max_size:
        ratio = max_size / float(max(tw, th))
        tw = round_to_multiple(max(64, int(tw * ratio)), 8)
        th = round_to_multiple(max(64, int(th * ratio)), 8)
    return tw, th


def masks_for_images(mask: torch.Tensor, image_batch: int, mode: str, metadata=None, bboxes=None):
    mask = normalize_mask(mask)
    n = mask.shape[0]
    if n == 0:
        return []
    if image_batch < 1:
        raise ValueError("IMAGE batch is empty")
    if mode == "regions_for_first_image":
        return [(0, i) for i in range(n)]
    if mode == "one_per_image":
        if n not in (1, image_batch):
            raise ValueError("one_per_image needs one shared mask or exactly one mask per image")
        return [(i, 0 if n == 1 else i) for i in range(image_batch)]
    if mode != "auto":
        raise ValueError(f"Unknown mask batch mode: {mode}")
    if metadata and any("image_index" in m or "frame" in m for m in metadata):
        if len(metadata) != n:
            raise ValueError("Frame metadata must contain exactly one record per mask")
        assignments = []
        for i, item in enumerate(metadata):
            frame = item.get("image_index", item.get("frame"))
            if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < image_batch:
                raise ValueError(f"Invalid image_index for mask {i}: {frame}")
            assignments.append((frame, i))
        return assignments
    if isinstance(bboxes, list) and (not bboxes or isinstance(bboxes[0], list)):
        if len(bboxes) != image_batch or sum(map(len, bboxes)) != n:
            raise ValueError("Connect SAM3 individual masks and their matching per-frame bboxes")
        return [(frame, i) for i, frame in enumerate(f for f, boxes in enumerate(bboxes) for _ in boxes)]
    if image_batch == 1:
        return [(0, i) for i in range(n)]
    if n == image_batch:
        return [(i, i) for i in range(image_batch)]
    if n == 1:
        return [(i, 0) for i in range(image_batch)]
    raise ValueError("Ambiguous mask batch: connect detector region_metadata or SAM3 bboxes, or explicitly select regions_for_first_image")


def boxes_from_masks(mask: torch.Tensor, threshold: float = 0.5, min_area: int = 16, max_boxes: int = 64):
    mask = normalize_mask(mask)
    boxes = []
    for i in range(mask.shape[0]):
        bbox = mask_bbox(mask[i:i + 1], threshold)
        if bbox is None:
            continue
        x1, y1, x2, y2, area = bbox
        if area < min_area:
            continue
        boxes.append({
            "x": float(x1), "y": float(y1), "width": float(x2 - x1), "height": float(y2 - y1),
            "score": 1.0, "mask_index": i,
        })
        if len(boxes) >= max_boxes:
            break
    return boxes


def normalize_florence_payload(data):
    """Accept Florence task dictionaries, Kijai frame lists and plain xyxy boxes."""
    if data is None:
        raise ValueError("Florence JSON is missing")
    if isinstance(data, dict):
        if "bboxes" not in data and "boxes" not in data:
            tasks = [v for k, v in data.items() if k.startswith("<") and isinstance(v, dict)]
            if len(tasks) != 1:
                raise ValueError("Florence JSON must contain bboxes/boxes or one grounding task")
            return normalize_florence_payload(tasks[0])
        raw = data.get("bboxes", data.get("boxes", []))
        labels = data.get("labels", [])
        if labels and len(labels) != len(raw):
            raise ValueError("Florence labels and boxes have different lengths")
        return [_florence_box(box, labels[i] if labels else "region") for i, box in enumerate(raw)]
    if not isinstance(data, (list, tuple)):
        raise ValueError("Unsupported Florence JSON; expected boxes or grounding results")
    if not data:
        return []
    def is_box(v):
        return (isinstance(v, dict) and ("x" in v or "box" in v)) or (
            isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(n, (int, float)) for n in v))
    if all(is_box(v) for v in data):
        return [_florence_box(v) for v in data]
    frames = [normalize_florence_payload(v) for v in data]
    if any(frame and isinstance(frame[0], list) for frame in frames):
        raise ValueError("Florence JSON has too many nested frame dimensions")
    return frames


def _florence_box(box, label="region"):
    if isinstance(box, dict):
        label = box.get("label", label)
        if all(k in box for k in ("x", "y", "width", "height")):
            x1, y1 = float(box["x"]), float(box["y"])
            x2, y2 = x1 + float(box["width"]), y1 + float(box["height"])
        else:
            x1, y1, x2, y2 = map(float, box["box"])
    else:
        x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("Florence coordinates must be finite")
    x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
    if x2 == x1 or y2 == y1:
        raise ValueError("Florence returned a zero-area box")
    return {"x": x1, "y": y1, "width": x2-x1, "height": y2-y1, "label": str(label)}


def combine_masks(masks: Iterable[torch.Tensor], operation: str, height: int | None = None, width: int | None = None, batch_mode: str = "pairwise") -> torch.Tensor:
    prepared = [normalize_mask(m) for m in masks if m is not None]
    if not prepared:
        raise ValueError("At least one mask is required")
    height = height or prepared[0].shape[-2]
    width = width or prepared[0].shape[-1]
    prepared = [resize_mask(m, height, width) if m.shape[-2:] != (height, width) else m for m in prepared]
    device = prepared[0].device
    prepared = [m.to(device) for m in prepared]
    if batch_mode == "collapse":
        prepared = [m.amax(dim=0, keepdim=True) if m.shape[0] else m.new_zeros((1, height, width)) for m in prepared]
    elif batch_mode == "pairwise":
        counts = {m.shape[0] for m in prepared}
        if 0 in counts:
            if counts != {0}:
                raise ValueError("Cannot pair empty masks with a nonempty batch")
            return prepared[0]
        if any(n not in (1, max(counts)) for n in counts):
            raise ValueError("Mask Fusion batches must match or contain one shared mask")
    else:
        raise ValueError("Unknown Mask Fusion batch_mode")
    out = prepared[0]
    for m in prepared[1:]:
        if operation == "union": out = torch.maximum(out, m)
        elif operation == "intersection": out = torch.minimum(out, m)
        elif operation == "subtract": out = (out - m).clamp(0.0, 1.0)
        elif operation == "xor": out = torch.abs(out - m)
        else: raise ValueError(f"Unknown operation: {operation}")
    return out.clamp(0.0, 1.0)
