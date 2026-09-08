from __future__ import annotations

import json
import re
import math
from typing import Any, Iterable

import torch

SMART_REGIONS_VERSION = 1


def new_regions(source_shape: Iterable[int] | None = None) -> dict[str, Any]:
    shape = list(source_shape) if source_shape is not None else None
    return {
        "type": "SMART_REGIONS",
        "version": SMART_REGIONS_VERSION,
        "source_shape": shape,
        "regions": [],
    }


def validate_regions(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "SMART_REGIONS":
        raise TypeError("Expected a SMART_REGIONS value")
    if not isinstance(value.get("regions"), list):
        raise TypeError("SMART_REGIONS['regions'] must be a list")
    if value.get("version") != SMART_REGIONS_VERSION:
        raise ValueError(f"Unsupported SMART_REGIONS version: {value.get('version')}")
    return value


def add_region(region_data: dict[str, Any], region: dict[str, Any]) -> None:
    validate_regions(region_data)
    item = dict(region)
    item.setdefault("region_id", len(region_data["regions"]))
    item.setdefault("label", "region")
    item.setdefault("confidence", 1.0)
    region_data["regions"].append(item)


def merge_regions(values: Iterable[dict[str, Any] | None]) -> dict[str, Any]:
    valid = [validate_regions(v) for v in values if v is not None]
    if not valid:
        return new_regions()
    source_shape = next((v.get("source_shape") for v in valid if v.get("source_shape")), None)
    out = new_regions(source_shape)
    for ds in valid:
        other_shape = ds.get("source_shape")
        if source_shape and other_shape and list(other_shape) != list(source_shape):
            raise ValueError(f"Cannot merge SMART_REGIONS values with different source shapes: {source_shape} vs {other_shape}")
        for region in ds["regions"]:
            copied = dict(region)
            copied["region_id"] = len(out["regions"])
            out["regions"].append(copied)
    return out


def public_region_metadata(region: dict[str, Any]) -> dict[str, Any]:
    hidden = {"detailed_crop", "source_crop", "mask_crop"}
    out: dict[str, Any] = {}
    for k, v in region.items():
        if k in hidden:
            continue
        if isinstance(v, torch.Tensor):
            out[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "device": str(v.device)}
        else:
            out[k] = v
    return out


def summary_json(region_data: dict[str, Any]) -> str:
    ds = validate_regions(region_data)
    payload = {
        "type": ds.get("type"),
        "version": ds.get("version"),
        "source_shape": ds.get("source_shape"),
        "region_count": len(ds["regions"]),
        "regions": [public_region_metadata(r) for r in ds["regions"]],
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_metadata(value: str | None) -> list[dict[str, Any]]:
    if not value or not str(value).strip():
        return []
    try:
        data = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("region_metadata must be valid JSON") from exc
    if isinstance(data, dict):
        if isinstance(data.get("regions"), list):
            data = data["regions"]
        else:
            data = [data]
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError("region_metadata must be an object or a list of objects")
    return data


def label_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def region_label(index: int, labels: list[str], metadata: list[dict[str, Any]]) -> str:
    if index < len(labels):
        return labels[index]
    if index < len(metadata):
        item = metadata[index]
        for key in ("label", "class", "name"):
            if item.get(key) not in (None, ""):
                return str(item[key])
    return "region"


def region_confidence(index: int, metadata: list[dict[str, Any]]) -> float:
    if index < len(metadata):
        item = metadata[index]
        for key in ("confidence", "score"):
            try:
                value = float(item[key])
            except KeyError:
                continue
            except (TypeError, ValueError) as exc:
                raise ValueError('Region confidence must be a number between 0 and 1') from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Region confidence must be finite and between 0 and 1")
            return value
    return 1.0


def select_regions(region_data: dict[str, Any], label_filter: str = "") -> list[dict[str, Any]]:
    ds = validate_regions(region_data)
    wanted = {x.strip().lower() for x in label_filter.split(",") if x.strip()}
    if not wanted:
        return list(ds["regions"])
    return [r for r in ds["regions"] if str(r.get("label", "region")).lower() in wanted]


def get_region(region_data: dict[str, Any], index: int) -> dict[str, Any]:
    ds = validate_regions(region_data)
    regions = ds["regions"]
    if not regions:
        raise ValueError("SMART_REGIONS contains no regions")
    idx = int(index)
    if idx < 0:
        idx += len(regions)
    if idx < 0 or idx >= len(regions):
        raise IndexError(f"Region index {index} is out of range for {len(regions)} regions")
    return regions[idx]


def local_tight_box(region: dict[str, Any]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(int, region["bbox"])
    cx1, cy1, cx2, cy2 = map(int, region["crop"])
    return x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1


def crop_region_tensor(region: dict[str, Any], variant: str = "detailed", crop_mode: str = "tight") -> tuple[torch.Tensor, torch.Tensor]:
    key = "detailed_crop" if variant == "detailed" else "source_crop"
    image = region.get(key)
    if image is None:
        if variant == "source":
            raise ValueError("This SMART_REGIONS was created without source crops. Enable store_source_crops on Smart Detailer.")
        raise ValueError("SMART_REGIONS region is missing detailed_crop")
    mask = region.get("mask_crop")
    if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("SMART_REGIONS region crop/mask payload is invalid")
    if crop_mode == "context":
        return image, mask
    lx1, ly1, lx2, ly2 = local_tight_box(region)
    h, w = image.shape[1:3]
    lx1, ly1 = max(0, lx1), max(0, ly1)
    lx2, ly2 = min(w, max(lx1 + 1, lx2)), min(h, max(ly1 + 1, ly2))
    return image[:, ly1:ly2, lx1:lx2, :], mask[:, ly1:ly2, lx1:lx2]


def safe_filename_component(value: Any) -> str:
    text = str(value or "region").strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return (text or "region")[:100]


def refresh_regions(region_data: dict[str, Any], image: torch.Tensor) -> dict[str, Any]:
    """Refresh stored detailed crops from a later/final IMAGE using existing crop coordinates."""
    source = validate_regions(region_data)
    ds = {**source, "regions": [dict(r) for r in source["regions"]]}
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("IMAGE used to refresh SMART_REGIONS must be [N,H,W,C]")
    source_shape = ds.get("source_shape")
    if source_shape and len(source_shape) >= 3:
        if int(source_shape[0]) != int(image.shape[0]) or int(source_shape[1]) != int(image.shape[1]) or int(source_shape[2]) != int(image.shape[2]):
            raise ValueError(f"Final IMAGE shape {list(image.shape)} does not match SMART_REGIONS source shape {source_shape}")
    for region in ds["regions"]:
        idx = int(region.get("image_index", 0))
        cx1, cy1, cx2, cy2 = map(int, region["crop"])
        if not (0 <= idx < image.shape[0] and 0 <= cx1 < cx2 <= image.shape[2] and 0 <= cy1 < cy2 <= image.shape[1]):
            raise ValueError("Region coordinates fall outside final IMAGE")
        region["detailed_crop"] = image[idx:idx + 1, cy1:cy2, cx1:cx2, :3].detach().float().cpu().clone()
    return ds
