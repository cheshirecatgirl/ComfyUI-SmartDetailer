from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


region_ops = load("smartdetailer_region_ops", ROOT / "region_ops.py")
detail = load("regions_core", ROOT / "regions.py")


def test_mask_bbox_and_resize():
    m = torch.zeros((1, 20, 30))
    m[:, 4:12, 7:21] = 1
    assert region_ops.mask_bbox(m, 0.5)[:4] == (7, 4, 21, 12)
    r = region_ops.resize_mask(m, 40, 60)
    assert r.shape == (1, 40, 60)


def test_regions_crop_and_merge():
    ds = detail.new_regions((1, 20, 30, 3))
    image = torch.rand((1, 12, 18, 3))
    mask = torch.ones((1, 12, 18))
    detail.add_region(ds, {
        "label": "face",
        "bbox": [7, 4, 21, 12],
        "crop": [5, 2, 23, 14],
        "detailed_crop": image,
        "mask_crop": mask,
    })
    tight, tight_mask = detail.crop_region_tensor(ds["regions"][0], "detailed", "tight")
    assert tight.shape == (1, 8, 14, 3)
    assert tight_mask.shape == (1, 8, 14)
    merged = detail.merge_regions([ds, ds])
    assert len(merged["regions"]) == 2
    assert [r["region_id"] for r in merged["regions"]] == [0, 1]


def test_metadata_and_filtering():
    parsed = detail.parse_metadata('[{"label":"hand","confidence":0.9}]')
    assert detail.region_label(0, [], parsed) == "hand"
    assert abs(detail.region_confidence(0, parsed) - 0.9) < 1e-6
    ds = detail.new_regions()
    detail.add_region(ds, {"label": "hand"})
    detail.add_region(ds, {"label": "face"})
    assert len(detail.select_regions(ds, "face")) == 1
    assert "region_count" in detail.summary_json(ds)
    ds2 = detail.new_regions((1, 10, 10, 3))
    detail.add_region(ds2, {"crop": [2, 2, 8, 8], "bbox": [3, 3, 7, 7], "mask_crop": torch.ones((1,6,6)), "detailed_crop": torch.zeros((1,6,6,3))})
    final = torch.ones((1, 10, 10, 3))
    refreshed = detail.refresh_regions(ds2, final)
    assert float(ds2["regions"][0]["detailed_crop"].mean()) == 0.0
    assert float(refreshed["regions"][0]["detailed_crop"].mean()) == 1.0


def test_mask_fusion():
    a = torch.zeros((1, 8, 8)); a[:, 1:5, 1:5] = 1
    b = torch.zeros((1, 8, 8)); b[:, 3:7, 3:7] = 1
    union = region_ops.combine_masks([a, b], "union")
    inter = region_ops.combine_masks([a, b], "intersection")
    assert union.sum() > a.sum()
    assert 0 < inter.sum() < a.sum()


def main():
    test_mask_bbox_and_resize()
    test_regions_crop_and_merge()
    test_metadata_and_filtering()
    test_mask_fusion()
    print("PASS: Smart Detailer core tests")


if __name__ == "__main__":
    main()
