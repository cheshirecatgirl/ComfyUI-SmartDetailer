"""Real ComfyUI schemas/encoders/saving; sampler doubles isolate boundary regressions."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ['COMFYUI_PATH']).resolve()
sys.path.insert(0, str(COMFY))
import comfy.options
comfy.options.enable_args_parsing()
_original_argv = sys.argv[:]
sys.argv = ['test', '--cpu']
import nodes as comfy_nodes
sys.argv = _original_argv
spec = importlib.util.spec_from_file_location('smartdetailer', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules['smartdetailer'] = package
spec.loader.exec_module(package)
from smartdetailer import nodes as n, region_ops as ops, regions as ds
from comfy_api.latest import io
import folder_paths

torch.set_num_threads(2)

class ImageVAE:
    """Small tensor codec fixture; not a trained VAE or image-quality test."""
    def spacial_compression_encode(self): return 8
    def encode(self, image):
        rgb = F.avg_pool2d(image.movedim(-1, 1), 8)
        return torch.cat([rgb, rgb[:, :1]], 1)
    def decode(self, latent): return F.interpolate(latent[:, :3], scale_factor=8).movedim(1, -1)


def detail(image=None, mask=None, **kw):
    if image is None: image = torch.full((1, 64, 80, 3), 0.25)
    if mask is None:
        mask = torch.zeros((1, image.shape[1], image.shape[2])); mask[:, 20:40, 24:48] = 1
    args = dict(image=image, mask=mask, model=object(), positive=[[torch.zeros((1, 2, 4)), {}]],
                negative=[[torch.zeros((1, 2, 4)), {}]], vae=ImageVAE(), seed=19, steps=2, cfg=1,
                denoise=0, guide_size=128, max_size=256, context_factor=1.6, sampler_name='euler',
                scheduler='normal', mask_grow=0, feather=0, mask_threshold=0.35,
                min_region_area=1, max_regions=16, region_order='large_first', mask_batch_mode='auto',
                upscale_only=True)
    args.update(kw)
    return n.SmartDetailer.execute(**args).result


class DetailerTests(unittest.TestCase):
    def test_class_conditioning_passes_accumulate_without_merge_nodes(self):
        masks = torch.zeros((2,64,80))
        masks[0,8:16,8:16] = 1
        masks[1,40:48,50:58] = 1
        metadata = json.dumps([{'label':'face'}, {'label':'hand'}])
        face = [[torch.full((1,2,4),.2), {}]]
        hand = [[torch.full((1,2,4),.8), {}]]
        seen = []
        def sample(api, model, vae, image, mask, pos, *args):
            seen.append(pos[0][0])
            return torch.ones_like(image)
        with patch.object(n, 'sample_crop', side_effect=sample):
            out, first = detail(mask=masks, region_metadata=metadata, label_filter=' FACE ', positive=face, denoise=.3)
            final, both = detail(image=out, mask=masks, region_metadata=metadata, label_filter='hand', positive=hand, previous_regions=first, denoise=.3)
            unchanged, same = detail(image=final, mask=masks, region_metadata=metadata, label_filter='absent', previous_regions=both, denoise=.3)
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0], face[0][0])
        self.assertIs(seen[1], hand[0][0])
        self.assertEqual([r['label'] for r in both['regions']], ['face','hand'])
        self.assertEqual([r['mask_index'] for r in both['regions']], [0,1])
        self.assertEqual([r['seed'] for r in both['regions']], [19,20])
        self.assertEqual(len(first['regions']), 1)
        self.assertTrue(torch.equal(unchanged, final))
        self.assertEqual(len(same['regions']), 2)

    def test_source_pixel_growth_and_feather_do_not_depend_on_guide_size(self):
        image = torch.full((1,128,128,3), .25)
        mask = torch.zeros((1,128,128)); mask[:,48:80,48:80] = 1
        sizes = []
        def sample(api, model, vae, crop, *args):
            sizes.append(crop.shape[1:3])
            return torch.ones_like(crop)
        with patch.object(n, 'sample_crop', side_effect=sample):
            a, _ = detail(image=image, mask=mask, guide_size=128, max_size=1024, mask_grow=4, feather=8, denoise=.3)
            b, _ = detail(image=image, mask=mask, guide_size=512, max_size=1024, mask_grow=4, feather=8, denoise=.3)
        self.assertNotEqual(sizes[0], sizes[1])
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(torch.equal(a[:, :32], image[:, :32]))
        self.assertGreater(float(a[:,48:80,48:80].mean()), .9)

    def test_all_registered_schemas(self):
        import asyncio
        extension = asyncio.run(package.comfy_entrypoint())
        classes = asyncio.run(extension.get_node_list())
        self.assertEqual(len(classes), 10)
        self.assertEqual(len({c.GET_SCHEMA().node_id for c in classes}), 10)
        import inspect
        for c in classes:
            schema = c.GET_SCHEMA()
            self.assertTrue(schema.node_id.startswith('SmartDetailer'))
            self.assertTrue(schema.category.startswith('Smart Detailer'))
            self.assertIsInstance(c.INPUT_TYPES(), dict)
            parameters = inspect.signature(c.execute).parameters
            for item in schema.inputs:
                self.assertIn(item.id, parameters)
        self.assertEqual([o.io_type for o in n.SmartDetailer.GET_SCHEMA().outputs], ['IMAGE', 'SMART_REGIONS'])

    def test_zero_denoise_exact_and_native_extract(self):
        image = torch.rand((1, 64, 80, 4))
        with patch.object(comfy_nodes, 'common_ksampler', side_effect=AssertionError('must not sample')):
            out, regions = detail(image=image, store_source_crops=True, feather=8)
        self.assertTrue(torch.equal(out, image))
        crop, mask, _, _ = n.SmartDetailerExtract.execute(regions, 0, 'detailed', 'tight').result
        self.assertEqual(crop.shape, (1, 20, 24, 3))
        self.assertEqual(mask.shape, (1, 20, 24))
        self.assertEqual(regions['regions'][0]['mask_crop'].untyped_storage().nbytes(), regions['regions'][0]['mask_crop'].numel()*4)

    def test_masked_sampler_preserves_input_and_model_identity(self):
        seen = []
        model = object()
        def sample(m, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, **kwargs):
            self.assertIs(m, model)
            self.assertTrue(torch.allclose(latent['samples'], torch.full_like(latent['samples'], .25)))
            seen.append((seed, latent['noise_mask'].shape))
            return ({'samples': torch.ones_like(latent['samples'])},)
        image = torch.full((1, 64, 80, 4), .25)
        with patch.object(comfy_nodes, 'common_ksampler', side_effect=sample):
            out, regions = detail(image=image, model=model, denoise=.3)
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(out[:, :20], image[:, :20]))
        self.assertTrue(torch.equal(out[..., 3], image[..., 3]))
        self.assertGreater(float(out[:, 22:38, 26:46, :3].mean()), .9)

    def test_real_inpaint_encoder_per_crop(self):
        def sample(model, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, **kwargs):
            self.assertIn('concat_latent_image', pos[0][1])
            self.assertIn('concat_mask', neg[0][1])
            self.assertEqual(latent['samples'].shape[1], 4)
            return (latent,)
        with patch.object(comfy_nodes, 'common_ksampler', side_effect=sample):
            detail(denoise=.3, sampling_mode='inpaint')

    def test_erase_mode_uses_native_encoder(self):
        with patch.object(comfy_nodes, 'common_ksampler', side_effect=lambda *a, **kw: (a[8],)):
            out, _ = detail(denoise=.3, sampling_mode='erase')
        self.assertGreater(float(out[:, 24:36, 28:44].mean()), .4)

    def test_empty_detections_do_not_sample(self):
        image = torch.rand((2, 33, 47, 3))
        with patch.object(comfy_nodes, 'common_ksampler', side_effect=AssertionError('must not sample')):
            out, regions = detail(image=image, mask=torch.empty((0, 8, 8)), denoise=0)
        self.assertTrue(torch.equal(image, out)); self.assertEqual(regions['regions'], [])
        self.assertIsNone(ops.mask_bbox(torch.zeros((1, 8, 8)), threshold=0))

    def test_no_ambiguous_mapping(self):
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            ops.masks_for_images(torch.ones((3, 8, 8)), 2, 'auto')
        with self.assertRaises(ValueError): ops.masks_for_images(torch.ones((2, 8, 8)), 3, 'one_per_image')
        self.assertEqual(ops.masks_for_images(torch.ones((3, 8, 8)), 2, 'auto', bboxes=[[{}, {}], [{}]]), [(0,0),(0,1),(1,2)])
        self.assertEqual(ops.masks_for_images(torch.ones((2, 8, 8)), 2, 'auto', metadata=[{'image_index':1}, {'image_index':1}]), [(1,0),(1,1)])
        with self.assertRaises(ValueError): ops.masks_for_images(torch.ones((2, 8, 8)), 2, 'auto', metadata=[{'frame':2}, {'frame':0}])

    def test_batched_sam_scores_and_frames(self):
        masks = torch.zeros((3,64,80)); masks[:, 20:40,24:48] = 1
        box = {'x':24,'y':20,'width':24,'height':20,'score':.9}
        _, regions = detail(image=torch.rand((2,64,80,3)), mask=masks, bboxes=[[box, box], [box]])
        self.assertEqual([r['image_index'] for r in regions['regions']], [0,0,1])
        self.assertEqual([r['confidence'] for r in regions['regions']], [.9,.9,.9])

    def test_seed_stable_under_order_change(self):
        masks = torch.zeros((2,64,80)); masks[0,20:30,20:30]=1; masks[1,30:50,30:50]=1
        _, a=detail(mask=masks, region_order='large_first')
        _, b=detail(mask=masks, region_order='small_first')
        self.assertEqual({r['mask_index']:r['seed'] for r in a['regions']}, {r['mask_index']:r['seed'] for r in b['regions']})

    def test_previous_pass_refresh_is_immutable(self):
        image, a = detail()
        original = a['regions'][0]['detailed_crop'].clone()
        _, b = detail(image=torch.ones_like(image), previous_regions=a)
        self.assertEqual(len(b['regions']),2)
        self.assertTrue(torch.equal(a['regions'][0]['detailed_crop'],original))
        self.assertEqual(float(b['regions'][0]['detailed_crop'].mean()),1)
        with self.assertRaises(ValueError): detail(image=torch.ones((1,128,80,3)),previous_regions=a)

    def test_framewise_mask_fusion(self):
        a=torch.zeros((2,8,8)); a[0,:4]=1
        b=torch.zeros((2,8,8)); b[1,4:]=1
        out=ops.combine_masks([a,b],'union')
        self.assertEqual(out.shape,(2,8,8)); self.assertEqual(float(out[0,4:].sum()),0)
        self.assertEqual(ops.combine_masks([a,b],'union',batch_mode='collapse').shape,(1,8,8))
        with self.assertRaises(ValueError): ops.combine_masks([a,torch.ones((3,8,8))],'union')

    def test_florence_frame_and_task_contracts(self):
        raw=[{'bboxes':[[1,2,10,12]], 'labels':['face']},{'bboxes':[], 'labels':[]}]
        boxes, meta=n.Florence2JSONToBoundingBoxes.execute(raw).result
        self.assertEqual(len(boxes),2); self.assertEqual(boxes[1],[])
        self.assertEqual(json.loads(meta)[0]['image_index'],0)
        nested=ops.normalize_florence_payload({'<CAPTION_TO_PHRASE_GROUNDING>': raw[0]})
        self.assertEqual(nested[0]['label'],'face')
        with self.assertRaises(ValueError): ops.normalize_florence_payload({'caption':'no regions'})

    def test_nan_and_bad_metadata_fail(self):
        with self.assertRaises(ValueError): detail(mask=torch.full((1,64,80),float('nan')))
        with self.assertRaises(ValueError): detail(region_metadata='broken json')
        with self.assertRaises(ValueError): ds.validate_regions({'type':'SMART_REGIONS','version':99,'regions':[]})

    def test_real_save_png_rgba_and_no_mutation(self):
        from PIL import Image
        _, regions=detail()
        old=regions['regions'][0]['detailed_crop'].clone()
        with tempfile.TemporaryDirectory() as folder, patch.object(folder_paths,'output_directory',folder):
            n.SmartDetailerSave.GET_SCHEMA()
            result=n.SmartDetailerSave.execute(regions,'smart_detailer','tight',True,True, final_image=torch.ones((1,64,80,3)))
            files=list(Path(folder).rglob('*.png'))
            self.assertEqual(len(files),2)
            rgba=next(p for p in files if '_mask_' not in p.name)
            with Image.open(rgba) as im: self.assertEqual(im.mode,'RGBA'); self.assertEqual(im.size,(24,20))
            self.assertTrue(torch.equal(old,regions['regions'][0]['detailed_crop']))
            with self.assertRaises(Exception): n.SmartDetailerSave.execute(regions,'../../outside','tight',False,False)


if __name__=='__main__': unittest.main()
