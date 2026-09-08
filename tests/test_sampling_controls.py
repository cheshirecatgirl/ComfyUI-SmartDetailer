import json
import unittest
from unittest.mock import patch

import torch
import comfy.controlnet

from test_detailer import n, ops, detail, ImageVAE, comfy_nodes
from smartdetailer.sampling import prepare_detail_model


class ModelFixture:
    def __init__(self, options=None):
        self.model_options = dict(options or {})

    def clone(self):
        return ModelFixture(self.model_options)

    def set_model_denoise_mask_function(self, function):
        self.model_options['denoise_mask_function'] = function


class SamplingControlTests(unittest.TestCase):
    def test_noise_feather_is_separate_from_composite_and_clones_model(self):
        model = ModelFixture()
        seen = []
        def sample(api, patched, vae, image, hard, pos, neg, *settings):
            soft = settings[-2]
            self.assertIn('denoise_mask_function', patched.model_options)
            self.assertIsNot(patched, model)
            self.assertTrue(torch.any((soft > 0) & (soft < 1)))
            self.assertFalse(torch.allclose(soft, hard))
            seen.append(soft)
            return torch.ones_like(image)
        image = torch.full((1,128,128,4), .25)
        mask = torch.zeros((1,128,128)); mask[:,48:80,48:80] = 1
        with patch.object(n, 'sample_crop', side_effect=sample):
            a, _ = detail(image=image, mask=mask, model=model, denoise=.3, noise_mask_feather=2, feather=4)
            b, _ = detail(image=image, mask=mask, model=model, denoise=.3, noise_mask_feather=8, feather=4)
        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(torch.equal(a[...,3], image[...,3]))
        self.assertTrue(torch.equal(a[:,:42], image[:,:42]))
        self.assertEqual(model.model_options, {})

    def test_existing_denoising_function_is_preserved(self):
        callback = object()
        model = ModelFixture({'denoise_mask_function': callback})
        self.assertIs(prepare_detail_model(model, True), model)
        self.assertIs(model.model_options['denoise_mask_function'], callback)
        self.assertIs(prepare_detail_model(model, False), model)

    def test_crop_specific_model_patches_are_rejected(self):
        for name in ('InpaintBlockPatch','DiffSynthCnetPatch','ZImageControlPatch'):
            model = ModelFixture({'transformer_options': {'patches': {'input_block_patch':[type(name, (), {})()]}}})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'image-specific'):
                prepare_detail_model(model)

    def test_native_controlnet_hint_crop_batch_schedule_and_ownership(self):
        control = comfy.controlnet.ControlNet()
        control.control_model_wrapped = None  # No weights needed at the hint-attachment boundary.
        image = torch.zeros((2,64,80,3))
        hint = torch.arange(80).float().view(1,1,80,1).expand(2,64,80,3).clone()/100
        hint[1] -= .5
        seen = []
        def sample(model, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, **kw):
            applied = pos[0][1]['control']
            self.assertIs(applied, neg[0][1]['control'])
            self.assertIsNot(applied, control)
            self.assertEqual(applied.strength, .7)
            self.assertEqual(applied.timestep_percent_range, (.2,.8))
            seen.append(applied.cond_hint_original)
            return (latent,)
        with patch.object(comfy_nodes, 'common_ksampler', side_effect=sample):
            _, regions = detail(image=image, denoise=.3, control_net=control, control_image=hint,
                                control_strength=.7, control_start=.2, control_end=.8)
        self.assertEqual(len(seen), 2)
        for got, region in zip(seen, regions['regions']):
            x1,y1,x2,y2 = region['crop']; w,h = region['detail_size']; frame = region['image_index']
            expected = torch.nn.functional.interpolate(hint[frame:frame+1,y1:y2,x1:x2].movedim(-1,1), (h,w), mode='bilinear', align_corners=False)
            self.assertTrue(torch.equal(got, expected))
        self.assertLess(float(seen[1].min()), 0)
        self.assertIsNone(control.cond_hint_original)

    def test_control_validation_and_zero_strength(self):
        control = comfy.controlnet.ControlNet()
        with self.assertRaisesRegex(ValueError, 'both'):
            detail(control_net=control)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            detail(control_net=control, control_image=torch.zeros((1,32,40,3)))
        with self.assertRaisesRegex(ValueError, 'start'):
            detail(control_start=.8, control_end=.2)
        def sample(model, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, **kw):
            self.assertNotIn('control', pos[0][1])
            return (latent,)
        with patch.object(comfy_nodes, 'common_ksampler', side_effect=sample):
            detail(denoise=.3, control_net=control, control_image=torch.zeros((1,64,80,3)), control_strength=0)

    def test_native_encoders_retain_soft_noise_masks_with_tiling(self):
        class TiledCodec(ImageVAE):
            def __init__(self): self.encoded = self.decoded = 0
            def encode(self, image): raise AssertionError('Expected tiled encode')
            def decode(self, samples): raise AssertionError('Expected tiled decode')
            def encode_tiled(self, image):
                self.encoded += 1
                return super().encode(image)
            def decode_tiled(self, samples):
                self.decoded += 1
                return super().decode(samples)
        def sample(model, seed, steps, cfg, sampler_name, scheduler, pos, neg, latent, **kw):
            mask = latent['noise_mask']
            self.assertTrue(torch.any((mask>0) & (mask<1)))
            return (latent,)
        for mode in ('masked','inpaint','erase'):
            codec = TiledCodec()
            with self.subTest(mode=mode), patch.object(comfy_nodes,'common_ksampler',side_effect=sample):
                detail(denoise=.3, model=ModelFixture(), vae=codec, sampling_mode=mode,
                       noise_mask_feather=4, vae_mode='tiled')
            self.assertGreater(codec.encoded, 0)
            self.assertEqual(codec.decoded, 1)

    def test_mask_erosion_at_tight_crop_and_empty_region_limit(self):
        masks = torch.zeros((2,64,80))
        masks[0,10:12,10:12] = 1
        masks[1,20:40,24:48] = 1
        def sample(*args): return torch.ones_like(args[3])
        with patch.object(n,'sample_crop',side_effect=sample):
            out, regions = detail(mask=masks, denoise=.3, mask_grow=-2, context_factor=1,
                                  max_regions=1, region_order='input_order')
        self.assertEqual([r['mask_index'] for r in regions['regions']], [1])
        self.assertEqual(regions['regions'][0]['region_id'], 0)
        self.assertTrue(torch.all(out[:,20:22] == .25))
        self.assertGreater(float(out[:,24:36,28:44].mean()), .99)

    def test_relative_area_filters_and_selection_orders(self):
        masks = torch.zeros((3,64,80))
        masks[0,4:8,4:8] = 1
        masks[1,20:40,50:70] = 1
        masks[2,24:36,32:44] = 1
        metadata = json.dumps([{'confidence':.9},{'confidence':.8},{'confidence':.7}])
        _, filtered = detail(mask=masks, min_region_ratio=.02, max_region_ratio=.04)
        self.assertEqual([r['mask_index'] for r in filtered['regions']], [2])
        for order, expected in [('confidence',[0,1,2]),('left_to_right',[0,2,1]),('center_first',[2,1,0])]:
            _, regions = detail(mask=masks, region_metadata=metadata, region_order=order)
            self.assertEqual([r['mask_index'] for r in regions['regions']], expected)
            self.assertEqual({r['mask_index']:r['seed'] for r in regions['regions']}, {0:19,1:20,2:21})
        with self.assertRaises(ValueError): detail(min_region_ratio=.9, max_region_ratio=.1)


if __name__ == '__main__':
    unittest.main()
