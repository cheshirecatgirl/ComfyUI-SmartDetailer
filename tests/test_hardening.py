import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from test_detailer import n, detail, ops
from test_detection import Boxes
from types import SimpleNamespace


class HardeningTests(unittest.TestCase):
    def test_comfy_execution_cache_uses_detector_fingerprint(self):
        import asyncio
        import execution
        from comfy_execution.graph import DynamicPrompt
        from test_detailer import comfy_nodes
        with TemporaryDirectory() as temp:
            path=Path(temp)/'detector.pt'; path.write_bytes(b'first')
            def cached_identity():
                prompt={'1':{'class_type':'SmartDetailerYOLODetector', 'inputs':{
                    'model_name':'a','model_path_override':str(path), 'confidence':.3,
                    'iou':.5,'imgsz':640,'max_detections':16}}}
                return asyncio.run(execution.IsChangedCache('test',DynamicPrompt(prompt),None).get('1'))
            with patch.dict(comfy_nodes.NODE_CLASS_MAPPINGS, {'SmartDetailerYOLODetector':n.YOLODetector}):
                first=cached_identity()
                self.assertIsInstance(first,list)
                self.assertEqual(first,cached_identity())
                path.write_bytes(b'replaced weights')
                self.assertNotEqual(first,cached_identity())

    def test_all_selected_confidences_validate_before_sampling(self):
        masks = torch.zeros((2,64,80)); masks[0,8:16,8:16]=1; masks[1,40:48,40:48]=1
        for invalid in (2, None, 'invalid', float('nan')):
            with self.subTest(invalid=invalid), patch.object(n, 'sample_crop') as sample:
                with self.assertRaisesRegex(ValueError, 'confidence'):
                    detail(mask=masks, denoise=.3, region_metadata=json.dumps([{'confidence':.9},{'confidence':invalid}]))
                sample.assert_not_called()

    def test_weight_fingerprint_tracks_all_files_and_missing_weights(self):
        with TemporaryDirectory() as temp:
            a,b=Path(temp)/'a.pt',Path(temp)/'b.pt'
            a.write_bytes(b'a'); b.write_bytes(b'b')
            with patch.object(n, '_resolve_detector_path', side_effect=lambda name, override: str(a if name=='a' else b)):
                fingerprint=lambda: n.YOLODetector.fingerprint_inputs('a', additional_models='b\na')
                first=fingerprint()
                self.assertEqual(len(first),2)
                self.assertEqual(first,fingerprint())
                old=b.stat()
                b.write_bytes(b'changed')
                os.utime(b, ns=(old.st_atime_ns, old.st_mtime_ns+1_000_000))
                self.assertNotEqual(first,fingerprint())
                b.unlink()
                with self.assertRaises(FileNotFoundError): fingerprint()

    def test_compact_segmentation_matches_full_mask_resize_and_union(self):
        mask=torch.zeros((1,63,79)); mask[:,8:14,9:18]=.7; mask[:,10:12,12:15]=1
        packed=n._pack_detector_mask(mask)
        self.assertEqual(packed[0].untyped_storage().nbytes(),6*9*4)
        box={'x':9,'y':8,'width':9,'height':6}
        for shape in ((63,79),(95,121)):
            out=torch.full((1,*shape),.2)
            n._write_detector_mask(out,(box,packed,{}))
            self.assertTrue(torch.allclose(out,torch.maximum(torch.full_like(out,.2),ops.resize_mask(mask,*shape))))
        empty=n._pack_detector_mask(torch.zeros_like(mask))
        self.assertIsNone(empty[0])

    def test_nms_rasterizes_only_surviving_boxes_and_releases_onnx(self):
        class Detector:
            predictor=object()
            def predict(self, **kwargs):
                return [SimpleNamespace(boxes=Boxes([[5,5,20,20]]*50), masks=None, names={0:'face'})]
        detector=Detector()
        with patch.object(n,'_resolve_detector_path',return_value='/tmp/test.onnx'), patch.object(n,'_get_yolo_model',return_value=detector), patch.object(n,'_write_detector_mask',wraps=n._write_detector_mask) as raster:
            _, masks, _=n.YOLODetector.execute(torch.zeros((1,63,79,3)),'a',.3,.5,640,100).result
        self.assertEqual(raster.call_count,1)
        self.assertEqual(masks.shape,(1,63,79))
        self.assertIsNone(detector.predictor)

    def test_per_class_conditioning_fallback_and_immutable_chain(self):
        positive=lambda value: [[torch.full((1,2,4),value),{}]]
        face,hand,default=positive(.2),positive(.7),positive(.9)
        negative=positive(0)
        first=n.SmartDetailerConditioning.execute(' FACE ',face,negative).result[0]
        overrides=n.SmartDetailerConditioning.execute('hand',hand,negative,first).result[0]
        self.assertEqual(set(first['entries']),{'face'})
        with self.assertRaisesRegex(ValueError,'already assigned'):
            n.SmartDetailerConditioning.execute('face',hand,negative,overrides)
        masks=torch.zeros((3,64,80))
        masks[0,8:16,8:16]=1; masks[1,24:32,24:32]=1; masks[2,40:48,40:48]=1
        seen=[]
        def sample(api, model, vae, image, mask, pos, neg, *args):
            seen.append(float(pos[0][0].mean())); return image
        with patch.object(n,'sample_crop',side_effect=sample):
            _,regions=detail(mask=masks,positive=default,negative=negative,region_labels='face,hand,other',denoise=.3,conditioning_overrides=overrides)
        self.assertTrue(torch.allclose(torch.tensor(seen),torch.tensor([.2,.7,.9])))
        self.assertEqual([r['conditioning_override'] for r in regions['regions']],[True,True,False])
        self.assertEqual([r['seed'] for r in regions['regions']],[19,20,21])

    def test_bad_later_class_conditioning_fails_before_first_sample(self):
        bad=[[torch.zeros((1,2,4)),{'control':object()}]]
        overrides=n.SmartDetailerConditioning.execute('hand',bad,bad).result[0]
        masks=torch.zeros((2,64,80)); masks[0,8:16,8:16]=1; masks[1,40:48,40:48]=1
        with patch.object(n,'sample_crop') as sample:
            with self.assertRaisesRegex(ValueError,'Full-image'):
                detail(mask=masks,region_labels='face,hand',denoise=.3,conditioning_overrides=overrides)
            sample.assert_not_called()
