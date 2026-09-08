"""Deterministic detector results isolate frame mapping, deduplication and cleanup."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from test_detailer import n, detail

class Boxes:
    def __init__(self, boxes):
        self.xyxy=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        self.conf=torch.ones(len(boxes))*.9
        self.cls=torch.zeros(len(boxes))
    def __len__(self): return len(self.xyxy)

class Detector:
    def __init__(self): self.calls=0; self.offloaded=False
    def predict(self, source, **kw):
        self.calls+=1
        assert source.size == (79,63) # deliberately not a stride multiple
        boxes=[[4,8,24,30],[40,10,60,40]] if self.calls % 2 else []
        return [SimpleNamespace(boxes=Boxes(boxes),masks=None,names={0:'face'})]
    def to(self,device): self.offloaded=(device=='cpu')

class DetectionTests(unittest.TestCase):
    def test_shared_model_paths_and_collision_preserve_all_files(self):
        import folder_paths
        from utils.extra_config import load_extra_path_config
        with TemporaryDirectory() as temp:
            root = Path(temp)
            local, shared, second = root/'local', root/'shared', root/'second'
            files = [shared/'bbox'/'face.pt', shared/'segm'/'person.ONNX',
                     second/'face.pt', local/'ultralytics'/'bbox'/'face.pt']
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (shared/'bbox'/'ignored.safetensors').touch()
            config = root/'extra_model_paths.yaml'
            config.write_text('shared:\n  base_path: '+shared.as_posix()+'\n  ultralytics: .\n  ultralytics_bbox: bbox\n  ultralytics_segm: segm\nsecond:\n  ultralytics_bbox: '+second.as_posix()+'\n')
            with patch.dict(folder_paths.folder_names_and_paths, {}, clear=True), patch.object(folder_paths,'models_dir',str(local)):
                load_extra_path_config(str(config))
                found = n._detector_model_map()
                self.assertEqual(set(found.values()), {str(p.resolve()) for p in files})
                self.assertEqual(len(found), 4)
                self.assertEqual(found['ultralytics/bbox/face.pt'], str(files[0].resolve()))
                for name, path in found.items():
                    self.assertEqual(n._resolve_detector_path(name, ''), path)
                self.assertEqual(n._detector_model_map(), found)

    def test_nms_threshold_confidence_and_different_labels(self):
        class OverlappingDetector(Detector):
            def predict(self, source, **kw):
                self.last_iou = kw['iou']
                boxes = Boxes([[5,5,25,25],[7,5,27,25],[5,5,25,25]])
                boxes.conf = torch.tensor([.7,.95,.8])
                boxes.cls = torch.tensor([0,0,1])
                return [SimpleNamespace(boxes=boxes,masks=None,names={0:'face',1:'eye'})]
        detector = OverlappingDetector()
        with patch.object(n,'_resolve_detector_path',return_value='/tmp/detector.pt'), patch.object(n,'_get_yolo_model',return_value=detector):
            boxes, masks, meta = n.YOLODetector.execute(torch.zeros((1,63,79,3)),'a',.3,.5,640,32).result
            self.assertEqual([b['label'] for b in boxes], ['face','eye'])
            self.assertEqual(boxes[0]['x'], 7)
            self.assertEqual(masks.shape[0], 2)
            self.assertEqual([m['mask_index'] for m in json.loads(meta)], [0,1])
            boxes, _, _ = n.YOLODetector.execute(torch.zeros((1,63,79,3)),'a',.3,.9,640,32).result
            self.assertEqual(len(boxes), 3)
            self.assertEqual(detector.last_iou, .9)

    def test_multi_detector_individual_frames_and_nms(self):
        with TemporaryDirectory() as temp:
            a,b=Path(temp)/'a.pt',Path(temp)/'b.pt'; a.touch(); b.touch()
            models={str(a.resolve()):Detector(),str(b.resolve()):Detector()}
            with patch.object(n,'_resolve_detector_path',side_effect=lambda name,override: str(a) if name=='a' else str(b)), patch.object(n,'_get_yolo_model',side_effect=models.get):
                boxes,masks,metadata=n.YOLODetector.execute(torch.zeros((2,63,79,3)),'a',.3,.5,640,32, additional_models='b\na').result
            self.assertEqual([len(f) for f in boxes],[2,0])
            self.assertEqual(masks.shape,(2,63,79))
            self.assertEqual([m['image_index'] for m in json.loads(metadata)],[0,0])
            self.assertTrue(all(m.offloaded for m in models.values()))
            _,regions=detail(image=torch.zeros((2,63,79,3)),mask=masks,region_metadata=metadata)
            self.assertEqual([r['image_index'] for r in regions['regions']],[0,0])

    def test_detector_failure_offloads_and_propagates(self):
        detector=Detector()
        with TemporaryDirectory() as temp:
            p=Path(temp)/'a.pt';p.touch()
            with patch.object(n,'_resolve_detector_path',return_value=str(p)),patch.object(n,'_get_yolo_model',return_value=detector),patch.object(detector,'predict',side_effect=RuntimeError('inference failed')):
                with self.assertRaisesRegex(RuntimeError,'inference failed'):
                    n.YOLODetector.execute(torch.zeros((1,63,79,3)),'a',.3,.5,640,32)
            self.assertTrue(detector.offloaded)

    def test_union_metadata_is_frame_aligned(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/'a.pt';p.touch()
            with patch.object(n,'_resolve_detector_path',return_value=str(p)),patch.object(n,'_get_yolo_model',return_value=Detector()):
                _,masks,metadata=n.YOLODetector.execute(torch.zeros((2,63,79,3)),'a',.3,.5,640,32,individual_masks=False).result
            self.assertEqual(masks.shape,(2,63,79));self.assertEqual(float(masks[1].sum()),0)
            self.assertEqual([m['image_index'] for m in json.loads(metadata)],[0,1])
            self.assertEqual(len(json.loads(metadata)[0]['detections']),2)

if __name__=='__main__': unittest.main()
