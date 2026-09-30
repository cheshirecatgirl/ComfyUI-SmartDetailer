import argparse
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
from test_detailer import n, comfy_nodes, ROOT
from comfy_extras.nodes_mask import ImageToMask

spec=importlib.util.spec_from_file_location('benchmark',ROOT/'tools/benchmark.py')
benchmark=importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class WorkflowTests(unittest.TestCase):
    def test_node_ids_remain_compatible(self):
        import asyncio
        classes=asyncio.run(n.SmartDetailerExtension().get_node_list())
        self.assertEqual({cls.GET_SCHEMA().node_id for cls in classes}, {
            'SmartDetailer','SmartDetailerConditioning','SmartDetailerExtract','SmartDetailerSave',
            'SmartDetailerMerge','SmartDetailerYOLODetector','SmartDetailerMaskToBoundingBoxes',
            'SmartDetailerFlorence2JSONToBoundingBoxes','SmartDetailerMaskFusion','SmartDetailerInfo'})
        main=n.SmartDetailer.GET_SCHEMA()
        self.assertEqual([output.io_type for output in main.outputs],['IMAGE','SMART_REGIONS'])
        self.assertEqual([output.io_type for output in n.YOLODetector.GET_SCHEMA().outputs],['BOUNDING_BOX','MASK','STRING'])

    def test_generated_workflow_links_and_required_inputs(self):
        for lora in (None,'character.safetensors'):
            args=argparse.Namespace(image='portrait.png',mask='mask.png',checkpoint='model.safetensors',
                positive='portrait',negative='',seed=19,lora=lora,lora_strength=.8)
            workflow=benchmark.prepare(args)
            classes={**comfy_nodes.NODE_CLASS_MAPPINGS,'SmartDetailer':n.SmartDetailer,'ImageToMask':ImageToMask}
            for node in workflow.values():
                cls=classes[node['class_type']]
                inputs=cls.INPUT_TYPES()
                self.assertTrue(set(inputs.get('required',{})) <= set(node['inputs']))
                for key,value in node['inputs'].items():
                    if isinstance(value,list):
                        upstream=classes[workflow[value[0]]['class_type']]
                        source_type=upstream.GET_SCHEMA().outputs[value[1]].io_type if hasattr(upstream,'GET_SCHEMA') else upstream.RETURN_TYPES[value[1]]
                        expected=(inputs.get('required',{}).get(key) or inputs.get('optional',{}).get(key))[0]
                        self.assertEqual(source_type,expected)
            if lora:
                self.assertEqual(workflow['5']['inputs']['clip'],['9',1])
                self.assertEqual(workflow['7']['inputs']['model'],['9',0])

    def test_pixel_metrics_detect_outside_change_without_quality_score(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            source=np.zeros((8,8,3),dtype=np.uint8)
            candidate=source.copy(); candidate[3,3]=255; candidate[0,0]=255
            mask=np.zeros((8,8),dtype=np.uint8); mask[2:6,2:6]=255
            for name,data in [('source',source),('candidate',candidate),('mask',mask)]:
                Image.fromarray(data).save(root/(name+'.png'))
            result=benchmark.image_metrics(root/'source.png',root/'candidate.png',root/'mask.png')
            self.assertEqual(result['outside_changed_pixels'],1)
            self.assertEqual(result['outside_max_abs_change'],1)
            self.assertEqual(result['outside_pixels'],48)
            Image.fromarray(np.full((8,8),255,dtype=np.uint8)).save(root/'mask.png')
            self.assertIsNone(benchmark.image_metrics(root/'source.png',root/'candidate.png',root/'mask.png')['outside_max_abs_change'])
