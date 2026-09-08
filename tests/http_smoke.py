import json, os, pathlib, subprocess, sys, tempfile, time, urllib.request
base = pathlib.Path(__file__).resolve().parents[1]
comfy = pathlib.Path(os.environ['COMFYUI_PATH']).resolve()
link = comfy / 'custom_nodes/ComfyUI-SmartDetailer'
assert not link.exists() and not link.is_symlink()
link.symlink_to(base, target_is_directory=True)
proc = None
try:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        shared = tmp / 'shared'
        shared.mkdir()
        (shared / 'face.pt').touch()
        config = tmp / 'paths.yaml'
        config.write_text(f'shared:\n  ultralytics_bbox: {shared}\n')
        with (tmp / 'server.log').open('w+') as log:
            try:
                proc = subprocess.Popen([sys.executable, 'main.py', '--cpu', '--listen', '127.0.0.1', '--port', '8191', '--disable-auto-launch', '--disable-api-nodes', '--disable-all-custom-nodes', '--whitelist-custom-nodes', 'ComfyUI-SmartDetailer', '--extra-model-paths-config', str(config), '--output-directory', str(tmp / 'output')], cwd=comfy, stdout=log, stderr=log)
                def request(path, data=None):
                    req = urllib.request.Request('http://127.0.0.1:8191' + path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=5) as response:
                        return json.load(response)
                for _ in range(160):
                    assert proc.poll() is None, 'ComfyUI stopped'
                    try:
                        info = request('/object_info')
                        break
                    except OSError:
                        time.sleep(.25)
                else:
                    raise AssertionError('ComfyUI startup timed out')
                assert len([name for name in info if name.startswith('SmartDetailer')]) == 10
                main = info['SmartDetailer']['input']
                for name in ['label_filter', 'noise_mask_feather', 'control_net', 'control_image', 'vae_mode', 'conditioning_overrides']:
                    assert name in main['optional'], name
                assert main['required']['mask_grow'][1]['min'] == -128
                models = info['SmartDetailerYOLODetector']['input']['required']['model_name'][1]['options']
                assert 'ultralytics/bbox/face.pt' in models, models
                prompt = json.loads((base / 'examples/mask_fusion_api.json').read_text())
                sys.path.insert(0, str(base / 'tools'))
                from benchmark import run_workflow
                first = run_workflow('http://127.0.0.1:8191', prompt, ['3'], 30)
                assert first['timing_eligible'] and first['server_execution_seconds'] is not None
                second = run_workflow('http://127.0.0.1:8191', prompt, ['3'], 30)
                assert not second['timing_eligible'] and second['cached_target_nodes'] == ['3']
                assert list((tmp / 'output').glob('smart_detailer_smoke*.png'))
                print('PASS: ten schemas, new controls, shared detector path, queued workflow, PNG output and benchmark cache detection')
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-12000:], file=sys.stderr)
                raise

finally:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    link.unlink()
