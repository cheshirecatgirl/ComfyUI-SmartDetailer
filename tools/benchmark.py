"""Prepare and run ComfyUI API workflows; measure pixels without scoring aesthetics."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def request(server, path, data=None):
    req = urllib.request.Request(server.rstrip('/') + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def prepare(args):
    def node(kind, **inputs): return {'class_type': kind, 'inputs': inputs}
    workflow = {
        '1': node('LoadImage', image=args.image),
        '2': node('LoadImage', image=args.mask),
        '3': node('ImageToMask', image=['2', 0], channel='red'),
        '4': node('CheckpointLoaderSimple', ckpt_name=args.checkpoint),
        '5': node('CLIPTextEncode', clip=['4', 1], text=args.positive),
        '6': node('CLIPTextEncode', clip=['4', 1], text=args.negative),
        '7': node('SmartDetailer', image=['1', 0], mask=['3', 0], model=['4', 0], vae=['4', 2],
            positive=['5', 0], negative=['6', 0], seed=args.seed, steps=20, cfg=5.0, denoise=.32,
            guide_size=1024, max_size=1536, context_factor=1.6, sampler_name='dpmpp_2m_sde',
            scheduler='karras', mask_grow=0, feather=0, mask_threshold=.35, min_region_area=1,
            max_regions=16, region_order='input_order', mask_batch_mode='auto', upscale_only=True,
            sampling_mode='masked', store_source_crops=True),
        '8': node('SaveImage', images=['7', 0], filename_prefix='smart_detailer_benchmark'),
    }
    if args.lora:
        workflow['9'] = node('LoraLoader', model=['4', 0], clip=['4', 1], lora_name=args.lora,
                             strength_model=args.lora_strength, strength_clip=args.lora_strength)
        workflow['7']['inputs']['model'] = ['9', 0]
        for index in ('5', '6'): workflow[index]['inputs']['clip'] = ['9', 1]
    return workflow


def run_workflow(server, workflow, target_nodes, timeout):
    if not target_nodes or any(node not in workflow for node in target_nodes):
        raise ValueError('Specify at least one valid target node to check sampling was executed')
    before = request(server, '/queue')
    if before.get('queue_running') or before.get('queue_pending'):
        raise RuntimeError('Benchmark needs an idle ComfyUI queue')
    started = time.perf_counter()
    queued = request(server, '/prompt', {'prompt': workflow})
    ident = queued['prompt_id']
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = request(server, '/history/' + ident)
        if ident in history:
            record = history[ident]
            break
        time.sleep(.2)
    else:
        raise TimeoutError(f'Prompt {ident} exceeded {timeout}s; inspect the ComfyUI queue')
    elapsed = time.perf_counter() - started
    status = record.get('status', {})
    if status.get('status_str') != 'success':
        raise RuntimeError(json.dumps(status))
    messages = status.get('messages', [])
    cached = {node for event, data in messages if event == 'execution_cached' for node in data.get('nodes', [])}
    timestamps = {event: data.get('timestamp') for event, data in messages}
    start, end = timestamps.get('execution_start'), timestamps.get('execution_success')
    return {
        'prompt_id': ident, 'wall_seconds': elapsed,
        'server_execution_seconds': (end-start)/1000 if start is not None and end is not None else None,
        'target_nodes': target_nodes, 'cached_target_nodes': sorted(cached.intersection(target_nodes)),
        'timing_eligible': not bool(cached.intersection(target_nodes)),
        'workflow_sha256': hashlib.sha256(json.dumps(workflow, sort_keys=True).encode()).hexdigest(),
        'workflow': workflow, 'system': request(server, '/system_stats'), 'history': record,
    }


def image_metrics(source_path, candidate_path, mask_path):
    import numpy as np
    from PIL import Image
    source = np.asarray(Image.open(source_path).convert('RGB')).astype('float32') / 255
    candidate = np.asarray(Image.open(candidate_path).convert('RGB')).astype('float32') / 255
    allowed = np.asarray(Image.open(mask_path).convert('L')) > 0
    if source.shape != candidate.shape or source.shape[:2] != allowed.shape:
        raise ValueError('Source, candidate and allowed-edit mask must have identical dimensions')
    delta = np.abs(candidate-source)
    outside = delta[~allowed]
    return {'width': source.shape[1], 'height': source.shape[0],
            'outside_pixels': int((~allowed).sum()),
            'outside_changed_pixels': int(np.any(outside > 0, axis=-1).sum()),
            'outside_max_abs_change': float(outside.max()) if outside.size else None,
            'outside_mean_abs_change': float(outside.mean()) if outside.size else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Build a workflow using filenames already installed in ComfyUI')
    for name in ('image', 'mask', 'checkpoint', 'positive'):
        prep.add_argument('--'+name, required=True)
    prep.add_argument('--negative', default='')
    prep.add_argument('--seed', type=int, default=19)
    prep.add_argument('--lora')
    prep.add_argument('--lora-strength', type=float, default=1.0)
    run = commands.add_parser('run', help='Queue a prepared or exported API workflow and record actual results')
    run.add_argument('workflow', type=Path)
    run.add_argument('--server', default='http://127.0.0.1:8188')
    run.add_argument('--target-node', action='append', required=True)
    run.add_argument('--timeout', type=float, default=900)
    compare = commands.add_parser('compare', help='Measure changes outside an allowed-edit mask')
    for name in ('source', 'candidate', 'mask'):
        compare.add_argument('--'+name, required=True, type=Path)
    for command in (prep, run, compare): command.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output already exists; choose a new file to preserve previous results')
    if args.command == 'prepare':
        result = prepare(args)
    elif args.command == 'run':
        if args.timeout <= 0: parser.error('Timeout must be positive')
        workflow = json.loads(args.workflow.read_text(encoding='utf-8'))
        result = run_workflow(args.server, workflow, args.target_node, args.timeout)
    else:
        result = image_metrics(args.source, args.candidate, args.mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        output.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(args.output)
    if args.command == 'run' and not result['timing_eligible']:
        parser.exit(2, 'Target output was cached; timing is ineligible. Restart ComfyUI with --cache-none.\n')


if __name__ == '__main__':
    main()
