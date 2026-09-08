# Workflows and benchmarks

`mask_fusion_api.json` is a checkpoint-free ComfyUI API workflow. The HTTP smoke test loads and executes it. Files created by the benchmark tool also use API format, not canvas format.

## Prepare a trained-model run

Place an image and a grayscale region mask in ComfyUI's input directory. White marks editable pixels. Use a checkpoint already installed in ComfyUI; no models are downloaded by this tool.

```sh
python tools/benchmark.py prepare --image portrait.png --mask face-mask.png --checkpoint illustrious.safetensors --positive "character description, detailed eyes" --negative "blur" --output runs/smart.json
```

Replace those example filenames with your installed files. For a character LoRA, add `--lora character.safetensors --lora-strength 0.8`. The generated workflow applies the LoRA to both MODEL and CLIP before encoding the prompts. Edit sampling settings in the JSON to match the model. Initial mask growth and feather are zero so the supplied mask also defines the exact allowed edit area.

Start ComfyUI with `--cache-none` for timed repetitions, leave its queue idle, then:

```sh
python tools/benchmark.py run runs/smart.json --target-node 7 --output runs/smart-result.json
```

The result includes the exact workflow, its hash, server/device information, output file records and observed timings. Existing reports are never overwritten. Cached target nodes make a run ineligible for timing and cause exit code 2 after saving the evidence. Wall time includes HTTP polling/queue overhead; server execution time covers the whole workflow, not just sampling. Neither measures peak VRAM.

## Compare implementations

Use the same source image, masks, checkpoint, VAE, LoRA strengths, prompts, seed list and sampling settings. For FaceDetailer/Impact, export an API workflow using those same regions through its supported mask/SEGS path. Run that file with `--target-node` pointing to its detailer. Comparing each tool's detector output separately measures detection as well as detailing; keep that as a separate experiment.

Run one warm-up, then at least five uncached repetitions per configuration. Retain each workflow/result and compare median timings on the same machine with no competing jobs. Log model hashes, extension commits, precision and attention backend alongside results; filenames alone do not identify weights. Report peak VRAM only when measured in the server process with a profiler, including its measurement method.

Test a fixed set covering faces, eyes, hands, feet, multiple characters, occlusions, image-edge regions and overlapping masks. Include the actual Pony/Illustrious checkpoints and character LoRAs you use. Keep two comparisons separate: matched settings for implementation effects, and explicitly documented tuned settings for each tool's best attainable results.

Inspect full images and crops in a shuffled, blinded order for identity, requested detail, seams, texture consistency and unintended changes. Retain failures. Pixel difference is not a perceptual-quality or anatomy score.

```sh
python tools/benchmark.py compare --source portrait.png --candidate result.png --mask face-mask.png --output runs/pixel-check.json
```

The comparison reports exact changes outside the allowed-edit mask. When enabling growth or feather, supply a mask covering the full intended blend support; otherwise legitimate edits will be counted as outside changes. An all-white mask has no outside pixels, so outside error values are reported as null.

## Multiple models and class conditioning

For another model pass, duplicate the checkpoint/CLIP/detailer section. Connect the first detailer's IMAGE to the next IMAGE and its SMART_REGIONS to `previous_regions`. Supply conditioning from the next model's CLIP.

When classes share a model, use **Smart Detailer Conditioning** to associate encoded positive/negative conditioning with exact labels. Chain its `conditioning` output into another instance's `previous` input, then connect the final mapping to `conditioning_overrides`. Unmatched labels use the detailer's normal prompts. Duplicate assignments raise an error. The integration tests execute this routing and verify that seeds and previous mappings remain unchanged.
