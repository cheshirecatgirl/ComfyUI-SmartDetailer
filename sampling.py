"""One native ComfyUI sampling path; no checkpoint or CLIP ownership."""
from __future__ import annotations

import math
import torch

from .region_ops import resize_mask, normalize_mask


def prepare_detail_model(model, differential=False):
    options = getattr(model, "model_options", {})
    patches = options.get("transformer_options", {}).get("patches", {})
    for group in patches.values():
        for patch in group:
            name = type(patch).__name__
            if name in {"InpaintBlockPatch", "DiffSynthCnetPatch", "ZImageControlPatch"}:
                raise ValueError(f"{name} contains image-specific data. Rebuild it for each crop outside Smart Detailer; a full-image patch cannot be reused here.")
    if differential and options.get("denoise_mask_function") is None:
        from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
        return DifferentialDiffusion.execute(model).result[0]
    return model


class _TiledVAE:
    """Use native tiled encoding in the standard ComfyUI inpaint encoders."""

    def __init__(self, vae):
        self.vae = vae

    def __getattr__(self, name):
        return getattr(self.vae, name)

    def encode(self, pixels):
        return self.vae.encode_tiled(pixels)

    def decode(self, samples):
        return self.vae.decode_tiled(samples)


def sampling_alignment(vae):
    ratio = int(vae.spacial_compression_encode())
    if ratio < 1:
        raise ValueError("VAE spatial compression must be positive")
    # Also accommodates image models that pack 2x2 latent patches.
    return math.lcm(8, ratio * 2)


def validate_conditioning(conditioning, batch):
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise ValueError('Conditioning must contain encoded entries')
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[1], dict):
            raise ValueError('Conditioning entries must contain an embedding and metadata')
        embedding, info = entry
        if any(info.get(key) is not None for key in ("control", "area", "gligen")):
            raise ValueError("Full-image ControlNet, GLIGEN and area conditioning cannot be reused on a resized detail crop. Connect plain text conditioning; regional CONDITIONING masks are supported.")
        if any(info.get(key) is not None for key in ("concat_latent_image", "concat_mask")):
            raise ValueError("Connect conditioning before InpaintModelConditioning and select sampling_mode=inpaint; inpaint inputs must be encoded for each crop.")
        if 'mask' in info:
            mask = normalize_mask(info['mask'])
            if mask.shape[0] not in (1, batch) or min(mask.shape[-2:]) < 1:
                raise ValueError('Conditioning mask batch must be 1 or match IMAGE batch, with nonempty dimensions')


def crop_conditioning(conditioning, image_index, batch, crop, target_hw, source_hw):
    """Copy batch-specific embeddings and transform full-image conditioning masks."""
    validate_conditioning(conditioning, batch)
    result = []
    x1, y1, x2, y2 = crop
    for embedding, source in conditioning:
        info = dict(source)
        if isinstance(embedding, torch.Tensor) and embedding.shape[0] == batch and batch > 1:
            embedding = embedding[image_index:image_index + 1]
        for key in ("pooled_output", "attention_mask"):
            value = info.get(key)
            if isinstance(value, torch.Tensor) and value.shape[0] == batch and batch > 1:
                info[key] = value[image_index:image_index + 1]
        if "mask" in info:
            mask = normalize_mask(info["mask"])
            if mask.shape[0] not in (1, batch):
                raise ValueError("Conditioning mask batch must be 1 or match IMAGE batch")
            mask = mask[0:1] if mask.shape[0] == 1 else mask[image_index:image_index + 1]
            if mask.shape[-2:] != source_hw:
                mask = resize_mask(mask, *source_hw)
            info["mask"] = resize_mask(mask[:, y1:y2, x1:x2], *target_hw)
        result.append([embedding, info])
    return result


@torch.inference_mode()
def sample_crop(api, model, vae, image, mask, positive, negative, seed, steps, cfg,
                sampler_name, scheduler, denoise, mode, noise_mask=None, vae_mode="auto"):
    if vae_mode == "tiled":
        vae = _TiledVAE(vae)
    elif vae_mode != "auto":
        raise ValueError("Invalid vae_mode")
    if mode == "masked":
        latent = {"samples": vae.encode(image), "noise_mask": mask.unsqueeze(1)}
    elif mode == "inpaint":
        positive, negative, latent = api.InpaintModelConditioning().encode(
            positive, negative, image, vae, mask, noise_mask=True)
    elif mode == "erase":
        latent = api.VAEEncodeForInpaint().encode(vae, image, mask, grow_mask_by=0)[0]
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
    if noise_mask is not None:
        latent["noise_mask"] = noise_mask.unsqueeze(1)
    if latent["samples"].ndim != 4:
        raise ValueError("Smart Detailer requires an image VAE with 4D latents; video VAEs are not supported")
    sampled = api.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                 positive, negative, latent, denoise=denoise)[0]
    decoded = vae.decode(sampled["samples"])
    if decoded.ndim != 4 or decoded.shape[0] != 1 or decoded.shape[-1] < 3:
        raise ValueError("VAE must decode a single RGB image crop")
    if not torch.isfinite(decoded).all():
        raise ValueError("VAE decoded NaN or infinity; check model/VAE precision")
    return decoded[..., :3]
