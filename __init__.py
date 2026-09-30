"""ComfyUI extension entry point with deferred runtime imports."""


async def comfy_entrypoint():
    from .nodes import comfy_entrypoint as _entrypoint
    return await _entrypoint()


__all__ = ["comfy_entrypoint"]
