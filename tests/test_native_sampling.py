"""Untrained tiny ComfyUI UNet/VAE exercise real sampling, never quality scoring."""
import unittest
import torch
from test_detailer import detail
import comfy.model_base
import comfy.model_patcher
import comfy.supported_models
import comfy.sd
import comfy.controlnet
from comfy.cldm.cldm import ControlNet
from comfy.ldm.models.autoencoder import AutoencoderKL


class NativeSamplingTests(unittest.TestCase):
    def test_native_unet_vae_sampling_and_two_model_passes(self):
        torch.manual_seed(7)
        cpu=torch.device('cpu')
        config=comfy.supported_models.SD15(dict(image_size=32, in_channels=4, model_channels=32,
            out_channels=4, num_res_blocks=1, channel_mult=[1,2], use_spatial_transformer=True,
            transformer_depth=[1,1], transformer_depth_output=[1,1,1,1],
            transformer_depth_middle=1, context_dim=32, dtype=torch.float32))
        model=comfy.model_base.BaseModel(config, device=cpu)
        # ComfyUI deliberately skips initialization because production loads checkpoints.
        # Explicit small deterministic test weights make this fixture well-defined.
        with torch.no_grad():
            for parameter in model.parameters(): parameter.uniform_(-.02,.02)
        patcher=comfy.model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)
        vae_config={'params': {'ddconfig':dict(double_z=True,z_channels=4,resolution=64,
            in_channels=3,out_ch=3,ch=32,ch_mult=[1,2,2,2],num_res_blocks=1,attn_resolutions=[],dropout=0),
            'embed_dim':4}}
        fixture=AutoencoderKL(**vae_config['params'])
        with torch.no_grad():
            for parameter in fixture.parameters(): parameter.uniform_(-.02,.02)
        vae=comfy.sd.VAE(sd=fixture.state_dict(), config=vae_config, device=cpu, dtype=torch.float32)
        conditioning=[[torch.zeros((1,4,32)),{}]]
        image=torch.full((1,64,80,3),.25)
        masks=torch.zeros((1,64,80)); masks[:,20:40,24:48]=1
        kwargs=dict(model=patcher, vae=vae,positive=conditioning,negative=conditioning,
                    denoise=.3,guide_size=64,max_size=128,steps=2)
        out,regions=detail(image=image,mask=masks,**kwargs)
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.equal(out,image))
        self.assertTrue(torch.equal(out[:,:20],image[:,:20]))
        # Switch model families and conditioning: SD1.5 pass -> SDXL pass.
        xl_config = comfy.supported_models.SDXL({**config.unet_config, 'model_channels':64,
            'adm_in_channels':2816, 'num_classes':'sequential'})
        xl_model = comfy.model_base.SDXL(xl_config, device=cpu)
        with torch.no_grad():
            for parameter in xl_model.parameters(): parameter.uniform_(-.02,.02)
        alternate = comfy.model_patcher.ModelPatcher(xl_model, load_device=cpu, offload_device=cpu)
        xl_conditioning = [[torch.zeros((1,4,32)), {'pooled_output':torch.zeros((1,1280))}]]
        kwargs.update(model=alternate, positive=xl_conditioning, negative=xl_conditioning)
        final,accumulated=detail(image=out,mask=masks,previous_regions=regions,**kwargs)
        self.assertEqual(len(accumulated['regions']),2)
        self.assertEqual(final.shape,image.shape)
        self.assertIsNot(patcher,alternate)

        # Native ControlNet and Differential Diffusion on a tiled VAE pass.
        control_model = ControlNet(image_size=32, in_channels=4, hint_channels=3,
            model_channels=32, num_res_blocks=1, channel_mult=[1,2],
            use_spatial_transformer=True, transformer_depth=[1,1],
            transformer_depth_middle=1, context_dim=32, num_heads=4, device=cpu)
        with torch.no_grad():
            for parameter in control_model.parameters(): parameter.uniform_(-.02,.02)
        control = comfy.controlnet.ControlNet(control_model, load_device=cpu)
        controlled, _ = detail(image=image, mask=masks, model=patcher, vae=vae,
            positive=conditioning, negative=conditioning, denoise=.3, guide_size=64,
            max_size=128, steps=2, control_net=control, control_image=image,
            noise_mask_feather=4, vae_mode='tiled')
        self.assertTrue(torch.isfinite(controlled).all())
        self.assertTrue(torch.equal(controlled[:,:20], image[:,:20]))
        self.assertNotIn('denoise_mask_function', patcher.model_options)
        self.assertIsNone(control.cond_hint_original)

if __name__=='__main__': unittest.main()
