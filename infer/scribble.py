import os
import torch
from utils import *
from PIL import Image
import einops
import random
import tensorrt as trt
from diffusers import UniPCMultistepScheduler


class Scribble():
    def __init__(self, engine_dir):
        self.engine_dir = engine_dir
        self.stream = cuda.Stream()
        self.load_engines()
        self.scheduler = UniPCMultistepScheduler()
        self.device = 'cuda'
        self.dtype = torch.float32

    def load_engines(self):
        # 预先加载各模型
        self.clip = Engine(os.path.join(self.engine_dir, "clip.plan"))
        self.clip.activate()
        self.clip.allocate_buffers({'tokens': (2, 77),
                                    'embeddings': (2, 77, 768)})

        self.control = Engine(os.path.join(self.engine_dir, "control.plan"))
        self.control.activate()
        self.control.allocate_buffers({'noise': (2, 4, 64, 64),
                                       'hint': (2, 3, 512, 512),
                                       't': (2,),
                                       'context': (2, 77, 768),
                                       'dbrs_0': (2, 320, 64, 64),
                                       'dbrs_1': (2, 320, 64, 64),
                                       'dbrs_2': (2, 320, 64, 64),
                                       'dbrs_3': (2, 320, 32, 32),
                                       'dbrs_4': (2, 640, 32, 32),
                                       'dbrs_5': (2, 640, 32, 32),
                                       'dbrs_6': (2, 640, 16, 16),
                                       'dbrs_7': (2, 1280, 16, 16),
                                       'dbrs_8': (2, 1280, 16, 16),
                                       'dbrs_9': (2, 1280, 8, 8),
                                       'dbrs_10': (2, 1280, 8, 8),
                                       'dbrs_11': (2, 1280, 8, 8),
                                       'mbrs_0': (2, 1280, 8, 8)})

        self.unet = Engine(os.path.join(self.engine_dir, "unet.plan"))
        self.unet.activate()
        self.unet.allocate_buffers({'u_noise': (2, 4, 64, 64),
                                    'u_t': (2,),
                                    'u_context': (2, 77, 768),
                                    'u_dbrs_0': (2, 320, 64, 64),
                                    'u_dbrs_1': (2, 320, 64, 64),
                                    'u_dbrs_2': (2, 320, 64, 64),
                                    'u_dbrs_3': (2, 320, 32, 32),
                                    'u_dbrs_4': (2, 640, 32, 32),
                                    'u_dbrs_5': (2, 640, 32, 32),
                                    'u_dbrs_6': (2, 640, 16, 16),
                                    'u_dbrs_7': (2, 1280, 16, 16),
                                    'u_dbrs_8': (2, 1280, 16, 16),
                                    'u_dbrs_9': (2, 1280, 8, 8),
                                    'u_dbrs_10': (2, 1280, 8, 8),
                                    'u_dbrs_11': (2, 1280, 8, 8),
                                    'u_mbrs_0': (2, 1280, 8, 8),
                                    'eps': (2, 4, 64, 64)})

        self.vae = Engine(os.path.join(self.engine_dir, "vae.plan"))
        self.vae.activate()
        self.vae.allocate_buffers({'sample': (1, 4, 64, 64),
                                   'decode_img': (512, 512, 3)})

    def process_img(self, control_img):
        c_img = np.array(Image.open(control_img))
        c_img = resize_image(HWC3(c_img), 512)
        H, W, C = c_img.shape
        detected_map = np.zeros_like(c_img, dtype=np.uint8)
        detected_map[np.min(c_img, axis=2) < 127] = 255
        control = torch.from_numpy(detected_map.copy()).float() / 255.0
        control = control.unsqueeze(0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()  # 1 3 512 512
        return control

    def clip_infer(self, text_p, text_n):
        tokens_p = tokenize(text_p)
        tokens_n = tokenize(text_n)
        tokens = torch.cat([tokens_n, tokens_p]).int()
        tokens_inp = cuda.DeviceView(ptr=tokens.data_ptr(), shape=tokens.shape, dtype=np.int32)
        embeddings = self.clip.infer({"tokens": tokens_inp}, self.stream)['embeddings']
        return embeddings # 2 77 768

    def control_infer(self, noise, hint, t, context):
        noise_inp = cuda.DeviceView(ptr=noise.data_ptr(), shape=noise.shape, dtype=np.float32)
        hint_inp = cuda.DeviceView(ptr=hint.data_ptr(), shape=hint.shape, dtype=np.float32)
        t_inp = cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=np.float32)
        context_inp = cuda.DeviceView(ptr=context.data_ptr(), shape=context.shape, dtype=np.float32)
        control_out = self.control.infer({'noise': noise_inp, 'hint': hint_inp, 't': t_inp, 'context': context_inp}, self.stream)
        return control_out

    def unet_infer(self, noise, t, context, c):
        print(c)
        noise_inp = cuda.DeviceView(ptr=noise.data_ptr(), shape=noise.shape, dtype=np.float32)
        t_inp = cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=np.float32)
        context_inp = cuda.DeviceView(ptr=context.data_ptr(), shape=context.shape, dtype=np.float32)
        dbrs0_inp = cuda.DeviceView(ptr=c['dbrs_0'].data_ptr(), shape=(2, 320, 64, 64), dtype=np.float32)
        dbrs1_inp = cuda.DeviceView(ptr=c['dbrs_1'].data_ptr(), shape=(2, 320, 64, 64), dtype=np.float32)
        dbrs2_inp = cuda.DeviceView(ptr=c['dbrs_2'].data_ptr(), shape=(2, 320, 64, 64), dtype=np.float32)
        dbrs3_inp = cuda.DeviceView(ptr=c['dbrs_3'].data_ptr(), shape=(2, 320, 32, 32), dtype=np.float32)
        dbrs4_inp = cuda.DeviceView(ptr=c['dbrs_4'].data_ptr(), shape=(2, 640, 32, 32), dtype=np.float32)
        dbrs5_inp = cuda.DeviceView(ptr=c['dbrs_5'].data_ptr(), shape=(2, 640, 32, 32), dtype=np.float32)
        dbrs6_inp = cuda.DeviceView(ptr=c['dbrs_6'].data_ptr(), shape=(2, 640, 16, 16), dtype=np.float32)
        dbrs7_inp = cuda.DeviceView(ptr=c['dbrs_7'].data_ptr(), shape=(2, 1280, 16, 16), dtype=np.float32)
        dbrs8_inp = cuda.DeviceView(ptr=c['dbrs_8'].data_ptr(), shape=(2, 1280, 16, 16), dtype=np.float32)
        dbrs9_inp = cuda.DeviceView(ptr=c['dbrs_9'].data_ptr(), shape=(2, 1280, 8, 8), dtype=np.float32)
        dbrs10_inp = cuda.DeviceView(ptr=c['dbrs_10'].data_ptr(), shape=(2, 1280, 8, 8), dtype=np.float32)
        dbrs11_inp = cuda.DeviceView(ptr=c['dbrs_11'].data_ptr(), shape=(2, 1280, 8, 8), dtype=np.float32)
        mbrs0_inp = cuda.DeviceView(ptr=c['mbrs_0'].data_ptr(), shape=(2, 1280, 8, 8), dtype=np.float32)
        eps = self.unet.infer({'u_noise': noise_inp, 'u_t': t_inp, 'u_context': context_inp, 'u_dbrs_0': dbrs0_inp,
                               'u_dbrs_1': dbrs1_inp,'u_dbrs_2': dbrs2_inp, 'u_dbrs_3': dbrs3_inp,
                               'u_dbrs_4': dbrs4_inp, 'u_dbrs_5': dbrs5_inp,'u_dbrs_6': dbrs6_inp,
                               'u_dbrs_7': dbrs7_inp, 'u_dbrs_8': dbrs8_inp, 'u_dbrs_9': dbrs9_inp,
                               'u_dbrs_10': dbrs10_inp, 'u_dbrs_11': dbrs11_inp, 'u_mbrs_0': mbrs0_inp}, self.stream)['eps']
        return eps

    def vae_infer(self, latent):
        sample_inp = cuda.DeviceView(ptr=latent.data_ptr(), shape=latent.shape, dtype=np.float32)
        decode_img = self.vae.infer({"sample": sample_inp}, self.stream)['decode_img']
        return decode_img

    def infer(self, prompts, neg_prompts, control, seed=None, scale=9.0, steps=20):
        control = self.process_img(control).float().to(self.device)
        if seed is None:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.scheduler.set_timesteps(steps)
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER) as runtime:
            embeddings = self.clip_infer(neg_prompts, prompts)
            latents = torch.randn([1, 4, 64, 64], device=self.device, dtype=self.dtype, generator=generator)
            latents = latents * self.scheduler.init_noise_sigma
            torch.cuda.synchronize()
            for step_index, timestep in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_index)
                timestep_input = torch.tensor([timestep.float(), timestep.float()])
                control_input = torch.cat([control] * 2)
                control_outs = self.control_infer(latent_model_input, control_input, timestep_input, embeddings)
                eps = self.unet_infer(latent_model_input, timestep_input, embeddings, control_outs)
                noise_pred_uncond, noise_pred_text = eps.chunk(2)
                noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, latents, step_index, timestep)

            image = self.vae_infer(latents)
        return image


if __name__ == '__main__':
    entry = Scribble("../trt/build/engine")
    img = entry.infer(prompts="hot air balloon, best quality, extremely detailed, sunset, beach",
                      neg_prompts="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                      control="../src/test_imgs/user_3.png",)

    img = Image.fromarray(img)
    img.save("out.jpg")
