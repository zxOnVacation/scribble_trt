import os
import time

import torch
from utils import *
from PIL import Image
import einops
import random
import tensorrt as trt
from diffusers import DDIMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler


class Scribble():
    def __init__(self, engine_dir):
        self.engine_dir = engine_dir
        self.stream = cuda.Stream()
        self.load_engines()
        # self.scheduler = UniPCMultistepScheduler.from_config('./config')
        self.scheduler = DPMSolverMultistepScheduler.from_config('./config')
        self.device = 'cuda'
        self.dtype = torch.float32
        self.tokenizer = tokenize()

    def load_engines(self):
        # 预先加载各模型
        self.clip = Engine(os.path.join(self.engine_dir, "clip.plan"))
        self.clip.activate()
        self.clip.allocate_buffers({'tokens': (2, 77),
                                    'embeddings': (2, 77, 768)})

        self.unet = Engine(os.path.join(self.engine_dir, "unet_control.plan"))
        self.unet.activate()
        self.unet.allocate_buffers({'noise': (2, 4, 64, 64),
                                    't': (2,),
                                    'context': (2, 77, 768),
                                    'hint': (2, 3, 512, 512),
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
        start = time.time()
        tokens_p = self.tokenizer(text_p, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"].to("cuda")
        print('tokenizer_a cost %s ms' % ((time.time() - start) * 1000))
        start = time.time()
        tokens_n = self.tokenizer(text_n, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"].to("cuda")
        print('tokenizer_b cost %s ms' % ((time.time() - start) * 1000))
        tokens = torch.cat([tokens_p, tokens_n]).int()
        start = time.time()
        tokens_inp = cuda.DeviceView(ptr=tokens.data_ptr(), shape=tokens.shape, dtype=np.int32)
        embeddings = self.clip.infer({"tokens": tokens_inp}, self.stream)['embeddings']
        print('tokenizer infer cost %s ms' % ((time.time() - start) * 1000))
        return embeddings # 2 77 768

    def unet_infer(self, noise, t, context, hint):
        noise_inp = cuda.DeviceView(ptr=noise.data_ptr(), shape=noise.shape, dtype=np.float32)
        t_inp = cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=np.float32)
        context_inp = cuda.DeviceView(ptr=context.data_ptr(), shape=context.shape, dtype=np.float32)
        hint_inp = cuda.DeviceView(ptr=hint.data_ptr(), shape=hint.shape, dtype=np.float32)
        eps = self.unet.infer({'noise': noise_inp, 't': t_inp, 'context': context_inp, 'hint': hint_inp}, self.stream)['eps']
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
        start = time.time()
        self.scheduler.set_timesteps(steps)
        print('set timestamps cost %s ms' % ((time.time() - start) * 1000))
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER) as runtime:
            start = time.time()
            embeddings = self.clip_infer(prompts, neg_prompts)
            print('clip infer cost %s ms' % ((time.time() - start) * 1000))
            latents = torch.randn([1, 4, 64, 64], device=self.device, dtype=self.dtype, generator=generator)
            latents = latents * self.scheduler.init_noise_sigma
            for step_index, timestep in enumerate(self.scheduler.timesteps):
                start = time.time()
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_index)
                timestep_input = torch.tensor([timestep.float(), timestep.float()]).to(self.device).float()
                control_input = torch.cat([control] * 2)
                eps = self.unet_infer(latent_model_input, timestep_input, embeddings, control_input)
                print('unet cost %s ms' % ((time.time() - start) * 1000))
                start = time.time()
                noise_pred_text, noise_pred_uncond = eps.chunk(2)
                noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                print('step cost %s ms' % ((time.time() - start) * 1000))

            start = time.time()
            image = self.vae_infer(latents)
            print('vae cost %s ms' % ((time.time() - start) * 1000))
        return image


if __name__ == '__main__':
    entry = Scribble("../trt/build/engine")

    for i in range(30):
        t = time.time()
        img = entry.infer(prompts="hot air balloon, best quality, extremely detailed, sunset, beach",
                          neg_prompts="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                          control="../src/test_imgs/user_3.png",
                          steps=20)
        print('cost %s ms' % ((time.time() - t) * 1000))
        img = img.detach().cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img)
        img.save("out_%s.jpg" % i)

