from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import time
from PIL import Image



if __name__ == '__main__':
    # download an image
    image = load_image("https://github.com/lllyasviel/ControlNet/raw/main/test_imgs/user_3.png")
    # image = np.array(image)

    if torch.cuda.is_available():
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, revision='fp16', safety_checker=None)
        pipe = pipe.to('cuda')
        # pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_model_cpu_offload()
    else:
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float32)
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32)
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
    # pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision='fp16',controlnet=controlnet)
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # speed up diffusion process with faster scheduler and memory optimization


    for i in range(15):
        t = time.time()
        image = pipe(prompt="(hot air balloon:2.0), best quality, extremely detailed",
                     negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                     image=image,
                     guidance_scale=7.5,
                     height=512,
                     width=512,
                     num_inference_steps=100
        ).images[0]
        print('cost %s s' % str((time.time() - t) * 1000))
        image.save("generated-%s.png" % i)

