from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
import numpy as np
import torch
import time
import cv2
from PIL import Image



if __name__ == '__main__':
    # download an image
    image = load_image("https://github.com/lllyasviel/ControlNet/raw/main/test_imgs/user_3.png")
    # image = np.array(image)

    if torch.cuda.is_available():
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, revision='f16')
        pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_model_cpu_offload()
    else:
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float32)
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
    # pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision='fp16',controlnet=controlnet)
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # speed up diffusion process with faster scheduler and memory optimization


    for i in range(30):
        t = time.time()
        image = pipe(prompt="hot air balloon, on the beach, sunset, best quality, extremely detailed",
                     negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                     image=image,
                     guidance_scale=9.0,
                     height=512,
                     width=512,
                     num_inference_steps=20
        ).images[0]
        print('cost %s s' % str((time.time() - t) * 1000))
        image.save("generated-%s.png" % i)

