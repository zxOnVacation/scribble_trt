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

    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
    # pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision='fp16',controlnet=controlnet)
    # pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed
    # pipe.enable_xformers_memory_efficient_attention()
    #
    # pipe.enable_model_cpu_offload()

    for i in range(30):
        t = time.time()
        image = pipe(prompt="hot air balloon, on the beach, sunset, best quality, extremely detailed",
                     negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                     image=image,
                     guidance_scale=9.0,
                     height=512,
                     width=512,
                     num_inference_steps=1
        ).images[0]
        image.save("generated-%s.png" % i)
        print('cost %s s' % str((time.time() - t) * 1000))

