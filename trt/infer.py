from PIL import Image
import numpy as np
from utils import *
import torch
import einops
import random
from pytorch_lightning import seed_everything


def control(embeddings):
    input_data = np.load('./build/weights/control_input.npz')
    noise = torch.from_numpy(np.repeat(input_data['noise'], 2, axis=0)).float().cuda() # 2 4 64 64
    hint = torch.from_numpy(np.repeat(input_data['hint'], 2, axis=0)).float().cuda() # 2 3 512 512
    t = torch.from_numpy(np.repeat(input_data['t'], 2, axis=0)).float().cuda() # 2
    # context = torch.from_numpy(input_data['context']).to(dtype='torch.float32', device='cuda') # 2 77 768
    context = embeddings.float().cuda()
    noise_inp = cuda.DeviceView(ptr=noise.data_ptr(), shape=noise.shape, dtype=np.float32)
    hint_inp = cuda.DeviceView(ptr=hint.data_ptr(), shape=hint.shape, dtype=np.float32)
    t_inp = cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=np.float32)
    context_inp = cuda.DeviceView(ptr=context.data_ptr(), shape=context.shape, dtype=np.float32)
    dbrs_1 = engines['control'].infer({'noise': noise_inp, 'hint': hint_inp, 't': t_inp, 'context': context_inp})['dbrs_2']
    print(dbrs_1)




def pre_img(img_path):
    c_img = np.array(Image.open(img_path))
    c_img = resize_image(HWC3(c_img), 512)
    H, W, C = c_img.shape
    detected_map = np.zeros_like(c_img, dtype=np.uint8)
    detected_map[np.min(c_img, axis=2) < 127] = 255
    control = torch.from_numpy(detected_map.copy()).float() / 255.0
    control = control.unsqueeze(0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone() # 1 3 512 512
    return control

def clip(text_a, text_b):
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    tokens = torch.cat([tokens_a, tokens_b]).int() # 牢记要转成int32 好难过
    tokens_inp = cuda.DeviceView(ptr=tokens.data_ptr(), shape=tokens.shape, dtype=np.int32)
    embeddings = engines['clip'].infer({"tokens": tokens_inp})['embeddings']
    return embeddings # 2 77 768


def load_engines():
    clip_engine = Engine("./build/engine/clip.plan")
    clip_engine.activate()
    clip_engine.allocate_buffers({'tokens': (2, 77), 'embeddings': (2, 77, 768)})
    control_engine = Engine("./build/engine/control.plan")
    control_engine.activate()
    control_engine.allocate_buffers({'noise': (2, 4, 64, 64), 'hint': (2, 3, 512, 512), 't': (2,), 'context': (2, 77, 768),
                                     'dbrs_0': (2, 320, 64, 64), 'dbrs_1': (2, 320, 64, 64), 'dbrs_2': (2, 320, 64, 64)})

    return {"clip": clip_engine, "control": control_engine}




if __name__ == '__main__':
    c_img_path = '../src/test_imgs/user_3.png'
    prompt = "hot air balloon, best quality, extremely detailed, sunset, beach"
    neg_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    engines = load_engines()

    c_img = pre_img(c_img_path)
    seed = 9
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    embeddings = clip(prompt, neg_prompt)

    control(embeddings)
