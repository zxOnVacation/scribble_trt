from PIL import Image
import numpy as np
from utils import *
import torch
import einops
import random
from pytorch_lightning import seed_everything

input_data = np.load('./build/weights/control_input.npz')
noise = torch.from_numpy(np.repeat(input_data['noise'], 2, axis=0)).float().cuda()  # 2 4 64 64
hint = torch.from_numpy(np.repeat(input_data['hint'], 2, axis=0)).float().cuda()  # 2 3 512 512
t = torch.from_numpy(np.repeat(input_data['t'], 2, axis=0)).float().cuda()  # 2


def unet(embeddings, control_outs):
    c = control_outs
    noise_inp = cuda.DeviceView(ptr=noise.data_ptr(), shape=noise.shape, dtype=np.float32)
    t_inp = cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=np.float32)
    context_inp = cuda.DeviceView(ptr=embeddings.float().data_ptr(), shape=embeddings.shape, dtype=np.float32)
    dbrs0_inp = cuda.DeviceView(ptr=c['dbrs_0'].float().data_ptr(), shape=c['dbrs_0'].shape, dtype=np.float32)
    dbrs1_inp = cuda.DeviceView(ptr=c['dbrs_1'].float().data_ptr(), shape=c['dbrs_1'].shape, dtype=np.float32)
    dbrs2_inp = cuda.DeviceView(ptr=c['dbrs_2'].float().data_ptr(), shape=c['dbrs_2'].shape, dtype=np.float32)
    dbrs3_inp = cuda.DeviceView(ptr=c['dbrs_3'].float().data_ptr(), shape=c['dbrs_3'].shape, dtype=np.float32)
    dbrs4_inp = cuda.DeviceView(ptr=c['dbrs_4'].float().data_ptr(), shape=c['dbrs_4'].shape, dtype=np.float32)
    dbrs5_inp = cuda.DeviceView(ptr=c['dbrs_5'].float().data_ptr(), shape=c['dbrs_5'].shape, dtype=np.float32)
    dbrs6_inp = cuda.DeviceView(ptr=c['dbrs_6'].float().data_ptr(), shape=c['dbrs_6'].shape, dtype=np.float32)
    dbrs7_inp = cuda.DeviceView(ptr=c['dbrs_7'].float().data_ptr(), shape=c['dbrs_7'].shape, dtype=np.float32)
    dbrs8_inp = cuda.DeviceView(ptr=c['dbrs_8'].float().data_ptr(), shape=c['dbrs_8'].shape, dtype=np.float32)
    dbrs9_inp = cuda.DeviceView(ptr=c['dbrs_9'].float().data_ptr(), shape=c['dbrs_9'].shape, dtype=np.float32)
    dbrs10_inp = cuda.DeviceView(ptr=c['dbrs_10'].float().data_ptr(), shape=c['dbrs_10'].shape, dtype=np.float32)
    dbrs11_inp = cuda.DeviceView(ptr=c['dbrs_11'].float().data_ptr(), shape=c['dbrs_11'].shape, dtype=np.float32)
    mbrs0_inp = cuda.DeviceView(ptr=c['mbrs_0'].float().data_ptr(), shape=c['mbrs_0'].shape, dtype=np.float32)

    eps = engines['unet'].infer({'noise': noise_inp, 't': t_inp, 'context': context_inp, 'dbrs_0': dbrs0_inp, 'dbrs_1': dbrs1_inp, 'dbrs_2': dbrs2_inp, 'dbrs_3': dbrs3_inp, 'dbrs_4': dbrs4_inp, 'dbrs_5': dbrs5_inp,
                                 'dbrs_6': dbrs6_inp, 'dbrs_7': dbrs7_inp, 'dbrs_8': dbrs8_inp, 'dbrs_9': dbrs9_inp, 'dbrs_10': dbrs10_inp, 'dbrs_11': dbrs11_inp, 'mbrs_0': mbrs0_inp})['eps']
    print(eps)


def control(embeddings):
    # context = torch.from_numpy(input_data['context']).to(dtype='torch.float32', device='cuda') # 2 77 768
    context = embeddings.float().cuda()
    noise_inp = cuda.DeviceView(ptr=noise.data_ptr(), shape=noise.shape, dtype=np.float32)
    hint_inp = cuda.DeviceView(ptr=hint.data_ptr(), shape=hint.shape, dtype=np.float32)
    t_inp = cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=np.float32)
    context_inp = cuda.DeviceView(ptr=context.data_ptr(), shape=context.shape, dtype=np.float32)
    control_out = engines['control'].infer({'noise': noise_inp, 'hint': hint_inp, 't': t_inp, 'context': context_inp})
    return control_out




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
                                     'dbrs_0': (2, 320, 64, 64), 'dbrs_1': (2, 320, 64, 64), 'dbrs_2': (2, 320, 64, 64),
                                     'dbrs_3': (2, 320, 32, 32), 'dbrs_4': (2, 640, 32, 32), 'dbrs_5': (2, 640, 32, 32),
                                     'dbrs_6': (2, 640, 16, 16), 'dbrs_7': (2, 1280, 16, 16), 'dbrs_8': (2, 1280, 16, 16),
                                     'dbrs_9': (2, 1280, 8, 8), 'dbrs_10': (2, 1280, 8, 8), 'dbrs_11': (2, 1280, 8, 8), 'mbrs_0': (2, 1280, 8, 8)})
    unet_engine = Engine("./build/engine/unet.plan")
    unet_engine.activate()
    unet_engine.allocate_buffers({'noise': (2, 4, 64, 64), 't': (2,), 'context': (2, 77, 768), 'dbrs_0': (2, 320, 64, 64), 'dbrs_1': (2, 320, 64, 64), 'dbrs_2': (2, 320, 64, 64),
                                     'dbrs_3': (2, 320, 32, 32), 'dbrs_4': (2, 640, 32, 32), 'dbrs_5': (2, 640, 32, 32), 'dbrs_6': (2, 640, 16, 16), 'dbrs_7': (2, 1280, 16, 16), 'dbrs_8': (2, 1280, 16, 16),
                                     'dbrs_9': (2, 1280, 8, 8), 'dbrs_10': (2, 1280, 8, 8), 'dbrs_11': (2, 1280, 8, 8), 'mbrs_0': (2, 1280, 8, 8), 'eps': (2, 1280, 8 ,8)})

    return {"clip": clip_engine, "control": control_engine, "unet": unet_engine}




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
    control_outs = control(embeddings)
    unet(embeddings, control_outs)
