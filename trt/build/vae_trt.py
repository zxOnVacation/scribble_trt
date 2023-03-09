import ctypes
import torch as t
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPModel
import numpy as np
import tensorrt as trt
from cuda import cudart
import time
import math
logger = trt.Logger(trt.Logger.VERBOSE)
np.random.seed(1532578949)
t.manual_seed(1532578949)
t.cuda.manual_seed_all(1532578949)
t.backends.cudnn.deterministic = True
cudart.cudaDeviceSynchronize()
trt.init_libnvinfer_plugins(logger, "")
ctypes.cdll.LoadLibrary('libnvinfer_plugin.so')
from layers import *


def build_network(network, para, sample):
    scale = network.add_constant((1, 1, 1, 1), format(np.array([1.0 / 0.18215], dtype=np.float32)))
    sample = network.add_elementwise(sample, out(scale), trt.ElementWiseOperation.PROD) # * 1/0.18215
    sample = network.add_convolution(out(sample), 4, (1, 1), format(para["post_quant_conv.weight"]), format(para["post_quant_conv.bias"])) #post_conv
    sample = network.add_convolution(out(sample), 512, (3, 3), format(para["decoder.conv_in.weight"]), format(para["decoder.conv_in.bias"]))
    sample.padding = (1, 1)

    if 1: # mid-1
        sample = vae_mid_res(network, para, sample, 1, [512, 512])
    if 2: # mid-2
        sample = vae_mid_attn(network, para, sample, 1, [512, 512])
    if 3: # mid-3
        sample = vae_mid_res(network, para, sample, 2, [512, 512])
    if 4: # up-1
        sample = vae_up_res(network, para, sample, 3, 0, [512, 512])
        sample = vae_up_res(network, para, sample, 3, 1, [512, 512])
        sample = vae_up_res(network, para, sample, 3, 2, [512, 512])
        sample = up_trt(network, para, 3, sample, [512], vae=True)
    if 5: # up-2
        sample = vae_up_res(network, para, sample, 2, 0, [512, 512])
        sample = vae_up_res(network, para, sample, 2, 1, [512, 512])
        sample = vae_up_res(network, para, sample, 2, 2, [512, 512])
        sample = up_trt(network, para, 2, sample, [512], vae=True)
    if 6: # up-3
        sample = vae_up_res(network, para, sample, 1, 0, [512, 256], skip=True)
        sample = vae_up_res(network, para, sample, 1, 1, [256, 256])
        sample = vae_up_res(network, para, sample, 1, 2, [256, 256])
        sample = up_trt(network, para, 1, sample, [256], vae=True)
    if 7:
        sample = vae_up_res(network, para, sample, 0, 0, [256, 128], skip=True)
        sample = vae_up_res(network, para, sample, 0, 1, [128, 128])
        sample = vae_up_res(network, para, sample, 0, 2, [128, 128]) # 1 128 512 512
    if 8: # last
        sample = gn(network, sample, para['decoder.norm_out.weight'], para['decoder.norm_out.bias'], epsilon=1e-6, bSwish=1)
        sample = network.add_convolution(out(sample), 3, (3, 3), format(para["decoder.conv_out.weight"]), format(para["decoder.conv_out.bias"]))
        sample.padding = (1, 1)
    if 9: # 后处理
        sample = network.add_shuffle(out(sample))
        sample.first_transpose = (0, 2, 3, 1)
        sample.reshape_dims = (512, 512, 3)
        img_scale = network.add_constant((1, 1, 1), format(np.array([0.5 * 255.0], dtype=np.float32)))
        sample = network.add_elementwise(out(sample), out(img_scale), trt.ElementWiseOperation.PROD)
        sample = network.add_elementwise(out(sample), out(img_scale), trt.ElementWiseOperation.SUM)
        sample = network.add_activation(out(sample), trt.ActivationType.CLIP)
        sample.alpha = 0.0
        sample.beta = 255.0

    out(sample).name = 'decode_img'
    network.mark_output(out(sample))
    return network


def vae_trt():
    import os
    paraFile = './weights/vae.npz'
    bUseFP16Mode = True
    bUseTimeCache = True
    timeCacheFile = './vae.cache'
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 7 << 30)
    if bUseFP16Mode:
        config.set_flag(trt.BuilderFlag.FP16)
    timeCache = b""
    if bUseTimeCache and os.path.isfile(timeCacheFile):
        with open(timeCacheFile, "rb") as f:
            timeCache = f.read()
        if timeCache == None:
            print("Failed getting serialized timing cache!")
            return
        print("Succeeded getting serialized timing cache!")

    if bUseTimeCache:
        cache = config.create_timing_cache(timeCache)
        config.set_timing_cache(cache, False)

    #network build
    sample = network.add_input("sample", trt.float32, [1, 4, 64, 64])
    profile.set_shape(sample.name, (1, 4, 64, 64), (1, 4, 64, 64), (1, 4, 64, 64))
    config.add_optimization_profile(profile)
    para = np.load(paraFile)

    network = build_network(network, para, sample)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        return
    with open('./engine/vae.plan', "wb") as f:  # 将序列化网络保存为 .plan 文件
        f.write(engineString)
        print("Succeeded saving .plan file!")

    print('build done')

    if bUseTimeCache:
        timeCache = config.get_timing_cache()
        timeCacheString = timeCache.serialize()
        with open(timeCacheFile, "wb") as f:
            f.write(timeCacheString)
            print("Succeeded saving .cache file!")



if __name__ == '__main__':
    vae_trt()