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


def build_network(network, para, noise, hint, t, context):
    temb = time_embedding(network, para, t) # 2 1280
    hint_in = hint_block(network, para, hint, temb, context) # 2 320 64 64

    if 1:
        # 第一层
        noise_in = network.add_convolution(noise, 320, (3, 3), format(para['input_blocks.0.0.weight']), format(para['input_blocks.0.0.bias']))
        noise_in.padding = (1, 1) # 2 320 64 64
        noise_in = network.add_elementwise(out(noise_in), out(hint_in), trt.ElementWiseOperation.SUM) # 2 320 64 64
        out_0 = network.add_convolution(out(noise_in), 320, (1, 1), format(para['zero_convs.0.0.weight']), format(para['zero_convs.0.0.bias']))  # 2 320 64 64
        out(out_0).name = 'dbrs_0'
        network.mark_output(out(out_0))
    if 2:
        # 第二层
        noise_in = build_in_0(network, para, noise_in, 1, [320, 320], temb) # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 1, [320, 320], context, 64) # 2 320 64 64
        out_1 = network.add_convolution(out(noise_in), 320, (1, 1), format(para['zero_convs.1.0.weight']), format(para['zero_convs.1.0.bias']))  # 2 320 64 64
        out(out_1).name = 'dbrs_1'
        network.mark_output(out(out_1))
    if 3:
        # 第三层
        noise_in = build_in_0(network, para, noise_in, 2, [320, 320], temb) # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 2, [320, 320], context, 64) # 2 320 64 64
        out_2 = network.add_convolution(out(noise_in), 320, (1, 1), format(para['zero_convs.2.0.weight']), format(para['zero_convs.2.0.bias']))  # 2 320 64 64
        out(out_2).name = 'dbrs_2'
        network.mark_output(out(out_2))
    if 4:
        # 第四层
        noise_in = network.add_convolution(out(noise_in), 320, (3, 3), format(para['input_blocks.3.0.op.weight']), format(para['input_blocks.3.0.op.bias']))  # 2 320 32 32
        noise_in.stride = (2, 2)
        noise_in.padding = (1, 1)
        out_3 = network.add_convolution(out(noise_in), 320, (1, 1), format(para['zero_convs.3.0.weight']), format(para['zero_convs.3.0.bias']))  # 2 320 64 64
        out(out_3).name = 'dbrs_3'
        network.mark_output(out(out_3))
    if 5:
        # 第五层
        noise_in = build_in_0(network, para, noise_in, 4, [320, 640], temb, skip=True) # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 4, [640, 640], context, 32) # 2 320 64 64
        out_4 = network.add_convolution(out(noise_in), 640, (1, 1), format(para['zero_convs.4.0.weight']), format(para['zero_convs.4.0.bias']))  # 2 320 64 64
        out(out_4).name = 'dbrs_4'
        network.mark_output(out(out_4))
    if 6:
        # 第6层
        noise_in = build_in_0(network, para, noise_in, 5, [640, 640], temb)  # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 5, [640, 640], context, 32)  # 2 320 64 64
        out_5 = network.add_convolution(out(noise_in), 640, (1, 1), format(para['zero_convs.5.0.weight']), format(para['zero_convs.5.0.bias']))  # 2 320 64 64
        out(out_5).name = 'dbrs_5'
        network.mark_output(out(out_5))
    if 7:
        # 第7层
        noise_in = network.add_convolution(out(noise_in), 640, (3, 3), format(para['input_blocks.6.0.op.weight']), format(para['input_blocks.6.0.op.bias']))  # 2 320 32 32
        noise_in.stride = (2, 2)
        noise_in.padding = (1, 1)
        out_6 = network.add_convolution(out(noise_in), 640, (1, 1), format(para['zero_convs.6.0.weight']), format(para['zero_convs.6.0.bias']))  # 2 320 64 64
        out(out_6).name = 'dbrs_6'
        network.mark_output(out(out_6))
    if 8:
        # 第8层
        noise_in = build_in_0(network, para, noise_in, 7, [640, 1280], temb, skip=True) # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 7, [1280, 1280], context, 16) # 2 320 64 64
        out_7 = network.add_convolution(out(noise_in), 1280, (1, 1), format(para['zero_convs.7.0.weight']), format(para['zero_convs.7.0.bias']))  # 2 320 64 64
        out(out_7).name = 'dbrs_7'
        network.mark_output(out(out_7))
    if 9:
        # 第9层
        noise_in = build_in_0(network, para, noise_in, 8, [1280, 1280], temb, skip=False)  # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 8, [1280, 1280], context, 16)  # 2 320 64 64
        out_8 = network.add_convolution(out(noise_in), 1280, (1, 1), format(para['zero_convs.8.0.weight']), format(para['zero_convs.8.0.bias']))  # 2 320 64 64
        out(out_8).name = 'dbrs_8'
        network.mark_output(out(out_8))
    if 10:
        # 第10层
        noise_in = network.add_convolution(out(noise_in), 1280, (3, 3), format(para['input_blocks.9.0.op.weight']), format(para['input_blocks.9.0.op.bias']))  # 2 320 32 32
        noise_in.stride = (2, 2)
        noise_in.padding = (1, 1)
        out_9 = network.add_convolution(out(noise_in), 1280, (1, 1), format(para['zero_convs.9.0.weight']), format(para['zero_convs.9.0.bias']))  # 2 320 64 64
        out(out_9).name = 'dbrs_9'
        network.mark_output(out(out_9))
    if 11:
        # 第11层
        noise_in = build_in_0(network, para, noise_in, 10, [1280, 1280], temb, skip=False)  # 2 320 64 64
        out_10 = network.add_convolution(out(noise_in), 1280, (1, 1), format(para['zero_convs.10.0.weight']), format(para['zero_convs.10.0.bias']))  # 2 320 64 64
        out(out_10).name = 'dbrs_10'
        network.mark_output(out(out_10))
    if 12:
        # 第12层
        noise_in = build_in_0(network, para, noise_in, 11, [1280, 1280], temb, skip=False)  # 2 320 64 64
        out_11 = network.add_convolution(out(noise_in), 1280, (1, 1), format(para['zero_convs.11.0.weight']), format(para['zero_convs.11.0.bias']))  # 2 320 64 64
        out(out_11).name = 'dbrs_11'
        network.mark_output(out(out_11))
    if 13:
        noise_in = build_mid_0(network, para, noise_in, 0, [1280, 1280], temb, skip=False)
    if 14:
        noise_in = build_mid_1(network, para, noise_in, 1, [1280, 1280], context, 8)
    if 15:
        noise_in = build_mid_0(network, para, noise_in, 2, [1280, 1280], temb, skip=False)
        out_12 = network.add_convolution(out(noise_in), 1280, (1, 1), format(para['middle_block_out.0.weight']), format(para['middle_block_out.0.bias']))
        out(out_12).name = 'mbrs_0'
        network.mark_output(out(out_12))
    return network


def control_trt():
    import os
    paraFile = './weights/control.npz'
    bUseFP16Mode = True
    bUseTimeCache = True
    timeCacheFile = './control.cache'
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
    noise = network.add_input("noise", trt.float32, [2, 4, 64, 64])
    hint = network.add_input("hint", trt.float32, [2, 3, 512, 512])
    t = network.add_input("t", trt.float32, [2, ])
    context = network.add_input("context", trt.float32, [2, 77, 768])
    profile.set_shape(noise.name, (2, 4, 64, 64), (2, 4, 64, 64), (2, 4, 64, 64))
    profile.set_shape(hint.name, (2, 3, 512, 512), (2, 3, 512, 512), (2, 3, 512, 512))
    profile.set_shape(t.name, (2, ), (2, ), (2, ))
    profile.set_shape(context.name, (2, 77, 768), (2, 77, 768), (2, 77, 768))
    config.add_optimization_profile(profile)
    para = np.load(paraFile)

    network = build_network(network, para, noise, hint, t, context)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        return
    with open('./engine/control.plan', "wb") as f:  # 将序列化网络保存为 .plan 文件
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
    control_trt()