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


def build_network(network, para, noise, t, context, dbrs_0, dbrs_1, dbrs_2, dbrs_3, dbrs_4, dbrs_5, dbrs_6, dbrs_7, dbrs_8, dbrs_9, dbrs_10, dbrs_11, mbrs_0):
    temb = time_embedding(network, para, t) # 2 1280

    if 1: # 第一层
        noise_in = network.add_convolution(noise, 320, (3, 3), format(para['input_blocks.0.0.weight']), format(para['input_blocks.0.0.bias']))
        noise_in.padding = (1, 1) # 2 320 64 64
        in_0 = noise_in
    if 2: # 第二层
        noise_in = build_in_0(network, para, noise_in, 1, [320, 320], temb) # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 1, [320, 320], context, 64) # 2 320 64 64
        in_1 = noise_in
    if 3:# 第三层
        noise_in = build_in_0(network, para, noise_in, 2, [320, 320], temb) # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 2, [320, 320], context, 64) # 2 320 64 64
        in_2 = noise_in
    if 4:# 第四层
        noise_in = network.add_convolution(out(noise_in), 320, (3, 3), format(para['input_blocks.3.0.op.weight']), format(para['input_blocks.3.0.op.bias']))  # 2 320 32 32
        noise_in.stride = (2, 2)
        noise_in.padding = (1, 1)
        in_3 = noise_in
    if 5:# 第五层
        noise_in = build_in_0(network, para, noise_in, 4, [320, 640], temb, skip=True) # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 4, [640, 640], context, 32) # 2 320 64 64
        in_4 = noise_in
    if 6:# 第6层
        noise_in = build_in_0(network, para, noise_in, 5, [640, 640], temb)  # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 5, [640, 640], context, 32)  # 2 320 64 64
        in_5 = noise_in
    if 7:# 第7层
        noise_in = network.add_convolution(out(noise_in), 640, (3, 3), format(para['input_blocks.6.0.op.weight']), format(para['input_blocks.6.0.op.bias']))  # 2 320 32 32
        noise_in.stride = (2, 2)
        noise_in.padding = (1, 1)
        in_6 = noise_in
    if 8: # 第8层
        noise_in = build_in_0(network, para, noise_in, 7, [640, 1280], temb, skip=True) # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 7, [1280, 1280], context, 16) # 2 320 64 64
        in_7 = noise_in
    if 9:# 第9层
        noise_in = build_in_0(network, para, noise_in, 8, [1280, 1280], temb, skip=False)  # 2 320 64 64
        noise_in = build_in_1(network, para, noise_in, 8, [1280, 1280], context, 16)  # 2 320 64 64
        in_8 = noise_in
    if 10:# 第10层
        noise_in = network.add_convolution(out(noise_in), 1280, (3, 3), format(para['input_blocks.9.0.op.weight']), format(para['input_blocks.9.0.op.bias']))  # 2 320 32 32
        noise_in.stride = (2, 2)
        noise_in.padding = (1, 1)
        in_9 = noise_in
    if 11:# 第11层
        noise_in = build_in_0(network, para, noise_in, 10, [1280, 1280], temb, skip=False)  # 2 320 64 64
        in_10 = noise_in
    if 12:# 第12层
        noise_in = build_in_0(network, para, noise_in, 11, [1280, 1280], temb, skip=False)  # 2 320 64 64
        in_11 = noise_in
    if 13: # mid第1层
        noise_in = build_mid_0(network, para, noise_in, 0, [1280, 1280], temb, skip=False)
    if 14: # mid第2层
        noise_in = build_mid_1(network, para, noise_in, 1, [1280, 1280], context, 8)
    if 15: # mid第3层
        noise_in = build_mid_0(network, para, noise_in, 2, [1280, 1280], temb, skip=False)
        noise_in = network.add_elementwise(out(noise_in), mbrs_0, trt.ElementWiseOperation.SUM) # 2 1280 8 8
    if 16: # up-0
        c_in = network.add_elementwise(out(in_11), dbrs_11, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 2 2560 8 8
        noise_in = build_out_0(network, para, noise_in, 0, [2560, 1280], temb, skip=True)
    if 17: # up-1
        c_in = network.add_elementwise(out(in_10), dbrs_10, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)])
        noise_in = build_out_0(network, para, noise_in, 1, [2560, 1280], temb, skip=True)
    if 18: # up-2
        c_in = network.add_elementwise(out(in_9), dbrs_9, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)])
        noise_in = build_out_0(network, para, noise_in, 2, [2560, 1280], temb, skip=True)
        noise_in = up_trt(network, para, 2, noise_in, [1280])
    if 19: # up-3
        c_in = network.add_elementwise(out(in_8), dbrs_8, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, para, noise_in, 3, [2560, 1280], temb, skip=True)
        noise_in = build_out_1(network, para, noise_in, 3, [1280, 1280], context, 16)
    if 20: # up-4
        c_in = network.add_elementwise(out(in_7), dbrs_7, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, para, noise_in, 4, [2560, 1280], temb, skip=True)
        noise_in = build_out_1(network, para, noise_in, 4, [1280, 1280], context, 16)
    if 21: # up-5
        c_in = network.add_elementwise(out(in_6), dbrs_6, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, para, noise_in, 5, [1920, 1280], temb, skip=True)
        noise_in = build_out_1(network, para, noise_in, 5, [1280, 1280], context, 16)
        noise_in = up_trt(network, para, 2, noise_in, [1280])
    if 22: # up-6
        c_in = network.add_elementwise(out(in_5), dbrs_5, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)])
        noise_in = build_out_0(network, para, noise_in, 6, [1920, 640], temb, skip=True)
        noise_in = build_out_1(network, para, noise_in, 6, [640, 640], context, 32)
    if 23: # up-7
        c_in = network.add_elementwise(out(in_4), dbrs_4, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)])
        noise_in = build_out_0(network, para, noise_in, 7, [1280, 640], temb, skip=True)
        noise_in = build_out_1(network, para, noise_in, 7, [640, 640], context, 32)
    if 24: # up-8
        c_in = network.add_elementwise(out(in_3), dbrs_3, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, para, noise_in, 8, [960, 640], temb, skip=True)
        noise_in = build_out_1(network, para, noise_in, 8, [640, 640], context, 32)
        # noise_in = up_trt(network, para, 2, noise_in, [640])
    # if 25: # up-9
    #     c_in = network.add_elementwise(out(in_2), dbrs_2, trt.ElementWiseOperation.SUM)
    #     noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
    #     noise_in = build_out_0(network, para, noise_in, 9, [960, 320], temb, skip=True)
    #     noise_in = build_out_1(network, para, noise_in, 9, [320, 320], context, 64)
    # if 26: # up-10
    #     c_in = network.add_elementwise(out(in_1), dbrs_1, trt.ElementWiseOperation.SUM)
    #     noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
    #     noise_in = build_out_0(network, para, noise_in, 10, [640, 320], temb, skip=True)
    #     noise_in = build_out_1(network, para, noise_in, 10, [320, 320], context, 64)
    # if 27: # up-11
    #     c_in = network.add_elementwise(out(in_0), dbrs_0, trt.ElementWiseOperation.SUM)
    #     noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
    #     noise_in = build_out_0(network, para, noise_in, 11, [640, 320], temb, skip=True)
    #     noise_in = build_out_1(network, para, noise_in, 11, [320, 320], context, 64)
    # if 28:
    #     noise_in = gn(network, noise_in, para['out.0.weight'], para['out.0.bias'])
    #     noise_in = network.add_convolution(out(noise_in), 4, (3, 3), format(para['out.2.weight']), format(para['out.2.bias']))  # 2 320 32 32
    #     noise_in.padding = (1, 1)
    out(noise_in).name = 'eps'
    network.mark_output(out(noise_in))
    return network


def unet_trt():
    import os
    paraFile = './weights/unet.npz'
    bUseFP16Mode = True
    bUseTimeCache = True
    timeCacheFile = './unet.cache'
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
    noise = network.add_input("u_noise", trt.float32, [2, 4, 64, 64])
    t = network.add_input("u_t", trt.float32, [2, ])
    context = network.add_input("u_context", trt.float32, [2, 77, 768])
    dbrs_0 = network.add_input("u_dbrs_0", trt.float32, [2, 320, 64, 64])
    dbrs_1 = network.add_input("u_dbrs_1", trt.float32, [2, 320, 64, 64])
    dbrs_2 = network.add_input("u_dbrs_2", trt.float32, [2, 320, 64, 64])
    dbrs_3 = network.add_input("u_dbrs_3", trt.float32, [2, 320, 32, 32])
    dbrs_4 = network.add_input("u_dbrs_4", trt.float32, [2, 640, 32, 32])
    dbrs_5 = network.add_input("u_dbrs_5", trt.float32, [2, 640, 32, 32])
    dbrs_6 = network.add_input("u_dbrs_6", trt.float32, [2, 640, 16, 16])
    dbrs_7 = network.add_input("u_dbrs_7", trt.float32, [2, 1280, 16, 16])
    dbrs_8 = network.add_input("u_dbrs_8", trt.float32, [2, 1280, 16, 16])
    dbrs_9 = network.add_input("u_dbrs_9", trt.float32, [2, 1280, 8, 8])
    dbrs_10 = network.add_input("u_dbrs_10", trt.float32, [2, 1280, 8, 8])
    dbrs_11 = network.add_input("u_dbrs_11", trt.float32, [2, 1280, 8, 8])
    mbrs_0 = network.add_input("u_mbrs_0", trt.float32, [2, 1280, 8, 8])
    profile.set_shape(noise.name, (2, 4, 64, 64), (2, 4, 64, 64), (2, 4, 64, 64))
    profile.set_shape(t.name, (2, ), (2, ), (2, ))
    profile.set_shape(context.name, (2, 77, 768), (2, 77, 768), (2, 77, 768))
    profile.set_shape(dbrs_0.name, (2, 320, 64, 64), (2, 320, 64, 64), (2, 320, 64, 64))
    profile.set_shape(dbrs_1.name, (2, 320, 64, 64), (2, 320, 64, 64), (2, 320, 64, 64))
    profile.set_shape(dbrs_2.name, (2, 320, 64, 64), (2, 320, 64, 64), (2, 320, 64, 64))
    profile.set_shape(dbrs_3.name, (2, 320, 32, 32), (2, 320, 32, 32), (2, 320, 32, 32))
    profile.set_shape(dbrs_4.name, (2, 640, 32, 32), (2, 640, 32, 32), (2, 640, 32, 32))
    profile.set_shape(dbrs_5.name, (2, 640, 32, 32), (2, 640, 32, 32), (2, 640, 32, 32))
    profile.set_shape(dbrs_6.name, (2, 640, 16, 16), (2, 640, 16, 16), (2, 640, 16, 16))
    profile.set_shape(dbrs_7.name, (2, 1280, 16, 16), (2, 1280, 16, 16), (2, 1280, 16, 16))
    profile.set_shape(dbrs_8.name, (2, 1280, 16, 16), (2, 1280, 16, 16), (2, 1280, 16, 16))
    profile.set_shape(dbrs_9.name, (2, 1280, 8, 8), (2, 1280, 8, 8), (2, 1280, 8, 8))
    profile.set_shape(dbrs_10.name, (2, 1280, 8, 8), (2, 1280, 8, 8), (2, 1280, 8, 8))
    profile.set_shape(dbrs_11.name, (2, 1280, 8, 8), (2, 1280, 8, 8), (2, 1280, 8, 8))
    profile.set_shape(mbrs_0.name, (2, 1280, 8, 8), (2, 1280, 8, 8), (2, 1280, 8, 8))
    config.add_optimization_profile(profile)
    para = np.load(paraFile)

    network = build_network(network, para, noise, t, context, dbrs_0, dbrs_1, dbrs_2, dbrs_3, dbrs_4, dbrs_5, dbrs_6, dbrs_7, dbrs_8, dbrs_9, dbrs_10, dbrs_11, mbrs_0)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        return
    with open('./engine/unet.plan', "wb") as f:  # 将序列化网络保存为 .plan 文件
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
    unet_trt()