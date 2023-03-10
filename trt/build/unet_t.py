import ctypes
import torch as t
import numpy as np
import tensorrt as trt
from cuda import cudart
logger = trt.Logger(trt.Logger.VERBOSE)
np.random.seed(1532578949)
t.manual_seed(1532578949)
t.cuda.manual_seed_all(1532578949)
t.backends.cudnn.deterministic = True
cudart.cudaDeviceSynchronize()
trt.init_libnvinfer_plugins(logger, "")
ctypes.cdll.LoadLibrary('libnvinfer_plugin.so')
from layers_t import *


def build_network(network, parau, parac, noise, t, context, hint):
    tembu = time_embedding(network, parau, t) # 2 1280
    tembc = time_embedding(network, parac, t) # 2 1280
    hint_in_c = hint_block(network, parac, hint, tembc, context) # 2 320 64 64

    if 1: # 第一层
        noise_in = network.add_convolution(noise, 320, (3, 3), format(parau['input_blocks.0.0.weight']), format(parau['input_blocks.0.0.bias']))
        noise_in.padding = (1, 1) # 2 320 64 64
        in_0 = noise_in
        noise_inc = network.add_convolution(noise, 320, (3, 3), format(parac['input_blocks.0.0.weight']),format(parac['input_blocks.0.0.bias']))
        noise_inc.padding = (1, 1)  # 2 320 64 64
        noise_inc = network.add_elementwise(out(noise_inc), out(hint_in_c), trt.ElementWiseOperation.SUM)  # 2 320 64 64
        noise_inc = network.add_convolution(out(noise_inc), 320, (1, 1), format(parac['zero_convs.0.0.weight']), format(parac['zero_convs.0.0.bias']))  # 2 320 64 64
        inc_0 = noise_inc
    if 2: # 第二层
        noise_in = build_in_0(network, parau, noise_in, 1, [320, 320], tembu) # 2 320 64 64
        noise_in = build_in_1(network, parau, noise_in, 1, [320, 320], context, 64) # 2 320 64 64
        in_1 = noise_in
        noise_inc = build_in_0(network, parac, noise_inc, 1, [320, 320], tembc) # 2 320 64 64
        noise_inc = build_in_1(network, parac, noise_inc, 1, [320, 320], context, 64) # 2 320 64 64
        noise_inc = network.add_convolution(out(noise_inc), 320, (1, 1), format(parac['zero_convs.1.0.weight']), format(parac['zero_convs.1.0.bias']))  # 2 320 64 64
        inc_1 = noise_inc
    if 3:# 第三层
        noise_in = build_in_0(network, parau, noise_in, 2, [320, 320], tembu) # 2 320 64 64
        noise_in = build_in_1(network, parau, noise_in, 2, [320, 320], context, 64) # 2 320 64 64
        in_2 = noise_in
        noise_inc = build_in_0(network, parac, noise_inc, 2, [320, 320], tembc) # 2 320 64 64
        noise_inc = build_in_1(network, parac, noise_inc, 2, [320, 320], context, 64) # 2 320 64 64
        noise_inc = network.add_convolution(out(noise_inc), 320, (1, 1), format(parac['zero_convs.2.0.weight']), format(parac['zero_convs.2.0.bias']))  # 2 320 64 64
        inc_2 = noise_inc
    if 4:# 第四层
        noise_in = network.add_convolution(out(noise_in), 320, (3, 3), format(parau['input_blocks.3.0.op.weight']), format(parau['input_blocks.3.0.op.bias']))  # 2 320 32 32
        noise_in.stride = (2, 2)
        noise_in.padding = (1, 1)
        in_3 = noise_in
        noise_inc = network.add_convolution(out(noise_inc), 320, (3, 3), format(parac['input_blocks.3.0.op.weight']), format(parac['input_blocks.3.0.op.bias']))  # 2 320 32 32
        noise_inc.stride = (2, 2)
        noise_inc.padding = (1, 1)
        noise_inc = network.add_convolution(out(noise_inc), 320, (1, 1), format(parac['zero_convs.3.0.weight']), format(parac['zero_convs.3.0.bias']))  # 2 320 64 64
        inc_3 = noise_inc
    if 5:# 第五层
        noise_in = build_in_0(network, parau, noise_in, 4, [320, 640], tembu, skip=True) # 2 320 64 64
        noise_in = build_in_1(network, parau, noise_in, 4, [640, 640], context, 32) # 2 320 64 64
        in_4 = noise_in
        noise_inc = build_in_0(network, parac, noise_inc, 4, [320, 640], tembc, skip=True) # 2 320 64 64
        noise_inc = build_in_1(network, parac, noise_inc, 4, [640, 640], context, 32) # 2 320 64 64
        noise_inc = network.add_convolution(out(noise_inc), 640, (1, 1), format(parac['zero_convs.4.0.weight']), format(parac['zero_convs.4.0.bias']))  # 2 320 64 64
        inc_4 = noise_inc
    if 6:# 第6层
        noise_in = build_in_0(network, parau, noise_in, 5, [640, 640], tembu)  # 2 320 64 64
        noise_in = build_in_1(network, parau, noise_in, 5, [640, 640], context, 32)  # 2 320 64 64
        in_5 = noise_in
        noise_inc = build_in_0(network, parac, noise_inc, 5, [640, 640], tembc)  # 2 320 64 64
        noise_inc = build_in_1(network, parac, noise_inc, 5, [640, 640], context, 32)  # 2 320 64 64
        noise_inc = network.add_convolution(out(noise_inc), 640, (1, 1), format(parac['zero_convs.5.0.weight']), format(parac['zero_convs.5.0.bias']))  # 2 320 64 64
        inc_5 = noise_inc
    if 7:# 第7层
        noise_in = network.add_convolution(out(noise_in), 640, (3, 3), format(parau['input_blocks.6.0.op.weight']), format(parau['input_blocks.6.0.op.bias']))  # 2 320 32 32
        noise_in.stride = (2, 2)
        noise_in.padding = (1, 1)
        in_6 = noise_in
        noise_inc = network.add_convolution(out(noise_inc), 640, (3, 3), format(parac['input_blocks.6.0.op.weight']), format(parac['input_blocks.6.0.op.bias']))  # 2 320 32 32
        noise_inc.stride = (2, 2)
        noise_inc.padding = (1, 1)
        noise_inc = network.add_convolution(out(noise_inc), 640, (1, 1), format(parac['zero_convs.6.0.weight']), format(parac['zero_convs.6.0.bias']))  # 2 320 64 64
        inc_6 = noise_inc
    if 8: # 第8层
        noise_in = build_in_0(network, parau, noise_in, 7, [640, 1280], tembu, skip=True) # 2 320 64 64
        noise_in = build_in_1(network, parau, noise_in, 7, [1280, 1280], context, 16) # 2 320 64 64
        in_7 = noise_in
        noise_inc = build_in_0(network, parac, noise_inc, 7, [640, 1280], tembc, skip=True) # 2 320 64 64
        noise_inc = build_in_1(network, parac, noise_inc, 7, [1280, 1280], context, 16) # 2 320 64 64
        noise_inc = network.add_convolution(out(noise_inc), 1280, (1, 1), format(parac['zero_convs.7.0.weight']), format(parac['zero_convs.7.0.bias']))  # 2 320 64 64
        inc_7 = noise_inc
    if 9:# 第9层
        noise_in = build_in_0(network, parau, noise_in, 8, [1280, 1280], tembu, skip=False)  # 2 320 64 64
        noise_in = build_in_1(network, parau, noise_in, 8, [1280, 1280], context, 16)  # 2 320 64 64
        in_8 = noise_in
        noise_inc = build_in_0(network, parac, noise_inc, 8, [1280, 1280], tembc, skip=False)  # 2 320 64 64
        noise_inc = build_in_1(network, parac, noise_inc, 8, [1280, 1280], context, 16)  # 2 320 64 64
        noise_inc = network.add_convolution(out(noise_inc), 1280, (1, 1), format(parac['zero_convs.8.0.weight']), format(parac['zero_convs.8.0.bias']))  # 2 320 64 64
        inc_8 = noise_inc
    if 10:# 第10层
        noise_in = network.add_convolution(out(noise_in), 1280, (3, 3), format(parau['input_blocks.9.0.op.weight']), format(parau['input_blocks.9.0.op.bias']))  # 2 320 32 32
        noise_in.stride = (2, 2)
        noise_in.padding = (1, 1)
        in_9 = noise_in
        noise_inc = network.add_convolution(out(noise_inc), 1280, (3, 3), format(parac['input_blocks.9.0.op.weight']), format(parac['input_blocks.9.0.op.bias']))  # 2 320 32 32
        noise_inc.stride = (2, 2)
        noise_inc.padding = (1, 1)
        noise_inc = network.add_convolution(out(noise_inc), 1280, (1, 1), format(parac['zero_convs.9.0.weight']), format(parac['zero_convs.9.0.bias']))  # 2 320 64 64
        inc_9 = noise_inc
    if 11:# 第11层
        noise_in = build_in_0(network, parau, noise_in, 10, [1280, 1280], tembu, skip=False)  # 2 320 64 64
        in_10 = noise_in
        noise_inc = build_in_0(network, parac, noise_inc, 10, [1280, 1280], tembc, skip=False)  # 2 320 64 64
        noise_inc = network.add_convolution(out(noise_inc), 1280, (1, 1), format(parac['zero_convs.10.0.weight']), format(parac['zero_convs.10.0.bias']))  # 2 320 64 64
        inc_10 = noise_inc
    if 12:# 第12层
        noise_in = build_in_0(network, parau, noise_in, 11, [1280, 1280], tembu, skip=False)  # 2 320 64 64
        in_11 = noise_in
        noise_inc = build_in_0(network, parac, noise_inc, 11, [1280, 1280], tembc, skip=False)  # 2 320 64 64
        noise_inc = network.add_convolution(out(noise_inc), 1280, (1, 1), format(parac['zero_convs.11.0.weight']), format(parac['zero_convs.11.0.bias']))  # 2 320 64 64
        inc_11 = noise_inc
    if 13: # mid第1层
        noise_in = build_mid_0(network, parau, noise_in, 0, [1280, 1280], tembu, skip=False)
        noise_inc = build_mid_0(network, parac, noise_inc, 0, [1280, 1280], tembc, skip=False)
    if 14: # mid第2层
        noise_in = build_mid_1(network, parau, noise_in, 1, [1280, 1280], context, 8)
        noise_inc = build_mid_1(network, parac, noise_inc, 1, [1280, 1280], context, 8)
    if 15: # mid第3层
        noise_in = build_mid_0(network, parau, noise_in, 2, [1280, 1280], tembu, skip=False)
        noise_inc = build_mid_0(network, parac, noise_inc, 2, [1280, 1280], tembc, skip=False)
        noise_inc = network.add_convolution(out(noise_inc), 1280, (1, 1), format(parac['middle_block_out.0.weight']), format(parac['middle_block_out.0.bias']))
        noise_in = network.add_elementwise(out(noise_in), out(noise_inc), trt.ElementWiseOperation.SUM) # 2 1280 8 8
    if 16: # up-0
        c_in = network.add_elementwise(out(in_11), inc_11, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 2 2560 8 8
        noise_in = build_out_0(network, parau, noise_in, 0, [2560, 1280], tembu, skip=True)
    if 17: # up-1
        c_in = network.add_elementwise(out(in_10), inc_10, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)])
        noise_in = build_out_0(network, parau, noise_in, 1, [2560, 1280], tembu, skip=True)
    if 18: # up-2
        c_in = network.add_elementwise(out(in_9), inc_9, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)])
        noise_in = build_out_0(network, parau, noise_in, 2, [2560, 1280], tembu, skip=True)
        noise_in = up_trt(network, parau, 2, noise_in, [1280])
    if 19: # up-3
        c_in = network.add_elementwise(out(in_8), inc_8, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, parau, noise_in, 3, [2560, 1280], tembu, skip=True)
        noise_in = build_out_1(network, parau, noise_in, 3, [1280, 1280], context, 16)
    if 20: # up-4
        c_in = network.add_elementwise(out(in_7), inc_7, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, parau, noise_in, 4, [2560, 1280], tembu, skip=True)
        noise_in = build_out_1(network, parau, noise_in, 4, [1280, 1280], context, 16)
    if 21: # up-5
        c_in = network.add_elementwise(out(in_6), inc_6, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, parau, noise_in, 5, [1920, 1280], tembu, skip=True)
        noise_in = build_out_1(network, parau, noise_in, 5, [1280, 1280], context, 16)
        noise_in = up_trt(network, parau, 5, noise_in, [1280])
    if 22: # up-6
        c_in = network.add_elementwise(out(in_5), inc_5, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)])
        noise_in = build_out_0(network, parau, noise_in, 6, [1920, 640], tembu, skip=True)
        noise_in = build_out_1(network, parau, noise_in, 6, [640, 640], context, 32)
    if 23: # up-7
        c_in = network.add_elementwise(out(in_4), inc_4, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)])
        noise_in = build_out_0(network, parau, noise_in, 7, [1280, 640], tembu, skip=True)
        noise_in = build_out_1(network, parau, noise_in, 7, [640, 640], context, 32)
    if 24: # up-8
        c_in = network.add_elementwise(out(in_3), inc_3, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, parau, noise_in, 8, [960, 640], tembu, skip=True)
        noise_in = build_out_1(network, parau, noise_in, 8, [640, 640], context, 32)
        noise_in = up_trt(network, parau, 8, noise_in, [640])
    if 25: # up-9
        c_in = network.add_elementwise(out(in_2), inc_2, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, parau, noise_in, 9, [960, 320], tembu, skip=True)
        noise_in = build_out_1(network, parau, noise_in, 9, [320, 320], context, 64)
    if 26: # up-10
        c_in = network.add_elementwise(out(in_1), inc_1, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, parau, noise_in, 10, [640, 320], tembu, skip=True)
        noise_in = build_out_1(network, parau, noise_in, 10, [320, 320], context, 64)
    if 27: # up-11
        c_in = network.add_elementwise(out(in_0), inc_0, trt.ElementWiseOperation.SUM)
        noise_in = network.add_concatenation([out(noise_in), out(c_in)]) # 1 2560 16 16
        noise_in = build_out_0(network, parau, noise_in, 11, [640, 320], tembu, skip=True)
        noise_in = build_out_1(network, parau, noise_in, 11, [320, 320], context, 64)
    if 28:
        noise_in = gn(network, noise_in, parau['out.0.weight'], parau['out.0.bias'])
        noise_in = network.add_convolution(out(noise_in), 4, (3, 3), format(parau['out.2.weight']), format(parau['out.2.bias']))  # 2 320 32 32
        noise_in.padding = (1, 1)
    out(noise_in).name = 'eps'
    network.mark_output(out(noise_in))
    return network


def unet_trt():
    import os
    paraFileUnet = './weights/unet.npz'
    paraFileControl = './weights/control.npz'
    bUseFP16Mode = True
    bUseTimeCache = True
    timeCacheFile = './unet_control.cache'
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
    t = network.add_input("t", trt.float32, [2, ])
    context = network.add_input("context", trt.float32, [2, 77, 768])
    hint = network.add_input("hint", trt.float32, [2, 3, 512, 512])

    profile.set_shape(noise.name, (2, 4, 64, 64), (2, 4, 64, 64), (2, 4, 64, 64))
    profile.set_shape(t.name, (2, ), (2, ), (2, ))
    profile.set_shape(context.name, (2, 77, 768), (2, 77, 768), (2, 77, 768))
    profile.set_shape(hint.name, (2, 3, 512, 512), (2, 3, 512, 512), (2, 3, 512, 512))
    config.add_optimization_profile(profile)
    paraUnet = np.load(paraFileUnet)
    paraControl = np.load(paraFileControl)

    network = build_network(network, paraUnet, paraControl, noise, t, context, hint)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        return
    with open('./engine/unet_control.plan', "wb") as f:  # 将序列化网络保存为 .plan 文件
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