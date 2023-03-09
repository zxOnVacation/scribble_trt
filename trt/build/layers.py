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


def format(weights):
    return trt.Weights(np.ascontiguousarray(weights))

def matrix_mul(network, input_layer, weight, bias, weight_shape, bias_shape, flag=True):
    m1_weight = network.add_constant(weight_shape, format(weight.transpose(1, 0).reshape(weight_shape)))
    m1_out = network.add_matrix_multiply(out(input_layer), trt.MatrixOperation.NONE, out(m1_weight), trt.MatrixOperation.NONE)
    if flag:
        m1_bias = network.add_constant(bias_shape, format(bias.reshape(bias_shape)))
        m1_out = network.add_elementwise(out(m1_out), out(m1_bias), trt.ElementWiseOperation.SUM)
    return m1_out

def silu(network, input_layer):
    silu_sig = network.add_activation(out(input_layer), trt.ActivationType.SIGMOID)
    siluR = network.add_elementwise(out(silu_sig), out(input_layer), trt.ElementWiseOperation.PROD)
    return siluR

def time_embedding(network, para, t):
    freqs = np.exp(-math.log(10000) * torch.arange(start=0, end=160, dtype=torch.float32) / 160.0).reshape(1, 160) # 1 160
    freqs_c = network.add_constant((1, 160), format(freqs))
    ts = network.add_shuffle(t)
    ts.reshape_dims = (2, 1)
    time_freqs_c = network.add_matrix_multiply(out(ts), trt.MatrixOperation.NONE, out(freqs_c), trt.MatrixOperation.NONE)
    time_cos = network.add_unary(out(time_freqs_c), trt.UnaryOperation.COS)
    time_sin = network.add_unary(out(time_freqs_c), trt.UnaryOperation.SIN)
    time_emb = network.add_concatenation([out(time_cos), out(time_sin)])
    time_emb.axis = 1 # 1 320
    time_emb_0 = matrix_mul(network, time_emb, para['time_embed.0.weight'], para['time_embed.0.bias'], (320, 1280), (1, 1280))
    time_emb_silu = silu(network, time_emb_0)
    time_emb_1 = matrix_mul(network, time_emb_silu, para['time_embed.2.weight'], para['time_embed.2.bias'], (1280, 1280), (1, 1280)) # 1 1280
    return time_emb_1


def up_trt(network, para, index, input_layer, ints):
    def resize_trt():
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "ResizeNearest_TRT":
                pLists = []
                pLists.append(trt.PluginField("scale", np.float32(2.0), trt.PluginFieldType.FLOAT32))
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None
    up_out = network.add_plugin_v2([out(input_layer)], resize_trt())
    if index == 2:
        up_out = network.add_convolution(out(up_out), ints[0], (3, 3), format(para['output_blocks.%s.1.conv.weight' % index]), format(para['output_blocks.%s.1.conv.bias' % index]))  # 2 320 32 32
    else:
        up_out = network.add_convolution(out(up_out), ints[0], (3, 3), format(para['output_blocks.%s.2.conv.weight' % index]), format(para['output_blocks.%s.2.conv.bias' % index]))  # 2 320 32 32
    up_out.padding = (1, 1)
    return up_out


def ln(network, inputs, gamma_weights, beta_weights):
    def ln_plugin(epsilon=1e-5, axis=-1):
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "LayerNorm":
                pLists = []
                pLists.append(trt.PluginField("epsilon", np.float32(epsilon), trt.PluginFieldType.FLOAT32))
                pLists.append(trt.PluginField("axis", np.int32(axis), trt.PluginFieldType.INT32))
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None
    gamma_cons = network.add_constant(gamma_weights.shape, format(gamma_weights))
    beta_cons = network.add_constant(beta_weights.shape, format(beta_weights))
    ln_out = network.add_plugin_v2([out(inputs), out(gamma_cons), out(beta_cons)], ln_plugin())
    return ln_out

def gn(network, inputs, gamma_weights, beta_weights, epsilon=1e-5, bSwish=1):
    def gn_plugin(epsilon=1e-5, bSwish=1):
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "GroupNorm":
                pLists = []
                pLists.append(trt.PluginField("epsilon", np.float32(epsilon), trt.PluginFieldType.FLOAT32))
                pLists.append(trt.PluginField("bSwish", np.int32(bSwish), trt.PluginFieldType.INT32))
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None
    gamma_cons = network.add_constant(gamma_weights.shape, format(gamma_weights))
    beta_cons = network.add_constant(beta_weights.shape, format(beta_weights))
    gn_out = network.add_plugin_v2([out(inputs), out(gamma_cons), out(beta_cons)], gn_plugin(epsilon, bSwish))
    return gn_out

def hint_block(network, para, hint, temb, context):
    hint_in = network.add_convolution(hint, 16, (3, 3), format(para['input_hint_block.0.weight']), format(para['input_hint_block.0.bias']))
    hint_in.padding = (1, 1)
    hint_in = silu(network, hint_in)
    hint_in = network.add_convolution(out(hint_in), 16, (3, 3), format(para['input_hint_block.2.weight']), format(para['input_hint_block.2.bias']))
    hint_in.padding = (1, 1)
    hint_in = silu(network, hint_in)
    hint_in = network.add_convolution(out(hint_in), 32, (3, 3), format(para['input_hint_block.4.weight']), format(para['input_hint_block.4.bias']))
    hint_in.padding = (1, 1)
    hint_in.stride = (2, 2)
    hint_in = silu(network, hint_in)
    hint_in = network.add_convolution(out(hint_in), 32, (3, 3), format(para['input_hint_block.6.weight']), format(para['input_hint_block.6.bias']))
    hint_in.padding = (1, 1)
    hint_in = silu(network, hint_in)
    hint_in = network.add_convolution(out(hint_in), 96, (3, 3), format(para['input_hint_block.8.weight']), format(para['input_hint_block.8.bias']))
    hint_in.padding = (1, 1)
    hint_in.stride = (2, 2)
    hint_in = silu(network, hint_in)
    hint_in = network.add_convolution(out(hint_in), 96, (3, 3), format(para['input_hint_block.10.weight']), format(para['input_hint_block.10.bias']))
    hint_in.padding = (1, 1)
    hint_in = silu(network, hint_in)
    hint_in = network.add_convolution(out(hint_in), 256, (3, 3), format(para['input_hint_block.12.weight']), format(para['input_hint_block.12.bias']))
    hint_in.padding = (1, 1)
    hint_in.stride = (2, 2)
    hint_in = silu(network, hint_in)
    hint_in = network.add_convolution(out(hint_in), 320, (3, 3), format(para['input_hint_block.14.weight']), format(para['input_hint_block.14.bias']))
    hint_in.padding = (1, 1)
    return hint_in

def out(layer, i=0):
    return layer.get_output(i)

def ffn(network, para, input_layer, index, ints):
    def geglu():
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "SplitGeLU":
                pLists = []
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None
    noise_in = ln(network, input_layer, para['input_blocks.%s.1.transformer_blocks.0.norm3.weight' % index], para['input_blocks.%s.1.transformer_blocks.0.norm3.bias' % index])  # 2 4096 320
    noise_in = matrix_mul(network, noise_in, para['input_blocks.%s.1.transformer_blocks.0.ff.net.0.proj.weight' % index], para['input_blocks.%s.1.transformer_blocks.0.ff.net.0.proj.bias' % index], (1, ints[0], ints[0]*8), (1, 1, ints[0]*8))
    noise_in = network.add_plugin_v2([out(noise_in)], geglu()) # 2 4096 1280
    noise_in = matrix_mul(network, noise_in, para['input_blocks.%s.1.transformer_blocks.0.ff.net.2.weight' % index], para['input_blocks.%s.1.transformer_blocks.0.ff.net.2.bias' % index], (1, ints[0]*4, ints[0]), (1, 1, ints[0])) # 2 4096 320
    noise_in = network.add_elementwise(out(noise_in), out(input_layer), trt.ElementWiseOperation.SUM)
    return noise_in

def cross_attn(network, para, input_layer, index, ints, context):
    def fmhca():
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "fMHCA":
                pLists = []
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None

    noise_in = ln(network, input_layer, para['input_blocks.%s.1.transformer_blocks.0.norm2.weight' % index], para['input_blocks.%s.1.transformer_blocks.0.norm2.bias' % index]) # 1 4096 320
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, 1, -1, ints[0])
    q = matrix_mul(network, noise_in, para['input_blocks.%s.1.transformer_blocks.0.attn2.to_q.weight' % index], None, (1, 1, ints[0], ints[0]), None, False) # 1 1 4096 320
    q = network.add_shuffle(out(q))
    q.reshape_dims = (2, -1, 8, ints[0]//8)
    kv_context = network.add_shuffle(context)
    kv_context.reshape_dims = (2, 1, 77, 768)
    union_weights = np.zeros((1, 2, 768, ints[0]), dtype=np.float32)
    union_weights[:, 0, :, :] = para["input_blocks.%s.1.transformer_blocks.0.attn2.to_k.weight" % index].transpose(1, 0)
    union_weights[:, 1, :, :] = para["input_blocks.%s.1.transformer_blocks.0.attn2.to_v.weight" % index].transpose(1, 0)
    weights_constant = network.add_constant((1, 2, 768, ints[0]), format(union_weights))
    kv = network.add_matrix_multiply(out(kv_context), trt.MatrixOperation.NONE, out(weights_constant), trt.MatrixOperation.NONE) # 2 2 77 320
    if ints[0] < 1280:
        kv = network.add_shuffle(out(kv))
        kv.reshape_dims = (2, 2, 77, 8, ints[0]//8)
        kv.second_transpose = (0, 2, 3, 1, 4) # 2 77 8 2 40
        noise_in = network.add_plugin_v2([out(q), out(kv)], fmhca()) # 2 4096 4 80
    else:
        # 插件不支持 q 2 1 256 1280  kv 2 2 77 1280
        q = network.add_shuffle(out(q))
        q.reshape_dims = (2, -1, 8, ints[0] // 8)
        q.second_transpose = (0, 2, 1, 3) # 2 8 256 160
        k = network.add_slice(out(kv), (0, 0, 0, 0), (2, 1, 77, ints[0]), (1, 1, 1, 1))
        k = network.add_shuffle(out(k))
        k.reshape_dims = (2, -1, 8, ints[0] // 8)
        k.second_transpose = (0, 2, 1, 3) # 2 8 77 160
        v = network.add_slice(out(kv), (0, 1, 0, 0), (2, 1, 77, ints[0]), (1, 1, 1, 1))
        v = network.add_shuffle(out(v))
        v.reshape_dims = (2, -1, 8, ints[0] // 8)
        v.second_transpose = (0, 2, 1, 3) # 2 8 77 160
        qk = network.add_matrix_multiply(out(q), trt.MatrixOperation.NONE, out(k), trt.MatrixOperation.TRANSPOSE) # 2 8 256 77
        scale = network.add_constant((1, 1, 1, 1), format(np.array(1 / math.sqrt(ints[0] // 8), dtype=np.float32)))
        qk = network.add_elementwise(out(qk), out(scale), trt.ElementWiseOperation.PROD) # 2 8 256 77
        qk = network.add_softmax(out(qk))
        qk.axes = 1 << 3 # 2 8 256 77
        v = network.add_matrix_multiply(out(qk), trt.MatrixOperation.NONE, out(v), trt.MatrixOperation.NONE) # 2 8 256 160
        v = network.add_shuffle(out(v))
        v.first_transpose = (0, 2, 1, 3)
        v.reshape_dims = (2, -1, 8, ints[0]//8)
        noise_in = v
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, -1, ints[0])
    noise_in = matrix_mul(network, noise_in, para['input_blocks.%s.1.transformer_blocks.0.attn2.to_out.0.weight' % index], para['input_blocks.%s.1.transformer_blocks.0.attn2.to_out.0.bias' % index], (1, ints[0], ints[0]), (1, 1, ints[0]))
    noise_in = network.add_elementwise(out(noise_in), out(input_layer), trt.ElementWiseOperation.SUM)
    return noise_in

def self_attn(network, para, input_layer, index, ints):
    def fmha():
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "fMHA_V2":
                pLists = []
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None

    noise_in = ln(network, input_layer, para['input_blocks.%s.1.transformer_blocks.0.norm1.weight' % index], para['input_blocks.%s.1.transformer_blocks.0.norm1.bias' % index]) # 1 4096 320
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, 1, -1, ints[0])
    union_weights = np.zeros((1, 3, ints[0], ints[0]), dtype=np.float32)
    union_weights[:, 0, :, :] = para["input_blocks.%s.1.transformer_blocks.0.attn1.to_q.weight" % index].transpose(1, 0)
    union_weights[:, 1, :, :] = para["input_blocks.%s.1.transformer_blocks.0.attn1.to_k.weight" % index].transpose(1, 0)
    union_weights[:, 2, :, :] = para["input_blocks.%s.1.transformer_blocks.0.attn1.to_v.weight" % index].transpose(1, 0)
    weights_constant = network.add_constant((1, 3, ints[0], ints[0]), format(union_weights))
    noise_in = network.add_matrix_multiply(out(noise_in), trt.MatrixOperation.NONE, out(weights_constant), trt.MatrixOperation.NONE) #1 3 4096 320
    noise_in = network.add_shuffle(out(noise_in)) # 1 3 4096 320
    noise_in.reshape_dims = (2, 3, -1, 8, ints[0]//8)
    noise_in.second_transpose = (0, 2, 3, 1, 4)
    noise_in = network.add_plugin_v2([out(noise_in)], fmha()) # 1 4098 4 80
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, -1, ints[0])
    noise_in = matrix_mul(network, noise_in, para['input_blocks.%s.1.transformer_blocks.0.attn1.to_out.0.weight' % index], para['input_blocks.%s.1.transformer_blocks.0.attn1.to_out.0.bias' % index], (1, ints[0], ints[0]), (1, 1, ints[0]))
    noise_in = network.add_elementwise(out(noise_in), out(input_layer), trt.ElementWiseOperation.SUM)
    return noise_in

def ffn_out(network, para, input_layer, index, ints):
    def geglu():
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "SplitGeLU":
                pLists = []
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None
    noise_in = ln(network, input_layer, para['output_blocks.%s.1.transformer_blocks.0.norm3.weight' % index], para['output_blocks.%s.1.transformer_blocks.0.norm3.bias' % index])  # 2 4096 320
    noise_in = matrix_mul(network, noise_in, para['output_blocks.%s.1.transformer_blocks.0.ff.net.0.proj.weight' % index], para['output_blocks.%s.1.transformer_blocks.0.ff.net.0.proj.bias' % index], (1, ints[0], ints[0]*8), (1, 1, ints[0]*8))
    noise_in = network.add_plugin_v2([out(noise_in)], geglu()) # 2 4096 1280
    noise_in = matrix_mul(network, noise_in, para['output_blocks.%s.1.transformer_blocks.0.ff.net.2.weight' % index], para['output_blocks.%s.1.transformer_blocks.0.ff.net.2.bias' % index], (1, ints[0]*4, ints[0]), (1, 1, ints[0])) # 2 4096 320
    noise_in = network.add_elementwise(out(noise_in), out(input_layer), trt.ElementWiseOperation.SUM)
    return noise_in

def cross_attn_out(network, para, input_layer, index, ints, context):
    def fmhca():
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "fMHCA":
                pLists = []
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None

    noise_in = ln(network, input_layer, para['output_blocks.%s.1.transformer_blocks.0.norm2.weight' % index], para['output_blocks.%s.1.transformer_blocks.0.norm2.bias' % index]) # 1 4096 320
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, 1, -1, ints[0])
    q = matrix_mul(network, noise_in, para['output_blocks.%s.1.transformer_blocks.0.attn2.to_q.weight' % index], None, (1, 1, ints[0], ints[0]), None, False) # 1 1 4096 320
    q = network.add_shuffle(out(q))
    q.reshape_dims = (2, -1, 8, ints[0]//8)
    kv_context = network.add_shuffle(context)
    kv_context.reshape_dims = (2, 1, 77, 768)
    union_weights = np.zeros((1, 2, 768, ints[0]), dtype=np.float32)
    union_weights[:, 0, :, :] = para["output_blocks.%s.1.transformer_blocks.0.attn2.to_k.weight" % index].transpose(1, 0)
    union_weights[:, 1, :, :] = para["output_blocks.%s.1.transformer_blocks.0.attn2.to_v.weight" % index].transpose(1, 0)
    weights_constant = network.add_constant((1, 2, 768, ints[0]), format(union_weights))
    kv = network.add_matrix_multiply(out(kv_context), trt.MatrixOperation.NONE, out(weights_constant), trt.MatrixOperation.NONE) # 2 2 77 320
    if ints[0] < 1280:
        kv = network.add_shuffle(out(kv))
        kv.reshape_dims = (2, 2, 77, 8, ints[0]//8)
        kv.second_transpose = (0, 2, 3, 1, 4) # 2 77 8 2 40
        noise_in = network.add_plugin_v2([out(q), out(kv)], fmhca()) # 2 4096 4 80
    else:
        # 插件不支持 q 2 1 256 1280  kv 2 2 77 1280
        q = network.add_shuffle(out(q))
        q.reshape_dims = (2, -1, 8, ints[0] // 8)
        q.second_transpose = (0, 2, 1, 3) # 2 8 256 160
        k = network.add_slice(out(kv), (0, 0, 0, 0), (2, 1, 77, ints[0]), (1, 1, 1, 1))
        k = network.add_shuffle(out(k))
        k.reshape_dims = (2, -1, 8, ints[0] // 8)
        k.second_transpose = (0, 2, 1, 3) # 2 8 77 160
        v = network.add_slice(out(kv), (0, 1, 0, 0), (2, 1, 77, ints[0]), (1, 1, 1, 1))
        v = network.add_shuffle(out(v))
        v.reshape_dims = (2, -1, 8, ints[0] // 8)
        v.second_transpose = (0, 2, 1, 3) # 2 8 77 160
        qk = network.add_matrix_multiply(out(q), trt.MatrixOperation.NONE, out(k), trt.MatrixOperation.TRANSPOSE) # 2 8 256 77
        scale = network.add_constant((1, 1, 1, 1), format(np.array(1 / math.sqrt(ints[0] // 8), dtype=np.float32)))
        qk = network.add_elementwise(out(qk), out(scale), trt.ElementWiseOperation.PROD) # 2 8 256 77
        qk = network.add_softmax(out(qk))
        qk.axes = 1 << 3 # 2 8 256 77
        v = network.add_matrix_multiply(out(qk), trt.MatrixOperation.NONE, out(v), trt.MatrixOperation.NONE) # 2 8 256 160
        v = network.add_shuffle(out(v))
        v.first_transpose = (0, 2, 1, 3)
        v.reshape_dims = (2, -1, 8, ints[0]//8)
        noise_in = v
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, -1, ints[0])
    noise_in = matrix_mul(network, noise_in, para['output_blocks.%s.1.transformer_blocks.0.attn2.to_out.0.weight' % index], para['output_blocks.%s.1.transformer_blocks.0.attn2.to_out.0.bias' % index], (1, ints[0], ints[0]), (1, 1, ints[0]))
    noise_in = network.add_elementwise(out(noise_in), out(input_layer), trt.ElementWiseOperation.SUM)
    return noise_in

def self_attn_out(network, para, input_layer, index, ints):
    def fmha():
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "fMHA_V2":
                pLists = []
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None

    noise_in = ln(network, input_layer, para['output_blocks.%s.1.transformer_blocks.0.norm1.weight' % index], para['output_blocks.%s.1.transformer_blocks.0.norm1.bias' % index]) # 1 4096 320
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, 1, -1, ints[0])
    union_weights = np.zeros((1, 3, ints[0], ints[0]), dtype=np.float32)
    union_weights[:, 0, :, :] = para["output_blocks.%s.1.transformer_blocks.0.attn1.to_q.weight" % index].transpose(1, 0)
    union_weights[:, 1, :, :] = para["output_blocks.%s.1.transformer_blocks.0.attn1.to_k.weight" % index].transpose(1, 0)
    union_weights[:, 2, :, :] = para["output_blocks.%s.1.transformer_blocks.0.attn1.to_v.weight" % index].transpose(1, 0)
    weights_constant = network.add_constant((1, 3, ints[0], ints[0]), format(union_weights))
    noise_in = network.add_matrix_multiply(out(noise_in), trt.MatrixOperation.NONE, out(weights_constant), trt.MatrixOperation.NONE) #1 3 4096 320
    noise_in = network.add_shuffle(out(noise_in)) # 1 3 4096 320
    noise_in.reshape_dims = (2, 3, -1, 8, ints[0]//8)
    noise_in.second_transpose = (0, 2, 3, 1, 4)
    noise_in = network.add_plugin_v2([out(noise_in)], fmha()) # 1 4098 4 80
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, -1, ints[0])
    noise_in = matrix_mul(network, noise_in, para['output_blocks.%s.1.transformer_blocks.0.attn1.to_out.0.weight' % index], para['output_blocks.%s.1.transformer_blocks.0.attn1.to_out.0.bias' % index], (1, ints[0], ints[0]), (1, 1, ints[0]))
    noise_in = network.add_elementwise(out(noise_in), out(input_layer), trt.ElementWiseOperation.SUM)
    return noise_in

# build input第二阶段
def build_out_1(network, para, in_layer, index, ints, context, hw):
    noise_in = gn(network, in_layer, para['output_blocks.%s.1.norm.weight' % index], para['output_blocks.%s.1.norm.bias' % index], bSwish=0, epsilon=1e-6)
    noise_in = network.add_convolution(out(noise_in), ints[1], (1, 1), format(para["output_blocks.%s.1.proj_in.weight" % index]), format(para["output_blocks.%s.1.proj_in.bias" % index])) # 2 320 64 64
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.first_transpose = (0, 2, 3, 1)
    noise_in.reshape_dims = (2, -1, ints[1])
    ### slef-attention
    noise_in = self_attn_out(network, para, noise_in, index, [ints[1]])
    noise_in = cross_attn_out(network, para, noise_in, index, [ints[1]], context)
    noise_in = ffn_out(network, para, noise_in, index, [ints[1]]) # 2 4096 320
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.first_transpose = (0, 2, 1)
    noise_in.reshape_dims = (2, ints[1], hw, hw)
    noise_in = network.add_convolution(out(noise_in), ints[1], (1, 1), format(para['output_blocks.%s.1.proj_out.weight' % index]), format(para['output_blocks.%s.1.proj_out.bias' % index])) # 2 320 64 64
    noise_in = network.add_elementwise(out(noise_in), out(in_layer), trt.ElementWiseOperation.SUM) # 2 320 64 64
    return noise_in

def vae_mid_attn(network, para, in_layer, index, ints):
    sample = gn(network, in_layer, para["decoder.mid.attn_%s.norm.weight" % index], para["decoder.mid.attn_%s.norm.bias" % index], epsilon=1e-6, bSwish=0) # 1 512 64 64
    return sample
    q = network.add_convolution(out(sample), ints[1], (1, 1), format(para["decoder.mid.attn_1.q.weight"]), format(para["decoder.mid.attn_1.q.bias"])) # 1 512 64 64
    k = network.add_convolution(out(sample), ints[1], (1, 1), format(para["decoder.mid.attn_1.k.weight"]), format(para["decoder.mid.attn_1.k.bias"])) # 1 512 64 64
    v = network.add_convolution(out(sample), ints[1], (1, 1), format(para["decoder.mid.attn_1.v.weight"]), format(para["decoder.mid.attn_1.v.bias"])) # 1 512 64 64
    q = network.add_shuffle(out(q))
    q.reshape_dims = (1, 512, 4096)
    q.second_transpose = (0, 2, 1) # 1 4096 512
    k = network.add_shuffle(out(k))
    k.reshape_dims = (1, 512, 4096)
    v = network.add_shuffle(out(v))
    v.reshape_dims = (1, 512, 4096)
    qk = network.add_matrix_multiply(out(q), trt.MatrixOperation.NONE, out(k), trt.MatrixOperation.NONE) # 1 4096 4096
    qk_scale = network.add_constant((1, 1, 1), format(np.array([512**(-0.5)], dtype=np.float32)))
    qk = network.add_elementwise(out(qk), out(qk_scale), trt.ElementWiseOperation.PROD) # 1 4096 4096
    qk = network.add_softmax(out(qk)) # 1 4096 4096
    qk = network.add_shuffle(out(qk))
    qk.first_transpose = (0, 2, 1)
    v = network.add_matrix_multiply(out(v), trt.MatrixOperation.NONE, out(v), trt.MatrixOperation.NONE) # 1 512 4096
    v = network.add_shuffle(out(v))
    v.reshape_dims = (1, 512, 64, 64)
    sample = network.add_convolution(out(v), ints[1], (1, 1), format(para["decoder.mid.attn_1.proj_out.weight"]), format(para["decoder.mid.attn_1.proj_out.bias"])) # 1 512 64 64
    sample = network.add_elementwise(out(sample), out(in_layer), trt.ElementWiseOperation.SUM)
    return sample

def vae_mid_res(network, para, in_layer, index, ints, skip=False, prefix='mid.'):
    sample = gn(network, in_layer, para["decoder." + prefix + "block_%s.norm1.weight" % index], para["decoder." + prefix + "block_%s.norm1.bias" % index], epsilon=1e-6, bSwish=1)
    sample = network.add_convolution(out(sample), ints[1], (3, 3), format(para["decoder." + prefix + "block_%s.conv1.weight" % index]), format(para["decoder." + prefix + "block_%s.conv1.bias" % index]))
    sample.padding = (1, 1)  # 1 512 64 64
    sample = gn(network, sample, para["decoder." + prefix + "block_%s.norm2.weight" % index], para["decoder." + prefix + "block_%s.norm2.bias" % index], epsilon=1e-6, bSwish=1)
    sample = network.add_convolution(out(sample), ints[1], (3, 3), format(para["decoder." + prefix + "block_%s.conv2.weight" % index]), format(para["decoder." + prefix + "block_%s.conv2.bias" % index]))
    sample.padding = (1, 1)  # 1 512 64 64
    if skip:
        in_layer = network.add_convolution(out(in_layer), ints[1], (1, 1), format(para["output_blocks.%s.0.skip_connection.weight" % index]), format(para["output_blocks.%s.0.skip_connection.bias" % index]))
    sample = network.add_elementwise(out(in_layer), out(sample), trt.ElementWiseOperation.SUM)
    return sample

# build input第一阶段
def build_out_0(network, para, in_layer, index, ints, temb, skip=False):
    noise_in = gn(network, in_layer, para["output_blocks.%s.0.in_layers.0.weight" % index], para["output_blocks.%s.0.in_layers.0.bias" % index])
    noise_in = network.add_convolution(out(noise_in), ints[1], (3, 3), format(para["output_blocks.%s.0.in_layers.2.weight" % index]), format(para["output_blocks.%s.0.in_layers.2.bias" % index]))
    noise_in.padding = (1, 1)  # 2 320 64 64
    t_in = silu(network, temb)

    t_in = matrix_mul(network, t_in, para["output_blocks.%s.0.emb_layers.1.weight" % index], para["output_blocks.%s.0.emb_layers.1.bias" % index], (1280, ints[1]), (1, ints[1]))  # 2 320
    t_in = network.add_shuffle(out(t_in))
    t_in.reshape_dims = (2, ints[1], 1, 1)
    noise_in = network.add_elementwise(out(noise_in), out(t_in), trt.ElementWiseOperation.SUM)  # 2 320 64 64
    noise_in = gn(network, noise_in, para["output_blocks.%s.0.out_layers.0.weight" % index], para['output_blocks.%s.0.out_layers.0.bias' % index])
    noise_in = network.add_convolution(out(noise_in), ints[1], (3, 3), format(para["output_blocks.%s.0.out_layers.3.weight" % index]), format(para["output_blocks.%s.0.out_layers.3.bias" % index]))
    noise_in.padding = (1, 1)  # 1 320 64 64
    if skip:
        in_layer = network.add_convolution(out(in_layer), ints[1], (1, 1), format(para["output_blocks.%s.0.skip_connection.weight" % index]), format(para["output_blocks.%s.0.skip_connection.bias" % index]))
    noise_in = network.add_elementwise(out(in_layer), out(noise_in), trt.ElementWiseOperation.SUM)
    return noise_in

# build input第一阶段
def build_in_0(network, para, in_layer, index, ints, temb, skip=False):
    noise_in = gn(network, in_layer, para["input_blocks.%s.0.in_layers.0.weight" % index], para["input_blocks.%s.0.in_layers.0.bias" % index])
    noise_in = network.add_convolution(out(noise_in), ints[1], (3, 3), format(para["input_blocks.%s.0.in_layers.2.weight" % index]), format(para["input_blocks.%s.0.in_layers.2.bias" % index]))
    noise_in.padding = (1, 1)  # 2 320 64 64
    t_in = silu(network, temb)

    t_in = matrix_mul(network, t_in, para["input_blocks.%s.0.emb_layers.1.weight" % index], para["input_blocks.%s.0.emb_layers.1.bias" % index], (1280, ints[1]), (1, ints[1]))  # 2 320
    t_in = network.add_shuffle(out(t_in))
    t_in.reshape_dims = (2, ints[1], 1, 1)
    noise_in = network.add_elementwise(out(noise_in), out(t_in), trt.ElementWiseOperation.SUM)  # 2 320 64 64
    noise_in = gn(network, noise_in, para["input_blocks.%s.0.out_layers.0.weight" % index], para['input_blocks.%s.0.out_layers.0.bias' % index])
    noise_in = network.add_convolution(out(noise_in), ints[1], (3, 3), format(para["input_blocks.%s.0.out_layers.3.weight" % index]), format(para["input_blocks.%s.0.out_layers.3.bias" % index]))
    noise_in.padding = (1, 1)  # 1 320 64 64
    if skip:
        in_layer = network.add_convolution(out(in_layer), ints[1], (1, 1), format(para["input_blocks.%s.0.skip_connection.weight" % index]), format(para["input_blocks.%s.0.skip_connection.bias" % index]))
    noise_in = network.add_elementwise(out(in_layer), out(noise_in), trt.ElementWiseOperation.SUM)
    return noise_in

# build input第二阶段
def build_in_1(network, para, in_layer, index, ints, context, hw):
    noise_in = gn(network, in_layer, para['input_blocks.%s.1.norm.weight' % index], para['input_blocks.%s.1.norm.bias' % index], bSwish=0, epsilon=1e-6)
    noise_in = network.add_convolution(out(noise_in), ints[1], (1, 1), format(para["input_blocks.%s.1.proj_in.weight" % index]), format(para["input_blocks.%s.1.proj_in.bias" % index])) # 2 320 64 64
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.first_transpose = (0, 2, 3, 1)
    noise_in.reshape_dims = (2, -1, ints[1])
    ### slef-attention
    noise_in = self_attn(network, para, noise_in, index, [ints[1]])
    noise_in = cross_attn(network, para, noise_in, index, [ints[1]], context)
    noise_in = ffn(network, para, noise_in, index, [ints[1]]) # 2 4096 320
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.first_transpose = (0, 2, 1)
    noise_in.reshape_dims = (2, ints[1], hw, hw)
    noise_in = network.add_convolution(out(noise_in), ints[1], (1, 1), format(para['input_blocks.%s.1.proj_out.weight' % index]), format(para['input_blocks.%s.1.proj_out.bias' % index])) # 2 320 64 64
    noise_in = network.add_elementwise(out(noise_in), out(in_layer), trt.ElementWiseOperation.SUM) # 2 320 64 64
    return noise_in


# build mid第一阶段
def build_mid_0(network, para, in_layer, index, ints, temb, skip=False):
    noise_in = gn(network, in_layer, para["middle_block.%s.in_layers.0.weight" % index], para["middle_block.%s.in_layers.0.bias" % index])
    noise_in = network.add_convolution(out(noise_in), ints[1], (3, 3), format(para["middle_block.%s.in_layers.2.weight" % index]), format(para["middle_block.%s.in_layers.2.bias" % index]))
    noise_in.padding = (1, 1)  # 2 320 64 64
    t_in = silu(network, temb)

    t_in = matrix_mul(network, t_in, para["middle_block.%s.emb_layers.1.weight" % index], para["middle_block.%s.emb_layers.1.bias" % index], (1280, ints[1]), (1, ints[1]))  # 2 320
    t_in = network.add_shuffle(out(t_in))
    t_in.reshape_dims = (2, ints[1], 1, 1)
    noise_in = network.add_elementwise(out(noise_in), out(t_in), trt.ElementWiseOperation.SUM)  # 2 320 64 64
    noise_in = gn(network, noise_in, para["middle_block.%s.out_layers.0.weight" % index], para['middle_block.%s.out_layers.0.bias' % index])
    noise_in = network.add_convolution(out(noise_in), ints[1], (3, 3), format(para["middle_block.%s.out_layers.3.weight" % index]), format(para["middle_block.%s.out_layers.3.bias" % index]))
    noise_in.padding = (1, 1)  # 1 320 64 64
    if skip:
        in_layer = network.add_convolution(out(in_layer), ints[1], (1, 1), format(para["input_blocks.%s.0.skip_connection.weight" % index]), format(para["input_blocks.%s.0.skip_connection.bias" % index]))
    noise_in = network.add_elementwise(out(in_layer), out(noise_in), trt.ElementWiseOperation.SUM)
    return noise_in

# build mid第二阶段
def build_mid_1(network, para, in_layer, index, ints, context, hw):
    noise_in = gn(network, in_layer, para['middle_block.%s.norm.weight' % index], para['middle_block.%s.norm.bias' % index], bSwish=0, epsilon=1e-6)
    noise_in = network.add_convolution(out(noise_in), ints[1], (1, 1), format(para["middle_block.%s.proj_in.weight" % index]), format(para["middle_block.%s.proj_in.bias" % index])) # 2 320 64 64
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.first_transpose = (0, 2, 3, 1)
    noise_in.reshape_dims = (2, -1, ints[1])
    ### slef-attention
    noise_in = self_attn_mid(network, para, noise_in, index, [ints[1]])
    noise_in = cross_attn_mid(network, para, noise_in, index, [ints[1]], context)
    noise_in = ffn_mid(network, para, noise_in, index, [ints[1]]) # 2 4096 320
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.first_transpose = (0, 2, 1)
    noise_in.reshape_dims = (2, ints[1], hw, hw)
    noise_in = network.add_convolution(out(noise_in), ints[1], (1, 1), format(para['middle_block.%s.proj_out.weight' % index]), format(para['middle_block.%s.proj_out.bias' % index])) # 2 320 64 64
    noise_in = network.add_elementwise(out(noise_in), out(in_layer), trt.ElementWiseOperation.SUM) # 2 320 64 64
    return noise_in


def ffn_mid(network, para, input_layer, index, ints):
    def geglu():
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "SplitGeLU":
                pLists = []
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None
    noise_in = ln(network, input_layer, para['middle_block.%s.transformer_blocks.0.norm3.weight' % index], para['middle_block.%s.transformer_blocks.0.norm3.bias' % index])  # 2 4096 320
    noise_in = matrix_mul(network, noise_in, para['middle_block.%s.transformer_blocks.0.ff.net.0.proj.weight' % index], para['middle_block.%s.transformer_blocks.0.ff.net.0.proj.bias' % index], (1, ints[0], ints[0]*8), (1, 1, ints[0]*8))
    noise_in = network.add_plugin_v2([out(noise_in)], geglu()) # 2 4096 1280
    noise_in = matrix_mul(network, noise_in, para['middle_block.%s.transformer_blocks.0.ff.net.2.weight' % index], para['middle_block.%s.transformer_blocks.0.ff.net.2.bias' % index], (1, ints[0]*4, ints[0]), (1, 1, ints[0])) # 2 4096 320
    noise_in = network.add_elementwise(out(noise_in), out(input_layer), trt.ElementWiseOperation.SUM)
    return noise_in

def cross_attn_mid(network, para, input_layer, index, ints, context):
    def fmhca():
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "fMHCA":
                pLists = []
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None

    noise_in = ln(network, input_layer, para['middle_block.%s.transformer_blocks.0.norm2.weight' % index], para['middle_block.%s.transformer_blocks.0.norm2.bias' % index]) # 1 4096 320
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, 1, -1, ints[0])
    q = matrix_mul(network, noise_in, para['middle_block.%s.transformer_blocks.0.attn2.to_q.weight' % index], None, (1, 1, ints[0], ints[0]), None, False) # 1 1 4096 320
    q = network.add_shuffle(out(q))
    q.reshape_dims = (2, -1, 8, ints[0]//8)
    kv_context = network.add_shuffle(context)
    kv_context.reshape_dims = (2, 1, 77, 768)
    union_weights = np.zeros((1, 2, 768, ints[0]), dtype=np.float32)
    union_weights[:, 0, :, :] = para["middle_block.%s.transformer_blocks.0.attn2.to_k.weight" % index].transpose(1, 0)
    union_weights[:, 1, :, :] = para["middle_block.%s.transformer_blocks.0.attn2.to_v.weight" % index].transpose(1, 0)
    weights_constant = network.add_constant((1, 2, 768, ints[0]), format(union_weights))
    kv = network.add_matrix_multiply(out(kv_context), trt.MatrixOperation.NONE, out(weights_constant), trt.MatrixOperation.NONE) # 2 2 77 320
    if ints[0] < 1280:
        kv = network.add_shuffle(out(kv))
        kv.reshape_dims = (2, 2, 77, 8, ints[0]//8)
        kv.second_transpose = (0, 2, 3, 1, 4) # 2 77 8 2 40
        noise_in = network.add_plugin_v2([out(q), out(kv)], fmhca()) # 2 4096 4 80
    else:
        # 插件不支持 q 2 1 256 1280  kv 2 2 77 1280
        q = network.add_shuffle(out(q))
        q.reshape_dims = (2, -1, 8, ints[0] // 8)
        q.second_transpose = (0, 2, 1, 3) # 2 8 256 160
        k = network.add_slice(out(kv), (0, 0, 0, 0), (2, 1, 77, ints[0]), (1, 1, 1, 1))
        k = network.add_shuffle(out(k))
        k.reshape_dims = (2, -1, 8, ints[0] // 8)
        k.second_transpose = (0, 2, 1, 3) # 2 8 77 160
        v = network.add_slice(out(kv), (0, 1, 0, 0), (2, 1, 77, ints[0]), (1, 1, 1, 1))
        v = network.add_shuffle(out(v))
        v.reshape_dims = (2, -1, 8, ints[0] // 8)
        v.second_transpose = (0, 2, 1, 3) # 2 8 77 160
        qk = network.add_matrix_multiply(out(q), trt.MatrixOperation.NONE, out(k), trt.MatrixOperation.TRANSPOSE) # 2 8 256 77
        scale = network.add_constant((1, 1, 1, 1), format(np.array(1 / math.sqrt(ints[0] // 8), dtype=np.float32)))
        qk = network.add_elementwise(out(qk), out(scale), trt.ElementWiseOperation.PROD) # 2 8 256 77
        qk = network.add_softmax(out(qk))
        qk.axes = 1 << 3 # 2 8 256 77
        v = network.add_matrix_multiply(out(qk), trt.MatrixOperation.NONE, out(v), trt.MatrixOperation.NONE) # 2 8 256 160
        v = network.add_shuffle(out(v))
        v.first_transpose = (0, 2, 1, 3)
        v.reshape_dims = (2, -1, 8, ints[0]//8)
        noise_in = v
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, -1, ints[0])
    noise_in = matrix_mul(network, noise_in, para['middle_block.%s.transformer_blocks.0.attn2.to_out.0.weight' % index], para['middle_block.%s.transformer_blocks.0.attn2.to_out.0.bias' % index], (1, ints[0], ints[0]), (1, 1, ints[0]))
    noise_in = network.add_elementwise(out(noise_in), out(input_layer), trt.ElementWiseOperation.SUM)
    return noise_in

def self_attn_mid(network, para, input_layer, index, ints):
    def fmha():
        for creator in trt.get_plugin_registry().plugin_creator_list:
            if creator.name == "fMHA_V2":
                pLists = []
                return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
        return None

    noise_in = ln(network, input_layer, para['middle_block.%s.transformer_blocks.0.norm1.weight' % index], para['middle_block.%s.transformer_blocks.0.norm1.bias' % index]) # 1 4096 320
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, 1, -1, ints[0])
    union_weights = np.zeros((1, 3, ints[0], ints[0]), dtype=np.float32)
    union_weights[:, 0, :, :] = para["middle_block.%s.transformer_blocks.0.attn1.to_q.weight" % index].transpose(1, 0)
    union_weights[:, 1, :, :] = para["middle_block.%s.transformer_blocks.0.attn1.to_k.weight" % index].transpose(1, 0)
    union_weights[:, 2, :, :] = para["middle_block.%s.transformer_blocks.0.attn1.to_v.weight" % index].transpose(1, 0)
    weights_constant = network.add_constant((1, 3, ints[0], ints[0]), format(union_weights))
    noise_in = network.add_matrix_multiply(out(noise_in), trt.MatrixOperation.NONE, out(weights_constant), trt.MatrixOperation.NONE) #1 3 4096 320
    noise_in = network.add_shuffle(out(noise_in)) # 1 3 4096 320
    noise_in.reshape_dims = (2, 3, -1, 8, ints[0]//8)
    noise_in.second_transpose = (0, 2, 3, 1, 4)
    noise_in = network.add_plugin_v2([out(noise_in)], fmha()) # 1 4098 4 80
    noise_in = network.add_shuffle(out(noise_in))
    noise_in.reshape_dims = (2, -1, ints[0])
    noise_in = matrix_mul(network, noise_in, para['middle_block.%s.transformer_blocks.0.attn1.to_out.0.weight' % index], para['middle_block.%s.transformer_blocks.0.attn1.to_out.0.bias' % index], (1, ints[0], ints[0]), (1, 1, ints[0]))
    noise_in = network.add_elementwise(out(noise_in), out(input_layer), trt.ElementWiseOperation.SUM)
    return noise_in