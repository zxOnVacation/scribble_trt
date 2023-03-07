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
    ts.reshape_dims = (1, 1)
    time_freqs_c = network.add_matrix_multiply(out(ts), trt.MatrixOperation.NONE, out(freqs_c), trt.MatrixOperation.NONE)
    time_cos = network.add_unary(out(time_freqs_c), trt.UnaryOperation.COS)
    time_sin = network.add_unary(out(time_freqs_c), trt.UnaryOperation.SIN)
    time_emb = network.add_concatenation([out(time_cos), out(time_sin)])
    time_emb.axis = 1 # 1 320
    time_emb_0 = matrix_mul(network, time_emb, para['time_embed.0.weight'], para['time_embed.0.bias'], (320, 1280), (1, 1280))
    time_emb_silu = silu(network, time_emb_0)
    time_emb_1 = matrix_mul(network, time_emb_silu, para['time_embed.2.weight'], para['time_embed.2.bias'], (1280, 1280), (1, 1280)) # 1 1280
    return time_emb_1

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

def gn(network, inputs, gamma_weights, beta_weights):
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
    gn_out = network.add_plugin_v2([out(inputs), out(gamma_cons), out(beta_cons)], gn_plugin())
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


def union_weights(n, para):
    Wqkv = np.zeros((3, 768, 768), np.float32)
    Bqkv = np.zeros((3, 768), np.float32)
    Wqkv[0, :, :] = para['encoder.layers.%s.self_attn.%s_proj.weight' % (n, 'q')]
    Wqkv[1, :, :] = para['encoder.layers.%s.self_attn.%s_proj.weight' % (n, 'k')]
    Wqkv[2, :, :] = para['encoder.layers.%s.self_attn.%s_proj.weight' % (n, 'v')]
    Bqkv[0, :] = para['encoder.layers.%s.self_attn.%s_proj.bias' % (n, 'q')]
    Bqkv[1, :] = para['encoder.layers.%s.self_attn.%s_proj.bias' % (n, 'k')]
    Bqkv[2, :] = para['encoder.layers.%s.self_attn.%s_proj.bias' % (n, 'v')]
    Wqkv = np.ascontiguousarray(Wqkv.reshape((3, 12, 64, 12, 64)).transpose((1, 0, 2, 3, 4)))
    Bqkv = np.ascontiguousarray(Bqkv.reshape((3, 12, 64)).transpose((1, 0, 2)))
    return format(Wqkv), format(Bqkv)


def gen_masks():
    mask = torch.empty(1, 77, 77, dtype=torch.float32)
    mask.fill_(torch.tensor(torch.finfo(torch.float32).min))
    mask.triu_(1)
    mask = mask.detach().cpu().numpy().reshape(1, 1, 77, 77)
    return mask


def out(layer, i=0):
    return layer.get_output(i)


def qkv_cal(network, para, i, input_layer):
    union_weights = np.zeros((3, 1, 768, 768), dtype=np.float32)
    union_bias = np.zeros((3, 1, 1, 768), dtype=np.float32)
    union_weights[0, :, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.weight' % (i, 'q')].transpose(1, 0)
    union_bias[0, :, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.bias' % (i, 'q')]
    union_weights[1, :, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.weight' % (i, 'k')].transpose(1, 0)
    union_bias[1, :, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.bias' % (i, 'k')]
    union_weights[2, :, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.weight' % (i, 'v')].transpose(1, 0)
    union_bias[2, :, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.bias' % (i, 'v')]
    weights_constant = network.add_constant((3, 1, 768, 768), format(union_weights))
    bias_constant = network.add_constant((3, 1, 1, 768), format(union_bias))
    qkv_mat = network.add_matrix_multiply(out(input_layer), trt.MatrixOperation.NONE, out(weights_constant), trt.MatrixOperation.NONE) # 3 1 77 768
    qkv_mat = network.add_elementwise(out(qkv_mat), out(bias_constant), trt.ElementWiseOperation.SUM) # 3 2 77 768
    return qkv_mat


def attn(network, para, i, input_layer, q_scale, masks):
    qkv_mat = qkv_cal(network, para, i, input_layer) # 3 2 77 768
    q_proj = network.add_slice(out(qkv_mat), (0, 0, 0, 0), (1, 2, 77, 768), (1, 1, 1, 1))
    k_proj = network.add_slice(out(qkv_mat), (1, 0, 0, 0), (1, 2, 77, 768), (1, 1, 1, 1))
    v_proj = network.add_slice(out(qkv_mat), (2, 0, 0, 0), (1, 2, 77, 768), (1, 1, 1, 1)) # 1 2 77 768
    q_proj_re = network.add_shuffle(out(q_proj))
    q_proj_re.reshape_dims = (2, 77, 12, 64)
    q_proj_re.second_transpose = (0, 2, 1, 3) # 2 12 77 64
    k_proj_re = network.add_shuffle(out(k_proj))
    k_proj_re.reshape_dims = (2, 77, 12, 64)
    k_proj_re.second_transpose = (0, 2, 1, 3) # 2 12 77 64
    v_proj_re = network.add_shuffle(out(v_proj))
    v_proj_re.reshape_dims = (2, 77, 12, 64)
    v_proj_re.second_transpose = (0, 2, 1, 3) # 2 12 77 64
    q_proj_scale = network.add_elementwise(out(q_proj_re), out(q_scale), trt.ElementWiseOperation.PROD) # 2 12 77 64
    attn_weights = network.add_matrix_multiply(out(q_proj_scale), trt.MatrixOperation.NONE, out(k_proj_re), trt.MatrixOperation.TRANSPOSE) # 2 12 77 77
    attn_mask_weights = network.add_elementwise(out(attn_weights), out(masks), trt.ElementWiseOperation.SUM) # 2 12 77 77
    attn_norm_score = network.add_softmax(out(attn_mask_weights))
    attn_norm_score.axes = 1 << 3
    attn_v = network.add_matrix_multiply(out(attn_norm_score), trt.MatrixOperation.NONE, out(v_proj_re),trt.MatrixOperation.NONE)  # 2 12 77 64
    attn_v_re = network.add_shuffle(out(attn_v))
    attn_v_re.first_transpose = (0, 2, 1, 3)
    attn_v_re.reshape_dims = (1, 2, 77, 768)
    out_weight = network.add_constant((1, 1, 768, 768), format(para['text_model.encoder.layers.%s.self_attn.%s_proj.weight' % (i, 'out')].transpose(1, 0).reshape(1, 1, 768, 768)))
    out_bias = network.add_constant((1, 1, 1, 768), format(para['text_model.encoder.layers.%s.self_attn.%s_proj.bias' % (i, 'out')].reshape(1, 1, 1, 768)))
    attn_out = network.add_matrix_multiply(out(attn_v_re), trt.MatrixOperation.NONE, out(out_weight), trt.MatrixOperation.NONE) # 1 2 77 768
    attn_out = network.add_elementwise(out(attn_out), out(out_bias), trt.ElementWiseOperation.SUM) # 1 2 77 768
    return attn_out


def mlp(network, para, i, input_layer, gelu_scale):
    mlp1_weight = network.add_constant((1, 1, 768, 3072), format(para['text_model.encoder.layers.%s.mlp.fc%s.weight' % (i, 1)].transpose(1, 0).reshape(1, 1, 768, 3072)))
    mlp1_bias = network.add_constant((1, 1, 1, 3072), format(para['text_model.encoder.layers.%s.mlp.fc%s.bias' % (i, 1)].reshape(1, 1, 1, 3072)))
    mlp1_out = network.add_matrix_multiply(out(input_layer), trt.MatrixOperation.NONE, out(mlp1_weight), trt.MatrixOperation.NONE) # 1 77 768
    mlp1_out = network.add_elementwise(out(mlp1_out), out(mlp1_bias), trt.ElementWiseOperation.SUM) # 1 2 77 768

    mlp_1_i1 = network.add_elementwise(out(mlp1_out), out(gelu_scale), trt.ElementWiseOperation.PROD)
    mlp_1_i2 = network.add_activation(out(mlp_1_i1), trt.ActivationType.SIGMOID)
    mlp_gelu = network.add_elementwise(out(mlp1_out), out(mlp_1_i2), trt.ElementWiseOperation.PROD)

    mlp2_weight = network.add_constant((1, 1, 3072, 768), format(para['text_model.encoder.layers.%s.mlp.fc%s.weight' % (i, 2)].transpose(1, 0).reshape(1, 1, 3072, 768)))
    mlp2_bias = network.add_constant((1, 1, 1, 768), format(para['text_model.encoder.layers.%s.mlp.fc%s.bias' % (i, 2)].reshape(1, 1, 1, 768)))
    mlp2_out = network.add_matrix_multiply(out(mlp_gelu), trt.MatrixOperation.NONE, out(mlp2_weight), trt.MatrixOperation.NONE) # 1 77 768
    mlp2_out = network.add_elementwise(out(mlp2_out), out(mlp2_bias), trt.ElementWiseOperation.SUM) # 1 77 768
    return mlp2_out


# build input第一阶段
def build_in_0(network, para, in_layer, index, ints, temb, skip=False):
    noise_in = gn(network, in_layer, para["input_blocks.%s.0.in_layers.0.weight" % index], para["input_blocks.%s.0.in_layers.0.bias" % index])
    noise_in = network.add_convolution(out(noise_in), ints[0], (3, 3), format(para["input_blocks.%s.0.in_layers.2.weight" % index]), format(para["input_blocks.%s.0.in_layers.2.bias" % index]))
    noise_in.padding = (1, 1)  # 1 320 64 64
    t_in = silu(network, temb)
    t_in = matrix_mul(network, t_in, para["input_blocks.%s.0.emb_layers.1.weight" % index], para["input_blocks.%s.0.emb_layers.1.bias" % index], (1280, 320), (1, 320))  # 1 320
    t_in = network.add_shuffle(out(t_in))
    t_in.reshape_dims = (1, 320, 1, 1)
    noise_in = network.add_elementwise(out(noise_in), out(t_in), trt.ElementWiseOperation.SUM)  # 1 320 64 64
    noise_in = gn(network, noise_in, para["input_blocks.%s.0.out_layers.0.weight" % index], para['input_blocks.%s.0.out_layers.0.bias' % index])
    noise_in = network.add_convolution(out(noise_in), ints[1], (3, 3), format(para["input_blocks.%s.0.out_layers.3.weight" % index]), format(para["input_blocks.%s.0.out_layers.3.bias" % index]))
    noise_in.padding = (1, 1)  # 1 320 64 64
    if skip:
        pass
    else:
        noise_in = network.add_elementwise(out(in_layer), out(noise_in), trt.ElementWiseOperation.SUM)
    return noise_in



def build_network(network, para, noise, hint, t, context):
    temb = time_embedding(network, para, t) # 1 1280
    hint_in = hint_block(network, para, hint, temb, context) # 1 320 64 64

    if 1:
        # 第一层
        noise_in = network.add_convolution(noise, 320, (3, 3), format(para['input_blocks.0.0.weight']), format(para['input_blocks.0.0.bias']))
        noise_in.padding = (1, 1) # 1 320 64 64
        noise_in = network.add_elementwise(out(noise_in), out(hint_in), trt.ElementWiseOperation.SUM) # 1 320 64 64
        out_0 = network.add_convolution(out(noise_in), 320, (1, 1), format(para['zero_convs.0.0.weight']), format(para['zero_convs.0.0.bias']))  # 1 320 64 64
        out(out_0).name = 'dbrs_0'
        network.mark_output(out(out_0))
    if 2:
        # 第二层
        noise_in = build_in_0(network, para, noise_in, 1, [320, 320], temb) # 1 320 64 64
        out(noise_in).name = 'dbrs_1'
        network.mark_output(out(noise_in))
    return network





    total_token_embeddings = network.add_constant(para['text_model.embeddings.token_embedding.weight'].shape, format(para['text_model.embeddings.token_embedding.weight']))
    token_embedding = network.add_gather(out(total_token_embeddings), inputTensor, 0) # 2 77 768
    position_embeddings = network.add_constant((1, 77, 768), format(para['text_model.embeddings.position_embedding.weight'].reshape(1, 77, 768)))
    input_embedding = network.add_elementwise(out(token_embedding), out(position_embeddings), trt.ElementWiseOperation.SUM) # 2, 77, 768 embedding

    q_scale = network.add_constant((1, 1, 1, 1), format(np.array([0.125], dtype=np.float32)))
    masks = network.add_constant((1, 1, 77, 77), format(gen_masks()))
    gelu_scale = network.add_constant((1, 1, 1, 1), format(np.array([1.702], dtype=np.float32)))
    input_embedding = network.add_shuffle(out(input_embedding))
    input_embedding.reshape_dims = (1, 2, 77, 768)
    residual = input_embedding # 1 2 77 768
    for i in range(12):
        ln_0 = ln(network, residual, para['text_model.encoder.layers.%s.layer_norm1.weight' % i], para['text_model.encoder.layers.%s.layer_norm1.bias' % i]) # 1 2 77 768
        attn_out = attn(network, para, i, ln_0, q_scale, masks) # 1 2 77 768
        residual = network.add_elementwise(out(attn_out), out(residual), trt.ElementWiseOperation.SUM) # 1 2 77 768
        ln_1 = ln(network, residual, para['text_model.encoder.layers.%s.layer_norm2.weight' % i], para['text_model.encoder.layers.%s.layer_norm2.bias' % i]) # 1 2 77 768
        mlp_out = mlp(network, para, i, ln_1, gelu_scale)
        residual = network.add_elementwise(out(mlp_out), out(residual), trt.ElementWiseOperation.SUM) # 1 2 77 768

    output = ln(network, residual, para['text_model.final_layer_norm.weight'], para['text_model.final_layer_norm.bias'])
    output_reshape = network.add_shuffle(out(output))
    output_reshape.reshape_dims = (2, 77, 768)
    out(output_reshape).name = 'embeddings'
    network.mark_output(out(output_reshape))
    return network


def control_trt():
    paraFile = './weights/control.npz'
    bUseFP16Mode = True
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 7 << 30)
    if bUseFP16Mode:
        config.set_flag(trt.BuilderFlag.FP16)

    #network build
    noise = network.add_input("noise", trt.float32, [1, 4, 64, 64])
    hint = network.add_input("hint", trt.float32, [1, 3, 512, 512])
    t = network.add_input("t", trt.int32, [1, ])
    context = network.add_input("context", trt.float32, [2, 77, 768])
    profile.set_shape(noise.name, (1, 4, 64, 64), (1, 4, 64, 64), (1, 4, 64, 64))
    profile.set_shape(hint.name, (1, 3, 512, 512), (1, 3, 512, 512), (1, 3, 512, 512))
    profile.set_shape(t.name, (1, ), (1, ), (1, ))
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



if __name__ == '__main__':
    control_trt()