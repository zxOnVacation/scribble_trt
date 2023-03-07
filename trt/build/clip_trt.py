import ctypes
import torch as t
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPModel
import numpy as np
import tensorrt as trt
from cuda import cudart
import time



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

def ln_plugin(epsilon=1e-5, axis=-1):
    for creator in trt.get_plugin_registry().plugin_creator_list:
        if creator.name == "LayerNorm":
            pLists = []
            pLists.append(trt.PluginField("epsilon", np.float32(epsilon), trt.PluginFieldType.FLOAT32))
            pLists.append(trt.PluginField("axis", np.int32(axis), trt.PluginFieldType.INT32))
            return creator.create_plugin(creator.name, trt.PluginFieldCollection(pLists))
    return None


def ln(network, inputs, gamma_weights, beta_weights):
    gamma_cons = network.add_constant(gamma_weights.shape, format(gamma_weights))
    beta_cons = network.add_constant(beta_weights.shape, format(beta_weights))
    ln_out = network.add_plugin_v2([out(inputs), out(gamma_cons), out(beta_cons)], ln_plugin(axis=2))
    return ln_out


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
    mask = mask.detach().cpu().numpy().reshape(1, 77, 77)
    return mask


def out(layer, i=0):
    return layer.get_output(i)


def qkv_cal(network, para, i, input_layer):
    union_weights = np.zeros((3, 768, 768), dtype=np.float32)
    union_bias = np.zeros((3, 1, 768), dtype=np.float32)
    union_weights[0, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.weight' % (i, 'q')].transpose(1, 0)
    union_bias[0, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.bias' % (i, 'q')]
    union_weights[1, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.weight' % (i, 'k')].transpose(1, 0)
    union_bias[1, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.bias' % (i, 'k')]
    union_weights[2, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.weight' % (i, 'v')].transpose(1, 0)
    union_bias[2, :, :] = para['text_model.encoder.layers.%s.self_attn.%s_proj.bias' % (i, 'v')]
    weights_constant = network.add_constant((3, 768, 768), format(union_weights))
    bias_constant = network.add_constant((3, 1, 768), format(union_bias))
    qkv_mat = network.add_matrix_multiply(out(input_layer), trt.MatrixOperation.NONE, out(weights_constant), trt.MatrixOperation.NONE) # 3 77 768
    qkv_mat = network.add_elementwise(out(qkv_mat), out(bias_constant), trt.ElementWiseOperation.SUM) # 3 77 768
    return qkv_mat


def attn(network, para, i, input_layer, q_scale, masks):
    qkv_mat = qkv_cal(network, para, i, input_layer)
    q_proj = network.add_slice(out(qkv_mat), (0, 0, 0), (1, 77, 768), (1, 1, 1))
    k_proj = network.add_slice(out(qkv_mat), (1, 0, 0), (1, 77, 768), (1, 1, 1))
    v_proj = network.add_slice(out(qkv_mat), (2, 0, 0), (1, 77, 768), (1, 1, 1))
    q_proj_re = network.add_shuffle(out(q_proj))
    q_proj_re.reshape_dims = (77, 12, 64)
    q_proj_re.second_transpose = (1, 0, 2) # 12 77 64
    k_proj_re = network.add_shuffle(out(k_proj))
    k_proj_re.reshape_dims = (77, 12, 64)
    k_proj_re.second_transpose = (1, 0, 2) # 12 77 64
    v_proj_re = network.add_shuffle(out(v_proj))
    v_proj_re.reshape_dims = (77, 12, 64)
    v_proj_re.second_transpose = (1, 0, 2) # 12 77 64
    q_proj_scale = network.add_elementwise(out(q_proj_re), out(q_scale), trt.ElementWiseOperation.PROD) # 12 77 64
    attn_weights = network.add_matrix_multiply(out(q_proj_scale), trt.MatrixOperation.NONE, out(k_proj_re), trt.MatrixOperation.TRANSPOSE) # 12 77 77
    attn_mask_weights = network.add_elementwise(out(attn_weights), out(masks), trt.ElementWiseOperation.SUM) # 12 77 77
    attn_norm_score = network.add_softmax(out(attn_mask_weights))
    attn_norm_score.axes = 1 << 2
    attn_v = network.add_matrix_multiply(out(attn_norm_score), trt.MatrixOperation.NONE, out(v_proj_re),trt.MatrixOperation.NONE)  # 12 77 64
    attn_v_re = network.add_shuffle(out(attn_v))
    attn_v_re.first_transpose = (1, 0, 2)
    attn_v_re.reshape_dims = (1, 77, 768)
    out_weight = network.add_constant((1, 768, 768), format(para['encoder.layers.%s.self_attn.%s_proj.weight' % (i, 'out')].transpose(1, 0).reshape(1, 768, 768)))
    out_bias = network.add_constant((1, 1, 768), format(para['encoder.layers.%s.self_attn.%s_proj.bias' % (i, 'out')].reshape(1, 1, 768)))
    attn_out = network.add_matrix_multiply(out(attn_v_re), trt.MatrixOperation.NONE, out(out_weight), trt.MatrixOperation.NONE) # 1 77 768
    attn_out = network.add_elementwise(out(attn_out), out(out_bias), trt.ElementWiseOperation.SUM) # 1 77 768
    return attn_out


def mlp(network, para, i, input_layer, gelu_scale):
    mlp1_weight = network.add_constant((1, 768, 3072), format(para['encoder.layers.%s.mlp.fc%s.weight' % (i, 1)].transpose(1, 0).reshape(1, 768, 3072)))
    mlp1_bias = network.add_constant((1, 1, 3072), format(para['encoder.layers.%s.mlp.fc%s.bias' % (i, 1)].reshape(1, 1, 3072)))
    mlp1_out = network.add_matrix_multiply(out(input_layer), trt.MatrixOperation.NONE, out(mlp1_weight), trt.MatrixOperation.NONE) # 1 77 768
    mlp1_out = network.add_elementwise(out(mlp1_out), out(mlp1_bias), trt.ElementWiseOperation.SUM) # 1 77 768

    mlp_1_i1 = network.add_elementwise(out(mlp1_out), out(gelu_scale), trt.ElementWiseOperation.PROD)
    mlp_1_i2 = network.add_activation(out(mlp_1_i1), trt.ActivationType.SIGMOID)
    mlp_gelu = network.add_elementwise(out(mlp1_out), out(mlp_1_i2), trt.ElementWiseOperation.PROD)

    mlp2_weight = network.add_constant((1, 3072, 768), format(para['encoder.layers.%s.mlp.fc%s.weight' % (i, 2)].transpose(1, 0).reshape(1, 3072, 768)))
    mlp2_bias = network.add_constant((1, 1, 768), format(para['encoder.layers.%s.mlp.fc%s.bias' % (i, 2)].reshape(1, 1, 768)))
    mlp2_out = network.add_matrix_multiply(out(mlp_gelu), trt.MatrixOperation.NONE, out(mlp2_weight), trt.MatrixOperation.NONE) # 1 77 768
    mlp2_out = network.add_elementwise(out(mlp2_out), out(mlp2_bias), trt.ElementWiseOperation.SUM) # 1 77 768
    return mlp2_out


def build_network(network, para, inputTensor):
    total_token_embeddings = network.add_constant(para['text_model.embeddings.token_embedding.weight'].shape, format(para['text_model.embeddings.token_embedding.weight']))
    token_embedding = network.add_gather(out(total_token_embeddings), inputTensor, 0) # 2 77 768
    position_embeddings = network.add_constant((1, 77, 768), format(para['text_model.embeddings.position_embedding.weight'].reshape(1, 77, 768)))
    input_embedding = network.add_elementwise(out(token_embedding), out(position_embeddings), trt.ElementWiseOperation.SUM) # 2, 77, 768 embedding

    q_scale = network.add_constant((1, 1, 1), format(np.array([0.125], dtype=np.float32)))
    masks = network.add_constant((1, 77, 77), format(gen_masks()))
    gelu_scale = network.add_constant((1, 1, 1), format(np.array([1.702], dtype=np.float32)))

    residual = input_embedding
    for i in range(1):
        ln_0 = ln(network, residual, para['text_model.encoder.layers.%s.layer_norm1.weight' % i], para['text_model.encoder.layers.%s.layer_norm1.bias' % i])
        attn_out = attn(network, para, i, ln_0, q_scale, masks)
        out(attn_out).name = 'embeddings'
        network.mark_output(out(attn_out))
        return network


        residual = network.add_elementwise(out(attn_out), out(residual), trt.ElementWiseOperation.SUM)
        ln_1 = network.add_plugin_v2([out(residual), out(residual)], ln_plugin(para['encoder.layers.%s.layer_norm2.bias' % i], para['encoder.layers.%s.layer_norm2.weight' % i]))
        mlp_out = mlp(network, para, i, ln_1, gelu_scale)
        residual = network.add_elementwise(out(mlp_out), out(residual), trt.ElementWiseOperation.SUM)

    output = network.add_plugin_v2([out(residual), out(residual)], ln_plugin(para['final_layer_norm.bias'], para['final_layer_norm.weight']))
    output_reshape = network.add_shuffle(out(output))
    output_reshape.reshape_dims = (1, 77, 768)
    out(output_reshape).name = 'clip_text_embedding'
    network.mark_output(out(output_reshape))
    return network


def clip_trt():
    paraFile = './weights/clip.npz'
    bUseFP16Mode = True
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 7 << 30)
    if bUseFP16Mode:
        config.set_flag(trt.BuilderFlag.FP16)

    #network build
    inputTensor = network.add_input("tokens", trt.int32, [2, 77])
    profile.set_shape(inputTensor.name, (2, 77), (2, 77), (2, 77))
    config.add_optimization_profile(profile)
    para = np.load(paraFile)

    network = build_network(network, para, inputTensor)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        return
    with open('./engine/clip.plan', "wb") as f:  # 将序列化网络保存为 .plan 文件
        f.write(engineString)
        print("Succeeded saving .plan file!")

    print('build done')



if __name__ == '__main__':
    clip_trt()
