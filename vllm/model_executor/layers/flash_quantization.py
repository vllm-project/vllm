import os
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def move_to_device(tensor, device):
    """Move tensor to device."""
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device, non_blocking=True)
    else:
        return tensor 
    
def linear_quantize(name, from_p, profile):
    device = torch.cuda.current_device() 
    if name in profile:
        from_p = from_p * move_to_device(profile[name]['output_scale'], device) * move_to_device(profile[name]['input_scale'], device)
        if profile[name]['type'] == torch.int8:
            from_p = torch.round(from_p).clamp(min=-128, max=127).to(torch.int8)
        else:
            from_p = from_p.to(profile[name]['type'])
    # print("from p dtype:", from_p.dtype)
    return from_p 

input_linear_map = {
    'input_layernorm.weight': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'],
    'post_attention_layernorm.weight': ['mlp.gate_proj.weight', 'mlp.up_proj.weight'],
}

def flash_quantize_with_prune(weights, profile):
    logger.debug("flash_rl quantization with dead neuron pruning is enabled")
    weights = dict(weights)
    for name in profile.keys():
        weights[name] = linear_quantize(name, weights[name], profile)
    
    for name, tensor in weights.items():
        for layernorm, layers in input_linear_map.items():
            if layernorm in name:
                dead_neurons_mask = 1 
                for layer_i in layers:
                    layer_name = name.replace(layernorm, layer_i) 
                    dead_neurons_mask_i = (weights[layer_name].abs().float().sum(dim=0, keepdim=False) == 0) # 1 := needs to be pruned
                    dead_neurons_mask = dead_neurons_mask * dead_neurons_mask_i # to-be-pruned := needs to be pruned for all layers 
                weights[name].data = weights[name].data * (1 - dead_neurons_mask) # 1 := needs to be pruned
                    
    return weights.items()

def flash_quantize(weights, profile):
    logger.debug("flash_rl quantization is called")
    for name, tensor in weights:
        yield (name, linear_quantize(name, tensor, profile))
        if name in profile:
            del tensor

# using vllm kernels
def fp8_quantize_channel(name, from_p, profile):
    device = torch.cuda.current_device() 
    scale = torch.empty(
        (from_p.shape[0], 1),
        device=device,
        dtype=torch.float32,
    )
    output = torch.empty(
        from_p.shape, 
        device=device, 
        dtype=torch.float8_e4m3fn,
    )
    torch.ops._C.dynamic_per_token_scaled_fp8_quant(
        output, from_p.to(device), scale, None,
    )

    return (name, output.to(device)), (name+'_scale', scale.to(device))
    

def flash_quantize_fp8_channel(weights, profile):
    logger.debug("flash_rl quantization is called")
    for name, tensor in weights:
        if name in profile:
            weight, scale = fp8_quantize_channel(name, tensor, profile)
            del tensor
            yield weight
            yield scale
        else:
            yield (name, tensor)
            
# using vllm kernels
def fp8_quantize_tensor(name, from_p, profile):
    device = torch.cuda.current_device() 
    scale_scalar = torch.zeros(1, device=device, dtype=torch.float32)
    output = torch.empty(
        from_p.shape, 
        device=device, 
        dtype=torch.float8_e4m3fn,
    )
    torch.ops._C.dynamic_scaled_fp8_quant(
        output, from_p.to(device), scale_scalar,
    )
    scale = torch.empty(
        (from_p.shape[0], 1),
        device=device,
        dtype=torch.float32,
    )
    scale.fill_(scale_scalar.item())
    return (name, output.to(device)), (name+'_scale', scale)
    
def flash_quantize_fp8_tensor(weights, profile):
    logger.debug("flash_rl quantization is called")
    for name, tensor in weights:
        if name in profile:
            weight, scale = fp8_quantize_tensor(name, tensor, profile)
            del tensor
            yield weight
            yield scale
        else:
            yield (name, tensor)

quant_fn_map = {
    'int8': flash_quantize,
    'int8_wo_prune': flash_quantize,
    'int8_prune': flash_quantize_with_prune,
    'fp8': lambda weights, profile: weights,
    'fp8_vllm': lambda weights, profile: weights,
    'fp8_tensor': flash_quantize_fp8_tensor,
    'fp8_channel': flash_quantize_fp8_channel,
}

def get_quantize_fn(name):
    if name not in quant_fn_map:
        logger.warning(f"Quantization function {name} not found, using identity mapping.")
    return quant_fn_map.get(name, lambda x: x)
    
def least_square(from_p, to_p, dim=-1):
    to_p = to_p.float()

    beta = (to_p * from_p).sum(dim=dim, keepdim=True) / (from_p **2).sum(dim=dim, keepdim=True)
    
    return beta

def profiling_fp8(quantized_model, profile_save_to):
    m = AutoModelForCausalLM.from_pretrained(quantized_model, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(quantized_model)
    
    profile = [k.replace('_scale', '') for k, v in m.named_parameters() if '_scale' in k]
    
    delete_irrelevant_parameters(m)
    
    # m.save_pretrained(profile_save_to)
    # tokenizer.save_pretrained(profile_save_to)
    # torch.save(profile, os.path.join(profile_save_to, 'profile.pt'))
    torch.save(profile, profile_save_to)

def delete_irrelevant_parameters(model):
    for _, module in model.named_children():
        delete_irrelevant_parameters(module)
    
    param_to_delete = []
    for name, param in model.named_parameters():
        if '_scale' not in name:
            param_to_delete.append(name)
        
    for name in param_to_delete:
        delattr(model, name)
        
def profiling_int8(model, quantized_model, profile_save_to):
    m = AutoModelForCausalLM.from_pretrained(model, device_map="cpu")
    qmodel = AutoModelForCausalLM.from_pretrained(quantized_model, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(quantized_model)
    
    profile = dict()
    
    param = {k: v for k, v in m.named_parameters()}
    qparam = {k: v for k, v in qmodel.named_parameters()}
    
    layernorm_list = ['layernorm']
    input_linear_map = {
        'self_attn.q_proj.weight': 'input_layernorm.weight',
        'self_attn.k_proj.weight': 'input_layernorm.weight',
        'self_attn.v_proj.weight': 'input_layernorm.weight',
        'self_attn.o_proj.weight': None,
        'mlp.gate_proj.weight': 'post_attention_layernorm.weight',
        'mlp.up_proj.weight': 'post_attention_layernorm.weight',
    }
    extra_output_list = ['mlp.up_proj.weight']
    output_linear_map = {
        'mlp.down_proj.weight': 'mlp.up_proj.weight',
    }
    exclude_list = [
        'bias', 
        'lm_head.weight',
        'model.norm.weight',
        'embed_tokens',
    ]
    
    input_scale = dict()
    for k, v in param.items():
        for key in layernorm_list:
            if key in k:
                input_scale_k = qparam[k] / param[k].float()
                profile[k] = {
                    'input_scale': input_scale_k,
                    'output_scale': 1.,
                    'type': qparam[k].data.dtype,
                }
                input_scale[k] = input_scale_k 
    
    for k, v in param.items():
        for balance, smooth in input_linear_map.items():
            if balance in k:
                if smooth is None:
                    original_output_scale = qparam[k+'_scale'].view(-1, 1)
                    profile[k] = {
                        'input_scale': 1.0,
                        'output_scale': 1. / original_output_scale.float(),
                        'type': qparam[k].data.dtype,
                    }
                else:
                    input_name = k.replace(balance, smooth)
                    input_scale_k = input_scale[input_name].view(1, -1)
                    original_output_scale = qparam[k+'_scale'].view(-1, 1)
                    
                    if any(ei in k for ei in extra_output_list):
                        additional_output_scale = least_square(
                            (param[k].float() / input_scale_k / original_output_scale).view(param[k].shape[0], -1),
                            qparam[k].view(param[k].shape[0], -1)
                        ).view(-1, 1)
                        input_scale[k] = additional_output_scale
                        original_output_scale = original_output_scale.float() / additional_output_scale
                    profile[k] = {
                        'input_scale': 1. / input_scale_k,
                        'output_scale': 1. / original_output_scale.float(),
                        'type': qparam[k].data.dtype,
                    }
                break
                
    for k, v in param.items():
        for balance, smooth in output_linear_map.items():
            if balance in k:
                input_name = k.replace(balance, smooth)
                input_scale_k = input_scale[input_name].view(1, -1)
                original_output_scale = qparam[k+'_scale'].view(-1, 1)
                profile[k] = {
                    'input_scale': 1. / input_scale_k,
                    'output_scale': 1. / original_output_scale.float(),
                    'type': qparam[k].data.dtype,
                }
                break
    
    # delete_irrelevant_parameters(qmodel)
    
    # qmodel.save_pretrained(profile_save_to)
    # tokenizer.save_pretrained(profile_save_to)
    # torch.save(profile, os.path.join(profile_save_to, 'profile.pt'))
    torch.save(profile, profile_save_to)