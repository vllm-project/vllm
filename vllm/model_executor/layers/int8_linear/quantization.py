# adapt from https://github.com/Guangxuan-Xiao/torch-int
import torch
import numpy as np

@torch.no_grad()
def quantize_per_tensor_absmax(t):
    scale = t.abs().max() / 127
    if not t.is_cuda:
        # half rounding is not supported on CPU
        t = t.float()
    # use inplace operation to save memory
    t.div_(scale).round_()
    t_q = t.to(torch.int8)
    return t_q, scale

@torch.no_grad()
def quantize_weight_per_channel_absmax(w):
    # w: [out_channel, in_channel]
    scales = w.abs().max(dim=1)[0] / 127
    scales = scales.view(-1, 1)
    if not w.is_cuda:
        # half rounding is not supported on CPU
        w = w.float()
    # use inplace operation to save memory
    w.div_(scales).round_().clamp_(-128, 127)
    w_q = w.to(torch.int8)
    return w_q, scales


@torch.no_grad()
def dynamic_quantize_activation_per_tensor_zeropoint(t):
    max_val = t.max()[0]
    min_val = t.min()[0]
    quant_min = -127
    quant_max = 127
    nudged_scale = (max_val - min_val) / (quant_max - quant_min)
    zp = (max_val + min_val) / 2
    zp = (zp / nudged_scale).round() * nudged_scale
    t -= zp
    max_val = (max_val - min_val) / 2

    max_val = torch.clamp(max_val, min=1e-8) / 127
    q_act = (t / max_val).round().clamp(-128, 127).to(torch.int8)
    return q_act, max_val, zp


@torch.no_grad()
def dynamic_quantize_activation_per_tensor_absmax(t):
    max_val = t.abs().max()
    max_val = torch.clamp(max_val, min=1e-8) / 127
    q_act = (t / max_val).round().clamp(-128, 127).to(torch.int8)
    return q_act, max_val


@torch.no_grad()
def dynamic_quantize_activation_per_token_absmax(t):
    max_val = t.abs().max(dim=-1, keepdim=True)[0]
    max_val = torch.clamp(max_val, min=1e-8) / 127
    t.div_(max_val).round_().clamp_(-128, 127)
    q_act = t.to(torch.int8)
    return q_act, max_val

@torch.no_grad()
def fake_quantize_activation_per_tensor_absmax(t):
    max_val = t.abs().max()
    max_val = torch.clamp(max_val, min=1e-8) / 127
    t.div_(max_val).round_().clamp_(-128, 127).mul_(max_val)
    return t


@torch.no_grad()
def fake_quantize_activation_per_token_absmax(t):
    max_val = t.abs().max(dim=-1, keepdim=True)[0]
    max_val = torch.clamp(max_val, min=1e-8) / 127
    t.div_(max_val).round_().clamp_(-128, 127).mul_(max_val)
    return t


@torch.no_grad()
def dequantize_activation_w_per_channel_a_per_token(q_act, w_scales, a_scales):
    # q_act: [B, dim]
    # w_scales: [dim]
    # a_scales: [B 1]
    dtype = a_scales.dtype
    q_act = q_act.to(torch.float32)
    q_act.mul_(w_scales.reshape(1, -1)).mul_(a_scales.reshape(-1, 1))
    return q_act.to(dtype)

@torch.no_grad()
def dequantize_activation_w_per_channel_a_per_tensor(q_act, w_scales, a_scales):
    # q_act: [..., dim]
    # w_scales: [dim]
    # a_scales: [1]
    dtype = a_scales.dtype
    q_act = q_act.to(torch.float32)
    q_act = q_act * w_scales.reshape(1, -1) * a_scales
    return q_act.to(dtype)
