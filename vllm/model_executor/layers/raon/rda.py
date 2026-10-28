# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raon fused convolution, RDA recurrence, and gated output normalization."""

import functools
from pathlib import Path

import torch

from vllm import envs
from vllm.triton_utils import tl, triton


@triton.jit
def _value_scale(
    beta,
    delta,
    out,
    N: tl.constexpr,
    H: tl.constexpr,
    BS: tl.constexpr,
    DS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // H, index % H
    b = tl.load(beta + row * BS + col, index < N, other=0).to(tl.float32)
    d = tl.load(delta + row * DS + col, index < N, other=0).to(tl.float32)
    x = b + d
    log_b = tl.minimum(b, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(b)))
    log_x = tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))
    tl.store(out + index, tl.exp(log_x - log_b), index < N)


def value_scale(beta, delta):
    result = torch.empty(beta.shape, device=beta.device, dtype=torch.float32)
    _value_scale[(triton.cdiv(beta.numel(), 256),)](
        beta,
        delta,
        result,
        beta.numel(),
        beta.shape[-1],
        beta.stride(0),
        delta.stride(0),
        256,
        enable_fp_fusion=False,
    )
    return result


@functools.cache
def load_kernel(speculative: bool):
    from torch.utils.cpp_extension import load

    name = "rda_verify" if speculative else "rda_decode"
    directory = Path(envs.VLLM_CACHE_ROOT) / "raon" / name
    directory.mkdir(parents=True, exist_ok=True)
    flags = ["-O3"] if speculative else ["-O3", "--use_fast_math"]
    return load(
        name=f"raon_{name}",
        sources=[str(Path(__file__).with_name(name + ".cu"))],
        build_directory=str(directory),
        extra_cuda_cflags=flags,
    )


@torch.no_grad()
def prepare_rda(layer):
    """Materialize loaded constants before capture and refresh them in place."""
    weight = layer.conv1d.weight.detach().reshape(-1, layer.conv_size).float()
    bias = layer.conv1d.bias
    values = {
        "fast_conv": weight.contiguous(),
        "fast_bias": (
            weight.new_empty(0) if bias is None else bias.detach().float().flatten()
        ),
        "fast_decode_bias": (
            weight.new_zeros(weight.shape[0])
            if bias is None
            else bias.detach().float().flatten()
        ),
        "fast_a_log": layer.A_log.detach().float().flatten().contiguous(),
        "fast_dt_bias": layer.dt_bias.detach().float().flatten().contiguous(),
        "fast_norm": layer.o_norm.weight.detach().float().contiguous(),
    }
    for key, part in zip(
        ("fast_q", "fast_k", "fast_v"), weight.split(layer.local_projection_size, dim=0)
    ):
        values[key] = part.T.contiguous()
    for name, value in values.items():
        old = getattr(layer, name, None)
        if old is None:
            layer.register_buffer(name, value.clone(), persistent=False)
        else:
            if old.shape != value.shape or old.dtype != value.dtype:
                raise ValueError(f"Cannot change captured RDA buffer {name}")
            old.copy_(value)


def fused_forward(layer, qkv, g1, g2, beta, scale, output, metadata):
    """Handle pure decode/verify batches; mixed prefill uses the native path."""
    m = metadata
    if m.num_prefills or (m.num_spec_decodes and m.num_decodes):
        return False
    conv, state = layer.kv_cache
    heads, dim = layer.local_num_heads, layer.head_dim
    if not (
        conv.ndim == 3
        and conv.shape[2] == 3 * heads * dim
        and conv.stride(2) == 1
        and conv.dtype == torch.bfloat16
        and state.dtype == torch.float32
        and state.shape[1:] == (heads, dim, dim)
        and state.stride()[1:] == (dim * dim, dim, 1)
    ):
        raise ValueError(
            "Raon FAST requires time-major BF16 convolution and FP32 RDA state"
        )
    count = m.num_actual_tokens
    qkv, g1, beta, scale = qkv[:count], g1[:, :count], beta[:, :count], scale[:count]
    if m.num_spec_decodes:
        indices = m.spec_state_indices_tensor
        width = indices.shape[1]
        if width > 8 or conv.shape[1] < width + 2:
            raise ValueError(
                "Raon FAST verification supports at most seven draft tokens"
            )
        requests = m.num_spec_decodes
        split = 2 if requests * heads < 148 else 1
        load_kernel(True).run(
            qkv,
            layer.fast_conv,
            layer.fast_bias,
            conv,
            g1[0],
            layer.fast_a_log,
            layer.fast_dt_bias,
            beta[0],
            scale,
            0,
            g2[:count],
            layer.fast_norm,
            state,
            indices.to(torch.int32),
            m.spec_query_start_loc[: requests + 1].to(torch.int32),
            m.num_accepted_tokens[:requests].to(torch.int32),
            output[0, :count],
            requests,
            dim**-0.5,
            float(layer.gate_lower_bound),
            float(layer.o_norm.eps),
            split,
        )
    elif m.num_decodes and conv.shape[1] == 3:
        load_kernel(False).run(
            qkv,
            g1[0].reshape(count, -1),
            beta[0],
            conv,
            layer.fast_q,
            layer.fast_k,
            layer.fast_v,
            layer.fast_decode_bias,
            layer.fast_a_log,
            layer.fast_dt_bias,
            g2[:count].reshape(count, -1),
            layer.fast_norm,
            state,
            m.non_spec_state_indices_tensor[:count],
            output[0, :count].view(count, -1),
            scale,
            dim**-0.5,
            float(layer.o_norm.eps),
            float(layer.gate_lower_bound),
            True,
        )
    else:
        return False
    return True
