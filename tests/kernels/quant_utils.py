# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    group_broadcast)
from vllm.platforms import current_platform
from vllm.utils import round_up

# Using the default value (240.0) from pytorch will cause accuracy
# issue on dynamic quantization models. Here use 224.0 for rocm.
ROCM_FP8FNUZ_MAX = 224.0
FP8_DTYPE = current_platform.fp8_dtype()


def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='cuda')

def ref_dynamic_per_token_quant(x: torch.tensor,
                                quant_dtype: torch.dtype,
                                scale_ub: Optional[torch.tensor] = None) \
        -> tuple[torch.tensor, torch.tensor]:

    assert quant_dtype in [torch.int8, FP8_DTYPE]
    if scale_ub is not None:
        assert quant_dtype == FP8_DTYPE

    qtype_traits = torch.iinfo(quant_dtype) if quant_dtype == torch.int8 \
            else torch.finfo(quant_dtype)
    qtype_traits_max = ROCM_FP8FNUZ_MAX if current_platform.is_rocm() \
                                            and current_platform.is_fp8_fnuz() \
                                        else qtype_traits.max
    qtype_traits_min = -ROCM_FP8FNUZ_MAX if current_platform.is_rocm() \
                                            and current_platform.is_fp8_fnuz() \
                                        else qtype_traits.min
    qtype_max = as_float32_tensor(qtype_traits_max)
    s_1 = as_float32_tensor(1.0)
    s_512 = as_float32_tensor(512.0)

    # For fp8, in order to match the cuda kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    # Compute scales
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = as_float32_tensor(x_token_max)
    if scale_ub is not None:
        x_token_max = x_token_max.clamp(max=scale_ub)
    scales = (x_token_max / qtype_max)[:, None]

    # Quant
    if quant_dtype == torch.int8:
        iscales = as_float32_tensor(s_1 / scales)
        torch_out = as_float32_tensor(x) * iscales
        torch_out = torch_out.round()
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)
    else:
        assert quant_dtype == FP8_DTYPE
        min_scaling_factor = s_1 / (qtype_max * s_512)
        scales = scales.clamp(min=min_scaling_factor)
        torch_out = as_float32_tensor(x) / scales
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)

    return torch_out, scales


# The int8 version is very similar. Incorporate the int8 version, like in
# ref_dynamic_per_token_quant, when we have a dynamic_per_tensor int8 quant
# kernel
def ref_dynamic_per_tensor_fp8_quant(x: torch.tensor) \
                    -> tuple[torch.tensor, torch.tensor]:

    fp8_traits = torch.finfo(FP8_DTYPE)
    fp8_traits_max = ROCM_FP8FNUZ_MAX if current_platform.is_rocm() \
                                            and current_platform.is_fp8_fnuz() \
                                    else fp8_traits.max
    fp8_traits_min = -ROCM_FP8FNUZ_MAX if current_platform.is_rocm() \
                                            and current_platform.is_fp8_fnuz() \
                                    else fp8_traits.min
    fp8_max = as_float32_tensor(fp8_traits_max)
    one = as_float32_tensor(1.0)

    # For fp8, in order to match the cuda kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    x_max = as_float32_tensor(x.abs().max())
    ref_scale = x_max / fp8_max
    ref_iscale = one / ref_scale
    ref_out = (as_float32_tensor(x) * ref_iscale).clamp(
        fp8_traits_min, fp8_traits_max).to(FP8_DTYPE)
    return ref_out, ref_scale.view((1, ))


def native_w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype,
    compute_type: torch.dtype = torch.float32,
) -> torch.Tensor:
    """This function performs matrix multiplication with block-wise
    quantization using native torch.
    It is agnostic to the input data type and can be used for both int8 and
    fp8 data types.

    It takes two input tensors `A` and `B` (int8) with scales `As` and
    `Bs` (float32).
    The output is returned in the specified `output_dtype`.
    """
    A = A.to(compute_type)
    B = B.to(compute_type)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N, )
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0], f"{n_tiles} == {Bs.shape[0]}"
    assert k_tiles == Bs.shape[1], f"{k_tiles} == {Bs.shape[1]}"

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=compute_type, device=A.device)

    A_tiles = [
        A[:, i * block_k:min((i + 1) * block_k, K)] for i in range(k_tiles)
    ]
    B_tiles = [[
        B[
            j * block_n:min((j + 1) * block_n, N),
            i * block_k:min((i + 1) * block_k, K),
        ] for i in range(k_tiles)
    ] for j in range(n_tiles)]
    C_tiles = [
        C[:, j * block_n:min((j + 1) * block_n, N)] for j in range(n_tiles)
    ]
    As_tiles = [As[:, i:i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


def native_per_token_group_quant_fp8(x,
                                     group_size,
                                     eps=1e-10,
                                     dtype=torch.float8_e4m3fn):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch."""
    assert x.shape[-1] % group_size == 0, ("the last dimension of `x` must "
                                           "be divisible by `group_size`")
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1,
                        keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size, ))

    return x_q, x_s


def native_per_token_group_quant_int8(x,
                                      group_size,
                                      eps=1e-10,
                                      dtype=torch.int8):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch.

    It converts the tensor values into int8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    """
    assert (x.shape[-1] % group_size == 0
            ), "the last dimension of `x` must be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    iinfo = torch.iinfo(dtype)
    int8_min = iinfo.min
    int8_max = iinfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    # Use float32 for scale calculation for stability
    amax = x_.abs().max(dim=-1,
                        keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / int8_max
    x_q = (x_.to(torch.float32) / x_s).round().clamp(
        min=int8_min, max=int8_max).to(dtype)  # Round before clamping
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size, ))

    return x_q, x_s


DEFAULT_BLOCK_SHAPE = [128, 128]


def per_block_cast_to_int8(
    x: torch.Tensor,
    block_shape: list[int] = DEFAULT_BLOCK_SHAPE,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_m, block_n = block_shape
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((round_up(m, block_m), round_up(n, block_n)),
                           dtype=x.dtype,
                           device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (256.0 / x_amax)).to(torch.int8)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 256.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def dequant(
    t: torch.Tensor,
    scale: Optional[torch.Tensor],
    block_shape: Optional[list[int]],
    per_act_token_quant: bool,
    out_dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    if scale is not None:
        f32 = torch.float32
        if per_act_token_quant or block_shape is None:
            return (t.to(f32) * scale).to(out_dtype)
        else:
            return (t.to(f32) * group_broadcast(scale, t.shape)).to(out_dtype)
    else:
        return t.to(out_dtype)


def batched_dequant(
    t: torch.Tensor,
    scale: Optional[torch.Tensor],
    block_shape: Optional[list[int]],
    per_act_token_quant: bool,
    out_dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    if scale is not None:
        assert t.shape[0] == scale.shape[0]
        out = torch.empty_like(t, dtype=out_dtype)
        for e in range(t.shape[0]):
            out[e] = dequant(t[e], scale[e], block_shape, per_act_token_quant,
                             out_dtype)
        return out

    return t.to(out_dtype)


def native_batched_masked_quant_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    num_expert_tokens: torch.Tensor,
    A_scale: Optional[torch.Tensor] = None,
    B_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    per_act_token_quant: bool = False,
) -> torch.Tensor:
    num_expert_tokens_cpu = num_expert_tokens.clone()
    num_expert_tokens_cpu = num_expert_tokens_cpu.to(device="cpu")
    num_experts = num_expert_tokens.size(0)

    for e in range(num_experts):
        num_tokens = num_expert_tokens_cpu[e]
        if A.dtype.itemsize == 1 and block_shape is not None:
            assert A_scale is not None and B_scale is not None
            tmp = native_w8a8_block_matmul(A[e], B[e], A_scale[e], B_scale[e],
                                           block_shape, C.dtype)
            C[e, :num_tokens, :] = tmp[:num_tokens, :]
        elif A.dtype.itemsize == 1 and block_shape is None:
            assert A_scale is not None and B_scale is not None
            A_dq = dequant(A[e], A_scale[e], block_shape, per_act_token_quant)
            B_dq = dequant(B[e], B_scale[e], block_shape, per_act_token_quant)
            C[e, :num_tokens, :] = (
                A_dq[:num_tokens] @ B_dq.transpose(0, 1)).to(C.dtype)
        else:
            assert A_scale is None
            assert B_scale is None
            C[e, :num_tokens, :] = A[e, :num_tokens, :] @ B[e].transpose(0, 1)

    return C
