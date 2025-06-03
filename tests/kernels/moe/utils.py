import torch
from typing import Optional

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize,
    BatchedTritonExperts,
    NaiveBatchedExperts)
from vllm.utils import round_up

from tests.kernels.quant_utils import native_w8a8_block_matmul


def Xnative_w8a8_block_matmul(A: torch.Tensor,
                             B: torch.Tensor,
                             As: torch.Tensor,
                             Bs: torch.Tensor,
                             block_size: Optional[list[int]],
                             output_dtype=torch.bfloat16):
    """This function performs matrix multiplication with block-wise
    quantization using native torch.
    It is agnostic to the input data type and can be used for both int8 and
    fp8 data types.

    It takes two input tensors `A` and `B` (int8) with scales `As` and
    `Bs` (float32).
    The output is returned in the specified `output_dtype`.
    """
    if A.dtype.itemsize <= 2:
        compute_type = torch.bfloat16
    else:
        compute_type = torch.float32

    A = A.to(compute_type)
    B = B.to(compute_type).contiguous()
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1], (
        f"{(A.shape[-1] + block_k - 1) // block_k} == {As.shape[-1]}")
    assert A.shape[:-1] == As.shape[:-1], f"{A.shape} == {As.shape}"

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


# Note: same as torch_moe but with fused_topk factored out.
def torch_moe2(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    quant_type: Optional[torch.dtype] = None,
    per_act_token_quant=False,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    M, K = a.shape
    N = w1.shape[1]
    topk = topk_ids.shape[1]

    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)

    a, a_scale = moe_kernel_quantize_input(
        a,
        None,
        quant_type,
        per_act_token_quant,
        block_shape
    )

    out = torch.zeros(M * topk,
                      w2.shape[1],
                      dtype=torch.bfloat16,
                      device=a.device)
    num_experts = w1.shape[0]

    #inters = torch.zeros((num_experts, M, N), device=a.device, dtype=out.dtype)
    #acts = torch.zeros((num_experts, M, N//2), device=a.device, dtype=out.dtype)

    for i in range(num_experts):
        mask = (topk_ids == i).view(-1)
        if mask.sum():
            if quant_type is None:
                tmp1 = a[mask] @ w1[i].transpose(0, 1)
                tmp2 = SiluAndMul()(tmp1)
                out[mask] = tmp2 @ w2[i].transpose(0, 1)
            elif block_shape is not None:
                tmp1 = native_w8a8_block_matmul(a[mask], w1[i], a_scale[mask],
                                                w1_scale[i], block_shape, out.dtype)

                #print(f"TORCH INTER[{i}] {tmp1.shape}\n{tmp1}")
                #inters[i, :tmp1.shape[0]] = tmp1

                tmp2 = SiluAndMul()(tmp1)

                #print(f"TORCH ACT[{i}] {tmp2.shape}\n{tmp2}")
                #acts[i, :tmp2.shape[0]] = tmp2

                tmp2, b_scale = moe_kernel_quantize_input(tmp2, None, quant_type,
                                                          per_act_token_quant,
                                                          block_shape)

                out[mask] = native_w8a8_block_matmul(tmp2, w2[i], b_scale,
                                                     w2_scale[i], block_shape,
                                                     out.dtype)
            else:
                # XXXX need scales here
                compute_type = torch.bfloat16
                tmp1 = a[mask].to(compute_type) @ w1[i].transpose(0, 1).to(compute_type)
                tmp2 = SiluAndMul()(tmp1)
                out[mask] = (tmp2 @ w2[i].transpose(0, 1).to(compute_type)).to(out.dtype)

    #print(f"TORCH INTER {inters.shape}\n{inters}")
    #print(f"TORCH ACT {acts.shape}\n{acts}")

    return (out.view(M, -1, w2.shape[1]) *
            topk_weight.view(M, -1, 1).to(out.dtype)).sum(dim=1)


def triton_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    quant_type: Optional[torch.dtype] = None,
    per_act_token_quant=False,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    return fused_experts(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        per_channel_quant=per_act_token_quant,
        use_fp8_w8a8=quant_type == torch.float8_e4m3fn,
        block_shape=block_shape
    )

def batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    qtype: Optional[torch.dtype] = None,
    per_act_token: bool = False,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    max_num_tokens = round_up(a.shape[0], 64)

    fused_experts = FusedMoEModularKernel(
        BatchedPrepareAndFinalize(max_num_tokens,
                                  world_size=1,
                                  dp_size=1,
                                  rank=0,
                                  quant_dtype=qtype,
                                  block_shape=block_shape,
                                  per_act_token_quant=per_act_token),
        BatchedTritonExperts(max_num_tokens=max_num_tokens,
                             dp_size=1,
                             world_size=1,
                             use_fp8_w8a8=qtype == torch.float8_e4m3fn,
                             per_act_token_quant=per_act_token,
                             block_shape=block_shape)
    )

    return fused_experts(a,
                         w1,
                         w2,
                         topk_weight,
                         topk_ids,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale)


def naive_batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_experts = w1.shape[0]

    fused_experts = FusedMoEModularKernel(
        BatchedPrepareAndFinalize(a.shape[0], world_size=1, dp_size=1, rank=0),
        NaiveBatchedExperts(max_num_tokens=a.shape[0], dp_size=1, world_size=1))

    return fused_experts(a, w1, w2, topk_weight, topk_ids, num_experts)

