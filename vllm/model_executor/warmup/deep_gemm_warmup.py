# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup deep_gemm kernels.
DeepGEMM JIT's the kernels. The warmup aims to JIT all the kernels that would
be used during model execution beforehand.
"""

import torch
from tqdm import tqdm

import vllm.envs as envs
from vllm.distributed.parallel_state import get_dp_group
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M, deep_gemm_block_shape)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.utils.deep_gemm import fp8_gemm_nt, m_grouped_fp8_gemm_nt_contiguous


def _extract_data_from_linear_base_module(
        m: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Extract weights, weight scales and quantization block sizes from the given
    LinearBase module.
    """
    assert isinstance(m, LinearBase)
    assert isinstance(m.quant_method, Fp8LinearMethod)
    assert m.quant_method.block_quant
    assert m.quant_method.quant_config is not None

    w = m.weight
    ws = m.weight_scale
    quant_block_size = m.quant_method.quant_config.weight_block_size

    assert isinstance(w, torch.Tensor)
    assert isinstance(ws, torch.Tensor)
    assert quant_block_size is not None
    return (w, ws, quant_block_size)


def _extract_data_from_fused_moe_module(
    m: torch.nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Extract weights, weight scales and num_topk from FusedMoE module.
    """
    assert isinstance(m, FusedMoE)
    w13 = m.w13_weight
    w13_s = m.w13_weight_scale_inv
    w2 = m.w2_weight
    w2_s = m.w2_weight_scale_inv
    num_topk = m.top_k

    assert isinstance(w13, torch.Tensor)
    assert isinstance(w13_s, torch.Tensor)
    assert isinstance(w2, torch.Tensor)
    assert isinstance(w2_s, torch.Tensor)
    return w13, w13_s, w2, w2_s, num_topk


def _fp8_linear_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    """
    Return True if the input module/layer could be processed with DeepGEMM.
    """
    block_size = deep_gemm_block_shape()[0]
    if not (isinstance(module, LinearBase)
            and isinstance(module.quant_method, Fp8LinearMethod)
            and module.quant_method.block_quant):
        return False

    w, _, block_sizes = _extract_data_from_linear_base_module(module)
    return (block_sizes == deep_gemm_block_shape() and w.ndim == 2
            and w.shape[0] % block_size == 0 and w.shape[1] % block_size == 0)


def _fused_moe_grouped_gemm_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    if not isinstance(module, FusedMoE):
        return False

    moe_quant_config = module.quant_method.get_fused_moe_quant_config(module)

    if (moe_quant_config is None
            or moe_quant_config.quant_dtype != torch.float8_e4m3fn
            or moe_quant_config.block_shape != deep_gemm_block_shape()):
        return False

    if not isinstance(module.quant_method.fused_experts,
                      FusedMoEModularKernel):
        # fused_experts could invoke deep_gemm_moe_fp8
        return True

    mk: FusedMoEModularKernel = module.quant_method.fused_experts
    # Further check if the ModularKernel implementation uses the DeepGemmExperts
    return isinstance(mk.fused_experts,
                      (DeepGemmExperts, TritonOrDeepGemmExperts))


FP8_GEMM_NT_WARMUP_CACHE: set[torch.Size] = set()


def _deepgemm_fp8_gemm_nt_warmup(w: torch.Tensor, ws: torch.Tensor,
                                 max_tokens: int):
    if w.size() in FP8_GEMM_NT_WARMUP_CACHE:
        return

    n, k = w.size()
    block_m = deep_gemm_block_shape()[0]

    device = w.device
    a1q = torch.empty((max_tokens, k),
                      device=device,
                      dtype=torch.float8_e4m3fn)
    a1q_scales = torch.empty((max_tokens, k // block_m),
                             device=device,
                             dtype=torch.float32)
    out = torch.empty((max_tokens, n), device=device, dtype=torch.bfloat16)

    pbar = tqdm(total=max_tokens,
                desc=f"DeepGemm(fp8_gemm_nt) warmup (W={w.size()})")
    num_tokens = max_tokens
    while num_tokens > 0:
        fp8_gemm_nt((a1q[:num_tokens], a1q_scales[:num_tokens]), (w, ws),
                    out[:num_tokens])
        pbar.update(1)
        num_tokens -= 1

    FP8_GEMM_NT_WARMUP_CACHE.add(w.size())


GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE: set[torch.Size] = set()


def _deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(
        w1: torch.Tensor, w2: torch.Tensor, w1_scale: torch.Tensor,
        w2_scale: torch.Tensor, num_topk: int, max_tokens: int):
    if (w1.size() in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
            and w2.size() in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE):
        return

    assert w1.size(0) == w2.size(0), (
        "w1 and w2 must have the same number of experts")

    block_m = deep_gemm_block_shape()[0]
    num_experts = w1.size(0)
    device = w1.device

    # Assumes all ranks have the same max_num_batched_tokens
    max_tokens_across_dp = get_dp_group().world_size * max_tokens
    max_tokens = min(max_tokens_across_dp, envs.VLLM_FUSED_MOE_CHUNK_SIZE)

    # This is the maximum GroupedGemm M size that we expect to run
    # the grouped_gemm with.
    MAX_M = compute_aligned_M(max_tokens,
                              num_topk,
                              num_experts,
                              block_m,
                              expert_tokens_meta=None)
    # Distribute expert-ids evenly.
    MAX_BLOCKS = MAX_M // block_m
    expert_ids_block = torch.randint(low=0,
                                     high=num_experts,
                                     size=(MAX_BLOCKS, ),
                                     device=device,
                                     dtype=torch.int32)
    expert_ids = torch.repeat_interleave(expert_ids_block, block_m, dim=0)

    def _warmup(w: torch.Tensor, w_scale: torch.Tensor):

        _, n, k = w.size()
        a1q = torch.empty((MAX_M, k), device=device, dtype=torch.float8_e4m3fn)
        a1q_scales = torch.empty((MAX_M, k // block_m),
                                 device=device,
                                 dtype=torch.float32)
        out = torch.empty((MAX_M, n), device=device, dtype=torch.bfloat16)

        pbar = tqdm(
            total=MAX_BLOCKS,
            desc=
            f"DeepGemm(m_grouped_fp8_gemm_nt_contiguous) warmup (W={w.size()})"
        )
        num_tokens = MAX_M
        while num_tokens > 0:
            m_grouped_fp8_gemm_nt_contiguous(
                (a1q[:num_tokens], a1q_scales[:num_tokens]), (w, w_scale),
                out[:num_tokens], expert_ids[:num_tokens])
            pbar.update(1)
            num_tokens = num_tokens - block_m

    for w, ws in [(w1, w1_scale), (w2, w2_scale)]:
        if w.size() not in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE:
            _warmup(w, ws)
            GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE.add(w.size())


def deepgemm_fp8_gemm_nt_warmup(model: torch.nn.Module, max_tokens: int):
    dg_modules = [
        m for m in model.modules() if _fp8_linear_may_use_deep_gemm(m)
    ]

    for dgm in dg_modules:
        w, ws, _ = _extract_data_from_linear_base_module(dgm)
        _deepgemm_fp8_gemm_nt_warmup(w=w, ws=ws, max_tokens=max_tokens)


def deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(model: torch.nn.Module,
                                                   max_tokens: int):
    dg_modules = [
        m for m in model.modules()
        if _fused_moe_grouped_gemm_may_use_deep_gemm(m)
    ]

    for dgm in dg_modules:
        w13, w13_scale, w2, w2_scale, num_topk = (
            _extract_data_from_fused_moe_module(dgm))
        _deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(
            w13, w2, w13_scale, w2_scale, num_topk, max_tokens)


def deep_gemm_warmup(model: torch.nn.Module, max_tokens: int):
    deepgemm_fp8_gemm_nt_warmup(model, max_tokens)
    deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(model, max_tokens)
