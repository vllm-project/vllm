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
from vllm.distributed.parallel_state import get_dp_group, is_global_first_rank
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoEModularMethod
from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel import (
    FP8W8A16LinearKernel,
)
from vllm.utils.deep_gemm import (
    fp8_gemm_nt,
    get_mk_alignment_for_contiguous_layout,
    m_grouped_fp8_gemm_nt_contiguous,
)
from vllm.utils.math_utils import cdiv


def _generate_optimal_warmup_m_values(
    max_tokens: int, n: int, device: torch.device
) -> list[int]:
    """
    Generate M values that cover all possible DeepGEMM kernel configurations.
    Reference: https://github.com/deepseek-ai/DeepGEMM/blob/79f48ee15a82dd5fad5cd9beaa393c1f755e6b55/csrc/jit_kernels/heuristics/common.hpp

    Args:
        max_tokens: Maximum number of tokens to warmup for
        n: The actual N dimension from the weight tensor
        device: The torch device to get properties from.
    """

    # DeepGEMM's possible block sizes
    block_ms = [64, 128, 256]
    block_ns = list(range(16, min(257, n + 1), 16))
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    m_values = set()

    # Always include small cases
    m_values.update([1, 2, 4] + [i for i in range(8, 65, 8)])

    # Collect M values where different wave patterns occur
    for block_m in block_ms:
        for block_n in block_ns:
            if block_n > n:
                continue

            # Add key M boundaries for this block combination
            for wave in range(1, 11):  # Up to 10 waves
                # M where this block config transitions to next wave
                target_blocks = wave * num_sms
                m = target_blocks * block_m // cdiv(n, block_n)
                if 1 <= m <= max_tokens:
                    m_values.add(m)

            # Add block_m boundaries
            for multiple in range(1, max_tokens // block_m + 1):
                m = multiple * block_m
                if m <= max_tokens:
                    m_values.add(m)

    return sorted(m_values)


def _extract_data_from_linear_base_module(
    m: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Extract weights, weight scales and quantization block sizes from the given
    LinearBase module.
    """
    assert isinstance(m, LinearBase)
    assert isinstance(m.quant_method, Fp8LinearMethod)
    assert m.quant_method.block_quant
    assert m.quant_method.quant_config is not None

    w = m.weight
    ws = m.weight_scale_inv if hasattr(m, "weight_scale_inv") else m.weight_scale
    quant_block_size = m.quant_method.quant_config.weight_block_size

    assert isinstance(w, torch.Tensor)
    assert isinstance(ws, torch.Tensor)
    assert quant_block_size is not None
    return (w, ws, quant_block_size)


def _extract_data_from_fused_moe_module(
    m: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Extract weights, weight scales and num_topk from FusedMoE module.
    """
    assert isinstance(m, FusedMoE)
    w13 = m.w13_weight
    w13_s = (
        m.w13_weight_scale_inv
        if hasattr(m, "w13_weight_scale_inv")
        else m.w13_weight_scale
    )
    w2 = m.w2_weight
    w2_s = (
        m.w2_weight_scale_inv
        if hasattr(m, "w2_weight_scale_inv")
        else m.w2_weight_scale
    )
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

    # FIXME: this logic is brittle and incorrect - since we
    # could use DeepGEMM with for than just Fp8LinearMethod
    block_size = get_mk_alignment_for_contiguous_layout()[0]
    if not (
        isinstance(module, LinearBase)
        and isinstance(module.quant_method, Fp8LinearMethod)
        and module.quant_method.block_quant
        and not isinstance(module.quant_method.kernel, FP8W8A16LinearKernel)
    ):
        return False

    w, _, block_sizes = _extract_data_from_linear_base_module(module)
    return (
        block_sizes == get_mk_alignment_for_contiguous_layout()
        and w.ndim == 2
        and w.shape[0] % block_size == 0
        and w.shape[1] % block_size == 0
    )


def _fused_moe_grouped_gemm_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    if not (envs.VLLM_USE_DEEP_GEMM and envs.VLLM_MOE_USE_DEEP_GEMM):
        return False

    if not isinstance(module, FusedMoE):
        return False

    moe_quant_config = module.quant_method.get_fused_moe_quant_config(module)

    if (
        moe_quant_config is None
        or moe_quant_config.quant_dtype != torch.float8_e4m3fn
        or moe_quant_config.block_shape != get_mk_alignment_for_contiguous_layout()
    ):
        return False

    if not isinstance(module.quant_method, FusedMoEModularMethod):
        # modular kernels could invoke deep_gemm_moe_fp8
        return True

    # Further check if the ModularKernel implementation uses the DeepGemmExperts
    return isinstance(
        module.quant_method.moe_mk, (DeepGemmExperts, TritonOrDeepGemmExperts)
    )


FP8_GEMM_NT_WARMUP_CACHE: set[torch.Size] = set()


def _get_fp8_gemm_nt_m_values(w: torch.Tensor, max_tokens: int) -> list[int]:
    """Get the M values to warmup for a given weight tensor."""
    n, _ = w.size()
    device = w.device

    # Use optimal M values only if VLLM_DEEP_GEMM_WARMUP is set to "relax".
    # Otherwise warmup all token sizes to avoid JIT compilation in hotpath
    if envs.VLLM_DEEP_GEMM_WARMUP == "relax":
        return _generate_optimal_warmup_m_values(max_tokens, n, device)
    else:
        assert envs.VLLM_DEEP_GEMM_WARMUP == "full", (
            "Expected "
            'VLLM_DEEP_GEMM_WARMUP env to be set to "full" but got '
            f"{envs.VLLM_DEEP_GEMM_WARMUP}"
        )
        return list(range(1, max_tokens + 1))


def _deepgemm_fp8_gemm_nt_warmup(
    w: torch.Tensor,
    ws: torch.Tensor,
    max_tokens: int,
    pbar: tqdm | None = None,
):
    if w.size() in FP8_GEMM_NT_WARMUP_CACHE:
        return

    n, k = w.size()
    block_m = get_mk_alignment_for_contiguous_layout()[0]

    device = w.device
    a1q = torch.empty((max_tokens, k), device=device, dtype=torch.float8_e4m3fn)
    a1q_scales = torch.empty(
        (max_tokens, k // block_m), device=device, dtype=torch.float32
    )
    out = torch.empty((max_tokens, n), device=device, dtype=torch.bfloat16)

    m_values = _get_fp8_gemm_nt_m_values(w, max_tokens)

    for num_tokens in m_values:
        fp8_gemm_nt(
            (a1q[:num_tokens], a1q_scales[:num_tokens]), (w, ws), out[:num_tokens]
        )
        if pbar is not None:
            pbar.update(1)

    FP8_GEMM_NT_WARMUP_CACHE.add(w.size())


GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE: set[torch.Size] = set()


def _get_grouped_gemm_params(
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_topk: int,
    max_tokens: int,
) -> tuple[int, int, torch.Tensor]:
    assert w1.size(0) == w2.size(0), "w1 and w2 must have the same number of experts"

    block_m = get_mk_alignment_for_contiguous_layout()[0]
    num_experts = w1.size(0)
    device = w1.device

    # Assumes all ranks have the same max_num_batched_tokens
    max_tokens_across_dp = get_dp_group().world_size * max_tokens
    max_tokens = min(max_tokens_across_dp, envs.VLLM_FUSED_MOE_CHUNK_SIZE)

    # This is the maximum GroupedGemm M size that we expect to run
    # the grouped_gemm with.
    MAX_M = compute_aligned_M(
        max_tokens, num_topk, num_experts, block_m, expert_tokens_meta=None
    )
    # Distribute expert-ids evenly.
    MAX_BLOCKS = MAX_M // block_m
    expert_ids_block = torch.randint(
        low=0, high=num_experts, size=(MAX_BLOCKS,), device=device, dtype=torch.int32
    )
    expert_ids = torch.repeat_interleave(expert_ids_block, block_m, dim=0)

    return MAX_M, block_m, expert_ids


def _deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    num_topk: int,
    max_tokens: int,
    pbar: tqdm | None = None,
):
    if (
        w1.size() in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
        and w2.size() in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
    ):
        return

    MAX_M, block_m, expert_ids = _get_grouped_gemm_params(w1, w2, num_topk, max_tokens)
    device = w1.device

    def _warmup(w: torch.Tensor, w_scale: torch.Tensor):
        _, n, k = w.size()
        a1q = torch.empty((MAX_M, k), device=device, dtype=torch.float8_e4m3fn)
        a1q_scales = torch.empty(
            (MAX_M, k // block_m), device=device, dtype=torch.float32
        )
        out = torch.empty((MAX_M, n), device=device, dtype=torch.bfloat16)

        m_values = list(range(block_m, MAX_M + 1, block_m))

        for num_tokens in m_values:
            m_grouped_fp8_gemm_nt_contiguous(
                (a1q[:num_tokens], a1q_scales[:num_tokens]),
                (w, w_scale),
                out[:num_tokens],
                expert_ids[:num_tokens],
            )
            if pbar is not None:
                pbar.update(1)

    for w, ws in [(w1, w1_scale), (w2, w2_scale)]:
        if w.size() not in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE:
            _warmup(w, ws)
            GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE.add(w.size())


def deepgemm_fp8_gemm_nt_warmup(
    model: torch.nn.Module, max_tokens: int, pbar: tqdm | None = None
):
    dg_modules = [m for m in model.modules() if _fp8_linear_may_use_deep_gemm(m)]

    for dgm in dg_modules:
        w, ws, _ = _extract_data_from_linear_base_module(dgm)
        _deepgemm_fp8_gemm_nt_warmup(w=w, ws=ws, max_tokens=max_tokens, pbar=pbar)


def deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(
    model: torch.nn.Module, max_tokens: int, pbar: tqdm | None = None
):
    dg_modules = [
        m for m in model.modules() if _fused_moe_grouped_gemm_may_use_deep_gemm(m)
    ]

    for dgm in dg_modules:
        w13, w13_scale, w2, w2_scale, num_topk = _extract_data_from_fused_moe_module(
            dgm
        )
        _deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(
            w13, w2, w13_scale, w2_scale, num_topk, max_tokens, pbar=pbar
        )


def _count_warmup_iterations(model: torch.nn.Module, max_tokens: int) -> int:
    seen_fp8_sizes: set[torch.Size] = set(FP8_GEMM_NT_WARMUP_CACHE)
    seen_grouped_sizes: set[torch.Size] = set(
        GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
    )

    total = 0
    for m in model.modules():
        if _fp8_linear_may_use_deep_gemm(m):
            w, _, _ = _extract_data_from_linear_base_module(m)
            if w.size() not in seen_fp8_sizes:
                total += len(_get_fp8_gemm_nt_m_values(w, max_tokens))
                seen_fp8_sizes.add(w.size())
        elif _fused_moe_grouped_gemm_may_use_deep_gemm(m):
            w13, _, w2, _, num_topk = _extract_data_from_fused_moe_module(m)
            if w13.size() in seen_grouped_sizes and w2.size() in seen_grouped_sizes:
                continue
            MAX_M, block_m, _ = _get_grouped_gemm_params(w13, w2, num_topk, max_tokens)
            n_values = (MAX_M - block_m) // block_m + 1
            if w13.size() not in seen_grouped_sizes:
                total += n_values
                seen_grouped_sizes.add(w13.size())
            if w2.size() not in seen_grouped_sizes:
                total += n_values
                seen_grouped_sizes.add(w2.size())
    return total


def deep_gemm_warmup(model: torch.nn.Module, max_tokens: int):
    total = _count_warmup_iterations(model, max_tokens)
    if total == 0:
        return

    # Only show progress bar on rank 0 to avoid cluttered output
    if is_global_first_rank():
        with tqdm(total=total, desc="DeepGEMM warmup") as pbar:
            deepgemm_fp8_gemm_nt_warmup(model, max_tokens, pbar)
            deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(model, max_tokens, pbar)
    else:
        deepgemm_fp8_gemm_nt_warmup(model, max_tokens, None)
        deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(model, max_tokens, None)
