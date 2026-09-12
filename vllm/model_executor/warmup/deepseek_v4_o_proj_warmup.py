# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-compile the DeepGEMM kernels of the DeepSeek V4 output projection.

DeepGEMM JIT-compiles a kernel per configuration, and its configuration for
the o-projection einsum and the wo_b GEMM follows the token count. Batches
above the CUDA graph range are padded to O_PROJ_EAGER_BUCKET multiples in
deep_gemm_fp8_o_proj, so the set of configurations an eager step can need
is small and known; run each once here instead of paying ~3 s for it the
first time a prefill step of that size arrives.
"""

import time

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v4.nvidia.ops.o_proj import o_proj_warmup_num_tokens

logger = init_logger(__name__)


def _find_o_proj_attention(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if hasattr(module, "_einsum_recipe") and hasattr(module, "_o_proj"):
            return module
    return None


def deepseek_v4_o_proj_warmup(model: torch.nn.Module, max_num_tokens: int) -> None:
    attn = _find_o_proj_attention(model)
    if attn is None:
        return
    sizes = o_proj_warmup_num_tokens(max_num_tokens)
    if not sizes:
        return
    head_dim = attn.nope_head_dim + attn.rope_head_dim
    device = attn.wo_a.weight.device
    start = time.perf_counter()
    o = torch.zeros(sizes[-1], attn.n_local_heads, head_dim, dtype=torch.bfloat16, device=device)
    positions = torch.zeros(sizes[-1], dtype=torch.int64, device=device)
    for n in sizes:
        attn._o_proj(o[:n], positions[:n])
    torch.cuda.synchronize(device)
    logger.info(
        "Warmed up DeepSeek V4 o-projection kernels for %d token sizes in %.2fs",
        len(sizes),
        time.perf_counter() - start,
    )
