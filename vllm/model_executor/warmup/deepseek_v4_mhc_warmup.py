# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).

Every decision below is a pure function of ``vllm_config``. The warmup drives
``runner._dummy_run``, whose forward pass issues TP collectives, so every rank
must take the same branch; per-rank module state (layer attributes, parameter
devices) must not be consulted. The ported version gated on ``hc_pre``/
``hc_post`` attributes only the AMD and XPU layers expose, making it a silent
no-op on every CUDA build. One residual per-rank input remains: ``n_splits``
depends on the local GPU's SM count, so ranks on heterogeneous GPUs would
derive different ladders — homogeneous GPUs per TP group are assumed, as the
kernels themselves already do.
"""

import time

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384

_MHC_HF_CONFIG_ATTRS = ("hc_mult", "hc_sinkhorn_iters", "hc_eps")


def _uses_mhc_tilelang(vllm_config) -> bool:
    hf_config = vllm_config.model_config.hf_config
    return (
        current_platform.is_cuda_alike()
        and getattr(hf_config, "model_type", None) == "deepseek_v4"
        and all(hasattr(hf_config, attr) for attr in _MHC_HF_CONFIG_ATTRS)
    )


def _compile_key(num_tokens: int, hidden_size: int, hc_mult: int) -> tuple[int, int]:
    """The (n_splits, tile_n) constexpr pair a call at ``num_tokens`` compiles.

    Mirrors the derivation in the mHC wrappers. ``num_tokens`` itself is a
    ``T.dynamic`` dimension and never keys a compilation; only these two
    constexprs do, so one cubin serves every token count that maps to the
    same pair.
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split
    from vllm.utils.math_utils import cdiv

    if num_tokens <= 16:
        tile_n = 2 if num_tokens < 8 else 3
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
        return (n_splits, tile_n)
    return (compute_num_split(64, hc_mult * hidden_size, cdiv(num_tokens, 64)), 1)


def _token_sizes_to_warm(
    *,
    max_tokens: int,
    hidden_size: int,
    hc_mult: int,
    capture_sizes: list[int],
) -> list[int]:
    """One representative token count per compile key capture does not own.

    ``n_splits`` follows a ``num_sms // ceil(tokens/64)`` staircase that a
    power-of-two ladder samples unevenly; enumerating the keys directly cannot
    miss a step. Filtering is by key, not by size: capture size 32 compiles
    the same cubin as every token count in [17, 64], so that whole bucket
    needs no dummy run even though none of its sizes is a capture size.
    """
    covered = {
        _compile_key(size, hidden_size, hc_mult) for size in capture_sizes if size > 0
    }
    representative: dict[tuple[int, int], int] = {}
    for num_tokens in range(1, max_tokens + 1):
        key = _compile_key(num_tokens, hidden_size, hc_mult)
        if key not in covered:
            representative.setdefault(key, num_tokens)
    return sorted(representative.values())


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    runner,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    """Compile the mHC TileLang kernels at startup instead of mid-request.

    Driven with real dummy forwards rather than direct kernel calls: which
    TileLang variant a call resolves to depends on tensor shape (the first
    local layer's 2-D residual selects the broadcast kernel), and the pre
    kernels also JIT a prenorm GEMM (DeepGEMM/Triton) that direct TileLang
    compilation would leave cold. A forward pass cannot get either wrong.

    Args:
        runner: the model runner, used for ``vllm_config`` and ``_dummy_run``.
        max_tokens: scheduler ``max_num_batched_tokens``.
        cudagraph_capture_sizes: sizes graph capture already compiles; any
            compile key they own is skipped here.
    """
    if not _uses_mhc_tilelang(runner.vllm_config):
        return

    hf_config = runner.vllm_config.model_config.hf_config
    token_sizes = _token_sizes_to_warm(
        max_tokens=min(max_tokens, _AUTO_WARMUP_MAX_TOKENS),
        hidden_size=int(hf_config.hidden_size),
        hc_mult=int(hf_config.hc_mult),
        capture_sizes=cudagraph_capture_sizes or [],
    )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC TileLang kernels for token sizes: %s",
        token_sizes,
    )
    for num_tokens in token_sizes:
        runner._dummy_run(num_tokens)
    torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4 mHC TileLang warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
