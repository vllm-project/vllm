# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warmup and autotune helpers for FlashInfer sparse MLA backends."""

from typing import TYPE_CHECKING, cast

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.flashinfer import autotune_sparse_mla_only, has_flashinfer
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2GPUModelRunner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE_DSV4",
        "FLASHINFER_MLA_SPARSE_DSV4",
        "ROCM_FLASHMLA_SPARSE_DSV4",
        "FLASHMLA_SPARSE_DSV41",
        "FLASHINFER_MLA_SPARSE_DSV41",
        "DEEPSEEK_SPARSE_SWA",
    }
)
_FLASHINFER_MLA_SPARSE_BACKENDS = frozenset({"FLASHINFER_MLA_SPARSE_SM120"})
_DEEPSEEK_V4_FLASHINFER_MLA_SPARSE_BACKENDS = frozenset(
    {"FLASHINFER_MLA_SPARSE_DSV4", "FLASHINFER_MLA_SPARSE_DSV41"}
)

_FLASHINFER_SM120_SPARSE_MLA_DECODE_LABELS = {
    "FLASHINFER_MLA_SPARSE_SM120": "DSv3.2",
    "FLASHINFER_MLA_SPARSE_DSV4": "DSv4",
    "FLASHINFER_MLA_SPARSE_DSV41": "DSv4.1",
}

_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16

# Capture buckets the sparse-MLA decode kernel can serve: the crossover probe
# grid caps at 64, so larger buckets never route to the decode form.
_SPARSE_MLA_REFINE_TOKEN_CAP = 64


def _sparse_mla_refine_tokens(worker: "Worker") -> tuple[int, ...]:
    """Return capture buckets executable by the runner's uniform-decode dummy run."""
    sizes = (
        getattr(worker.vllm_config.compilation_config, "cudagraph_capture_sizes", None)
        or ()
    )
    runner = worker.model_runner
    is_v2 = _uses_v2_model_runner(runner)
    query_len = (
        cast("V2GPUModelRunner", runner).decode_query_len
        if is_v2
        else runner.uniform_decode_query_len
    )
    cap = min(
        _SPARSE_MLA_REFINE_TOKEN_CAP,
        runner.max_num_reqs * query_len,
        worker.scheduler_config.max_num_batched_tokens,
    )
    # V1 permits a final short request; V2 requires exact query-length multiples.
    return tuple(
        sorted({s for s in sizes if 0 < s <= cap and (not is_v2 or s % query_len == 0)})
    )


def autotune_hisparse_flashinfer_attention(runner: "GPUModelRunner") -> None:
    """Autotune each HiSparse FlashInfer sparse-MLA configuration."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseImpl,
    )

    tuned: set[tuple[object, ...]] = set()
    for layer in runner.vllm_config.compilation_config.static_forward_context.values():
        impl = getattr(layer, "impl", None)
        if not isinstance(impl, FlashInferMLASparseImpl):
            continue
        if getattr(layer, "hisparse_cache", None) is None:
            continue
        if impl.topk_indices_buffer is None:
            continue
        key = (
            impl.kv_cache_dtype,
            impl.num_heads,
            impl.qk_nope_head_dim,
            impl.qk_rope_head_dim,
            impl.kv_lora_rank,
            impl.topk_indices_buffer.shape[1],
        )
        if key in tuned:
            continue
        impl.autotune_hisparse_decode(layer)
        tuned.add(key)


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_deepseek_v4_sparse_mla_backend(runner: "GPUModelRunner") -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS:
                return True
    return False


def _flashinfer_sparse_mla_decode_label(
    runner: "GPUModelRunner",
    allowed_backends: frozenset[str],
) -> str | None:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in allowed_backends:
                return _FLASHINFER_SM120_SPARSE_MLA_DECODE_LABELS.get(name)
    return None


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _uses_v2_model_runner(runner: "GPUModelRunner") -> bool:
    vllm_config = getattr(runner, "vllm_config", None)
    return bool(getattr(vllm_config, "use_v2_model_runner", False))


def _run_flashinfer_sparse_mla_decode_autotune(
    worker: "Worker",
    num_tokens: int,
    allowed_backends: frozenset[str],
) -> bool:
    """Tune sparse calls on every rank; FlashInfer owns persistence.

    Eager token buckets do not cover every speculative or graph descriptor.
    """
    runner = worker.model_runner
    log_label = _flashinfer_sparse_mla_decode_label(runner, allowed_backends)
    if log_label is None:
        return False
    if worker.vllm_config.kernel_config.enable_flashinfer_autotune is not True:
        return False
    if not has_flashinfer() or not current_platform.is_device_capability_family(120):
        return False

    is_v2 = _uses_v2_model_runner(runner)
    if is_v2 and getattr(runner, "ubatch_runner", None) is not None:
        logger.warning(
            "Skipping FlashInfer SM120 sparse MLA decode autotune: "
            "V2 microbatching does not expose a serial warmup override."
        )
        return False

    logger.info("Autotuning FlashInfer SM120 sparse MLA %s decode.", log_label)
    skip_ops = set(envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS or ()) or None
    with torch.inference_mode(), autotune_sparse_mla_only(skip_ops=skip_ops):
        if not _run_sparse_mla_mixed_warmup(worker, num_tokens):
            return False
        for bucket in _sparse_mla_refine_tokens(worker):
            if is_v2:
                v2_runner = cast("V2GPUModelRunner", runner)
                v2_runner._dummy_run(
                    num_tokens=bucket,
                    skip_eplb=True,
                    is_profile=True,
                    skip_attn=False,
                    uniform_decode=True,
                )
            else:
                runner._dummy_run(
                    num_tokens=bucket,
                    skip_eplb=True,
                    is_profile=True,
                    force_attention=True,
                    uniform_decode=True,
                    allow_microbatching=False,
                )
    return True


def _run_sparse_mla_mixed_warmup(worker: "Worker", num_tokens: int) -> bool:
    runner = worker.model_runner
    if _uses_v2_model_runner(runner):
        v2_runner = cast("V2GPUModelRunner", runner)
        if runner.max_num_reqs >= 2:
            return run_mixed_prefill_decode_warmup(
                v2_runner,
                worker.execute_model,
                worker.sample_tokens,
                num_tokens,
                req_id_prefix="_sparse_mla_v2_warmup",
            )
        # A single request cannot form a mixed batch.
        v2_runner._dummy_run(
            num_tokens=num_tokens, skip_eplb=True, is_profile=True, skip_attn=False
        )
    else:
        runner._dummy_run(
            num_tokens=num_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
            allow_microbatching=False,
        )
    return True


def _flashinfer_sparse_mla_decode_autotune(
    worker: "Worker",
    num_tokens: int,
) -> bool:
    return _run_flashinfer_sparse_mla_decode_autotune(
        worker, num_tokens, _FLASHINFER_MLA_SPARSE_BACKENDS
    )


def _deepseek_v4_sparse_mla_decode_autotune(
    worker: "Worker",
    num_tokens: int,
) -> bool:
    return _run_flashinfer_sparse_mla_decode_autotune(
        worker, num_tokens, _DEEPSEEK_V4_FLASHINFER_MLA_SPARSE_BACKENDS
    )


def flashinfer_sparse_mla_decode_autotune_warmup(worker: "Worker") -> None:
    """Autotune generic FlashInfer sparse MLA decode when selected."""
    runner = worker.model_runner
    if runner.is_pooling_model:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens)
    if mixed_tokens <= 0:
        return
    _flashinfer_sparse_mla_decode_autotune(worker, mixed_tokens)


def deepseek_v4_sparse_mla_attention_warmup(worker: "Worker") -> None:
    """Warm DSv4 sparse-MLA mixed prefill+decode attention."""
    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens)
    if mixed_tokens <= 0:
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention for mixed tokens=%s.",
        mixed_tokens,
    )
    mixed_warmup_done = _deepseek_v4_sparse_mla_decode_autotune(worker, mixed_tokens)
    if not mixed_warmup_done:
        _run_sparse_mla_mixed_warmup(worker, mixed_tokens)
