# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _is_attention_backend(backend, name: str) -> bool:
    try:
        return backend.get_name() == name
    except NotImplementedError:
        return False


def _uses_triton_attention(runner: "GPUModelRunner") -> bool:
    return any(
        _is_attention_backend(group.backend, "TRITON_ATTN")
        for groups in runner.attn_groups
        for group in groups
    )


def _warmup_triton_nvfp4_prefill_kernels(runner: "GPUModelRunner") -> None:
    """Warm NVFP4 pure-prefill Triton kernels missed by dummy runs.

    The NVFP4 Triton path can bypass the paged cache for pure prefill and call
    `context_attention_fwd` directly. Hybrid models may have several Triton
    prefill specializations, for example full and sliding-window attention with
    different head sizes. Use tiny synthetic tensors with the real layer shapes
    so those variants compile before the JIT monitor is enabled.
    """
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

    warmup_tokens = min(runner.max_num_tokens, runner.max_model_len, 64)
    if warmup_tokens <= 0:
        return

    b_start_loc = torch.zeros((1,), dtype=torch.int32, device=runner.device)
    b_seq_len = torch.full((1,), warmup_tokens, dtype=torch.int32, device=runner.device)

    seen: set[tuple] = set()
    for groups in runner.attn_groups:
        for group in groups:
            if not _is_attention_backend(group.backend, "TRITON_ATTN"):
                continue

            layer_names = getattr(group, "layer_names", ())
            if not layer_names:
                continue

            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                runner.vllm_config,
                layer_type,
                layer_names,
            )
            for layer_name in layer_names:
                layer = layers.get(layer_name)
                if layer is None:
                    continue

                impl = cast(Any, layer.impl)
                if (
                    getattr(impl, "kv_cache_dtype", None) != "nvfp4"
                    or getattr(impl, "kv_sharing_target_layer_name", None) is not None
                    or getattr(impl, "alibi_slopes", None) is not None
                    or getattr(impl, "use_alibi_sqrt", False)
                    or getattr(impl, "sinks", None) is not None
                    or getattr(impl, "chunk_lookback", -1) != -1
                ):
                    continue

                sliding_window = getattr(impl, "sliding_window", (-1, -1))
                key = (
                    impl.num_heads,
                    impl.num_kv_heads,
                    impl.head_size,
                    impl.scale,
                    impl.logits_soft_cap,
                    sliding_window,
                    runner.dtype,
                )
                if key in seen:
                    continue
                seen.add(key)

                q = torch.zeros(
                    (warmup_tokens, impl.num_heads, impl.head_size),
                    dtype=runner.dtype,
                    device=runner.device,
                )
                k = torch.zeros(
                    (warmup_tokens, impl.num_kv_heads, impl.head_size),
                    dtype=runner.dtype,
                    device=runner.device,
                )
                v = torch.zeros_like(k)
                out = torch.empty_like(q)

                context_attention_fwd(
                    q=q,
                    k=k,
                    v=v,
                    o=out,
                    b_start_loc=b_start_loc,
                    b_seq_len=b_seq_len,
                    max_input_len=warmup_tokens,
                    is_causal=True,
                    softmax_scale=impl.scale,
                    softcap=impl.logits_soft_cap,
                    sliding_window_q=sliding_window[0],
                    sliding_window_k=sliding_window[1],
                )


def _warmup_triton_nvfp4_attention(runner: "GPUModelRunner") -> None:
    if (
        runner.is_pooling_model
        or runner.cache_config.cache_dtype != "nvfp4"
        or not runner.attn_groups
        or not _uses_triton_attention(runner)
    ):
        return

    num_tokens = runner.uniform_decode_query_len
    if num_tokens <= 0 or num_tokens > runner.max_num_tokens:
        return

    # Warm up a decode-shaped NVFP4 Triton attention variant before the JIT
    # monitor is activated.  A small batch is enough for Triton specialization:
    # the long context length is a runtime value, but keeping seq_lens > 1
    # exercises the decode cache-read path rather than pure self-attention.
    profile_seq_lens = min(runner.max_model_len, max(num_tokens + 1, 8192))
    logger.info("Warming up Triton NVFP4 attention.")
    runner._dummy_run(
        num_tokens=num_tokens,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        uniform_decode=True,
        profile_seq_lens=profile_seq_lens,
    )

    # NVFP4 prefill can bypass the paged cache and use the Triton context
    # attention kernel directly. Warm a representative chunk-sized pure-prefill
    # shape as well so this kernel does not JIT on the first long prompt.
    prefill_tokens = min(runner.max_num_tokens, runner.max_model_len, 8192)
    if prefill_tokens > 1:
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=False,
        )

    _warmup_triton_nvfp4_prefill_kernels(runner)


def _flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    factors = aot_compile_hash_factors(runner.vllm_config)
    return hashlib.sha256(str(factors).encode()).hexdigest()


def _resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / _flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def kernel_warmup(worker: "Worker"):
    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    minimax_m3_msa_warmup(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    _warmup_triton_nvfp4_attention(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_attention_backend(group.backend, "FLASHINFER")
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


# TODO: remove once FlashInfer upstream fixes the persistent file cache
# to resolve collisions like `use_8x4_sf_layout=True/False`, which causes
# invalid tactics to be chosen
_FLASHINFER_USE_PERSISTENT_CACHE = False


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Tuning is performed only on rank 0. The resulting cache is broadcast
    to every rank so all ranks dispatch the same kernel tactic.
    """
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    if not _FLASHINFER_USE_PERSISTENT_CACHE:
        with torch.inference_mode(), fi_utils.autotune():
            runner._dummy_run(
                num_tokens=runner.scheduler_config.max_num_batched_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        get_world_group().barrier()
        return

    world = get_world_group()
    is_leader = world.rank_in_group == 0

    cache_path = _resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # When autotuning with number of tokens m, flashinfer will autotune
    # operations for all number of tokens up to m, so we only need to
    # run with the max number of tokens.
    dummy_run_kwargs = dict(
        num_tokens=runner.scheduler_config.max_num_batched_tokens,
        skip_eplb=True,
        is_profile=True,
    )

    with torch.inference_mode():
        if is_leader:
            with fi_utils.autotune(tune_mode=True, cache=str(cache_path)):
                runner._dummy_run(**dummy_run_kwargs)
        else:
            runner._dummy_run(**dummy_run_kwargs)

    # Broadcast autotune cache from rank 0 to all other ranks so every
    # rank loads the same set of chosen tactics.
    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)

    if tune_results is None:
        logger.warning(
            "No FlashInfer autotune cache entries found."
            "Falling back to default tactics."
        )
    else:
        if not is_leader and world.local_rank == 0:
            with open(cache_path, "wb") as f:
                f.write(tune_results)
        world.barrier()
        from flashinfer.autotuner import AutoTuner

        AutoTuner.get().load_configs(str(cache_path))
        logger.info(
            "FlashInfer autotune cache loaded on rank %d from %s.",
            world.rank_in_group,
            cache_path,
        )
