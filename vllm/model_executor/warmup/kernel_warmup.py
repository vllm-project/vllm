# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.config.mamba import MambaBackendEnum
from vllm.logger import init_logger
from vllm.model_executor.warmup.b12x_warmup import b12x_warmup
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.model_executor.warmup.fa4_cutedsl_warmup import (
    fa4_cutedsl_warmup,
)
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_file,
    write_flashinfer_autotune_cache,
)
from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    deepseek_v4_sparse_mla_attention_warmup,
    flashinfer_sparse_mla_decode_autotune_warmup,
)
from vllm.model_executor.warmup.kimi_k3_triton_warmup import (
    kimi_k3_triton_warmup,
)
from vllm.model_executor.warmup.qwen_triton_warmup import qwen_triton_warmup
from vllm.model_executor.warmup.sparse_mla_triton_warmup import (
    sparse_mla_triton_warmup,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_LL_BF16_WARMUP_M_RANGE = range(1, 17)


def _ll_bf16_router_shapes_from_model(
    model: torch.nn.Module,
) -> tuple[tuple[int, int], ...]:
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    shapes: set[tuple[int, int]] = set()
    for module in model.modules():
        if not isinstance(module, GateLinear):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        n, k = weight.shape
        if k % 8 == 0:
            shapes.add((int(k), int(n)))
    return tuple(sorted(shapes))


def _warmup_ll_bf16_router_gemm(model: torch.nn.Module) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available as is_ll_bf16_gemm_available,
    )
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm_kernel,
    )

    if not is_ll_bf16_gemm_available():
        return

    shapes = _ll_bf16_router_shapes_from_model(model)
    if not shapes:
        logger.debug_once(
            "Skipping ll_bf16 router GEMM warmup: no bf16 GateLinear shapes found."
        )
        return

    logger.info_once("Warming up ll_bf16 router GEMM kernels for shapes: %s.", shapes)
    ll_bf16_gemm_kernel.warmup(
        shapes=shapes,
        m_values=_LL_BF16_WARMUP_M_RANGE,
    )


def kernel_warmup(worker: "Worker", *, process_local_only: bool = False):
    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )

    if not worker.use_v2_model_runner:
        # The KV-block zeroing kernel is driven by the scheduler's
        # `new_block_ids_to_zero`, so no dummy run ever reaches it.
        zeroer = getattr(worker.model_runner, "_kv_block_zeroer", None)
        if zeroer is not None:
            zeroer.warmup(worker.model_runner.kv_cache_config.num_blocks)

    if worker.vllm_config.kernel_config.enable_jit_warmup:
        logger.info("JIT kernel warmup starting.")
        jit_warmup_start = time.perf_counter()
        try:
            worker.model_runner.jit_warmup_registry.warmup()
        except Exception:
            logger.exception(
                "JIT kernel warmup failed after %.2fs.",
                time.perf_counter() - jit_warmup_start,
            )
            raise
        logger.info(
            "JIT kernel warmup finished in %.2fs.",
            time.perf_counter() - jit_warmup_start,
        )

    qwen_triton_warmup(worker.model_runner, worker.vllm_config.model_config)

    compilation_config = worker.vllm_config.compilation_config
    cudagraph_capture_sizes = list(compilation_config.cudagraph_capture_sizes or [])

    # DSv4 mHC TileLang kernels (hc_pre/hc_post/hc_head_op) run every decoder
    # layer per token; warm them across token sizes first so the first real
    # request doesn't pay JIT cost. No-op for non-DSv4 models (gated inside).
    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )

    # Run next so input-prep kernels JIT against pristine runner state.
    if worker.vllm_config.kernel_config.enable_jit_warmup:
        kimi_k3_triton_warmup(worker)
        fa4_cutedsl_warmup(worker)
        sparse_mla_triton_warmup(worker)

    if current_platform.has_device_capability(90):
        _warmup_ll_bf16_router_gemm(worker.get_model())

    if worker.vllm_config.kernel_config.enable_cutedsl_warmup:
        # TODO(roberto): Remove after registered CuTeDSL warmups are migrated
        # to the shared JIT warmup infrastructure.
        # https://github.com/vllm-project/vllm/pull/47451
        cutedsl_warmup()

    if process_local_only:
        return

    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    deepseek_v4_sparse_mla_attention_warmup(worker)

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

    b12x_warmup(worker, cudagraph_capture_sizes)

    minimax_m3_msa_warmup(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info_once("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info_once("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


def _flashinfer_autotune_skip_ops(runner: "GPUModelRunner") -> set[str] | None:
    if envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS is not None:
        return set(envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS) or None

    from vllm.model_executor.kernels.linear import (
        FlashInferCuteDslNvFp4LinearKernel,
    )

    for module in runner.get_model().modules():
        for holder_name in ("quant_method", "scheme"):
            kernel = getattr(getattr(module, holder_name, None), "kernel", None)
            # CuTe-DSL mm_fp4 tuning JIT-compiles every tactic and its
            # fallback is already the heuristic; all mm_fp4 backends share
            # the "fp4_gemm" op name, so skip only when cute-dsl is selected.
            if isinstance(kernel, FlashInferCuteDslNvFp4LinearKernel):
                return {"fp4_gemm"}
    return None


def _flashinfer_replayssm_autotune_kwargs(
    runner: "GPUModelRunner", max_token_prefill_kwargs: dict[str, Any]
) -> tuple[int, dict[str, Any]] | None:
    config = runner.vllm_config
    if not (
        config.cache_config.use_replayssm
        and config.mamba_config.backend == MambaBackendEnum.FLASHINFER
    ):
        return None
    use_v2_model_runner = config.use_v2_model_runner
    v2_runner: Any = runner
    query_len = (
        v2_runner.decode_query_len
        if use_v2_model_runner
        else runner.uniform_decode_query_len
    )
    max_num_reqs = min(
        runner.scheduler_config.max_num_seqs,
        runner.max_num_tokens // query_len,
        runner.kv_cache_config.num_blocks - 1,
    )
    if max_num_reqs <= 0:
        logger.warning_once(
            "Skipping FlashInfer ReplaySSM autotuning because no non-padding "
            "state slot is available."
        )
        return None

    decode_kwargs = {
        **max_token_prefill_kwargs,
        "num_tokens": max_num_reqs * query_len,
        "uniform_decode": True,
    }
    if use_v2_model_runner:
        decode_kwargs["dummy_first_block_id"] = 1
    else:
        decode_kwargs.update(
            allow_microbatching=False,
            force_attention=True,
            profile_seq_lens=query_len + 1,
        )
    return max_num_reqs, decode_kwargs


@contextmanager
def _temporary_replayssm_autotune_slots(
    runner: "GPUModelRunner", max_num_reqs: int
) -> Iterator[None]:
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
        reset_replayssm_ring_trackers,
    )

    reset_tensors: list[torch.Tensor] = []
    seen: set[int] = set()
    tracker_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    seen_tracker_pairs: set[tuple[int, int]] = set()
    for module in runner.get_model().modules():
        if not isinstance(module, MambaMixer2) or not module.use_replayssm:
            continue
        tracker_pair = (
            module._replayssm_ring_start,
            module._replayssm_prev_num_accepted,
        )
        tracker_ptrs = (tracker_pair[0].data_ptr(), tracker_pair[1].data_ptr())
        if tracker_ptrs not in seen_tracker_pairs:
            tracker_pairs.append(tracker_pair)
            seen_tracker_pairs.add(tracker_ptrs)
        tensors = (
            *module.kv_cache,
            *tracker_pair,
        )
        for tensor in tensors:
            if not tensor.numel():
                continue
            data_ptr = tensor.data_ptr()
            if data_ptr not in seen:
                reset_tensors.append(tensor)
                seen.add(data_ptr)

    use_v2_model_runner = runner.vllm_config.use_v2_model_runner
    v2_runner: Any = runner
    block_tables = saved_block_ids = None
    if not use_v2_model_runner:
        block_tables = runner.input_batch.block_table.block_tables
        saved_block_ids = tuple(
            block_table.block_table.np[:max_num_reqs, 0].copy()
            for block_table in block_tables
        )
        dummy_block_ids = range(1, max_num_reqs + 1)
        for block_table in block_tables:
            block_table.block_table.np[:max_num_reqs, 0] = dummy_block_ids

    if tracker_pairs and tracker_pairs[0][0].is_cuda:
        if use_v2_model_runner:
            # Match the strided first-column view used by production prefill;
            # Triton specializes this tracker reset separately from a fresh,
            # contiguous arange tensor.
            state_batch_indices = v2_runner.block_tables.get_dummy_block_tables(
                max_num_reqs, first_block_id=1
            )[0][:, 0]
        else:
            state_batch_indices = torch.arange(
                1,
                max_num_reqs + 1,
                dtype=torch.int32,
                device=tracker_pairs[0][0].device,
            )
        for ring_start, prev_num_accepted in tracker_pairs:
            reset_replayssm_ring_trackers(
                ring_start,
                prev_num_accepted,
                state_batch_indices,
            )

    try:
        yield
    finally:
        if use_v2_model_runner:
            v2_runner.block_tables.get_dummy_block_tables(max_num_reqs)
        else:
            assert block_tables is not None and saved_block_ids is not None
            for block_table, block_ids in zip(block_tables, saved_block_ids):
                block_table.block_table.np[:max_num_reqs, 0] = block_ids
            runner.input_batch.block_table.commit_block_table(max_num_reqs)
        for tensor in reset_tensors:
            tensor[1 : max_num_reqs + 1].zero_()


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Every rank profiles the same tactics. When distributed, per-tactic
    timings are averaged over the world CPU group so all ranks select the
    same tactic.
    """
    from flashinfer.autotuner import AutoTuner, set_autotune_process_group

    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    world = get_world_group()
    is_leader = world.rank_in_group == 0
    tuner = AutoTuner.get()

    autotune_kwargs: dict = {}
    skip_ops = _flashinfer_autotune_skip_ops(runner)
    if skip_ops:
        logger.info_once(
            "Skipping FlashInfer autotuning for ops %s",
            tuple(sorted(skip_ops)),
        )
        autotune_kwargs["skip_ops"] = skip_ops

    cache_path = resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info_once("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # When autotuning with number of tokens m, flashinfer will autotune
    # operations for all number of tokens up to m, so we only need to
    # run with the max number of tokens.
    # Randomize inputs to avoid every token pick the same experts,
    # which lead to some EP ranks receiving no tokens and skipping their
    # MoE kernel entirely, and cause hang due to all-reduce collective
    # during synchronized autotuning.
    max_token_prefill_kwargs = dict(
        num_tokens=runner.scheduler_config.max_num_batched_tokens,
        skip_eplb=True,
        is_profile=True,
        randomize_inputs=True,
    )
    replayssm_autotune = _flashinfer_replayssm_autotune_kwargs(
        runner, max_token_prefill_kwargs
    )

    # Read cached autotune results and broadcast to all ranks.
    cached_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached_results = f.read()
    cached_results = world.broadcast_object(cached_results, src=0)
    if cached_results is not None:
        write_flashinfer_autotune_cache(cache_path, cached_results)
        world.barrier()
        tuner.load_configs(str(cache_path))

    group = world.cpu_group if world.world_size > 1 else None
    set_autotune_process_group(group)
    try:
        with (
            torch.inference_mode(),
            fi_utils.autotune(tune_mode=True, **autotune_kwargs),
        ):
            runner._dummy_run(**max_token_prefill_kwargs)
            if replayssm_autotune is not None:
                max_num_reqs, max_batch_decode_kwargs = replayssm_autotune
                with _temporary_replayssm_autotune_slots(runner, max_num_reqs):
                    runner._dummy_run(**max_batch_decode_kwargs)
    finally:
        set_autotune_process_group(None)

    if world.world_size > 1:
        world.barrier()
    if is_leader:
        tuner.save_configs(str(cache_path))
