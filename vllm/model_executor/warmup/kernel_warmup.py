# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import sys
import time
from typing import TYPE_CHECKING, Any, cast

import torch

import vllm.envs as envs
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


def _warmup_kimi_k3_gemm_rs_ar() -> None:
    # Kimi-K3 model construction imports this module only when GEMM-RS/AR is
    # enabled and initializes its singleton before kernel_warmup runs. Avoid
    # importing it here so other models do not compile the RS/AR variants.
    module = sys.modules.get("vllm.models.kimi_k3.nvidia.ops.cute_dsl.gemm_rs_ar")
    if module is None:
        return
    compiled = module.warmup_gemm_rs_ar()
    if compiled:
        logger.info_once("Warmed up %d Kimi-K3 GEMM-RS/AR variants.", compiled)


def _is_attention_backend(backend, name: str) -> bool:
    try:
        return backend.get_name() == name
    except NotImplementedError:
        return False


def _has_attention_backend(runner: "GPUModelRunner", name: str) -> bool:
    return any(
        _is_attention_backend(group.backend, name)
        for groups in runner.attn_groups
        for group in groups
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

    _warmup_kimi_k3_gemm_rs_ar()

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
        is_deep_gemm_supported() and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
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


_FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS = 32


def _flashinfer_autotune_token_counts(runner: "GPUModelRunner") -> tuple[int, ...]:
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    linear_backend = runner.vllm_config.kernel_config.linear_backend
    if (
        linear_backend == "flashinfer_cutedsl"
        and max_tokens > _FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS
    ):
        return max_tokens, _FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS
    return (max_tokens,)


def _run_flashinfer_mla_decode_autotune(
    runner: "GPUModelRunner",
    batch_size: int,
    query_len: int,
    max_seq_len: int,
) -> None:
    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        flashinfer_mla_decode_autotune,
    )

    runner_any = cast(Any, runner)
    if runner.vllm_config.use_v2_model_runner:
        block_tables = runner_any.block_tables.get_dummy_block_tables(batch_size)
    else:
        block_tables = tuple(
            table.get_device_tensor(batch_size)
            for table in runner.input_batch.block_table.block_tables
        )

    static_forward_context = (
        runner.vllm_config.compilation_config.static_forward_context
    )
    for groups in runner.attn_groups:
        for group in groups:
            if not _is_attention_backend(group.backend, "FLASHINFER_MLA"):
                continue
            layer = static_forward_context[group.layer_names[0]]
            flashinfer_mla_decode_autotune(
                layer,
                block_tables[group.kv_cache_group_id],
                batch_size,
                query_len,
                max_seq_len,
            )


def _run_flashinfer_deferred_moe_autotune(runner: "GPUModelRunner") -> None:
    """Tune the unfinalized decode path without another model forward."""
    import vllm.utils.flashinfer as fi_utils
    from vllm.model_executor.layers.fused_moe import MoERunner

    max_num_batched_tokens = runner.scheduler_config.max_num_batched_tokens
    tuned: set[tuple] = set()
    for module in runner.get_model().modules():
        if not isinstance(module, MoERunner):
            continue

        moe_config = module.moe_config
        if not moe_config.use_deferred_moe_finalize:
            continue

        max_deferred_tokens = moe_config.defer_moe_finalize_max_num_tokens
        num_tokens = max_num_batched_tokens
        if max_deferred_tokens > 0:
            num_tokens = min(num_tokens, max_deferred_tokens)
        if not moe_config.should_defer_moe_finalize(num_tokens):
            continue

        routed_experts = module.routed_experts
        quant_method = routed_experts.quant_method
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        if (
            not quant_method.is_monolithic
            or moe_kernel is None
            or not moe_kernel.supports_deferred_moe_finalize()
        ):
            continue

        # Layers agreeing on these properties produce identical autotune cache
        # keys, so one dispatcher call already covers the rest.
        signature = (
            num_tokens,
            moe_config.hidden_dim,
            moe_config.num_experts,
            moe_config.in_dtype,
            moe_config.router_logits_dtype,
            type(quant_method),
            type(moe_kernel),
            tuple(routed_experts.w13_weight.shape),
            routed_experts.w13_weight.dtype,
        )
        if signature in tuned:
            continue
        tuned.add(signature)

        tuning_buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(num_tokens)
        logger.info(
            "Running FlashInfer deferred MoE autotune with token buckets %s.",
            tuning_buckets,
        )
        device = routed_experts.w13_weight.device
        hidden_states = torch.randn(
            num_tokens,
            moe_config.hidden_dim,
            dtype=moe_config.in_dtype,
            device=device,
        )
        router_logits = torch.randn(
            num_tokens,
            moe_config.num_experts,
            dtype=moe_config.router_logits_dtype,
            device=device,
        )
        with fi_utils.autotune(tuning_buckets=tuning_buckets):
            routed_experts.forward_monolithic(
                x=hidden_states,
                router_logits=router_logits,
            )


def _run_flashinfer_autotune_dummy_runs(runner: "GPUModelRunner") -> None:
    for num_tokens in _flashinfer_autotune_token_counts(runner):
        logger.info("Running FlashInfer autotune with %d tokens.", num_tokens)
        runner._dummy_run(
            num_tokens=num_tokens,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
        )

    # Some decode consumers fuse the final top-k reduction, changing the
    # FlashInfer MoE output from (tokens, hidden) to (tokens, 0). The generic
    # full-model profile does not exercise that path. Invoke only each live
    # deferred MoE dispatcher so its capture keys are tuned without repeating
    # the whole model forward.
    _run_flashinfer_deferred_moe_autotune(runner)

    if not _has_attention_backend(runner, "FLASHINFER_MLA"):
        return
    if runner.vllm_config.parallel_config.decode_context_parallel_size > 1:
        # Native DCP selects CuTeDSL explicitly; there is no backend choice to
        # autotune, and its decode metadata has a different distributed shape.
        return

    import vllm.utils.flashinfer as fi_utils

    runner_any = cast(Any, runner)
    use_v2_model_runner = runner.vllm_config.use_v2_model_runner
    query_len = (
        runner_any.decode_query_len
        if use_v2_model_runner
        else runner.uniform_decode_query_len
    )
    max_decode_batch_size = min(
        runner.scheduler_config.max_num_seqs,
        runner.scheduler_config.max_num_batched_tokens // query_len,
    )
    if max_decode_batch_size == 0:
        return
    # FlashInfer MLA otherwise profiles every batch bucket supported by any
    # candidate runner (currently through B8192 for TRTLLM-GEN), regardless of
    # the dummy input size. Limit the sweep to batches this scheduler can
    # execute; scheduler settings are part of the autotune cache fingerprint.
    tuning_buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(
        max_decode_batch_size
    )
    max_model_len = (
        runner_any.model_state.max_model_len
        if use_v2_model_runner
        else runner.max_model_len
    )
    logger.info(
        "Running FlashInfer MLA decode autotune with %d tokens, "
        "batch buckets %s, and max sequence length %d.",
        max_decode_batch_size * query_len,
        tuning_buckets,
        max_model_len,
    )
    with fi_utils.autotune(tuning_buckets=tuning_buckets):
        # FULL CUDA graph capture uses max_seq_len=max_model_len, and
        # FlashInfer includes it in the MLA runner cache key. Invoke only the
        # dense MLA dispatcher with the initialized layer/cache geometry so
        # this key matches capture without rerunning the whole model (and MoE).
        _run_flashinfer_mla_decode_autotune(
            runner,
            max_decode_batch_size,
            query_len,
            max_model_len,
        )


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
    # Randomize inputs to avoid every token pick the same experts,
    # which lead to some EP ranks receiving no tokens and skipping their
    # MoE kernel entirely, and cause hang due to all-reduce collective
    # during synchronized autotuning.
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
            _run_flashinfer_autotune_dummy_runs(runner)
    finally:
        set_autotune_process_group(None)

    if world.world_size > 1:
        world.barrier()
    if is_leader:
        tuner.save_configs(str(cache_path))
