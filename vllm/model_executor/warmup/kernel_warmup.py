# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.hybrid_gdn_mamba_mrope_warmup import (
    has_hybrid_gdn_mamba_mrope,
    hybrid_gdn_mamba_mrope_warmup,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.worker.gpu.warmup import warmup_kernels

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


class _NoOpKVConnector:
    def set_disabled(self, disabled: bool) -> None:
        pass


def _clear_pending_execute_model_state(worker: "Worker") -> None:
    runner = worker.model_runner
    if getattr(runner, "execute_model_state", None) is None:
        return
    try:
        worker.sample_tokens(None)
    except Exception:
        runner.execute_model_state = None
        if hasattr(runner, "kv_connector_output"):
            runner.kv_connector_output = None


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


def _warmup_scheduler_output_kernels(worker: "Worker") -> None:
    """Run scheduler-output based warmup on V1 runners.

    vllm.v1.worker.gpu.warmup.warmup_kernels is written for the V2 runner but
    uses mostly shared runner state.  Add the small compatibility aliases needed
    by the V1 runner only for this warmup call.
    """
    runner = worker.model_runner
    runner_attrs: Any = runner
    added_num_speculative_steps = not hasattr(runner_attrs, "num_speculative_steps")
    added_kv_connector = not hasattr(runner_attrs, "kv_connector")
    added_is_last_pp_rank = not hasattr(runner_attrs, "is_last_pp_rank")
    if added_num_speculative_steps:
        runner_attrs.num_speculative_steps = getattr(runner_attrs, "num_spec_tokens", 0)
    if added_kv_connector:
        runner_attrs.kv_connector = _NoOpKVConnector()
    if added_is_last_pp_rank:
        from vllm.distributed.parallel_state import get_pp_group

        runner_attrs.is_last_pp_rank = get_pp_group().is_last_rank
    try:
        warmup_kernels(runner, worker.execute_model, worker.sample_tokens)
    except Exception:
        _clear_pending_execute_model_state(worker)
        raise
    finally:
        if added_is_last_pp_rank:
            del runner_attrs.is_last_pp_rank
        if added_kv_connector:
            del runner_attrs.kv_connector
        if added_num_speculative_steps:
            del runner_attrs.num_speculative_steps


@torch.inference_mode()
def _warmup_single_request_decode_kernels(worker: "Worker") -> None:
    """Warm up single-request scheduler variants used by real chat requests."""
    from vllm import SamplingParams
    from vllm.utils.math_utils import cdiv
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )
    from vllm.v1.request import Request

    runner = worker.model_runner
    kv_cache_config = runner.kv_cache_config
    kv_cache_groups = kv_cache_config.kv_cache_groups
    if kv_cache_config.num_blocks <= 1 or not kv_cache_groups:
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    prompt_len = min(64, max(1, max_tokens))

    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]

    def _block_counts(seq_len: int) -> list[int]:
        return [cdiv(seq_len, block_size) for block_size in group_block_sizes]

    while prompt_len > 1:
        decode_block_counts = _block_counts(prompt_len + 1)
        if sum(decode_block_counts) <= kv_cache_config.num_blocks - 1:
            break
        prompt_len //= 2
    else:
        decode_block_counts = _block_counts(prompt_len + 1)
        if sum(decode_block_counts) > kv_cache_config.num_blocks - 1:
            return

    prefill_block_counts = _block_counts(prompt_len)
    decode_block_deltas = [
        decode - prefill
        for decode, prefill in zip(decode_block_counts, prefill_block_counts)
    ]
    num_kv_cache_groups = len(kv_cache_groups)
    req_id = "_hybrid_single_request_warmup_"
    prompt_token_ids = list(range(prompt_len))
    sampling_params = SamplingParams.for_sampler_warmup()

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        blocks = list(range(next_block_id, next_block_id + num_blocks))
        next_block_id += num_blocks
        return blocks

    request = Request(req_id, prompt_token_ids, sampling_params, None)
    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = [
        NewRequestData.from_request(
            request,
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            prefill_token_ids=prompt_token_ids,
        )
    ]
    prefill_output.num_scheduled_tokens = {req_id: prompt_len}
    prefill_output.total_num_scheduled_tokens = prompt_len
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    kv_connector = getattr(runner, "kv_connector", None)
    if kv_connector is not None:
        kv_connector.set_disabled(True)
    try:
        try:
            worker.execute_model(prefill_output)
            worker.sample_tokens(None)

            cached_req_data = CachedRequestData.make_empty()
            cached_req_data.req_ids = [req_id]
            cached_req_data.num_computed_tokens = [prompt_len]
            cached_req_data.num_output_tokens = [1]
            cached_req_data.new_block_ids = [
                (
                    tuple(_alloc_blocks(n) for n in decode_block_deltas)
                    if any(decode_block_deltas)
                    else None
                )
            ]

            decode_output = SchedulerOutput.make_empty()
            decode_output.scheduled_cached_reqs = cached_req_data
            decode_output.num_scheduled_tokens = {req_id: 1}
            decode_output.total_num_scheduled_tokens = 1
            decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

            worker.execute_model(decode_output)
            worker.sample_tokens(None)
        except Exception:
            _clear_pending_execute_model_state(worker)
            raise
        finally:
            cleanup_output = SchedulerOutput.make_empty()
            cleanup_output.finished_req_ids = {req_id}
            try:
                worker.execute_model(cleanup_output)
            except Exception:
                logger.debug(
                    "Hybrid single-request warmup cleanup failed.",
                    exc_info=True,
                )
    finally:
        if kv_connector is not None:
            kv_connector.set_disabled(False)


def kernel_warmup(worker: "Worker"):
    model = worker.get_model()

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    has_hybrid_warmup_targets = has_hybrid_gdn_mamba_mrope(model)
    hybrid_gdn_mamba_mrope_warmup(
        model,
        model_dtype=worker.model_runner.dtype,
    )
    if has_hybrid_warmup_targets and not worker.model_runner.is_pooling_model:
        num_tokens = min(worker.scheduler_config.max_num_batched_tokens, 96)
        if num_tokens > 0:
            try:
                worker.model_runner._dummy_run(
                    num_tokens=num_tokens,
                    skip_eplb=True,
                    is_profile=True,
                    force_attention=True,
                    create_mixed_batch=True,
                )
            except Exception:
                logger.warning(
                    "Hybrid GDN/Mamba/MRoPE dummy-run warmup failed. "
                    "First inference may JIT compile runtime-specific variants.",
                    exc_info=True,
                )
        if not getattr(worker, "use_v2_model_runner", False):
            try:
                _warmup_scheduler_output_kernels(worker)
            except Exception:
                logger.warning(
                    "Hybrid GDN/Mamba/MRoPE scheduler-output warmup failed. "
                    "First inference may JIT compile scheduler-specific variants.",
                    exc_info=True,
                )
            try:
                _warmup_single_request_decode_kernels(worker)
            except Exception:
                logger.warning(
                    "Hybrid GDN/Mamba/MRoPE single-request warmup failed. "
                    "First single-request inference may JIT compile decode variants.",
                    exc_info=True,
                )

    kv_block_zeroer = getattr(worker.model_runner, "_kv_block_zeroer", None)
    if kv_block_zeroer is not None:
        kv_block_zeroer.warmup()

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
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
