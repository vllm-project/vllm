# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.structured_output.utils import apply_grammar_bitmask

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "V4_FLASHMLA_SPARSE",
        "DEEPSEEK_SPARSE_SWA",
    }
)
_DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16
_DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS = 1024
_DEEPSEEK_V4_MTP_UNIFORM_DECODE_WARMUP_REQUESTS = (1, 2)
_DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS = tuple(range(1, 17)) + (
    32,
    64,
    128,
    256,
    512,
)


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


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _is_deepseek_v4_mtp_spec_decode(runner: "GPUModelRunner") -> bool:
    spec_config = getattr(runner, "speculative_config", None)
    return (
        getattr(spec_config, "method", None) == "mtp"
        and getattr(runner, "num_spec_tokens", 0) > 0
    )


def _deepseek_v4_mtp_uniform_decode_warmup_requests(
    runner: "GPUModelRunner",
    max_tokens: int,
    max_reqs: int,
) -> tuple[int, ...]:
    if not _is_deepseek_v4_mtp_spec_decode(runner):
        return ()

    query_len = getattr(
        runner,
        "uniform_decode_query_len",
        1 + getattr(runner, "num_spec_tokens", 0),
    )
    if query_len <= 0:
        return ()

    max_warmup_reqs = min(max_reqs, max_tokens // query_len)
    return tuple(
        reqs
        for reqs in _DEEPSEEK_V4_MTP_UNIFORM_DECODE_WARMUP_REQUESTS
        if reqs <= max_warmup_reqs
    )


def _deepseek_v4_slot_mapping_warmup(runner: "GPUModelRunner") -> None:
    max_tokens = getattr(runner, "max_num_tokens", 1)
    block_table = runner.input_batch.block_table

    # Snapshot the runner buffers we mutate so warmup never leaks state into
    # the first real request.
    saved_query_start_loc_np: np.ndarray | None = None
    saved_query_start_loc_gpu: torch.Tensor | None = None
    if hasattr(runner, "query_start_loc"):
        saved_query_start_loc_np = runner.query_start_loc.np[:2].copy()
        saved_query_start_loc_gpu = runner.query_start_loc.gpu[:2].clone()

    try:
        for requested_tokens in _DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS:
            num_tokens = _clamp_warmup_tokens(requested_tokens, max_tokens)
            if num_tokens <= 0:
                continue

            positions_source = torch.arange(
                num_tokens, dtype=torch.int64, device=runner.device
            )
            if hasattr(runner, "query_start_loc"):
                runner.query_start_loc.np[0] = 0
                runner.query_start_loc.np[1] = num_tokens
                runner.query_start_loc.copy_to_gpu(2)
                query_start_loc = runner.query_start_loc.gpu[:2]
            else:
                query_start_loc = torch.tensor(
                    [0, num_tokens], dtype=torch.int32, device=runner.device
                )

            if hasattr(runner, "positions"):
                saved_positions: torch.Tensor | None = (
                    runner.positions[:num_tokens].clone()
                )
                runner.positions[:num_tokens].copy_(positions_source)
                positions = runner.positions[:num_tokens]
            else:
                saved_positions = None
                positions = positions_source

            try:
                block_table.commit_block_table(1)
                block_table.compute_slot_mapping(1, query_start_loc, positions)
            finally:
                if saved_positions is not None:
                    runner.positions[:num_tokens].copy_(saved_positions)
    finally:
        if saved_query_start_loc_np is not None:
            runner.query_start_loc.np[:2] = saved_query_start_loc_np
            assert saved_query_start_loc_gpu is not None
            runner.query_start_loc.gpu[:2].copy_(saved_query_start_loc_gpu)


def _deepseek_v4_structured_output_bitmask_warmup(
    runner: "GPUModelRunner",
) -> None:
    vocab_size = runner.model_config.get_vocab_size()
    if vocab_size <= 0:
        return

    dtypes = [torch.float32]
    model_dtype = getattr(runner.model_config, "dtype", None)
    if isinstance(model_dtype, torch.dtype) and model_dtype not in dtypes:
        dtypes.append(model_dtype)

    bitmask_width = (vocab_size + 31) // 32
    req_id = "_deepseek_v4_warmup_"
    grammar_bitmask = np.full((1, bitmask_width), fill_value=-1, dtype=np.int32)
    grammar_output = GrammarOutput(
        structured_output_request_ids=[req_id], grammar_bitmask=grammar_bitmask
    )

    for dtype in dtypes:
        for req_ids in ([req_id], [req_id, "_deepseek_v4_warmup_unmasked_"]):
            logits = torch.zeros(
                (len(req_ids), vocab_size), dtype=dtype, device=runner.device
            )
            input_batch = SimpleNamespace(req_ids=req_ids)
            apply_grammar_bitmask(
                SchedulerOutput.make_empty(), grammar_output, input_batch, logits
            )


@torch.inference_mode()
def _deepseek_v4_request_prep_warmup(worker: "Worker") -> None:
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return
    if not current_platform.is_cuda_alike():
        return

    logger.info("Warming up DeepSeek V4 request preparation kernels.")
    _deepseek_v4_slot_mapping_warmup(runner)

    if getattr(runner, "is_last_pp_rank", True):
        try:
            _deepseek_v4_structured_output_bitmask_warmup(runner)
        except ImportError:
            logger.debug(
                "Skipping DeepSeek V4 structured output bitmask warmup because "
                "xgrammar is unavailable."
            )

    torch.accelerator.synchronize()


def _run_deepseek_v4_mtp_spec_decode_warmup_kernels(
    *,
    device: torch.device,
    num_reqs: int,
    num_spec_tokens: int,
    vocab_size: int,
    block_size: int,
    max_model_len: int,
) -> None:
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.rejection_sampler import rejection_sample
    from vllm.v1.spec_decode.utils import (
        eagle_prepare_inputs_padded_kernel,
        eagle_prepare_next_token_padded_kernel,
        eagle_step_update_slot_mapping_and_metadata,
        next_power_of_2,
    )

    num_sampled_tokens = num_spec_tokens + 1
    sampled_token_ids = torch.arange(
        num_reqs * num_sampled_tokens, dtype=torch.int32, device=device
    ).reshape(num_reqs, num_sampled_tokens)
    sampled_token_ids.remainder_(vocab_size)
    discard_request_mask = torch.zeros(num_reqs, dtype=torch.bool, device=device)
    backup_next_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    next_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=device)
    valid_sampled_tokens_count = torch.empty(num_reqs, dtype=torch.int32, device=device)
    eagle_prepare_next_token_padded_kernel[(num_reqs,)](
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        next_token_ids,
        valid_sampled_tokens_count,
        vocab_size,
        num_sampled_tokens,
        num_reqs,
        sampled_token_ids.stride(0),
        BLOCK_SIZE_TOKENS=next_power_of_2(num_sampled_tokens),
    )

    cu_num_draft_tokens = torch.arange(
        num_spec_tokens,
        num_reqs * num_spec_tokens + 1,
        num_spec_tokens,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc = torch.arange(
        0,
        (num_reqs + 1) * num_sampled_tokens,
        num_sampled_tokens,
        dtype=torch.int32,
        device=device,
    )
    token_indices_to_sample = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected_tokens = torch.empty(num_reqs, dtype=torch.int32, device=device)
    eagle_prepare_inputs_padded_kernel[(num_reqs,)](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc,
        token_indices_to_sample,
        num_rejected_tokens,
        num_reqs,
    )

    positions = torch.arange(num_reqs, dtype=torch.int64, device=device)
    block_table_tensor = torch.zeros((num_reqs, 1), dtype=torch.int32, device=device)
    seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
    out_clamped_positions = torch.empty_like(positions)
    out_slot_mapping = torch.empty(num_reqs, dtype=torch.int64, device=device)
    eagle_step_update_slot_mapping_and_metadata(
        positions,
        block_table_tensor,
        seq_lens,
        block_size,
        max_model_len,
        out_clamped_positions,
        out_slot_mapping,
        input_batch_size=num_reqs,
    )

    total_draft_tokens = num_reqs * num_spec_tokens
    draft_token_ids = torch.arange(total_draft_tokens, dtype=torch.int32, device=device)
    draft_token_ids.remainder_(vocab_size)
    draft_probs = torch.rand(
        total_draft_tokens, vocab_size, dtype=torch.float32, device=device
    )
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    target_logits = torch.randn(
        total_draft_tokens, vocab_size, dtype=torch.float32, device=device
    )
    bonus_token_ids = torch.zeros((num_reqs, 1), dtype=torch.int32, device=device)
    sampling_metadata = SamplingMetadata(
        temperature=torch.full((num_reqs,), 0.7, dtype=torch.float32, device=device),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0, device=device),
        presence_penalties=torch.empty(0, device=device),
        repetition_penalties=torch.empty(0, device=device),
        output_token_ids=[[] for _ in range(num_reqs)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        logprob_token_ids=None,
        spec_token_ids=[[] for _ in range(num_reqs)],
    )
    rejection_sample(
        draft_token_ids=draft_token_ids,
        num_draft_tokens=[num_spec_tokens] * num_reqs,
        max_spec_len=num_spec_tokens,
        cu_num_draft_tokens=cu_num_draft_tokens,
        draft_probs=draft_probs,
        target_logits=target_logits,
        bonus_token_ids=bonus_token_ids,
        sampling_metadata=sampling_metadata,
    )


def _deepseek_v4_sparse_mla_attention_warmup(worker: "Worker") -> None:
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(
        _DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens
    )
    prefill_tokens = _clamp_warmup_tokens(
        _DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS, max_tokens
    )
    uniform_decode_reqs = _deepseek_v4_mtp_uniform_decode_warmup_requests(
        runner,
        max_tokens=max_tokens,
        max_reqs=worker.scheduler_config.max_num_seqs,
    )
    if mixed_tokens <= 0 and prefill_tokens <= 0 and not uniform_decode_reqs:
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention "
        "for mixed tokens=%s, prefill tokens=%s, and MTP uniform decode "
        "requests=%s.",
        mixed_tokens,
        prefill_tokens,
        list(uniform_decode_reqs),
    )
    if mixed_tokens > 0:
        runner._dummy_run(
            num_tokens=mixed_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
    if prefill_tokens > 0:
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_single_prefill=True,
        )
    query_len = getattr(runner, "uniform_decode_query_len", 0)
    for num_reqs in uniform_decode_reqs:
        runner._dummy_run(
            num_tokens=num_reqs * query_len,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=True,
        )

    if uniform_decode_reqs and current_platform.is_cuda_alike():
        vocab_size = runner.model_config.get_vocab_size()
        block_size = getattr(runner.cache_config, "block_size", None) or 16
        logger.info(
            "Warming up DeepSeek V4 MTP spec-decode kernels for request "
            "counts=%s and %d draft tokens.",
            list(uniform_decode_reqs),
            runner.num_spec_tokens,
        )
        for num_reqs in uniform_decode_reqs:
            _run_deepseek_v4_mtp_spec_decode_warmup_kernels(
                device=runner.device,
                num_reqs=num_reqs,
                num_spec_tokens=runner.num_spec_tokens,
                vocab_size=vocab_size,
                block_size=block_size,
                max_model_len=runner.max_model_len,
            )
        torch.accelerator.synchronize()


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

    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        ),
    )

    _deepseek_v4_sparse_mla_attention_warmup(worker)
    _deepseek_v4_request_prep_warmup(worker)

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
