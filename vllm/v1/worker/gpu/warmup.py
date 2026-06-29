# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    TQFullAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

_TQ_CONTINUATION_DECODE_THRESHOLD = 128
_TQ_WARMUP_PROMPT_CHUNK_LEN = 256

logger = init_logger(__name__)


def run_mixed_prefill_decode_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    num_tokens: int,
    *,
    mixed_step_context: AbstractContextManager[object] | None = None,
    req_id_prefix: str = "_v2_mixed_warmup",
) -> bool:
    """Run a V2 mixed prefill+decode step through normal scheduler inputs."""
    if model_runner.is_pooling_model or num_tokens < 3:
        return False

    decode_req_id = f"{req_id_prefix}_decode_"
    prefill_req_id = f"{req_id_prefix}_prefill_"
    decode_prompt_len = 2
    decode_scheduled_tokens = 1
    prefill_len = num_tokens - decode_scheduled_tokens
    decode_token_ids = list(range(decode_prompt_len))
    prefill_token_ids = list(range(prefill_len))

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)
    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
    decode_prefill_block_counts = [
        cdiv(decode_prompt_len, block_size) for block_size in group_block_sizes
    ]
    decode_block_counts = [
        cdiv(decode_prompt_len + decode_scheduled_tokens, block_size)
        for block_size in group_block_sizes
    ]
    decode_block_deltas = [
        decode - prefill
        for decode, prefill in zip(decode_block_counts, decode_prefill_block_counts)
    ]
    prefill_block_counts = [
        cdiv(prefill_len, block_size) for block_size in group_block_sizes
    ]
    required_blocks = sum(decode_block_counts) + sum(prefill_block_counts)
    if model_runner.kv_cache_config.num_blocks <= required_blocks:
        logger.warning(
            "Skipping V2 mixed prefill+decode warmup because only %d KV blocks "
            "are available for %d required warmup blocks.",
            model_runner.kv_cache_config.num_blocks,
            required_blocks,
        )
        return False

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        block_ids = list(range(next_block_id, next_block_id + num_blocks))
        next_block_id += num_blocks
        return block_ids

    sampling_params = SamplingParams(max_tokens=2, temperature=0.0)

    decode_prefill_output = SchedulerOutput.make_empty()
    decode_prefill_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=decode_req_id,
            prompt_token_ids=decode_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(_alloc_blocks(n) for n in decode_prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=decode_token_ids,
        ),
    ]
    decode_prefill_output.num_scheduled_tokens = {
        decode_req_id: decode_prompt_len,
    }
    decode_prefill_output.total_num_scheduled_tokens = decode_prompt_len
    decode_prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    decode_new_blocks = tuple(_alloc_blocks(n) for n in decode_block_deltas)
    cached_decode_req = CachedRequestData.make_empty()
    cached_decode_req.req_ids = [decode_req_id]
    cached_decode_req.num_computed_tokens = [decode_prompt_len]
    cached_decode_req.num_output_tokens = [1]
    cached_decode_req.new_block_ids = [
        decode_new_blocks if any(decode_block_deltas) else None
    ]

    mixed_output = SchedulerOutput.make_empty()
    mixed_output.scheduled_cached_reqs = cached_decode_req
    mixed_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=prefill_req_id,
            prompt_token_ids=prefill_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=prefill_token_ids,
        ),
    ]
    mixed_output.num_scheduled_tokens = {
        decode_req_id: decode_scheduled_tokens,
        prefill_req_id: prefill_len,
    }
    mixed_output.total_num_scheduled_tokens = num_tokens
    mixed_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {decode_req_id, prefill_req_id}

    context = mixed_step_context or nullcontext()
    model_runner.kv_connector.set_disabled(True)
    try:
        worker_execute_model(decode_prefill_output)
        worker_sample_tokens(None)
        with context:
            worker_execute_model(mixed_output)
            worker_sample_tokens(None)
        worker_execute_model(cleanup_output)
    finally:
        model_runner.kv_connector.set_disabled(False)
    return True


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run two execute_model + sample_tokens iterations to JIT compile
    triton kernels. We must call the provided worker's execute_model for
    pipeline parallel coordination.

    The first iteration simulates a prefill with requests of
    decode_query_len + 1 prompt tokens each. The second iteration simulates
    a decode step with all requests generating decode_query_len tokens.
    """
    num_spec_steps = model_runner.num_speculative_steps
    decode_query_len = model_runner.decode_query_len
    # Use decode_query_len + 1 tokens so the prefill batch's per-request query
    # length exceeds decode_query_len, preventing it from being misclassified as
    # a uniform decode batch.
    prompt_len = decode_query_len + 1
    prompt_token_ids = list(range(prompt_len))
    # After prefill, decode generates decode_query_len tokens.
    decode_len = prompt_len + decode_query_len

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    # Compute per-request block counts for each KV cache group.
    def _warmup_block_count(num_tokens: int, spec: Any) -> int:
        num_blocks = cdiv(num_tokens, spec.block_size)
        if isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align":
            # Align mode reserves extra blocks beyond the token range for the
            # speculative-decode running-state snapshots.
            num_blocks += spec.num_speculative_blocks
        return num_blocks

    kv_cache_specs = [g.kv_cache_spec for g in kv_cache_groups]
    prefill_block_counts = [_warmup_block_count(prompt_len, s) for s in kv_cache_specs]
    decode_block_counts = [_warmup_block_count(decode_len, s) for s in kv_cache_specs]
    decode_block_deltas = [
        d - p for d, p in zip(decode_block_counts, prefill_block_counts)
    ]
    max_blocks_per_req = sum(decode_block_counts)

    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, decode_query_len),
        # Reserve block 0 (null block) and ensure we have enough blocks.
        max(1, (model_runner.kv_cache_config.num_blocks - 1) // max_blocks_per_req),
    )

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    # Assign distinct block IDs per request per group. 0 null block, start from 1.
    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    # Step 1: Prefill all requests with 1 + decode_query_len prompt tokens each.
    new_reqs = [
        NewRequestData.from_request(
            Request(req_ids[i], prompt_token_ids, sampling_params, pooling_params),
            block_ids=tuple(_alloc_blocks(n) for n in prefill_block_counts),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    # Disable KV connector for warmup run.
    model_runner.kv_connector.set_disabled(True)
    worker_execute_model(prefill_output)

    if not model_runner.is_pooling_model:
        # Warm up sampler and perform a decode step for non-pooling models.

        grammar_output = None
        if model_runner.is_last_pp_rank:
            # Build a GrammarOutput to exercise the structured output bitmask
            # kernel during the prefill step.
            vocab_size = model_runner.model_config.get_vocab_size()
            bitmask_width = (vocab_size + 31) // 32
            grammar_bitmask = np.full(
                (len(req_ids), bitmask_width), fill_value=-1, dtype=np.int32
            )
            grammar_output = GrammarOutput(
                structured_output_request_ids=req_ids, grammar_bitmask=grammar_bitmask
            )

        worker_sample_tokens(grammar_output)

        # Step 2: Decode all requests with decode_query_len tokens each.
        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = list(req_ids)
        cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
        cached_req_data.num_output_tokens = [1] * num_reqs
        new_block = any(decode_block_deltas)
        cached_req_data.new_block_ids = [
            tuple(_alloc_blocks(n) for n in decode_block_deltas) if new_block else None
            for _ in range(num_reqs)
        ]

        decode_output = SchedulerOutput.make_empty()
        decode_output.scheduled_cached_reqs = cached_req_data
        decode_output.num_scheduled_tokens = {
            req_id: decode_query_len for req_id in req_ids
        }
        if num_spec_steps > 0:
            decode_output.scheduled_spec_decode_tokens = {
                req_id: [0] * num_spec_steps for req_id in req_ids
            }
        decode_output.total_num_scheduled_tokens = sum(
            decode_output.num_scheduled_tokens.values()
        )
        decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

        worker_execute_model(decode_output)
        worker_sample_tokens(None)

    # Clean up - process finish_req_ids.
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    worker_execute_model(cleanup_output)
    if not model_runner.is_pooling_model:
        _warmup_turboquant_attention_kernels(model_runner, worker_execute_model)
    model_runner.kv_connector.set_disabled(False)
    torch.accelerator.synchronize()


def _kv_cache_spec_uses_turboquant(kv_cache_spec: KVCacheSpec) -> bool:
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        return any(
            _kv_cache_spec_uses_turboquant(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        )
    if isinstance(kv_cache_spec, AttentionSpec):
        return isinstance(kv_cache_spec, TQFullAttentionSpec)
    return False


def _uses_turboquant_kv_cache(model_runner: GPUModelRunner) -> bool:
    kv_cache_config = getattr(model_runner, "kv_cache_config", None)
    if kv_cache_config is None:
        return False
    return any(
        _kv_cache_spec_uses_turboquant(group.kv_cache_spec)
        for group in kv_cache_config.kv_cache_groups
    )


def _get_max_model_len(model_runner: GPUModelRunner) -> int:
    max_model_len = _get_positive_int(model_runner, "max_model_len", 0)
    if max_model_len > 0:
        return max_model_len
    model_config = getattr(model_runner, "model_config", None)
    return _get_positive_int(model_config, "max_model_len", 0)


def _warmup_turboquant_attention_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
) -> None:
    """Warm TQ continuation-prefill through scheduler outputs.

    Qwen TQ selector models mix ordinary and TQ attention layers. Run the same
    request lifecycle used by the scheduler so the TQ backend sees the real
    cached-prefill metadata without direct synthetic kernel calls.
    """
    if not _uses_turboquant_kv_cache(model_runner):
        return

    scheduler_config = getattr(model_runner, "scheduler_config", None)
    if scheduler_config is None:
        return

    max_model_len = _get_max_model_len(model_runner)
    max_num_batched_tokens = _get_positive_int(
        scheduler_config,
        "max_num_batched_tokens",
        0,
    )
    decode_query_len = _get_positive_int(model_runner, "decode_query_len", 1)
    if max_model_len <= 0 or max_num_batched_tokens <= 0:
        return

    continuation_len = min(_TQ_WARMUP_PROMPT_CHUNK_LEN, max_num_batched_tokens)
    if continuation_len <= _TQ_CONTINUATION_DECODE_THRESHOLD:
        return
    prefix_len = min(
        _TQ_WARMUP_PROMPT_CHUNK_LEN,
        max_num_batched_tokens,
        max_model_len - continuation_len - decode_query_len,
    )
    if prefix_len <= 0 or decode_query_len > max_num_batched_tokens:
        return

    prompt_len = prefix_len + continuation_len

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    group_block_sizes = [g.kv_cache_spec.block_size for g in kv_cache_groups]
    prefix_block_counts = [cdiv(prefix_len, bs) for bs in group_block_sizes]
    continuation_block_counts = [cdiv(prompt_len, bs) for bs in group_block_sizes]
    if sum(continuation_block_counts) > model_runner.kv_cache_config.num_blocks - 1:
        return

    continuation_block_deltas = [
        c - p for c, p in zip(continuation_block_counts, prefix_block_counts)
    ]

    next_block_id = 1

    def _alloc_blocks(num_blocks: int) -> list[int]:
        nonlocal next_block_id
        return list(range(next_block_id, next_block_id := next_block_id + num_blocks))

    req_id = "_warmup_tq_"
    prompt_token_ids = list(range(prompt_len))
    sampling_params = SamplingParams.for_sampler_warmup()

    new_req = NewRequestData.from_request(
        Request(req_id, prompt_token_ids, sampling_params, None),
        block_ids=tuple(_alloc_blocks(n) for n in prefix_block_counts),
        prefill_token_ids=prompt_token_ids,
    )

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = [new_req]
    prefill_output.num_scheduled_tokens = {req_id: prefix_len}
    prefill_output.total_num_scheduled_tokens = prefix_len
    prefill_output.num_common_prefix_blocks = [0] * len(kv_cache_groups)
    worker_execute_model(prefill_output)

    cached_req_data = CachedRequestData.make_empty()
    cached_req_data.req_ids = [req_id]
    cached_req_data.num_computed_tokens = [prefix_len]
    cached_req_data.num_output_tokens = [0]
    new_block = any(continuation_block_deltas)
    cached_req_data.new_block_ids = [
        tuple(_alloc_blocks(n) for n in continuation_block_deltas)
        if new_block
        else None
    ]

    continuation_output = SchedulerOutput.make_empty()
    continuation_output.scheduled_cached_reqs = cached_req_data
    continuation_output.num_scheduled_tokens = {req_id: continuation_len}
    continuation_output.total_num_scheduled_tokens = continuation_len
    continuation_output.num_common_prefix_blocks = [0] * len(kv_cache_groups)
    worker_execute_model(continuation_output)

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {req_id}
    worker_execute_model(cleanup_output)


def _get_positive_int(obj: Any, name: str, default: int) -> int:
    value = getattr(obj, name, default)
    return value if isinstance(value, int) and value > 0 else default


@torch.inference_mode()
def warmup_v1_attention_kernels(model_runner: Any) -> None:
    """Warm old V1 attention kernels through the existing dummy-run path.

    V1 profiling does not always build attention metadata, and CUDA graph
    capture locks the workspace after capture. Run small forced-attention dummy
    batches before capture so attention backends can size workspace without
    backend-specific synthetic calls.
    """
    dummy_run = getattr(model_runner, "_dummy_run", None)
    if dummy_run is None:
        return
    if getattr(model_runner, "is_pooling_model", False):
        return
    if not getattr(model_runner, "attn_groups", None):
        return

    scheduler_config = getattr(model_runner, "scheduler_config", None)
    if scheduler_config is None:
        return

    max_num_tokens = _get_positive_int(model_runner, "max_num_tokens", 0)
    max_model_len = _get_positive_int(model_runner, "max_model_len", 0)
    max_num_seqs = _get_positive_int(scheduler_config, "max_num_seqs", 1)
    max_num_batched_tokens = _get_positive_int(
        scheduler_config,
        "max_num_batched_tokens",
        max_num_tokens,
    )
    if max_num_tokens <= 0 or max_model_len <= 0 or max_num_batched_tokens <= 0:
        return

    decode_query_len = _get_positive_int(model_runner, "uniform_decode_query_len", 1)
    max_decode_tokens = min(
        max_num_tokens,
        max_num_batched_tokens,
        max_num_seqs * decode_query_len,
    )
    if max_decode_tokens >= decode_query_len:
        decode_seq_len = min(max_model_len, max(decode_query_len + 1, 8192))
        decode_kwargs = {
            "num_tokens": max_decode_tokens,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": True,
        }
        if decode_seq_len > decode_query_len:
            decode_kwargs["profile_seq_lens"] = decode_seq_len
        dummy_run(**decode_kwargs)

    # Keep pure prefill intentionally small. Cached prefill must use the
    # scheduler chunk size because attention backends specialize on the maximum
    # query length, and a small cached-prefill dummy run misses chunked-prefill
    # kernels used by long prompts.
    prefill_tokens = min(max_num_tokens, max_num_batched_tokens, max_model_len, 64)
    if prefill_tokens > 1:
        dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=False,
            num_reqs_override=1,
        )

    cached_prefill_tokens = min(max_num_tokens, max_num_batched_tokens, max_model_len)
    cached_prefill_seq_len = min(
        max_model_len,
        max(cached_prefill_tokens + 1, 8192),
    )
    if cached_prefill_tokens > 1 and cached_prefill_seq_len > cached_prefill_tokens:
        dummy_run(
            num_tokens=cached_prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=False,
            num_reqs_override=1,
            profile_seq_lens=cached_prefill_seq_len,
            profile_as_cached_prefill=True,
        )
        cached_prefill_req_counts = [min(max_num_seqs, cached_prefill_tokens, 3)]
        if max_num_seqs >= 16 and cached_prefill_tokens >= 16:
            cached_prefill_req_counts.append(16)
        for cached_prefill_reqs in dict.fromkeys(cached_prefill_req_counts):
            if cached_prefill_reqs <= 1:
                continue
            dummy_run(
                num_tokens=cached_prefill_tokens,
                skip_eplb=True,
                is_profile=True,
                force_attention=True,
                uniform_decode=False,
                num_reqs_override=cached_prefill_reqs,
                profile_seq_lens=cached_prefill_seq_len,
                profile_as_cached_prefill=True,
            )

    torch.accelerator.synchronize()
