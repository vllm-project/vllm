# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    flashinfer_replayssm_autotune_supported,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _replayssm_autotune_kwargs(
    runner: "GPUModelRunner",
) -> tuple[int, dict[str, Any]] | None:
    config = runner.vllm_config
    if not (
        config.cache_config.use_replayssm
        and config.mamba_config.backend == MambaBackendEnum.FLASHINFER
    ):
        return None
    if not flashinfer_replayssm_autotune_supported():
        logger.info_once(
            "Skipping FlashInfer ReplaySSM autotuning because "
            "flashinfer.mamba.checkpointing_ssu.CheckpointingSSURunner "
            "is unavailable."
        )
        return None
    v2_runner: Any = runner
    query_len = (
        v2_runner.decode_query_len
        if config.use_v2_model_runner
        else runner.uniform_decode_query_len
    )
    max_num_reqs = min(
        runner.scheduler_config.max_num_seqs,
        runner.max_num_tokens // query_len,
        runner.kv_cache_config.num_blocks - 1,
    )
    decode_kwargs = {
        "num_tokens": max_num_reqs * query_len,
        "skip_eplb": True,
        "is_profile": True,
        "randomize_inputs": True,
        "uniform_decode": True,
    }
    if config.use_v2_model_runner:
        decode_kwargs["valid_dummy_state_slots"] = True
    else:
        decode_kwargs.update(
            allow_microbatching=False,
            force_attention=True,
            profile_seq_lens=query_len + 1,
        )
    return max_num_reqs, decode_kwargs


@contextmanager
def _temporary_replayssm_autotune_state(
    runner: "GPUModelRunner", max_num_reqs: int
) -> Iterator[None]:
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
        reset_replayssm_ring_trackers,
        update_replayssm_ring_trackers,
    )

    reset_tensors: dict[int, torch.Tensor] = {}
    tracker_specs: dict[int, tuple[torch.Tensor, torch.Tensor, int, int]] = {}
    for module in runner.get_model().modules():
        if not isinstance(module, MambaMixer2) or not module.use_replayssm:
            continue
        assert module.replayssm_buffer_len is not None
        ring_start = module._replayssm_ring_start
        prev_num_accepted = module._replayssm_prev_num_accepted
        tracker_specs.setdefault(
            ring_start.data_ptr(),
            (
                ring_start,
                prev_num_accepted,
                module.replayssm_buffer_len,
                module.kv_cache[2].size(2),
            ),
        )
        tensors = (
            *module.kv_cache,
            ring_start,
            prev_num_accepted,
        )
        for tensor in tensors:
            if tensor.numel():
                reset_tensors.setdefault(tensor.data_ptr(), tensor)

    v2_runner: Any = runner
    block_tables = saved_block_ids = None
    if not runner.vllm_config.use_v2_model_runner:
        block_tables = runner.input_batch.block_table.block_tables
        saved_block_ids = tuple(
            block_table.block_table.np[:max_num_reqs, 0].copy()
            for block_table in block_tables
        )
        dummy_block_ids = range(1, max_num_reqs + 1)
        for block_table in block_tables:
            block_table.block_table.np[:max_num_reqs, 0] = dummy_block_ids
        runner.input_batch.block_table.commit_block_table(max_num_reqs)

    first_tracker = next(iter(tracker_specs.values()), None)
    if first_tracker is not None and first_tracker[0].is_cuda:
        state_slots = torch.arange(
            1, max_num_reqs + 1, dtype=torch.int32, device=first_tracker[0].device
        )
        for (
            ring_start,
            prev_num_accepted,
            logical_window,
            ring_buffer_len,
        ) in tracker_specs.values():
            # Compile reset (prefill) and advance (decode) before inference.
            # The final reset leaves the decode tuning run in a clean state.
            reset_replayssm_ring_trackers(ring_start, prev_num_accepted, state_slots)
            update_replayssm_ring_trackers(
                ring_start,
                prev_num_accepted,
                state_slots,
                logical_window,
                ring_buffer_len,
            )
            reset_replayssm_ring_trackers(ring_start, prev_num_accepted, state_slots)

    try:
        yield
    finally:
        if runner.vllm_config.use_v2_model_runner:
            v2_runner.block_tables.get_dummy_block_tables(max_num_reqs)
        else:
            assert block_tables is not None and saved_block_ids is not None
            for block_table, block_ids in zip(block_tables, saved_block_ids):
                block_table.block_table.np[:max_num_reqs, 0] = block_ids
            runner.input_batch.block_table.commit_block_table(max_num_reqs)
        for tensor in reset_tensors.values():
            tensor[1 : max_num_reqs + 1].zero_()


def replayssm_autotune_warmup(runner: "GPUModelRunner") -> None:
    autotune = _replayssm_autotune_kwargs(runner)
    if autotune is None:
        return
    max_num_reqs, decode_kwargs = autotune
    with _temporary_replayssm_autotune_state(runner, max_num_reqs):
        runner._dummy_run(**decode_kwargs)
