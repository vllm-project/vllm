# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    flashinfer_replayssm_autotune_supported,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _replayssm_autotune_kwargs(
    runner: "GPUModelRunner",
    max_token_prefill_kwargs: dict[str, Any],
) -> tuple[int, dict[str, Any]] | None:
    config = runner.vllm_config
    if not config.use_flashinfer_replayssm:
        return None
    if (
        config.cache_config.replayssm_checkpoint_mode != "separate"
        and not flashinfer_replayssm_autotune_supported()
    ):
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
    runner: "GPUModelRunner",
    max_num_reqs: int,
    *,
    prepare_dummy_block_tables: bool = True,
) -> Iterator[None]:
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    reset_tensors: list[torch.Tensor] = []
    reset_trackers: dict[int, torch.Tensor] = {}
    for module in runner.get_model().modules():
        if not isinstance(module, MambaMixer2) or not module.use_replayssm:
            continue
        assert module.replayssm_buffer_len is not None
        ring_start = module._replayssm_ring_start
        prev_num_accepted = module._replayssm_prev_num_accepted
        for tensor in module.kv_cache:
            if tensor.numel():
                # Overlaid groups may start at the same address with different
                # widths. Preserve each bounded view; clearing twice is harmless.
                reset_tensors.append(tensor)
        for tensor in (ring_start, prev_num_accepted):
            if tensor.numel():
                reset_trackers.setdefault(tensor.data_ptr(), tensor)
        if len(module.kv_cache) > 5:
            # Compile quantization before graph capture. Only use dummy slots.
            state_indices = torch.arange(
                1,
                max_num_reqs + 1,
                dtype=torch.int32,
                device=module.kv_cache[1].device,
            )
            initial = torch.zeros(
                (1, *module.kv_cache[1].shape[1:]),
                dtype=torch.float32,
                device=module.kv_cache[1].device,
            ).expand(max_num_reqs, -1, -1, -1)
            module.initialize_replayssm_state(initial, state_indices)

    v2_runner: Any = runner
    block_tables = saved_block_ids = None
    if prepare_dummy_block_tables and not runner.vllm_config.use_v2_model_runner:
        block_tables = runner.input_batch.block_table.block_tables
        saved_block_ids = tuple(
            block_table.block_table.np[:max_num_reqs, 0].copy()
            for block_table in block_tables
        )
        dummy_block_ids = range(1, max_num_reqs + 1)
        for block_table in block_tables:
            block_table.block_table.np[:max_num_reqs, 0] = dummy_block_ids
        runner.input_batch.block_table.commit_block_table(max_num_reqs)

    try:
        yield
    finally:
        if prepare_dummy_block_tables:
            if runner.vllm_config.use_v2_model_runner:
                v2_runner.block_tables.get_dummy_block_tables(max_num_reqs)
            else:
                assert block_tables is not None and saved_block_ids is not None
                for block_table, block_ids in zip(block_tables, saved_block_ids):
                    block_table.block_table.np[:max_num_reqs, 0] = block_ids
                runner.input_batch.block_table.commit_block_table(max_num_reqs)
        for tensor in reset_tensors:
            tensor[1 : max_num_reqs + 1].zero_()
        for tensor in reset_trackers.values():
            tensor.zero_()


def replayssm_lifecycle_warmup(runner: "GPUModelRunner") -> None:
    """Warm request lifecycle kernels with bounded, non-padding cache slots."""
    config = runner.vllm_config
    if config.cache_config.replayssm_checkpoint_mode == "separate":
        # CuTe compiles both masked decode kernels before CUDA graph capture.
        replayssm_autotune_warmup(runner)
        return
    if not config.use_flashinfer_replayssm:
        return
    num_slots = min(runner.kv_cache_config.num_blocks - 1, 2)
    if num_slots <= 0:
        return

    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.replayssm_utils import ReplaySSMModelContext

    use_v2 = config.use_v2_model_runner
    v2_runner: Any = runner
    grouped = []
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not isinstance(spec, MambaSpec) or not spec.use_flashinfer_replayssm:
            continue
        mixers = [
            config.compilation_config.static_forward_context[name]
            for name in group.layer_names
        ]
        if not mixers:
            continue
        table = (
            v2_runner.block_tables.input_block_tables[group_id]
            if use_v2
            else runner.input_batch.block_table[group_id].get_device_tensor(1)
        )
        # Match the runtime stride without changing its block-table contents.
        scratch_table = torch.empty_strided(
            (1, 2), table.stride(), dtype=table.dtype, device=table.device
        )
        scratch_table[0, 0] = 1
        scratch_table[0, 1] = num_slots
        grouped.append((mixers, spec, scratch_table))
    if not grouped:
        return

    with _temporary_replayssm_autotune_state(
        runner, num_slots, prepare_dummy_block_tables=False
    ):
        for mixers, _, _ in grouped:
            for mixer in mixers:
                for state in mixer.kv_cache:
                    state[1 : num_slots + 1].zero_()
                mixer._replayssm_ring_start.zero_()
                mixer._replayssm_prev_num_accepted.zero_()

        context = ReplaySSMModelContext.create(grouped, runner.max_num_reqs)
        device = grouped[0][2].device
        align = grouped[0][1].mamba_cache_mode == "align"
        block_size = grouped[0][1].block_size
        query_len = (
            v2_runner.decode_query_len if use_v2 else runner.uniform_decode_query_len
        )
        idx_mapping = (
            torch.zeros(1, dtype=torch.int64, device=device) if use_v2 else None
        )
        live_cols = torch.ones(1, dtype=torch.int32, device=device) if align else None
        if align:
            context.reset_new_slots(
                idx_mapping=idx_mapping,
                src_cols=torch.full((1,), -1, dtype=torch.int32, device=device),
                dst_cols=live_cols,
                num_reqs=1,
            )
        context.postprocess(
            idx_mapping=idx_mapping,
            query_metadata=torch.tensor(
                [0, query_len] if use_v2 else [query_len],
                dtype=torch.int32,
                device=device,
            ),
            query_metadata_is_cumulative=use_v2,
            num_computed_tokens=torch.full(
                (1,),
                block_size if use_v2 else block_size - 1,
                dtype=torch.int32,
                device=device,
            ),
            num_computed_is_post_step=use_v2,
            num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),
            is_prefilling=torch.zeros(1, dtype=torch.bool, device=device),
            live_cols=live_cols,
            num_reqs=1,
        )


def replayssm_autotune_warmup(runner: "GPUModelRunner") -> None:
    max_token_prefill_kwargs = {
        "num_tokens": runner.scheduler_config.max_num_batched_tokens,
        "skip_eplb": True,
        "is_profile": True,
        "randomize_inputs": True,
    }
    autotune = _replayssm_autotune_kwargs(runner, max_token_prefill_kwargs)
    if autotune is None:
        return
    max_num_reqs, decode_kwargs = autotune
    with _temporary_replayssm_autotune_state(runner, max_num_reqs):
        runner._dummy_run(**decode_kwargs)
