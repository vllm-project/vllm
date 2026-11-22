# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.worker.replayssm_utils import (
    replayssm_state,
    uses_replayssm,
    uses_ring_trackers,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


@contextmanager
def _zero_state_slots_after(layers: Iterable[Any], num_slots: int) -> Generator[None]:
    """Zero the Mamba layers' state slots 1..num_slots, and ReplaySSM trackers,
    once warmup has run on them."""
    reset_tensors: list[torch.Tensor] = []
    reset_trackers: dict[int, torch.Tensor] = {}
    for layer in layers:
        for tensor in layer.kv_cache:
            if tensor.numel():
                # Overlaid groups may start at the same address with different
                # widths. Preserve each bounded view; clearing twice is harmless.
                reset_tensors.append(tensor)
        if uses_replayssm(layer):
            state = replayssm_state(layer)
            for tensor in (state.ring_start, state.prev_num_accepted):
                if tensor.numel():
                    reset_trackers.setdefault(tensor.data_ptr(), tensor)

    try:
        yield
    finally:
        for tensor in reset_tensors:
            tensor[1 : num_slots + 1].zero_()
        for tensor in reset_trackers.values():
            tensor.zero_()


def replayssm_lifecycle_warmup(runner: "GPUModelRunner") -> None:
    """Warm request lifecycle kernels with bounded, non-padding cache slots."""
    config = runner.vllm_config
    if not config.cache_config.use_replayssm:
        return
    from vllm.v1.worker.extensible_kv_cache import (
        ensure_kv_cache_blocks,
        num_committable_kv_blocks,
    )

    num_slots = min(num_committable_kv_blocks(runner) - 1, 2)
    if num_slots <= 0:
        return

    from vllm.v1.core.kv_cache_utils import (
        is_mamba_group,
        is_mamba_prefix_caching_group,
    )
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.replayssm_utils import ReplaySSMModelContext

    grouped = []
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if not is_mamba_group(group):
            continue
        mixers = [
            layer
            for name in group.layer_names
            if uses_ring_trackers(
                layer := config.compilation_config.static_forward_context[name]
            )
        ]
        if not mixers:
            continue
        spec = group.kv_cache_specs[group.layer_names[0]]
        assert isinstance(spec, MambaSpec)
        prefix_caching = is_mamba_prefix_caching_group(group)
        table = runner.block_tables.input_block_tables[group_id]
        # Match the runtime stride without changing its block-table contents.
        scratch_table = torch.empty_strided(
            (1, 2), table.stride(), dtype=table.dtype, device=table.device
        )
        scratch_table[0, 0] = 1
        scratch_table[0, 1] = num_slots
        grouped.append((mixers, spec, scratch_table))
        # Homogeneous layer/group counts are runtime values, not JIT keys.
        # One group warms the kernels without overlapping shared scratch pages.
        break
    if not grouped:
        return

    ensure_kv_cache_blocks(runner, 1 + num_slots)
    all_mixers = [mixer for mixers, _, _ in grouped for mixer in mixers]
    with _zero_state_slots_after(all_mixers, num_slots):
        for mixers, _, _ in grouped:
            for mixer in mixers:
                for state in mixer.kv_cache:
                    state[1 : num_slots + 1].zero_()
                state = replayssm_state(mixer)
                state.ring_start.zero_()
                state.prev_num_accepted.zero_()

        context = ReplaySSMModelContext.create(
            grouped, runner.max_num_reqs, materialize_prefixes=prefix_caching
        )
        device = grouped[0][2].device
        block_size = grouped[0][1].block_size
        query_len = runner.decode_query_len
        idx_mapping = torch.zeros(1, dtype=torch.int64, device=device)
        live_cols = (
            torch.ones(1, dtype=torch.int32, device=device) if prefix_caching else None
        )
        if prefix_caching:
            context.reset_new_slots(
                idx_mapping=idx_mapping,
                src_cols=torch.full((1,), -1, dtype=torch.int32, device=device),
                dst_cols=live_cols,
                num_reqs=1,
            )
        context.postprocess(
            idx_mapping=idx_mapping,
            query_metadata=torch.tensor(
                [0, query_len],
                dtype=torch.int32,
                device=device,
            ),
            query_metadata_is_cumulative=True,
            num_computed_tokens=torch.full(
                (1,),
                block_size,
                dtype=torch.int32,
                device=device,
            ),
            num_computed_is_post_step=True,
            num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),
            is_prefilling=torch.zeros(1, dtype=torch.bool, device=device),
            live_cols=live_cols,
            num_reqs=1,
        )


def mamba_ssu_autotune_warmup(runner: "GPUModelRunner") -> None:
    """Let each Mamba decode kernel that tunes do so at the largest decode
    batch, on distinct committed state slots."""
    from vllm.v1.worker.extensible_kv_cache import (
        ensure_kv_cache_blocks,
        num_committable_kv_blocks,
    )

    forward_context = runner.vllm_config.compilation_config.static_forward_context
    layers = [
        layer
        for layer in forward_context.values()
        if (kernel := getattr(layer, "ssu_kernel", None)) is not None
        and kernel.needs_autotune()
    ]
    if not layers:
        return
    # Block 0 is the null block; the slots must be committed memory.
    num_reqs = min(
        runner.scheduler_config.max_num_seqs,
        runner.max_num_tokens // runner.decode_query_len,
        num_committable_kv_blocks(runner) - 1,
    )
    if num_reqs <= 0:
        logger.warning_once(
            "Skipping Mamba decode kernel autotuning because no non-padding "
            "state slot is available."
        )
        return
    ensure_kv_cache_blocks(runner, 1 + num_reqs)
    with _zero_state_slots_after(layers, num_reqs):
        for layer in layers:
            layer.ssu_kernel.autotune(layer, num_reqs)
