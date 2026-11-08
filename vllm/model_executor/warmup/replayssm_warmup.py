# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import gdn_replayssm_executed_width
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    flashinfer_replayssm_autotune_supported,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


def _replayssm_autotune_kwargs(
    runner: "GPUModelRunner",
    max_token_prefill_kwargs: dict[str, Any],
) -> tuple[int, dict[str, Any]] | None:
    config = runner.vllm_config
    if not config.use_flashinfer_replayssm or config.is_gdn_replayssm_enabled():
        return None
    if not flashinfer_replayssm_autotune_supported():
        logger.info_once(
            "Skipping FlashInfer ReplaySSM autotuning because "
            "flashinfer.mamba.checkpointing_ssu.CheckpointingSSURunner "
            "is unavailable."
        )
        return None
    from vllm.v1.worker.extensible_kv_cache import num_committable_kv_blocks

    query_len = runner.decode_query_len
    # Block 0 is the null block; the dummy slots must be committed memory.
    max_num_reqs = min(
        runner.scheduler_config.max_num_seqs,
        runner.max_num_tokens // query_len,
        num_committable_kv_blocks(runner) - 1,
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
        "valid_dummy_state_slots": True,
    }
    return max_num_reqs, decode_kwargs


@contextmanager
def _temporary_replayssm_autotune_state(
    runner: "GPUModelRunner",
    max_num_reqs: int,
    *,
    prepare_dummy_block_tables: bool = True,
) -> Iterator[None]:
    """Run FlashInfer ReplaySSM dummy steps, then restore the touched state."""
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    reset_tensors: list[torch.Tensor] = []
    reset_trackers: dict[int, torch.Tensor] = {}
    for module in runner.get_model().modules():
        if not isinstance(module, MambaBase) or not module.use_flashinfer_replayssm:
            continue
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

    try:
        yield
    finally:
        if prepare_dummy_block_tables:
            runner.block_tables.get_dummy_block_tables(max_num_reqs)
        for tensor in reset_tensors:
            tensor[1 : max_num_reqs + 1].zero_()
        for tensor in reset_trackers.values():
            tensor.zero_()


def replayssm_lifecycle_warmup(runner: "GPUModelRunner") -> None:
    """Warm request lifecycle kernels with bounded, non-padding cache slots."""
    config = runner.vllm_config
    if not config.use_flashinfer_replayssm:
        return
    from vllm.v1.worker.extensible_kv_cache import (
        ensure_kv_cache_blocks,
        num_committable_kv_blocks,
    )

    num_slots = min(num_committable_kv_blocks(runner) - 1, 2)
    if num_slots <= 0:
        return

    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.replayssm_utils import ReplaySSMModelContext

    grouped = []
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if not isinstance(spec, MambaSpec):
            continue
        mixers = [
            layer
            for name in group.layer_names
            if isinstance(
                layer := config.compilation_config.static_forward_context[name],
                MambaBase,
            )
            and layer.use_flashinfer_replayssm
        ]
        if not mixers:
            continue
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
        query_len = runner.decode_query_len
        idx_mapping = torch.zeros(1, dtype=torch.int64, device=device)
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


def replayssm_autotune_warmup(runner: "GPUModelRunner") -> None:
    from vllm.v1.worker.extensible_kv_cache import ensure_kv_cache_blocks

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
    ensure_kv_cache_blocks(runner, 1 + max_num_reqs)
    with _temporary_replayssm_autotune_state(runner, max_num_reqs):
        runner._dummy_run(**decode_kwargs)


def _warm_gdn_replayssm_materializer(
    runner: "GPUModelRunner", max_num_reqs: int
) -> None:
    """Compile the GDN prefix materializer for each current cache layout.

    FlashInfer specializes on state and ring shapes and strides. A later
    pool-capacity change can require another compilation.
    """
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
    from vllm.model_executor.layers.mamba.gdn.replayssm import (
        load_gdn_replayssm_materializer,
    )

    materialize = load_gdn_replayssm_materializer()
    warmed: set[tuple[Any, ...]] = set()
    for module in runner.get_model().modules():
        if not (isinstance(module, GatedDeltaNetAttention) and module.use_replayssm):
            continue
        state = module.kv_cache[1]
        u_cache, k_cache, g_cache = module.replayssm_rings
        signature = tuple(
            (t.device, t.dtype, tuple(t.shape), tuple(t.stride()))
            for t in (state, u_cache, k_cache, g_cache)
        )
        if signature in warmed:
            continue
        warmed.add(signature)
        # Every request index is inactive, so the kernel touches no state.
        inactive = torch.full(
            (max_num_reqs,), -1, dtype=torch.int32, device=state.device
        )
        materialize(
            state=state,
            src_slots=inactive,
            dst_slots=inactive,
            k_cache=k_cache,
            u_cache=u_cache,
            g_cache=g_cache,
            cache_base=torch.zeros_like(inactive),
            count=inactive,
            active_request_indices=inactive,
            num_active=torch.zeros(1, dtype=torch.int32, device=state.device),
        )


def gdn_replayssm_warmup(runner: "GPUModelRunner") -> None:
    """Warm captured GDN decode sizes and current materializer layouts."""
    config = runner.vllm_config
    if not config.is_gdn_replayssm_enabled():
        return
    from vllm.v1.worker.extensible_kv_cache import (
        ensure_kv_cache_blocks,
        num_committable_kv_blocks,
    )

    query_len = config.num_speculative_tokens + 1
    assert query_len <= gdn_replayssm_executed_width(config.num_speculative_tokens)
    max_num_reqs = min(
        runner.scheduler_config.max_num_seqs,
        num_committable_kv_blocks(runner) - 1,
    )
    if max_num_reqs <= 0:
        raise ValueError(
            "FlashInfer GDN ReplaySSM warmup requires a non-padding state slot"
        )
    max_decode_reqs = min(max_num_reqs, runner.max_num_tokens // query_len)
    manager = runner.cudagraph_manager
    assert manager is not None
    capture_reqs = {
        desc.num_reqs
        for descs in manager._capture_descs.values()
        for desc in descs
        if desc.uniform_token_count == query_len
        and desc.num_reqs is not None
        and 0 < desc.num_reqs <= max_decode_reqs
    }
    if not capture_reqs and max_decode_reqs > 0:
        capture_reqs.add(max_decode_reqs)

    decode_kwargs: dict[str, Any] = {
        "uniform_decode": True,
        "skip_eplb": True,
        "is_profile": True,
        "randomize_inputs": True,
        "valid_dummy_state_slots": True,
    }
    ensure_kv_cache_blocks(runner, 1 + max_decode_reqs)
    with _temporary_replayssm_autotune_state(runner, max_decode_reqs):
        for num_reqs in sorted(capture_reqs):
            runner._dummy_run(num_tokens=num_reqs * query_len, **decode_kwargs)
        if config.cache_config.mamba_cache_mode == "align":
            _warm_gdn_replayssm_materializer(runner, max_num_reqs)
