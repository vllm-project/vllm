# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    flashinfer_replayssm_autotune_supported,
)
from vllm.utils.math_utils import round_up

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
) -> Generator[None]:
    """Run ReplaySSM dummy steps, then restore the touched state."""
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    reset_tensors: list[torch.Tensor] = []
    reset_trackers: dict[int, torch.Tensor] = {}
    for module in runner.get_model().modules():
        if not isinstance(module, MambaBase) or not module.use_replayssm_trackers:
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
    if not config.use_replayssm_trackers:
        return
    from vllm.v1.worker.extensible_kv_cache import (
        ensure_kv_cache_blocks,
        num_committable_kv_blocks,
    )

    num_slots = min(num_committable_kv_blocks(runner) - 1, 2)
    if num_slots <= 0:
        return

    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.v1.core.kv_cache_utils import (
        is_mamba_group,
        is_mamba_prefix_caching_group,
    )
    from vllm.v1.worker.mamba_utils import _get_mamba_spec_for_layer
    from vllm.v1.worker.replayssm_utils import ReplaySSMModelContext

    grouped = []
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        if not is_mamba_group(group):
            continue
        names = [
            name
            for name in group.layer_names
            if isinstance(
                layer := config.compilation_config.static_forward_context[name],
                MambaBase,
            )
            and layer.use_replayssm_trackers
        ]
        if not names:
            continue
        spec = _get_mamba_spec_for_layer(group, names[0])
        prefix_caching = is_mamba_prefix_caching_group(group)
        mixers = [
            config.compilation_config.static_forward_context[name] for name in names
        ]
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

    warmed: set[tuple[Any, ...]] = set()
    for module in runner.get_model().modules():
        if not (isinstance(module, GatedDeltaNetAttention) and module.use_replayssm):
            continue
        state = module.kv_cache[1]
        materialize = load_gdn_replayssm_materializer(state.dtype)
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


def hmma_warmup_batch_sizes(
    num_v_heads: int, num_sms: int, max_num_reqs: int
) -> list[int]:
    """Return one batch size for each launch tactic of the HMMA kernel.

    FlashInfer compiles one HMMA kernel per `min_blocks_per_mp`, which it sets
    to `min(ceil(num_v_heads * batch / num_sms) + 1, 7)`, that is, 2 to 7.
    `GDN_WY_MBP` overrides that policy with one value.
    """
    batches = {1}
    if not os.environ.get("GDN_WY_MBP"):
        batches.update(i * num_sms // num_v_heads + 1 for i in range(1, 6))
    return sorted(b for b in batches if b <= max_num_reqs)


def _warm_gdn_replayssm_hmma(runner: "GPUModelRunner", max_num_reqs: int) -> None:
    """Compile the HMMA kernel for each layout and tactic outside full graphs.

    Full-graph decode passes Q/K/V as strided views of the QKVZ projection.
    Mixed prefill/decode steps pack them into compact rows. FlashInfer
    compiles each row stride and launch tactic separately, so warm both
    strides at every tactic. With flashinfer-ai/flashinfer#8975 the compiled
    kernels then serve every batch and pool size.
    """
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
    from vllm.model_executor.layers.mamba.gdn.replayssm import run_gdn_replayssm

    modules = [
        module
        for module in runner.get_model().modules()
        if isinstance(module, GatedDeltaNetAttention)
        and module.use_replayssm
        and module.replayssm_backend == "hmma"
    ]
    if not modules:
        return
    provider = importlib.import_module(
        "flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush"
    )
    if not getattr(provider, "SUPPORTS_DYNAMIC_STRIDED_SHAPES", False):
        logger.warning_once(
            "This FlashInfer version compiles the HMMA GDN ReplaySSM kernel "
            "for each new batch and pool size, so the first mixed "
            "prefill/decode steps of each size can stall for about a second. "
            "FlashInfer with flashinfer-ai/flashinfer#8975 avoids this."
        )
        return
    warmed: set[tuple[Any, ...]] = set()
    for module in modules:
        state = module.kv_cache[1]
        u_cache, k_cache, g_cache = module.replayssm_rings
        signature = tuple(
            (t.device, t.dtype, tuple(t.shape[1:]), tuple(t.stride()))
            for t in (state, u_cache, k_cache, g_cache)
        )
        if signature in warmed:
            continue
        warmed.add(signature)
        width = module.replayssm_executed_width
        assert width is not None
        _, num_v_heads, head_v_dim, head_k_dim = state.shape
        num_k_heads = k_cache.shape[1]
        qkv_width = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        # Compact rows, and views of the QKVZ projection rows.
        row_strides = (qkv_width, qkv_width + num_v_heads * head_v_dim)
        num_sms = torch.cuda.get_device_properties(state.device).multi_processor_count
        for num_reqs in hmma_warmup_batch_sizes(num_v_heads, num_sms, max_num_reqs):
            num_tokens = num_reqs * width
            # Rows with state index -1 make the kernel exit before it reads or
            # writes any state or ring. Each cursor row starts at a 16-byte
            # boundary, as in the metadata builder.
            cursors = torch.zeros(
                3, round_up(num_reqs, 4), dtype=torch.int32, device=state.device
            )[:, :num_reqs]
            cursors[0].fill_(-1)
            query_start_loc = torch.arange(
                0, num_tokens + 1, width, dtype=torch.int32, device=state.device
            )
            gates = torch.zeros(
                num_tokens, num_v_heads, dtype=torch.bfloat16, device=state.device
            )
            for row_stride in row_strides:
                qkv = torch.zeros(
                    num_tokens, row_stride, dtype=torch.bfloat16, device=state.device
                )[:, :qkv_width]
                run_gdn_replayssm(
                    kernel=module._replayssm_kernel,
                    executed_width=width,
                    mixed_qkv=qkv,
                    a=gates,
                    b=gates,
                    A_log=module.A_log,
                    dt_bias=module.dt_bias,
                    checkpoint_state=state,
                    replayssm_cache=(u_cache, k_cache, g_cache),
                    cursors=cursors,
                    query_start_loc=query_start_loc,
                    output_indices=torch.empty(
                        0, dtype=torch.int64, device=state.device
                    ),
                    num_k_heads=num_k_heads,
                    num_v_heads=num_v_heads,
                    head_k_dim=head_k_dim,
                    head_v_dim=head_v_dim,
                    uniform_query_length=width,
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
        _warm_gdn_replayssm_hmma(runner, max_decode_reqs)
        if config.cache_config.enable_prefix_caching:
            _warm_gdn_replayssm_materializer(runner, max_num_reqs)
