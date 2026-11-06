# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the Triton MLA decode attention kernels.

Eager-only layers (e.g. an EAGLE drafter that is never graph-captured) first
run decode attention on live traffic. Batch size only shapes the launch grid,
so one call per layer config and ``seq_lens`` dtype covers every batch: the
plain decode path passes int32 ``seq_lens`` while multi-token expansion in
subclasses produces int64. ``NUM_KV_SPLITS`` is a constexpr, so every split
count the runtime can select must be warmed separately.
"""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.warmup.runner_utils import (
    block_table,
    kv_cache_group_ids,
    target_and_draft_models,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.triton_mla import (
    TritonMLAImpl,
    _compute_num_kv_splits,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.v1.kv_cache_interface import is_quantized_kv_cache

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker


def _num_kv_splits_values(max_model_len: int, sm_count: int) -> tuple[int, ...]:
    """Every num_kv_splits value runtime can pick for this config.

    ``TritonMLAImpl`` selects ``_compute_num_kv_splits(max_seq_len, sm_count)``
    per batch: a power of two bounded by the SM-based maximum. The bound is
    itself reachable for long sequences, so the set is every power of two up
    to it plus the bound when it is not a power of two.
    """
    max_splits = _compute_num_kv_splits(max_model_len, sm_count)
    values = []
    splits = 1
    while splits <= max_splits:
        values.append(splits)
        splits *= 2
    if values[-1] != max_splits:
        values.append(max_splits)
    return tuple(values)


@torch.inference_mode()
def triton_mla_decode_warmup(worker: "Worker") -> None:
    runner = worker.model_runner
    if runner.is_pooling_model:
        return
    layer_to_gid = kv_cache_group_ids(worker)
    num_kv_splits_values: tuple[int, ...]
    if envs.VLLM_BATCH_INVARIANT:
        num_kv_splits_values = (1,)
    else:
        num_kv_splits_values = _num_kv_splits_values(
            worker.vllm_config.model_config.max_model_len,
            current_platform.num_compute_units(),
        )
    device = runner.device
    seen: set[tuple] = set()
    for model in target_and_draft_models(worker):
        for layer in model.modules():
            if not isinstance(layer, MLAAttention):
                continue
            impl = layer.impl
            kv_cache = layer.kv_cache
            if not isinstance(impl, TritonMLAImpl) or kv_cache.numel() == 0:
                continue
            # Same view as MLAAttention.forward_impl: a uint8-stored fp8 cache
            # reaches forward_mqa as fp8.
            if is_quantized_kv_cache(layer.kv_cache_dtype) and (
                layer.kv_cache_dtype not in ("fp8_ds_mla", "nvfp4_ds_mla")
            ):
                kv_cache = kv_cache.view(current_platform.fp8_dtype())
            # Subclasses may override forward_mqa with the non-MLA tiling.
            is_mla_variants = (
                (True,)
                if type(impl).forward_mqa is TritonMLAImpl.forward_mqa
                else (True, False)
            )
            gid = layer_to_gid[layer.layer_name]
            block_table_row = block_table(worker, gid, num_reqs=1)
            num_heads = impl.num_heads * impl.dcp_world_size
            dtype = runner.model_config.dtype
            key = (
                num_heads,
                impl.kv_lora_rank,
                kv_cache.dtype,
                tuple(kv_cache.shape[1:]),
                block_table_row.stride(0),
                dtype,
                is_mla_variants,
            )
            if key in seen:
                continue
            seen.add(key)

            cache = kv_cache.unsqueeze(2)
            kv_c_cache = cache[..., : impl.kv_lora_rank]
            q = torch.zeros((1, num_heads, cache.shape[-1]), dtype=dtype, device=device)
            o = torch.empty(
                (1, num_heads, impl.kv_lora_rank), dtype=dtype, device=device
            )
            lse = torch.empty((1, num_heads), dtype=dtype, device=device)
            for num_kv_splits in num_kv_splits_values:
                attn_logits = torch.empty(
                    (1, num_heads, num_kv_splits, impl.kv_lora_rank + 1),
                    dtype=torch.float32,
                    device=device,
                )
                for is_mla in is_mla_variants:
                    for seq_lens_dtype in (torch.int32, torch.int64):
                        decode_attention_fwd(
                            q,
                            cache,
                            kv_c_cache,
                            o,
                            lse,
                            block_table_row,
                            torch.zeros(1, dtype=seq_lens_dtype, device=device),
                            attn_logits,
                            num_kv_splits,
                            impl.scale,
                            cache.size(1),
                            k_scale=layer._k_scale,
                            v_scale=layer._k_scale,
                            is_mla=is_mla,
                        )
