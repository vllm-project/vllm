# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connect loaded Qwen4Exp QSA modules to their kernel-owned warmup."""

import sys
from typing import TYPE_CHECKING, cast

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _warmup_qwen4_exp_ple(worker: "Worker") -> None:
    ple_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.ple_layer")
    if ple_module is None:
        return
    ple = next(
        (
            layer
            for layer in worker.get_model().modules()
            if isinstance(layer, ple_module.Qwen4ExpPLELayer)
        ),
        None,
    )
    if ple is None:
        return

    from vllm.model_executor.layers.mamba.mamba_utils import (
        is_conv_state_dim_first,
    )
    from vllm.models.qwen4_exp.nvidia.ops.ple import ple_conv

    conv_state = ple.kv_cache[0]
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    state_width = ple.conv_state_len + ple.num_spec_tokens
    conv_state = torch.empty_strided(
        (1, ple.hc_hidden_size, state_width),
        conv_state.stride(),
        dtype=conv_state.dtype,
        device=conv_state.device,
    )
    inputs = torch.empty(
        (2, ple.hc_hidden_size),
        dtype=ple.model_config.dtype,
        device=conv_state.device,
    )
    residual = torch.empty_like(inputs)
    outer_residual = torch.empty_like(inputs)
    # Runtime state indices are a tail slice, while the query-start tensor is
    # aligned for a pure-prefill batch. Reproduce both pointer cache-key classes.
    state_indices_storage = torch.zeros(2, dtype=torch.int32, device=conv_state.device)
    state_indices = state_indices_storage[1:]
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32, device=conv_state.device)
    has_initial_states = torch.ones(1, dtype=torch.bool, device=conv_state.device)
    ple_conv(
        inputs,
        residual,
        conv_state,
        ple.conv1d.weight.squeeze(1).to(dtype=inputs.dtype),
        state_indices,
        outer_residual,
        mode="prefill",
        dilation=ple.short_conv_dilation,
        query_start_loc=query_start_loc,
        has_initial_states=has_initial_states,
    )
    logger.info("Warmed up Qwen4Exp PLE prefill short-convolution kernels.")


@torch.inference_mode()
def qwen4_exp_qsa_triton_warmup(worker: "Worker") -> None:
    """Warm every reachable QSA indexer and sparse-attention specialization."""

    qsa_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.indexer_qsa")
    if qsa_module is None:
        return
    indexer = next(
        (
            layer
            for layer in worker.get_model().modules()
            if isinstance(layer, qsa_module.QSAIndexer)
        ),
        None,
    )
    if indexer is None:
        return

    runner = worker.model_runner
    prefix = indexer.compressed_key_cache.prefix
    group_id = next(
        i
        for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
        if prefix in group.layer_names
    )
    if worker.use_v2_model_runner:
        runner_v2 = cast("GPUModelRunnerV2", runner)
        block_table = runner_v2.block_tables.input_block_tables[group_id]
        max_decode_query_len = runner_v2.decode_query_len
    else:
        block_table = runner.input_batch.block_table[group_id].get_device_tensor(
            runner.max_num_reqs
        )
        max_decode_query_len = runner.uniform_decode_query_len

    from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import (
        warmup_qsa_mqa_paged_decode,
        warmup_qsa_mqa_paged_prefill,
    )

    k_cache = indexer.compressed_key_cache.kv_cache
    assert k_cache.numel()
    profiles = warmup_qsa_mqa_paged_decode(
        k_cache,
        block_table,
        num_heads=indexer.index_n_heads,
        head_dim=indexer.index_head_dim,
        max_decode_query_len=max_decode_query_len,
        max_num_reqs=runner.max_num_reqs,
        max_num_batched_tokens=runner.max_num_tokens,
    )
    logger.info("Warmed up Qwen4Exp QSA decode kernels: %s.", profiles)
    warmup_qsa_mqa_paged_prefill(
        k_cache,
        block_table,
        num_heads=indexer.index_n_heads,
        head_dim=indexer.index_head_dim,
    )
    logger.info("Warmed up Qwen4Exp QSA prefill scorer.")

    qsa_attention_module = sys.modules.get("vllm.models.qwen4_exp.nvidia.qsa")
    if qsa_attention_module is None:
        return
    attention = next(
        (
            layer
            for layer in worker.get_model().modules()
            if isinstance(layer, qsa_attention_module.Qwen4ExpQSAAttention)
        ),
        None,
    )
    if attention is None:
        return

    attention_group_id = next(
        i
        for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
        if attention.layer_name in group.layer_names
    )
    if worker.use_v2_model_runner:
        attention_block_table = runner_v2.block_tables.input_block_tables[
            attention_group_id
        ]
    else:
        attention_block_table = runner.input_batch.block_table[
            attention_group_id
        ].get_device_tensor(runner.max_num_reqs)

    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        warmup_qsa_sparse_paged_attention,
    )

    key_cache, value_cache = attention.kv_cache.transpose(1, 2).split(
        attention.head_dim, dim=-1
    )
    attention_profiles = warmup_qsa_sparse_paged_attention(
        key_cache,
        value_cache,
        attention_block_table,
        num_query_heads=attention.num_heads,
        head_dim=attention.head_dim,
        selection_width=attention.indexer.output_width,
        max_num_batched_tokens=runner.max_num_tokens,
        has_output_gate=attention.attn_output_gate,
    )
    logger.info(
        "Warmed up Qwen4Exp QSA sparse-attention kernels for row counts: %s.",
        attention_profiles,
    )
    _warmup_qwen4_exp_ple(worker)
