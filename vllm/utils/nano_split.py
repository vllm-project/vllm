# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

from vllm.compilation.nanoflow import manager as nano_manager
from vllm.compilation.nanoflow.split_utils import NanoOpInfo
from vllm.forward_context import (ForwardContext, get_forward_context,
                                  override_forward_context)
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


def _get_cumsum_and_arange(
    num_tokens: np.ndarray,
    cumsum_dtype: Optional[np.dtype] = None,
) -> tuple[np.ndarray, np.ndarray]:
    cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
    total_num_tokens = cu_num_tokens[-1]
    cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
    arange = np.arange(total_num_tokens) - cumsums_offsets
    return cu_num_tokens, arange


def prepare_nano_split_and_set_hooks(
    scheduler_output: SchedulerOutput,
    input_batch: InputBatch,
    attn_metadata_builders: list[AttentionMetadataBuilder],
    kv_cache_config: KVCacheConfig,
) -> None:
    prev_forward_context = get_forward_context()
    req_ids = input_batch.req_ids
    batch_size = len(req_ids)
    tokens = [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids]
    num_scheduled_tokens = torch.tensor(tokens, dtype=torch.int32)
    cached_seqlens = input_batch.num_computed_tokens_cpu[:batch_size].tolist()
    split_config = nano_manager.prepare_nano_split(batch_size, tokens,
                                                   cached_seqlens)
    if split_config.num_nano_batches == 1:
        return

    cu_num_tokens = torch.cumsum(num_scheduled_tokens,
                                 dim=0,
                                 dtype=torch.int32)
    query_start_loc_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
    query_start_loc_cpu[0] = 0
    query_start_loc_cpu[1:batch_size + 1] = cu_num_tokens
    seq_lens_cpu = torch.zeros(batch_size, dtype=torch.int32)
    seq_lens_cpu[:batch_size] = (
        input_batch.num_computed_tokens_cpu_tensor[:batch_size] +
        num_scheduled_tokens)
    query_start_loc = query_start_loc_cpu.to(input_batch.device)
    seq_lens = seq_lens_cpu.to(input_batch.device)

    attn_metadatas = []
    start_req_idx = 0
    end_req_idx = 0
    for nano_batch_idx in range(split_config.num_nano_batches):
        start_req_idx = split_config.batch_indices[nano_batch_idx]
        end_req_idx = split_config.batch_indices[nano_batch_idx + 1]
        nano_batch_req_ids = req_ids[start_req_idx:end_req_idx]
        start_token_idx = split_config.split_indices[nano_batch_idx]
        end_token_idx = split_config.split_indices[nano_batch_idx + 1]

        # Gather per-request info for this group
        nano_batch_num_scheduled_tokens = np.array(
            [
                scheduler_output.num_scheduled_tokens[rid]
                for rid in nano_batch_req_ids
            ],
            dtype=np.int32,
        )
        nano_batch_cu_num_tokens, nano_batch_arange = _get_cumsum_and_arange(
            nano_batch_num_scheduled_tokens)
        nano_batch_total_tokens = int(nano_batch_cu_num_tokens[-1])
        nano_batch_req_indices = np.repeat(np.arange(len(nano_batch_req_ids)),
                                           nano_batch_num_scheduled_tokens)

        # Compute positions for this group
        nano_batch_positions_np = np.empty(nano_batch_total_tokens,
                                           dtype=np.int64)
        np.add(
            input_batch.num_computed_tokens_cpu[start_req_idx:end_req_idx]
            [nano_batch_req_indices],
            nano_batch_arange,
            out=nano_batch_positions_np,
        )

        # Prepare attention metadata for each KV cache group
        nano_batch_attn_metadata = {}
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                kv_cache_config.kv_cache_groups):
            blk_table = input_batch.block_table[kv_cache_group_id]
            blk_table_tensor = blk_table.get_device_tensor(
            )[start_req_idx:end_req_idx]
            slot_mapping = blk_table.slot_mapping[
                start_token_idx:end_token_idx]

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc[start_req_idx:end_req_idx + 1]
                - query_start_loc[start_req_idx],
                query_start_loc_cpu=query_start_loc_cpu[
                    start_req_idx:end_req_idx + 1] -
                query_start_loc_cpu[start_req_idx],
                seq_lens=seq_lens[start_req_idx:end_req_idx],
                seq_lens_cpu=seq_lens_cpu[start_req_idx:end_req_idx],
                num_computed_tokens_cpu=input_batch.
                num_computed_tokens_cpu_tensor[start_req_idx:end_req_idx],
                num_reqs=split_config.batch_sizes[nano_batch_idx],
                num_actual_tokens=nano_batch_total_tokens,
                max_query_len=int(max(nano_batch_num_scheduled_tokens)),
                block_table_tensor=blk_table_tensor,
                slot_mapping=slot_mapping,
            )

            # NOTE(yi): does not support chunked local attention or cascade
            # attention
            common_prefix_len = 0
            builder = attn_metadata_builders[kv_cache_group_id]
            attn_metadata_i = builder.build(
                common_prefix_len=common_prefix_len,
                common_attn_metadata=common_attn_metadata,
            )

            for layer_name in kv_cache_group_spec.layer_names:
                nano_batch_attn_metadata[layer_name] = attn_metadata_i

        attn_metadatas.append(nano_batch_attn_metadata)

    assert (end_req_idx == batch_size
            ), f"invalid nano batch size: {split_config.batch_sizes}"
    forward_contexts = [
        ForwardContext(
            no_compile_layers=prev_forward_context.no_compile_layers,
            attn_metadata=attn_metadata,
            virtual_engine=prev_forward_context.virtual_engine,
            dp_metadata=prev_forward_context.dp_metadata,
            skip_cuda_graphs=True,
        ) for attn_metadata in attn_metadatas
    ]

    @contextmanager
    def op_hook(op_info: NanoOpInfo):
        previous_context = get_forward_context()
        override_forward_context(forward_contexts[op_info.idx])
        try:
            yield
        finally:
            override_forward_context(previous_context)

    nano_manager.set_op_hook(op_hook)
