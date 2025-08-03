from contextlib import contextmanager

import numpy as np
from vllm.compilation.nano_utils import NanoOpInfo
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    make_local_attention_virtual_batches,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.compilation import nano_manager


def prepare_nano_split_and_set_hooks(
    gpu_model_runner: "GPUModelRunner",
    scheduler_output: "SchedulerOutput",
) -> None:
    input_batch = gpu_model_runner.input_batch
    req_ids = input_batch.req_ids
    batch_size = len(req_ids)
    num_tokens = [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids]
    cached_seqlens = input_batch.num_computed_tokens_cpu[
        :batch_size
    ].tolist()
    split_config = nano_manager.prepare_nano_split(batch_size, num_tokens, cached_seqlens)

    attn_metadatas = []
    start_req_idx = 0
    end_req_idx = 0
    for nano_batch_size in split_config.batch_sizes:
        start_req_idx = end_req_idx
        end_req_idx = start_req_idx + nano_batch_size
        nano_batch_req_ids = req_ids[start_req_idx:end_req_idx]

        # Gather per-request info for this group
        nano_batch_num_scheduled_tokens = np.array(
            [scheduler_output.num_scheduled_tokens[rid] for rid in nano_batch_req_ids],
            dtype=np.int32,
        )
        nano_batch_cu_num_tokens, nano_batch_arange = (
            gpu_model_runner._get_cumsum_and_arange(nano_batch_num_scheduled_tokens)
        )
        nano_batch_total_tokens = int(nano_batch_cu_num_tokens[-1])
        nano_batch_req_indices = np.repeat(
            np.arange(len(nano_batch_req_ids)), nano_batch_num_scheduled_tokens
        )

        # Compute positions for this group
        nano_batch_positions_np = np.empty(nano_batch_total_tokens, dtype=np.int64)
        np.add(
            input_batch.num_computed_tokens_cpu[
                start_req_idx:end_req_idx
            ][nano_batch_req_indices],
            nano_batch_arange,
            out=nano_batch_positions_np,
        )

        # Prepare attention metadata for each KV cache group
        nano_batch_attn_metadata = {}
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
            gpu_model_runner.kv_cache_config.kv_cache_groups
        ):
            blk_table = input_batch.block_table[kv_cache_group_id]
            blk_table_tensor = blk_table.get_device_tensor()[start_req_idx:end_req_idx]
            slot_mapping = blk_table.slot_mapping[:nano_batch_total_tokens]

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=gpu_model_runner.query_start_loc[
                    start_req_idx : end_req_idx + 1
                ]
                - gpu_model_runner.query_start_loc[start_req_idx],
                query_start_loc_cpu=gpu_model_runner.query_start_loc_cpu[
                    start_req_idx : end_req_idx + 1
                ]
                - gpu_model_runner.query_start_loc_cpu[start_req_idx],
                seq_lens=gpu_model_runner.seq_lens[start_req_idx:end_req_idx],
                seq_lens_cpu=gpu_model_runner.seq_lens_cpu[start_req_idx:end_req_idx],
                num_computed_tokens_cpu=input_batch.num_computed_tokens_cpu_tensor[
                    start_req_idx:end_req_idx
                ],
                num_reqs=nano_batch_size,
                num_actual_tokens=nano_batch_total_tokens,
                max_query_len=int(max(nano_batch_num_scheduled_tokens)),
                block_table_tensor=blk_table_tensor,
                slot_mapping=slot_mapping,
            )

            if isinstance(kv_cache_group_spec.kv_cache_spec, ChunkedLocalAttentionSpec):
                common_attn_metadata = make_local_attention_virtual_batches(
                    kv_cache_group_spec.kv_cache_spec.attention_chunk_size,
                    common_attn_metadata,
                    gpu_model_runner.cache_config.block_size,
                )

            # NOTE(yi): does not support cascade attention
            common_prefix_len = 0
            builder = gpu_model_runner.attn_metadata_builders[kv_cache_group_id]
            attn_metadata_i = builder.build(
                common_prefix_len=common_prefix_len,
                common_attn_metadata=common_attn_metadata,
            )

            for layer_name in kv_cache_group_spec.layer_names:
                nano_batch_attn_metadata[layer_name] = attn_metadata_i

        attn_metadatas.append(nano_batch_attn_metadata)

    assert (
        end_req_idx == batch_size
    ), f"invalid nano batch size: {split_config.batch_sizes}"
    forward_contexts = [
        ForwardContext(
            no_compile_layers=gpu_model_runner.vllm_config.compilation_config.static_forward_context,
            virtual_engine=0,
            attn_metadata=attn_metadata,
            dp_metadata=None,
            skip_cuda_graphs=True,
        )
        for attn_metadata in attn_metadatas
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
