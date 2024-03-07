from typing import List, Optional

import torch

from vllm._C import cache_ops
from vllm._C import ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention.ops.prefix_prefill import (
    context_attention_fwd)

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttentionImpl:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        cache_ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            input_metadata.slot_mapping.flatten(),
            input_metadata.kv_cache_dtype,
        )

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output = torch.empty_like(query)

        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = (
            (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
            _PARTITION_SIZE)
        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        use_v1 = input_metadata.max_context_len <= 8192 and (
            max_num_partitions == 1 or num_seqs * num_heads > 512)
        if use_v1:
            # Run PagedAttention V1.
            ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                input_metadata.block_tables,
                input_metadata.context_lens,
                block_size,
                input_metadata.max_context_len,
                alibi_slopes,
                input_metadata.kv_cache_dtype,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                input_metadata.block_tables,
                input_metadata.context_lens,
                block_size,
                input_metadata.max_context_len,
                alibi_slopes,
                input_metadata.kv_cache_dtype,
            )
        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        alibi_slopes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        context_attention_fwd(
            query,
            key,
            value,
            output,
            key_cache,
            value_cache,
            input_metadata.block_tables,  # [BS, max_block_per_request]
            input_metadata.start_loc,
            input_metadata.prompt_lens,
            input_metadata.context_lens,
            input_metadata.max_seq_len,
            alibi_slopes,
        )
        return output
