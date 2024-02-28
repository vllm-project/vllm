"""Multi-head attention."""
from typing import List, Optional, Tuple

import importlib
import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalFromBottomRightMask, LowerTriangularMaskWithTensorBias)

from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.triton_kernel.prefix_prefill import (
    context_attention_fwd)
from vllm.utils import is_hip
from vllm.model_executor.kv_buffer import KVBuffer

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttention(nn.Module):
    """MHA/MQA/GQA layer with PagedAttention.

     This class takes flattened 1D query, key, and value tensors as input. The
    input 1D tensors can either contain prompt tokens or generation tokens, in
    addition to paddings.
    If the input tensors contain prompt tokens, the layout is as follows:
    |<---------------------- num_valid_tokens ---------------------->|
    |<--------------- num_prompt_tokens -------------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|<--padding-->|
    Otherwise, the layout is as follows:
    |<------------------ num_valid_tokens ------------------->|
    |<------- num_generation_tokens (M) ------->|
    |<--generation_0-->|...|<--generation_M-1-->|<--padding-->|
    The prompts might have different lengths, while the generation tokens always
    have length 1. The paddings are appended to make the input length a multiple
    of 8, which is desirable for Tensor Cores.

    The class does the following:
    1. Reshape and store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention using either
        xformers or the PagedAttention custom op.
    3. Output a flattened 1D tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

        self.use_ref_attention = self.check_use_ref_attention()

    def check_use_ref_attention(self) -> bool:
        if not is_hip():
            return False
        # For ROCm, check whether flash attention is installed or not.
        # if not, use_ref_attention needs to be True
        return importlib.util.find_spec("flash_attn") is None

    def ref_masked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        seq_len, _, _ = query.shape
        attn_mask = torch.triu(torch.ones(seq_len,
                                          seq_len,
                                          dtype=query.dtype,
                                          device=query.device),
                               diagonal=1)
        attn_mask = attn_mask * torch.finfo(query.dtype).min

        attn_weights = self.scale * torch.einsum("qhd,khd->hqk", query,
                                                 key).float()
        attn_weights = attn_weights + attn_mask.float()
        attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
        out = torch.einsum("hqk,khd->qhd", attn_weights, value)
        return out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        kv_buffer: KVBuffer,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Preallocating the output tensor.
        output = torch.empty_like(query)

        num_valid_tokens = input_metadata.num_valid_tokens
        num_current_prompt_tokens = input_metadata.num_current_prompt_tokens
        num_generation_tokens = input_metadata.num_generation_tokens
        # print(num_valid_tokens, num_current_prompt_tokens, num_generation_tokens)

        if num_current_prompt_tokens > 0:
            # Set attention bias if not provided. This typically happens at the
            # very attention layer of every iteration.
            # FIXME(woosuk): This is a hack.
            if len(input_metadata.attn_bias) == 0:
                for seq_id, current_prompt_chunk_len, processed_prompt_len in zip(
                        input_metadata.prompt_seq_ids,
                        input_metadata.current_prompt_chunk_lens,
                        input_metadata.processed_prompt_lens,
                ):
                    if self.alibi_slopes is None:
                        kv_cache_len = current_prompt_chunk_len + processed_prompt_len
                        attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
                            [current_prompt_chunk_len], [kv_cache_len])
                        if self.sliding_window is not None:
                            processed_prompt_len = min(processed_prompt_len, self.sliding_window)
                            attn_bias = attn_bias.make_local_attention_from_bottomright(self.sliding_window)
                        input_metadata.attn_bias[seq_id] = attn_bias
                    else:
                        raise RuntimeError("ALiBi is not yet supported")
                        input_metadata.attn_bias = _make_alibi_bias(
                            self.alibi_slopes, self.num_kv_heads, batch_size,
                            seq_len, query.dtype)

            # Compute attention op for prompts.
            kv_tensors = self.get_prefill_kv_tensors(
                key[:num_current_prompt_tokens],
                value[:num_current_prompt_tokens],
                input_metadata,
                kv_buffer,
            )

            # we need to work with query[:num_current_prompt_tokens]
            offset = 0
            for seq_id, current_prompt_chunk_len, kv_tensor in zip(
                    input_metadata.prompt_seq_ids,
                    input_metadata.current_prompt_chunk_lens,
                    kv_tensors,
            ):
                seq_query = query[offset:offset + current_prompt_chunk_len]
                seq_output = output[offset:offset + current_prompt_chunk_len]
                seq_key = kv_tensor[0]
                seq_value = kv_tensor[1]
                offset += current_prompt_chunk_len

                if self.num_kv_heads != self.num_heads:
                    # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
                    # project the key and value tensors to the desired number of
                    # heads.
                    # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                    seq_query = seq_query.view(seq_query.shape[0],
                                            self.num_kv_heads,
                                            self.num_queries_per_kv,
                                            seq_query.shape[-1])
                    seq_key = seq_key[:, :,
                                    None, :].expand(seq_key.shape[0],
                                                    self.num_kv_heads,
                                                    self.num_queries_per_kv,
                                                    seq_key.shape[-1])
                    seq_value = seq_value[:, :, None, :].expand(
                        seq_value.shape[0], self.num_kv_heads,
                        self.num_queries_per_kv, seq_value.shape[-1])

                # TODO(woosuk): Too many view operations. Let's try to reduce them
                # in the future for code readability.
                if self.alibi_slopes is None:
                    seq_query = seq_query.unsqueeze(0)
                    seq_key = seq_key.unsqueeze(0)
                    seq_value = seq_value.unsqueeze(0)
                else:
                    raise RuntimeError("ALiBi support not implemented")
                    query = query.unflatten(0, (batch_size, seq_len))
                    key = key.unflatten(0, (batch_size, seq_len))
                    value = value.unflatten(0, (batch_size, seq_len))

                out = xops.memory_efficient_attention_forward(
                    seq_query,
                    seq_key,
                    seq_value,
                    attn_bias=input_metadata.attn_bias[seq_id],
                    p=0.0,
                    scale=self.scale,
                    op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0] if
                    (is_hip()) else None,
                )
                seq_output.copy_(out.view_as(seq_output))

            # Prepare the k_cache_buffer and the v_cache_buffer for the next iteration
            self.update_kv_cache_buffer(
                key_cache,
                value_cache,
                input_metadata,
                kv_buffer,
            )
        if num_generation_tokens > 0:
            output = _paged_attention(
                output[num_current_prompt_tokens:num_valid_tokens],
                query[num_current_prompt_tokens:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def get_prefill_kv_tensors(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
        kv_buffer: KVBuffer,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        kv_tensors = []

        offset = 0
        for seq_id, current_prompt_chunk_len, processed_prompt_len, total_prompt_len in zip(
                input_metadata.prompt_seq_ids,
                input_metadata.current_prompt_chunk_lens,
                input_metadata.processed_prompt_lens,
                input_metadata.total_prompt_lens,
        ):
            seq_k = key[offset:offset + current_prompt_chunk_len]
            seq_v = value[offset:offset + current_prompt_chunk_len]
            offset += current_prompt_chunk_len

            if current_prompt_chunk_len == total_prompt_len:
                kv_buffer.add_request_with_kv_tensors(seq_id, seq_k, seq_v)
            else:
                if processed_prompt_len == 0 or input_metadata.is_profiling_iteration:
                    kv_buffer.add_request(seq_id, total_prompt_len)

                # Skip check during profiling phase
                assert input_metadata.is_profiling_iteration or processed_prompt_len == kv_buffer.get_offset(
                    seq_id
                ), (f"processed_prompt_len={processed_prompt_len}"
                    f"kv_buffer.get_offset(seq_id)={kv_buffer.get_offset(seq_id)}"
                    )
                kv_buffer.extend(seq_id, seq_k, seq_v)

            kv_tensors.append(kv_buffer.get_kv_tensors(seq_id))

        return kv_tensors

    def update_kv_cache_buffer(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        kv_buffer: KVBuffer,
    ) -> None:
        offset = 0
        for seq_id, current_prompt_chunk_len, processed_prompt_len, total_prompt_len in zip(
                input_metadata.prompt_seq_ids,
                input_metadata.current_prompt_chunk_lens,
                input_metadata.processed_prompt_lens,
                input_metadata.total_prompt_lens,
        ):
            if processed_prompt_len + current_prompt_chunk_len != total_prompt_len:
                continue

            if input_metadata.is_profiling_iteration:
                kv_buffer.free_request(seq_id)
                continue

            key, value = kv_buffer.get_kv_tensors(seq_id)
            if self.sliding_window is not None:
                start_index = max(0, total_prompt_len - self.sliding_window)
                key = key[start_index:]
                value = value[start_index:]

            slot_mapping = input_metadata.prefix_plus_current_prompt_tokens_slot_mapping[
                offset:offset + len(key)]
            offset += len(key)
            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
            )

            # TODO: need to handle restarted and aborted sequences
            # in the current state, we can have memory leaks
            kv_buffer.free_request(seq_id)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> LowerTriangularMaskWithTensorBias:
    bias = torch.arange(seq_len, dtype=dtype)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    attn_bias = LowerTriangularMaskWithTensorBias(bias)
    return attn_bias


def _paged_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    input_metadata: InputMetadata,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
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
