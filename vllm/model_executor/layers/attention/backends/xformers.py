"""Attention layer with xFormers and PagedAttention."""
import importlib
from typing import List, Optional

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention.ops.paged_attn import (
    PagedAttentionImpl)
from vllm.utils import is_hip, _get_aligned_size


class XFormersBackend:

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        suppored_head_sizes = PagedAttentionImpl.get_supported_head_sizes()
        if head_size not in suppored_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {suppored_head_sizes}.")

        self.use_ref_attention = _check_use_ref_attention()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Forward pass with xFormers and PagedAttention.

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
        num_tokens, hidden_size = query.shape
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            PagedAttentionImpl.reshape_and_cache(key, value, key_cache,
                                                 value_cache, input_metadata)

        num_prompt_tokens = input_metadata.num_prompt_tokens
        num_generation_tokens = input_metadata.num_generation_tokens

        # print(f"SANG-TODO original query: {query.size()}")
        decode_query = query[num_prompt_tokens:]
        query = query[:num_prompt_tokens]
        key = key[:num_prompt_tokens]
        value = value[:num_prompt_tokens]
        # print(f"SANG-TODO {num_prompt_tokens=} {num_generation_tokens=}")
        # print(f"SANG-TODO {query.size()=} {decode_query.size()=}")

        assert query.shape[0] == num_prompt_tokens
        assert decode_query.shape[0] == num_generation_tokens

        output = torch.empty_like(query)

        if num_prompt_tokens > 0:
            prefill_input_metadata = input_metadata.prefill_input_metadata()
            # Prompt run.
            # key_cache and value_cache is None when it is a profiling run.
            # block tables are empty if the prompt has never been computed.
            if (key_cache is None or value_cache is None
                    or prefill_input_metadata.block_tables.numel() == 0):
                # normal attention
                if self.num_kv_heads != self.num_heads:
                    # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
                    # project the key and value tensors to the desired number of
                    # heads.
                    # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                    query = query.view(query.shape[0], self.num_kv_heads,
                                       self.num_queries_per_kv,
                                       query.shape[-1])
                    key = key[:, :,
                              None, :].expand(key.shape[0], self.num_kv_heads,
                                              self.num_queries_per_kv,
                                              key.shape[-1])
                    value = value[:, :,
                                  None, :].expand(value.shape[0],
                                                  self.num_kv_heads,
                                                  self.num_queries_per_kv,
                                                  value.shape[-1])

                if self.use_ref_attention:
                    output[:num_prompt_tokens] = _ref_masked_attention(
                        query,
                        key,
                        value,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_size,
                        self.scale,
                    )
                    # Using view got RuntimeError: view size is not compatible
                    # with input tensor's size and stride (at least one
                    # dimension spans across two contiguous subspaces).
                    # Use reshape instead.
                    return output.reshape(num_tokens, hidden_size)

                output[:num_prompt_tokens] = self._multi_query_kv_attention(
                    query, key, value, prefill_input_metadata)
            else:
                # prefix-enabled attention
                # print("SANG-TODO prefix")
                # print(prefill_input_metadata.prompt_lens)
                # print(prefill_input_metadata.context_lens)
                # print(prefill_input_metadata.max_context_len)
                # print(prefill_input_metadata.start_loc)
                # print(prefill_input_metadata.max_seq_len)
                output[:num_prompt_tokens] = PagedAttentionImpl.forward_prefix(
                    query,
                    key,
                    value,
                    key_cache,
                    value_cache,
                    prefill_input_metadata,
                    self.alibi_slopes,
                )
                # assert output.shape[0] == num_prompt_tokens
                # print("SANG-TODO prefix output: ")
                # for n in range(num_prompt_tokens):
                #     print(f"{n=}")
                #     print(output[n][0])

        if num_generation_tokens > 0:
            decoding_input_metadata = input_metadata.decode_input_metadata()
            # Decoding run.
            output[num_prompt_tokens:] = PagedAttentionImpl.forward_decode(
                decode_query,
                key_cache,
                value_cache,
                decoding_input_metadata,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def _multi_query_kv_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Attention for 1D query of multiple prompts. Multiple prompt
        tokens are flattened in to `query` input.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        # Set attention bias if not provided. This typically happens at
        # the very attention layer of every iteration.
        # FIXME(woosuk): This is a hack.
        if input_metadata.attn_bias is None:
            if self.alibi_slopes is None:
                attn_bias = BlockDiagonalCausalMask.from_seqlens(
                    input_metadata.prompt_lens)
                if self.sliding_window is not None:
                    attn_bias = attn_bias.make_local_attention(
                        self.sliding_window)
                input_metadata.attn_bias = [attn_bias]
            else:
                input_metadata.attn_bias = _make_alibi_bias(
                    self.alibi_slopes, self.num_kv_heads, query.dtype,
                    input_metadata)

        op = xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0] if (
            is_hip()) else None
        # No alibi slopes.
        # TODO(woosuk): Too many view operations. Let's try to reduce
        # them in the future for code readability.
        if self.alibi_slopes is None:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=input_metadata.attn_bias[0],
                p=0.0,
                scale=self.scale,
                op=op)

            return out.view_as(query)

        # Attention with alibi slopes.
        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        output = torch.empty_like(query)
        start = 0
        for i, prompt_len in enumerate(input_metadata.prompt_lens):
            end = start + prompt_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=input_metadata.attn_bias[i],
                p=0.0,
                scale=self.scale,
                op=op)
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.squeeze(0))
            start += prompt_len
        return output


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    input_metadata: InputMetadata,
) -> LowerTriangularMaskWithTensorBias:
    attn_biases = []
    for prompt_len in input_metadata.prompt_lens:
        bias = torch.arange(prompt_len, dtype=dtype)
        # NOTE(zhuohan): HF uses
        #     `bias = bias[None, :].repeat(prompt_len, 1)`
        # here. We find that both biases give the same results, but
        # the bias below more accurately follows the original ALiBi
        # paper.
        # Calculate a matrix where each element represents ith element- jth
        # element.
        bias = bias[None, :] - bias[:, None]

        padded_len = _get_aligned_size(prompt_len, 8)
        num_heads = alibi_slopes.shape[0]
        bias = torch.empty(
            1,  # batch size
            num_heads,
            prompt_len,
            padded_len,
            device=alibi_slopes.device,
            dtype=dtype,
        )[:, :, :, :prompt_len].copy_(bias)
        bias.mul_(alibi_slopes[:, None, None])
        if num_heads != num_kv_heads:
            bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
        attn_biases.append(LowerTriangularMaskWithTensorBias(bias))

    return attn_biases


def _check_use_ref_attention() -> bool:
    if not is_hip():
        return False
    # For ROCm, check whether flash attention is installed or not.
    # if not, use_ref_attention needs to be True
    return importlib.util.find_spec("flash_attn") is None


def _ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
) -> torch.Tensor:
    query = query.view(-1, num_heads, head_size)
    key = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size)

    seq_len, _, _ = query.shape
    attn_mask = torch.triu(torch.ones(seq_len,
                                      seq_len,
                                      dtype=query.dtype,
                                      device=query.device),
                           diagonal=1)
    attn_mask = attn_mask * torch.finfo(query.dtype).min

    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out
