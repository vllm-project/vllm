# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

from vllm.utils import direct_register_custom_op

from .utils import get_lora_op_configs


def next_power_of_2(n: int) -> int:
    """
    Returns the smallest power of two >= n
    """
    return 1 << ((n - 1).bit_length())


@triton.jit
def _bgmv_embed_kernel(
    tokens,  # pointer to tokens array
    embed_tokens_all,  # pointer to embedded tokens - all
    embed_tokens_base,  # pointer to embedded tokens - base
    token_indices,  # pointer to token indices
    embeddings,  # pointer to output embeddings
    num_tokens,  # number of tokens
    REAL_HIDDEN_DIM: tl.constexpr,  # actual hidden dimension
    HIDDEN_DIM_P2: tl.constexpr,  # hidden dimension padded up to power of 2
    VOCAB_SIZE: tl.constexpr,  # vocabulary size
    BLOCK_N: tl.constexpr  # block size (number of tokens per block)
):
    # Calculate the block offset for this program instance
    start_idx = tl.program_id(0) * BLOCK_N

    # Offsets for token IDs in this block
    offs_n = start_idx + tl.arange(0, BLOCK_N)
    mask_n = offs_n < num_tokens  # valid token mask

    # Read token indices and LoRA indices
    cur_tokens = tl.load(tokens + offs_n, mask=mask_n, other=0)
    lora_index = tl.load(token_indices + offs_n, mask=mask_n, other=-1)

    #
    # For the hidden dimension, we tile with HIDDEN_DIM_P2 threads
    # but we only load/store up to REAL_HIDDEN_DIM columns.
    #
    hidden_range = tl.arange(0, HIDDEN_DIM_P2)  # [0..HIDDEN_DIM_P2)
    mask_h = hidden_range < REAL_HIDDEN_DIM  # extra mask for columns
    mask = mask_n[:,
                  None] & mask_h[None, :]  # combined mask (tokens & columns)

    # ------------------------------------------------
    # 1) Load embeddings from embed_tokens_base
    # ------------------------------------------------
    # embed_tokens_base is laid out as [vocab_size, REAL_HIDDEN_DIM]
    # offset for row = cur_tokens[i], col = hidden_range
    offsets_base = cur_tokens[:,
                              None] * REAL_HIDDEN_DIM + hidden_range[None, :]
    # masked load (don’t load if token is invalid or col >= REAL_HIDDEN_DIM)
    embeddings_base = tl.load(embed_tokens_base + offsets_base,
                              mask=mask,
                              other=0.0)

    # Start our final block from the base embeddings
    embeddings_block = embeddings_base

    # ------------------------------------------------
    # 2) For tokens with a valid LoRA index, load from embed_tokens_all
    # ------------------------------------------------
    # mask for tokens that actually have a LoRA index
    mask_all = (lora_index != -1) & mask_n

    # embed_tokens_all is shaped [num_loras, vocab_size, REAL_HIDDEN_DIM]
    # We flatten it as if:
    #   “base offset” = lora_index * (vocab_size * REAL_HIDDEN_DIM)
    #   plus offset for the token row
    #   plus hidden_range
    #
    base_offsets_all = tl.where(mask_all,
                                lora_index * VOCAB_SIZE * REAL_HIDDEN_DIM, 0)
    # tile offsets expression
    offsets_all = base_offsets_all[:, None] + offsets_base
    # masked load
    embeddings_all = tl.load(embed_tokens_all + offsets_all,
                             mask=mask,
                             other=0.0)
    # Overwrite wherever lora_index != -1
    #   Note: mask_all[:, None] is only for the “token” axis;
    # we still need mask_h for columns
    combined_mask_all = mask_all[:, None] & mask_h[None, :]
    embeddings_block = tl.where(combined_mask_all, embeddings_all,
                                embeddings_block)

    # ------------------------------------------------
    # 3) Store result to output
    # ------------------------------------------------
    # embeddings is shaped [num_tokens, REAL_HIDDEN_DIM]
    # offset for row = offs_n[i], col = hidden_range
    output_offsets = offs_n[:, None] * REAL_HIDDEN_DIM + hidden_range[None, :]
    # same combined mask needed for storing
    tl.store(embeddings + output_offsets, embeddings_block, mask=mask)


@torch.inference_mode()
def _bgmv_embed(
    tokens: torch.Tensor,
    embed_tokens_all: torch.Tensor,
    embed_tokens_base: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        tokens:           [num_tokens] int64
        embed_tokens_all: [num_loras, vocab_size, hidden_dim]
        embed_tokens_base:[vocab_size, hidden_dim]
        token_indices:    [num_tokens]  (LoRA indices, -1 means no LoRA)
    
    Returns:
        embeddings:       [num_tokens, hidden_dim]
    """
    assert embed_tokens_all.dtype == embed_tokens_base.dtype
    assert tokens.dtype == torch.int64
    assert token_indices.dtype == torch.int64

    assert embed_tokens_base.is_contiguous()
    assert embed_tokens_all.is_contiguous()

    vocab_size, real_hidden_dim = embed_tokens_all.shape[-2:]
    num_tokens = tokens.shape[0]
    embeddings = torch.zeros((num_tokens, real_hidden_dim),
                             dtype=embed_tokens_all.dtype,
                             device=embed_tokens_all.device)

    # Triton requires a power-of-2 block-size dimension for performance
    hidden_dim_p2 = next_power_of_2(real_hidden_dim)

    # Adjust your config call so that Triton blocks use hidden_dim_p2
    config = get_lora_op_configs("embed", num_tokens, hidden_dim_p2)
    # config typically returns a dict, e.g. {"BLOCK_N": some_value}

    # Kernel launch
    grid = lambda meta: (triton.cdiv(num_tokens, meta['BLOCK_N']), )
    _bgmv_embed_kernel[grid](
        tokens,
        embed_tokens_all,
        embed_tokens_base,
        token_indices,
        embeddings,
        num_tokens,
        REAL_HIDDEN_DIM=real_hidden_dim,
        HIDDEN_DIM_P2=hidden_dim_p2,
        VOCAB_SIZE=vocab_size,
        **config,
    )
    return embeddings


try:
    direct_register_custom_op(op_name="bgmv_embed",
                              op_func=_bgmv_embed,
                              mutates_args=[],
                              fake_impl=None)
    bgmv_embed = torch.ops.vllm.bgmv_embed
except AttributeError:
    bgmv_embed = _bgmv_embed
