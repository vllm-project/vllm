import torch
import triton
import triton.language as tl

from .utils import get_lora_op_configs


@triton.jit
def _bgmv_embed_kernel(
    tokens,  # pointer to tokens array
    embed_tokens_all,  # pointer to embedded tokens - all
    embed_tokens_base,  # pointer to embedded tokens - base
    token_indices,  # pointer to token indices
    embeddings,  # pointer to output embeddings
    num_tokens,  # number of tokens
    HIDDEN_DIM: tl.constexpr,  # hidden dimension
    VOCAB_SIZE: tl.constexpr,  # vocabulary size
    BLOCK_N: tl.constexpr  # block size (number of tokens per block)
):
    # Calculate the starting index for this block
    start_idx = tl.program_id(0) * BLOCK_N
    # Create an array of offsets for the tokens in this block
    offs_n = start_idx + tl.arange(0, BLOCK_N)
    # Create a mask to handle cases where we exceed num_tokens
    mask = offs_n < num_tokens

    # Load lora_index and tokens for the current block (masked)
    lora_index = tl.load(token_indices + offs_n, mask=mask, other=-1)
    cur_tokens = tl.load(tokens + offs_n, mask=mask, other=0)

    # Compute offsets into the embedding matrices
    hidden_range = tl.arange(0, HIDDEN_DIM)
    offsets_embed = cur_tokens[:, None] * HIDDEN_DIM + hidden_range[
        None, :]  # Shape: (BLOCK_N, HIDDEN_DIM)

    # Load embeddings from embed_tokens_base
    embeddings_base = tl.load(embed_tokens_base + offsets_embed,
                              mask=mask[:, None],
                              other=0.0)

    # Initialize embeddings_block with embeddings_base
    embeddings_block = embeddings_base

    # Create a mask for tokens that require loading from embed_tokens_all
    mask_all = (lora_index != -1) & mask

    # For tokens with lora_index != -1, load from embed_tokens_all

    # Calculate base offsets for tokens with lora_index != -1
    # Use tl.where to avoid invalid memory accesses
    base_offsets_all = tl.where(mask_all, lora_index * HIDDEN_DIM * VOCAB_SIZE,
                                0)
    # Calculate full offsets into embed_tokens_all
    full_offsets_all = base_offsets_all[:, None] + offsets_embed
    # Load embeddings from embed_tokens_all
    embeddings_all = tl.load(embed_tokens_all + full_offsets_all,
                             mask=mask_all[:, None],
                             other=0.0)
    # Overwrite embeddings_block where lora_index != -1
    embeddings_block = tl.where(mask_all[:, None], embeddings_all,
                                embeddings_block)

    # Calculate the offsets where embeddings should be stored
    output_offsets = offs_n[:, None] * HIDDEN_DIM + hidden_range[None, :]

    # Store embeddings_block to the output embeddings array
    tl.store(embeddings + output_offsets, embeddings_block, mask=mask[:, None])


@torch.inference_mode()
def _bgmv_embed(
    tokens: torch.Tensor,
    embed_tokens_all: torch.Tensor,
    embed_tokens_base: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        tokens - [num_tokens] - input tokens
        embed_tokens_all - [num_loras, vocab_size, hidden_dim] 
            modules_to_save embeddings
        embed_tokens_base - [vocab_size, hidden_dim] - base layer 
            embeddings will be applied to tokens with index=-1
        token_indices - [num_tokens] LoRA indices from 0 to num_loras,
             -1 means no LoRA, embed_tokens_base will be used

        returns:
        embeddings: [num_tokens, hidden_dim]
    """
    assert embed_tokens_all.dtype == embed_tokens_base.dtype
    assert tokens.dtype == torch.int64
    assert token_indices.dtype == torch.int64

    assert embed_tokens_base.is_contiguous()
    assert embed_tokens_all.is_contiguous()

    vocab_size, hidden_dim = embed_tokens_all.shape[-2:]
    num_tokens = tokens.shape[0]
    embeddings = torch.zeros((num_tokens, hidden_dim),
                             dtype=embed_tokens_all.dtype,
                             device=embed_tokens_all.device)

    grid = lambda meta: (triton.cdiv(num_tokens, meta['BLOCK_N']), )

    config = get_lora_op_configs("embed", num_tokens, hidden_dim)

    _bgmv_embed_kernel[grid](
        tokens,
        embed_tokens_all,
        embed_tokens_base,
        token_indices,
        embeddings,
        num_tokens,
        HIDDEN_DIM=hidden_dim,
        VOCAB_SIZE=vocab_size,
        **config,
    )
    return embeddings


try:
    bgmv_embed = torch.library.custom_op("lora::bgmv_embed",
                                         _bgmv_embed,
                                         mutates_args=[])
except AttributeError:
    bgmv_embed = _bgmv_embed
