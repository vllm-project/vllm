import torch
import triton
import triton.language as tl

from .utils import get_lora_op_configs

@triton.jit
def _bgmv_embed_kernel(
        tokens,
        embed_tokens_all,
        embed_tokens_base,
        token_indices,
        embeddings,
        num_tokens,
        HIDDEN_DIM:tl.constexpr,
        VOCAB_SIZE:tl.constexpr,
        BLOCK_N:tl.constexpr
    ):

    
    cur_token_idx = tl.program_id(axis=0)

    if cur_token_idx>=num_tokens:
        return

    lora_index = tl.load(token_indices + cur_token_idx)

    cur_token = tl.load(tokens + cur_token_idx)

    offsets_embed = cur_token*HIDDEN_DIM + tl.arange(0, HIDDEN_DIM)

    if lora_index == -1:
        embedding = tl.load(embed_tokens_base + offsets_embed)
    else:
        embedding = tl.load(embed_tokens_all + lora_index*HIDDEN_DIM*VOCAB_SIZE + offsets_embed)

    tl.store(embeddings + cur_token_idx * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM), embedding)


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
        embed_tokens_all - [num_loras, vocab_size, hidden_dim] - modules_to_save embeddings
        embed_tokens_base - [vocab_size, hidden_dim] - base layer embeddings will be applied to tokens with index=-1
        token_indices - [num_tokens] LoRA indices from 0 to num_loras, -1 means no LoRA, embed_tokens_base will be used

        returns:
        embeddings: [num_tokens, hidden_dim]
    """
    
    #tokens=tokens.long()
    #token_indices=token_indices.long()

    assert embed_tokens_all.dtype == embed_tokens_base.dtype
    assert tokens.dtype == torch.int64, f"tokens must be of dtype torch.int64 but got {tokens.dtype}"
    assert token_indices.dtype == torch.int64, f"token_indices must be of dtype torch.int64 but got {token_indices.dtype}"

    assert embed_tokens_base.is_contiguous()
    assert embed_tokens_all.is_contiguous()

    vocab_size, hidden_dim = embed_tokens_all.shape[-2:]
    num_tokens=tokens.shape[0]
    embeddings = torch.zeros((num_tokens, hidden_dim),
                         dtype=embed_tokens_all.dtype,
                         device=embed_tokens_all.device)


    grid = lambda meta: (triton.cdiv(num_tokens, meta['BLOCK_N']),)

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
