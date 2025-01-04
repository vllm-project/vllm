import torch
import triton
import triton.language as tl

from .utils import get_lora_op_configs


@triton.jit
def _bgmv_sample_kernel(hidden_state_ptr, lm_heads_all_ptr, lm_head_base_ptr,
                        logits_ptr, sampling_indices_tensor_ptr,
                        HIDDEN_DIM: tl.constexpr, VOCAB_SIZE: tl.constexpr,
                        BLOCK_N: tl.constexpr):

    cur_token = tl.program_id(axis=0)

    logits_start_idx = tl.program_id(axis=1) * BLOCK_N

    lora_index = tl.load(sampling_indices_tensor_ptr + cur_token)

    hidden_state = tl.load(hidden_state_ptr + HIDDEN_DIM * cur_token +
                           tl.arange(0, HIDDEN_DIM))
    hidden_state = hidden_state.expand_dims(0)

    offsets_embed = tl.arange(0, HIDDEN_DIM)
    offsets_logits = logits_start_idx + tl.arange(0, BLOCK_N)

    offset_base_layer = offsets_embed[
        None, :] + offsets_logits[:, None] * HIDDEN_DIM
    offset_lora = lora_index * (VOCAB_SIZE * HIDDEN_DIM) + offset_base_layer

    if lora_index == -1:
        weights = tl.load(lm_head_base_ptr + offset_base_layer)
    else:
        weights = tl.load(lm_heads_all_ptr + offset_lora)

    logits = tl.sum(weights * hidden_state, axis=1)

    tl.store(logits_ptr + cur_token * VOCAB_SIZE + offsets_logits, logits)


@torch.inference_mode()
def _bgmv_sample(
    hidden_state: torch.Tensor,
    lm_heads_all: torch.Tensor,
    lm_head_base: torch.Tensor,
    sampling_indices_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        hidden_state - [num_tokens, hidden_dim]
        lm_heads_all - [num_loras, vocab_size, hidden_dim]
        sampling_indices_tensor - [num_tokens] - indexes from 0 to num_loras-1
    """
    assert hidden_state.dtype == lm_heads_all.dtype
    assert hidden_state.size(-1) == lm_heads_all.size(-1)
    assert hidden_state.is_contiguous()
    assert lm_heads_all.is_contiguous()

    vocab_size = lm_heads_all.shape[-2]
    logits = torch.zeros((hidden_state.size(0), vocab_size),
                         dtype=hidden_state.dtype,
                         device=hidden_state.device)

    num_tokens = sampling_indices_tensor.shape[0]
    hidden_dim = hidden_state.shape[-1]

    grid = lambda meta: (num_tokens, triton.cdiv(vocab_size, meta['BLOCK_N']))

    config = get_lora_op_configs("sample", num_tokens, hidden_dim)

    _bgmv_sample_kernel[grid](
        hidden_state,
        lm_heads_all,
        lm_head_base,
        logits,
        sampling_indices_tensor,
        HIDDEN_DIM=hidden_dim,
        VOCAB_SIZE=vocab_size,
        **config,
    )
    return logits


try:
    bgmv_sample = torch.library.custom_op("lora::bgmv_sample",
                                          _bgmv_sample,
                                          mutates_args=[])
except AttributeError:
    bgmv_sample = _bgmv_sample
