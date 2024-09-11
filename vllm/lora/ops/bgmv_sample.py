
import torch
import triton
import triton.language as tl

from .utils import get_lora_op_configs



@triton.jit
def _bgmv_sample_kernel(
    hidden_state_ptr,
    lm_heads_all_ptr,
    logits_ptr,
    sampling_indices_tensor_ptr,
    HIDDEN_DIM: tl.constexpr,
    VOCAB_SIZE: tl.constexpr
):

    curr_logit_idx=tl.program_id(axis=0)

    cur_token = tl.program_id(axis=1)
    

    lora_index = tl.load(sampling_indices_tensor_ptr + cur_token)
    if lora_index == -1: #TODO multiply to base_layer.weights
        return

    hidden_state=tl.load(hidden_state_ptr+HIDDEN_DIM*cur_token+tl.arange(0,HIDDEN_DIM))
    hidden_state=hidden_state.expand_dims(0)

    offsets=tl.arange(0, HIDDEN_DIM)[None,:]

    weights=tl.load(lm_heads_all_ptr+lora_index*(VOCAB_SIZE*HIDDEN_DIM)+curr_logit_idx*HIDDEN_DIM+offsets)


    logit=tl.sum(weights*hidden_state)

    tl.store(logits_ptr+cur_token*VOCAB_SIZE+curr_logit_idx, logit)
    

@torch.inference_mode()
def _bgmv_sample(
    hidden_state: torch.Tensor,
    lm_heads_all: torch.Tensor,
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

    vocab_size=lm_heads_all.shape[-2]
    logits = torch.zeros((hidden_state.size(0), vocab_size),
                                 dtype=hidden_state.dtype,
                                 device=hidden_state.device)

    print(hidden_state.shape, lm_heads_all.shape, logits.shape, sampling_indices_tensor.shape)

    num_tokens = sampling_indices_tensor.shape[0]
    hidden_dim=hidden_state.shape[-1]


    grid = lambda meta: (vocab_size,num_tokens,)

    _bgmv_sample_kernel[grid](
        hidden_state,
        lm_heads_all,
        logits,
        sampling_indices_tensor,
        HIDDEN_DIM=hidden_dim,
        VOCAB_SIZE=vocab_size,
        #**config,
    )
    return logits



try:
    bgmv_sample = torch.library.custom_op("lora::bgmv_sample",
                                          _bgmv_sample,
                                          mutates_args=[])
except AttributeError:
    bgmv_sample = _bgmv_sample
