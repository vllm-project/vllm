import torch
import triton
import triton.language as tl

from vllm.utils import direct_register_custom_op

from .utils import get_lora_op_configs


def next_power_of_two(n: int) -> int:
    """
    Returns the smallest power-of-two integer >= n.
    For example:
    - n=1 -> 1
    - n=3 -> 4
    - n=5 -> 8
    - n=8 -> 8
    - n=9 -> 16
    """
    return 1 << ((n - 1).bit_length())


@triton.jit
def _bgmv_sample_kernel_arbitrary(
        hidden_state_ptr,  # float32[ ..., HIDDEN_DIM ]
        lm_heads_all_ptr,  # float32[ VOCAB_SIZE * HIDDEN_DIM_P2 * num_lora? ]
        lm_head_base_ptr,  # float32[ VOCAB_SIZE * HIDDEN_DIM_P2 ]
        logits_ptr,  # float32[ ..., VOCAB_SIZE ]
        sampling_indices_tensor_ptr,  # int32[ ... ]
        HIDDEN_DIM: tl.constexpr,
        HIDDEN_DIM_P2: tl.constexpr,  # internal power-of-two dimension
        VOCAB_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr):
    """
    A Triton kernel that:
    - Rounds HIDDEN_DIM up to HIDDEN_DIM_P2 (a power of two)
    - Applies a mask for the out-of-bounds region (if any)
    """

    # Program IDs for parallelization
    cur_token = tl.program_id(axis=0)
    logits_start_idx = tl.program_id(axis=1) * BLOCK_N

    # Read which LoRA index to use for this token
    lora_index = tl.load(sampling_indices_tensor_ptr + cur_token)

    # Create index range [0..HIDDEN_DIM_P2)
    offsets_embed = tl.arange(0, HIDDEN_DIM_P2)
    # Boolean mask to disable out-of-range indices
    mask = offsets_embed < HIDDEN_DIM

    # Load hidden_state (size = HIDDEN_DIM_P2),
    # using a mask to skip invalid positions
    hidden_ptr = hidden_state_ptr + cur_token * HIDDEN_DIM + offsets_embed
    hidden_state = tl.load(hidden_ptr, mask=mask, other=0.0)
    # Expand dims so we can do [Block_N, HIDDEN_DIM_P2] * [1, HIDDEN_DIM_P2]
    hidden_state = hidden_state.expand_dims(0)

    # Offsets in the logits dimension
    offsets_logits = logits_start_idx + tl.arange(0, BLOCK_N)

    # Compute memory offsets for loading weights
    # We scale by HIDDEN_DIM_P2 because that's the *internally used* dimension
    offset_base_layer = offsets_embed[None, :] + (offsets_logits[:, None] *
                                                  HIDDEN_DIM)
    offset_lora = lora_index * (VOCAB_SIZE * HIDDEN_DIM) + offset_base_layer

    # Depending on lora_index, load from lm_head_base_ptr or lm_heads_all_ptr
    if lora_index == -1:
        weights = tl.load(lm_head_base_ptr + offset_base_layer,
                          mask=mask[None, :],
                          other=0.0)
    else:
        weights = tl.load(lm_heads_all_ptr + offset_lora,
                          mask=mask[None, :],
                          other=0.0)

    # Compute logits by summation over the masked dimension
    logits = tl.sum(weights * hidden_state, axis=1)

    # Store the result
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

    assert num_tokens / config['BLOCK_N'] < 65535, (
        "increase BLOCK_N,"
        "triton can not handle grid size larger 2**31-1")

    HIDDEN_DIM_P2 = next_power_of_two(hidden_dim)

    # For example, if you want to 2D-launch the kernel:
    # - dimension 0 = number of tokens
    # - dimension 1 = how many BLOCK_N segments needed for the vocab
    _bgmv_sample_kernel_arbitrary[grid](hidden_state,
                                        lm_heads_all,
                                        lm_head_base,
                                        logits,
                                        sampling_indices_tensor,
                                        HIDDEN_DIM=hidden_dim,
                                        HIDDEN_DIM_P2=HIDDEN_DIM_P2,
                                        VOCAB_SIZE=vocab_size,
                                        **config)
    return logits


try:
    direct_register_custom_op(op_name="bgmv_sample",
                              op_func=_bgmv_sample,
                              mutates_args=[],
                              fake_impl=None)
    bgmv_sample = torch.ops.vllm.bgmv_sample

except AttributeError:
    bgmv_sample = _bgmv_sample
