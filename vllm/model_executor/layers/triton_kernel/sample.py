import torch
import triton
import triton.language as tl
import math
from typing import Tuple, Optional

_EPS = 1e-6

# This is a hardcoded limit in Triton (max block size).
MAX_TRITON_N_COLS = 131072


def get_num_triton_sampler_splits(n_cols: int) -> int:
    """Get the number of splits to use for Triton sampling.

    Triton has a limit on the number of columns it can handle, so we need to
    split the tensor and call the kernel multiple times if it's too large.
    """
    return math.ceil(n_cols / MAX_TRITON_N_COLS)


def _multi_split_sample(
    probs: torch.Tensor,
    seeds: torch.Tensor,
    n_splits: int,
    sampled_tokens_size: Tuple[int, int],
    sampled_logprobs_size: Tuple[int, int],
    sample_indices: torch.Tensor,
    *,
    logprobs: Optional[torch.Tensor] = None,
    modify_greedy_probs: bool = False,
    save_logprobs: bool = False,
):
    """Sample tokens where vocab size is split into multiple parts
    (too large for Triton otherwise)."""
    assert seeds.ndim == 2 and seeds.shape[0] == n_splits
    split_probs = probs.tensor_split(n_splits, 1)
    split_logprobs = logprobs.tensor_split(n_splits, 1)
    sampled_tokens_tmp = [
        torch.empty(sampled_tokens_size, dtype=torch.long, device=probs.device)
        for _ in range(n_splits)
    ]
    sampled_logprobs_tmp = [
        torch.empty(sampled_logprobs_size,
                    dtype=probs.dtype,
                    device=probs.device) for _ in range(n_splits)
    ]
    # We are purposefuly using sampled_tokens_size as we need to always
    # save modified probs in this case.
    sampled_modified_probs_tmp = [
        torch.empty(sampled_tokens_size,
                    dtype=probs.dtype,
                    device=probs.device) for _ in range(n_splits)
    ]
    for i in range(n_splits):
        # TODO(yard1): See if we can remove the contiguous() calls.
        # Will need kernel support.
        _sample(
            split_probs[i],
            split_logprobs[i],
            sample_indices,
            sampled_tokens_tmp[i],
            sampled_logprobs_tmp[i],
            sampled_modified_probs_tmp[i],
            seeds[i],
            modify_greedy_probs=modify_greedy_probs,
            # Don't save logprobs in kernel, we need to gather them
            # below
            save_logprobs=False,
            save_modified_probs=True,
        )
        if i > 0:
            # Add offset to sampled tokens
            sampled_tokens_tmp[i].add_(i * split_probs[i - 1].shape[1])
    sampled_tokens = torch.stack(sampled_tokens_tmp)
    sampled_modified_probs = torch.stack(sampled_modified_probs_tmp)
    # Reduce the results from the splits.
    sampled_modified_probs, indices = torch.max(sampled_modified_probs,
                                                dim=0,
                                                keepdim=True)
    sampled_tokens = sampled_tokens.gather(0, indices).squeeze(0)
    if save_logprobs:
        sampled_logprobs = torch.stack(sampled_logprobs_tmp)
        sampled_logprobs = sampled_logprobs.gather(0, indices).squeeze(0)
    else:
        sampled_logprobs = None
    sampled_modified_probs = sampled_modified_probs.squeeze(0)
    if modify_greedy_probs:
        # We need to modify the greedy probs for the sampled tokens.
        # We can't do this in the kernel as we need to know the
        # sampled tokens.
        probs.fill_(0.0)
        probs.scatter_(1, sampled_tokens, 1.0)
    return (sampled_tokens, sampled_logprobs, sampled_modified_probs)


def sample(
    probs: torch.Tensor,
    seeds: torch.Tensor,
    *,
    max_best_of: int = 1,
    sample_indices: Optional[torch.Tensor] = None,
    logprobs: Optional[torch.Tensor] = None,
    modify_greedy_probs: bool = False,
    save_logprobs: bool = False,
    _save_modified_probs: bool = False,  # pylint: disable=invalid-name
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Sample tokens from probs. with per-sequence seeds.

    Can sample from a subset of sequences through sample_indices.

    Args:
        probs: Probabilities to sample from.
            shape = [batch_size, vocab_size]
        seeds: Per-sequence seed values.
            shape = [n, math.ceil(vocab_size / MAX_TRITON_N_COLS)]
        max_best_of: Number of samples to generate per sequence.
            Sequence seed will be incremented by 1 each time.
        sample_indices: Indices of sequences to sample from.
            If not provided, will sample from all sequences.
            shape = [n]
        logprobs: Log-probabilities of the sampled tokens.
            Only used for saving the logprobs if save_logprobs is True.
            shape = [batch_size, vocab_size]
        modify_greedy_probs: Whether to modify the greedy probabilities
            for speculative sampling (sampled token = 1.0,
            everything else = 0.0).
        save_logprobs: Whether to save the log-probabilities of the
            sampled tokens to a tensor.
        _save_modified_probs: Whether to save the modified probabilities
            (including gumbel noise) of the sampled tokens to a tensor.
            DOES NOT include the modification done by modify_greedy_probs
            (because we want to use the unmodified probs to pick the best
            split in case of multi-split sampling).
            This is exposed only for testing.

    Returns:
        sampled_tokens: shape = [n, max_best_of]
        sampled_logprobs: shape = [n, max_best_of] if save_logprobs else None
        sampled_modified_probs: shape = [n, max_best_of]
            if save_modified_probs else None
    """
    if sample_indices is None:
        sample_indices = torch.arange(0, probs.shape[0], device=probs.device)

    sampled_tokens_size = (sample_indices.size(0), max_best_of)
    if save_logprobs:
        if logprobs is None:
            raise ValueError(
                "logprobs tensor must be provided if save_logprobs is True")
        sampled_logprobs_size = sampled_tokens_size
    else:
        # Empty tensors to invoke the kernel
        sampled_logprobs_size = (0, 0)
        logprobs = probs

    if _save_modified_probs:
        sampled_modified_probs_size = sampled_tokens_size
    else:
        # Empty tensors to invoke the kernel
        sampled_modified_probs_size = (0, 0)

    # If the number of columns in probs is too large for Triton to handle,
    # we split the tensor and sample from each split separately, and then
    # do an argmax+gather to combine the results.
    n_splits = get_num_triton_sampler_splits(probs.shape[1])
    if n_splits > 1:
        (sampled_tokens, sampled_logprobs,
         sampled_modified_probs) = _multi_split_sample(
             probs,
             seeds,
             n_splits,
             sampled_tokens_size,
             sampled_logprobs_size,
             sample_indices,
             logprobs=logprobs,
             modify_greedy_probs=modify_greedy_probs,
             save_logprobs=save_logprobs)
    else:
        sampled_tokens = torch.empty(sampled_tokens_size,
                                     dtype=torch.long,
                                     device=probs.device)
        sampled_logprobs = torch.empty(sampled_logprobs_size,
                                       dtype=probs.dtype,
                                       device=probs.device)
        sampled_modified_probs = torch.empty(sampled_modified_probs_size,
                                             dtype=probs.dtype,
                                             device=probs.device)

        _sample(probs,
                logprobs,
                sample_indices,
                sampled_tokens,
                sampled_logprobs,
                sampled_modified_probs,
                seeds,
                modify_greedy_probs=modify_greedy_probs,
                save_logprobs=save_logprobs,
                save_modified_probs=_save_modified_probs)
    return (sampled_tokens, sampled_logprobs if save_logprobs else None,
            sampled_modified_probs if _save_modified_probs else None)


def _sample(probs: torch.Tensor,
            logprobs: torch.Tensor,
            sample_indices: torch.Tensor,
            output_samples: torch.Tensor,
            output_logprobs: torch.Tensor,
            output_modified_probs: torch.Tensor,
            seeds: torch.Tensor,
            *,
            modify_greedy_probs: bool = False,
            save_logprobs: bool = True,
            save_modified_probs: bool = False) -> None:
    # Operates in place.
    n_samples = sample_indices.shape[0]
    n_cols = probs.shape[1]
    n_best = output_samples.shape[1] if len(output_samples.shape) > 1 else 1
    # The block size is the smallest power of two greater than the number of
    # columns in probs
    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4
    # Manual tuning. This seems to give best performance on A100 for
    # simple kernels like this.
    if block_size >= 8192:
        num_warps = 32
    elif block_size >= 4096:
        num_warps = 16
    elif block_size >= 2048:
        num_warps = 8
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel
    # instance per row of the probs matrix
    _sample_triton_kernel[(n_samples, n_best)](
        sample_indices,
        output_samples,
        output_logprobs,
        output_modified_probs,
        probs,
        logprobs,
        seeds,
        output_samples.stride(0),
        probs.stride(0),
        n_samples,
        n_cols,
        n_best,
        num_warps=num_warps,
        block_size=block_size,
        modify_greedy_probs=modify_greedy_probs,
        save_logprobs=save_logprobs,
        save_modified_probs=save_modified_probs,
    )


@triton.jit
def _sample_triton_kernel(
        sample_indices_ptr: torch.Tensor, output_ptr: torch.Tensor,
        output_logprobs_ptr: torch.Tensor,
        output_modified_probs_ptr: torch.Tensor, probs_ptr: torch.Tensor,
        logprobs_ptr: torch.Tensor, seed_ptr: torch.Tensor,
        output_row_stride: int, probs_row_stride: int, n_samples: int,
        n_cols: int, n_best: int, block_size: tl.constexpr,
        modify_greedy_probs: tl.constexpr, save_logprobs: tl.constexpr,
        save_modified_probs: tl.constexpr):
    # The rows are independent, so we parallelize across those
    sample_idx = tl.program_id(0)
    best_idx = tl.program_id(1)

    # Load the row index from DRAM
    row_idx = tl.load(sample_indices_ptr + sample_idx)

    # The stride represents how much we need to increase the
    # pointer to advance 1 row
    row_start_ptr = probs_ptr + row_idx * probs_row_stride

    # The block size is the next power of two greater than n_cols,
    # so we can fit each row in a single block
    col_offsets = tl.arange(0, block_size)
    probs_ptrs = row_start_ptr + col_offsets

    # Load the row into SRAM, using a mask since block_size may be > than n_cols
    row = tl.load(probs_ptrs, mask=col_offsets < n_cols, other=float("-inf"))
    seed = tl.load(seed_ptr + sample_idx)
    uses_random_sampling = seed != 0

    if uses_random_sampling:
        random_uniform = tl.rand(seed + best_idx, col_offsets)

        # tl.rand returns values in [0, 1), so we clamp lower bound
        # to _EPS to avoid log(0) and thus division by nan later
        lb = tl.full(random_uniform.shape, _EPS, random_uniform.dtype)
        random_uniform = tl.maximum(random_uniform, lb)
        # Use the inversion method to turn uniform samples
        # into exponential samples
        random_exponential = -tl.log(random_uniform)

        row /= random_exponential

    sampled_value, sampled_token = tl.max(row, axis=0, return_indices=True)
    # clamp sampled token to n_cols - 1
    # this should not be necessary, but we do it
    # just in case
    if sampled_token >= n_cols:
        sampled_token = n_cols - 1
    # Write back output to DRAM
    output_row_start_ptr = (output_ptr + sample_idx * output_row_stride +
                            best_idx)
    tl.store(output_row_start_ptr, sampled_token)

    if modify_greedy_probs:  # noqa
        if not uses_random_sampling:
            # Set the probability of the sampled token to 1, all other
            # tokens to zero. This is used in speculative decoding where
            # the sampling method must be encoded within the sampled
            # probability distributions.
            row = tl.where(col_offsets == sampled_token, 1.0, 0.0)
            tl.store(probs_ptrs, row, mask=col_offsets < n_cols)

    if save_modified_probs:
        output_row_start_ptr = (output_modified_probs_ptr +
                                sample_idx * output_row_stride + best_idx)
        tl.store(output_row_start_ptr, sampled_value)

    if save_logprobs:
        # Load the row into SRAM, using a mask since block_size
        # may be > than n_cols
        sampled_logprob = tl.load(logprobs_ptr + row_idx * probs_row_stride +
                                  sampled_token)
        # Write back output to DRAM
        output_row_start_ptr = (output_logprobs_ptr +
                                sample_idx * output_row_stride + best_idx)
        tl.store(output_row_start_ptr, sampled_logprob)
