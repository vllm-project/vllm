# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k/top-p with FlashInfer's Cake radix sampler (FlashInfer >= 0.7.1).

Cake selects the exact per-row top-k (1 <= k <= 1024), applies top-p within
it and draws a token in at most two kernels. It is used in two ways:

- ``cake_sample`` draws tokens directly, replacing FlashInfer's fused
  top-k/top-p sampler.
- ``apply_top_k_top_p_cake`` keeps the masked-logits contract of
  ``apply_top_k_top_p``. Cake leaves the kept set as a prefix of its sorted
  slab, so one pass keeps ``prob > tau``, or ``prob == tau`` with
  ``index <= cutoff`` (Cake breaks ties by lower index). Sampling and
  rejection kernels downstream are unchanged.
"""

import functools

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import has_flashinfer_cake_sampling

# Width of Cake's per-row top-k slab.
CAKE_MAX_TOP_K = 1024
# Keep large mask batches on Triton: the extra passes make Cake's crossover
# depend on the input distribution. Direct sampling has no such cutoff.
CAKE_MASK_MAX_ROWS = 64
_MASK_BLOCK_SIZE = 4096


@functools.cache
def _route_ok(device_index: int, vocab_size: int) -> bool:
    import flashinfer.cake_sampling as cake

    probe = torch.empty(
        1, vocab_size, device=torch.device("cuda", device_index), dtype=torch.float32
    )
    return cake.cake_sampling_route(probe, 1, 1) == "pipeline"


def cake_eligible(logits: torch.Tensor, k_max: int | None) -> bool:
    """Whether Cake serves a batch whose largest top-k is ``k_max``.

    ``k_max`` comes from host-side sampling state, so no device sync is
    needed. Rows without top-k carry ``k == vocab_size`` and opt the batch out.
    """
    if (
        k_max is None
        or not envs.VLLM_USE_FLASHINFER_CAKE_SAMPLER
        or not current_platform.is_cuda()
        or not has_flashinfer_cake_sampling()
    ):
        return False
    vocab_size = logits.shape[-1]
    if not 1 <= k_max <= CAKE_MAX_TOP_K or k_max >= vocab_size:
        return False
    return _route_ok(logits.device.index or 0, vocab_size)


_WORKSPACES: dict[tuple[int, torch.device], tuple[torch.Tensor, ...]] = {}


def _workspace(batch: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Slab probs, slab ids, counts, renormalized probs and samples buffers."""
    # Power-of-two capacities keep the cache within 4x the largest batch. The
    # buffers are never freed, so captured CUDA graphs may hold their views.
    capacity = 1 << (batch - 1).bit_length()
    key = (capacity, device)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        workspace = (
            torch.empty(capacity, CAKE_MAX_TOP_K, device=device, dtype=torch.float32),
            torch.empty(capacity, CAKE_MAX_TOP_K, device=device, dtype=torch.int32),
            torch.empty(capacity, device=device, dtype=torch.int32),
            torch.empty(capacity, CAKE_MAX_TOP_K, device=device, dtype=torch.float32),
            torch.empty(capacity, device=device, dtype=torch.int32),
        )
        _WORKSPACES[key] = workspace
    return tuple(buffer[:batch] for buffer in workspace)


def cake_sample(
    logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor | None, k_max: int
) -> torch.Tensor:
    """Samples one token per row from top-k (then top-p) filtered logits."""
    import flashinfer.cake_sampling as cake
    import flashinfer.sampling

    probs = flashinfer.sampling.softmax(logits)
    slab_probs, slab_ids, counts, _, _ = _workspace(probs.shape[0], probs.device)
    return cake.top_k_top_p_sampling_from_probs(
        probs,
        k,
        p if p is not None else 1.0,
        top_k_max=k_max,
        workspace=(slab_probs, slab_ids, counts),
    )


@triton.jit
def _slab_mask_kernel(
    logits_ptr,
    logits_stride,
    probs_ptr,
    probs_stride,
    slab_probs_ptr,
    slab_ids_ptr,
    renorm_ptr,
    counts_ptr,
    slab_stride,
    vocab_size,
    mask_value,
    K_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    # Top-p keeps a prefix of the sorted slab: renorm is zero past the kept
    # prefix and undefined past `count`.
    slots = tl.arange(0, K_MAX)
    count = tl.load(counts_ptr + row)
    renorm = tl.load(
        renorm_ptr + row * slab_stride + slots, mask=slots < count, other=0.0
    )
    last = tl.maximum(tl.sum((renorm > 0).to(tl.int32)) - 1, 0)
    tau = tl.load(slab_probs_ptr + row * slab_stride + last)
    cutoff = tl.load(slab_ids_ptr + row * slab_stride + last)

    offs = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_vocab = offs < vocab_size
    probs = tl.load(probs_ptr + row * probs_stride + offs, mask=in_vocab, other=0.0)
    logits = tl.load(logits_ptr + row * logits_stride + offs, mask=in_vocab)
    keep = (probs > tau) | ((probs == tau) & (offs <= cutoff))
    tl.store(
        logits_ptr + row * logits_stride + offs,
        tl.where(keep, logits, mask_value),
        mask=in_vocab,
    )


def apply_top_k_top_p_cake(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor | None,
    k_max: int,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """Masks ``logits`` in place to Cake's top-k-then-top-p kept set."""
    import flashinfer.cake_sampling as cake
    import flashinfer.sampling

    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    batch, vocab_size = logits.shape
    probs = flashinfer.sampling.softmax(logits)
    slab_probs, slab_ids, counts, renorm, samples = _workspace(batch, logits.device)
    # The draw itself is discarded; a fixed Philox state leaves the default
    # generator untouched.
    cake.top_k_top_p_sampling_from_probs(
        probs,
        k,
        p if p is not None else 1.0,
        top_k_max=k_max,
        philox_seed=0,
        philox_offset=0,
        out=samples,
        renorm_out=renorm,
        workspace=(slab_probs, slab_ids, counts),
    )
    _slab_mask_kernel[(batch, triton.cdiv(vocab_size, _MASK_BLOCK_SIZE))](
        logits,
        logits.stride(0),
        probs,
        probs.stride(0),
        slab_probs,
        slab_ids,
        renorm,
        counts,
        slab_probs.stride(0),
        vocab_size,
        mask_value,
        K_MAX=triton.next_power_of_2(k_max),
        BLOCK_SIZE=_MASK_BLOCK_SIZE,
    )
    return logits
