# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional

import torch
import torch.nn as nn

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    import flashinfer.sampling
    is_flashinfer_available = True
except ImportError:
    is_flashinfer_available = False


class TopKTopPSampler(nn.Module):

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda:
            if is_flashinfer_available:
                if envs.VLLM_USE_FLASHINFER_SAMPLER is not False:
                    # NOTE(woosuk): The V0 sampler doesn't use FlashInfer for
                    # sampling unless VLLM_USE_FLASHINFER_SAMPLER=1 (i.e., by
                    # default it is unused). For backward compatibility, we set
                    # `VLLM_USE_FLASHINFER_SAMPLER` as None by default and
                    # interpret it differently in V0 and V1 samplers: In V0,
                    # None means False, while in V1, None means True. This is
                    # why we use the condition
                    # `envs.VLLM_USE_FLASHINFER_SAMPLER is not False` here.
                    logger.info("Using FlashInfer for top-p & top-k sampling.")
                    self.forward = self.forward_cuda
                else:
                    logger.warning(
                        "FlashInfer is available, but it is not enabled. "
                        "Falling back to the PyTorch-native implementation of "
                        "top-p & top-k sampling. For the best performance, "
                        "please set VLLM_USE_FLASHINFER_SAMPLER=1.")
                    self.forward = self.forward_native
            else:
                logger.warning(
                    "FlashInfer is not available. Falling back to the PyTorch-"
                    "native implementation of top-p & top-k sampling. For the "
                    "best performance, please install FlashInfer.")
                self.forward = self.forward_native
        else:
            self.forward = self.forward_native

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: Dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """PyTorch-native implementation of top-k and top-p sampling."""
        logits = apply_top_k_top_p(logits, k, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators)

    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: Dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """More optimized implementation for top-k and top-p sampling."""
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        if k is None and p is None:
            # We prefer `random_sample` over `flashinfer_sample` when sorting is
            # not needed. This is because `random_sample` does not require
            # CPU-GPU synchronization while `flashinfer_sample` does.
            return random_sample(probs, generators)
        return flashinfer_sample(probs, k, p, generators)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    This function sorts the logits tensor, which can be slow for large batches.
    """
    if k is None and p is None:
        return logits
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def random_sample(
    probs: torch.Tensor,
    generators: Dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def flashinfer_sample(
    probs: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    generators: Dict[int, torch.Generator],
) -> torch.Tensor:
    """Sample from the probabilities using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.
    
    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.

    NOTE: This function includes CPU-GPU synchronization, while `random_sample`
    does not. Call this function at the end of the forward pass to minimize
    the synchronization overhead.
    """
    assert not (k is None and p is None)
    max_top_k_round = 32
    batch_size = probs.shape[0]
    uniform_samples = torch.empty((max_top_k_round, batch_size),
                                  device=probs.device)
    if len(generators) != batch_size:
        uniform_samples.uniform_()
    if generators:
        for i, generator in generators.items():
            uniform_samples[:, i].uniform_(generator=generator)

    if k is None:
        # Top-p only.
        next_token_ids, success = flashinfer.sampling.top_p_sampling_from_probs(
            probs, uniform_samples, p, deterministic=True)
    elif p is None:
        # Top-k only.
        next_token_ids, success = flashinfer.sampling.top_k_sampling_from_probs(
            probs, uniform_samples, k, deterministic=True)
    else:
        # Both top-k and top-p.
        next_token_ids, success = (
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, uniform_samples, k, p, deterministic=True))

    # NOTE: CPU-GPU synchronization happens here.
    if not success.all():
        if k is not None:
            probs = flashinfer.sampling.top_k_renorm_prob(probs, k)
        if p is not None:
            probs = flashinfer.sampling.top_p_renorm_prob(probs, p)
        next_token_ids = flashinfer.sampling.sampling_from_probs(
            probs, uniform_samples[0], deterministic=True)
    return next_token_ids.view(-1)
