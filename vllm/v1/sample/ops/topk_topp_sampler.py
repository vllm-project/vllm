from typing import Dict

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    import flashinfer.sampling
    use_flashinfer = True
except ImportError:
    use_flashinfer = False


class TopKTopPSampler(nn.Module):

    def __init__(self):
        super().__init__()
        if current_platform.is_cuda:
            if use_flashinfer:
                self.forward = self.forward_cuda
            else:
                logger.warning(
                    "flashinfer.sampling is not available. Falling back to "
                    "Pytorch-native implementation of sampling. For the best "
                    "performance, please install FalshInfer.")
                self.forward = self.forward_native
        else:
            self.forward = self.forward_native

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: Dict[int, torch.Generator],
        no_top_k: bool,
        k: torch.Tensor,
        no_top_p: bool,
        p: torch.Tensor,
    ) -> torch.Tensor:
        logits = apply_top_k_top_p(logits, no_top_k, k, no_top_p, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators)

    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: Dict[int, torch.Generator],
        no_top_k: bool,
        k: torch.Tensor,
        no_top_p: bool,
        p: torch.Tensor,
    ) -> torch.Tensor:
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        if no_top_k and no_top_p:
            return random_sample(probs, generators)
        return flashinfer_sample(probs, no_top_k, k, no_top_p, p, generators)


def apply_top_k_top_p(
    logits: torch.Tensor,
    no_top_k: bool,
    k: torch.Tensor,
    no_top_p: bool,
    p: torch.Tensor,
) -> torch.Tensor:
    if no_top_k and no_top_p:
        return logits
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if not no_top_k:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if not no_top_p:
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
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        # This might still be done here unnecessarily if there are greedies
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def flashinfer_sample(
    probs: torch.Tensor,
    no_top_k: bool,
    k: torch.Tensor,
    no_top_p: bool,
    p: torch.Tensor,
    generators: Dict[int, torch.Generator],
) -> torch.Tensor:
    assert not (no_top_k and no_top_p)
    max_top_k_round = 32
    batch_size = probs.shape[0]
    uniform_samples = torch.empty((max_top_k_round, batch_size),
                                  device=probs.device)
    if len(generators) != batch_size:
        uniform_samples.uniform_()
    if generators:
        for i, generator in generators.items():
            uniform_samples[:, i].uniform_(generator=generator)

    if no_top_k:
        # Top-p only.
        next_token_ids, success = flashinfer.sampling.top_p_sampling_from_probs(
            probs, uniform_samples, p, deterministic=True)
    elif no_top_p:
        # Top-k only.
        next_token_ids, success = flashinfer.sampling.top_k_sampling_from_probs(
            probs, uniform_samples, k, deterministic=True)
    else:
        # Both top-k and top-p.
        next_token_ids, success = (
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, uniform_samples, k, p, deterministic=True))

    if not success.all():
        if not no_top_k:
            probs = flashinfer.sampling.top_k_renorm_prob(probs, k)
        if not no_top_p:
            probs = flashinfer.sampling.top_p_renorm_prob(probs, p)
        next_token_ids = flashinfer.sampling.sampling_from_probs(
            probs, uniform_samples[0], deterministic=True)
    return next_token_ids.view(-1)
