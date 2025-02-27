# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import random_sample

try:
    import flashinfer.sampling as fs
    is_flashinfer_available = True
except ImportError:
    is_flashinfer_available = False

logger = init_logger(__name__)
INVALID_TOKEN_ID = -1


class RejectionSampler(nn.Module):

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
                    logger.info("Using FlashInfer for rejection sampling.")
                    self.forward_method = self.flashinfer_sample
                else:
                    logger.warning(
                        "FlashInfer is available, but it is not enabled. "
                        "Falling back to the PyTorch-native implementation of "
                        "rejection sampling. For the best performance, "
                        "please set VLLM_USE_FLASHINFER_SAMPLER=1.")
                    self.forward_method = self.forward_native
            else:
                logger.warning(
                    "FlashInfer is not available. Falling back to the PyTorch-"
                    "native implementation of rejection sampling. For the "
                    "best performance, please install FlashInfer.")
                self.forward_method = self.forward_native
        else:
            self.forward_method = self.forward_native

    def forward(
        self,
        draft_token_ids: List[List[int]],
        draft_probs: Optional[
            torch.Tensor],  # [batch_size, max_spec_len, vocab_size]
        target_probs: torch.
        Tensor,  # [batch_size, max_spec_len + 1, vocab_size]
        sampling_metadata: SamplingMetadata
    ) -> SamplerOutput:
        return self.forward_method(draft_token_ids, draft_probs, target_probs,
                                   sampling_metadata)

    def flashinfer_sample(
        self,
        draft_token_ids: List[List[int]],
        draft_probs: Optional[torch.Tensor],
        target_probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # NOTE: The following input preparationg can be moved
        # to the model runner with a persistent manner for better
        # performance.
        sample_lens = [len(x) + 1 for x in draft_token_ids]
        # Convert draft token IDs to a tensor, split by sample_lens, then pad.
        draft_token_ids = [
            torch.tensor(x, dtype=int, device='cpu') for x in draft_token_ids
        ]
        draft_token_ids_tensor = pad_sequence(draft_token_ids,
                                              batch_first=True,
                                              padding_value=INVALID_TOKEN_ID)

        batch_size = draft_token_ids_tensor.size(0)
        max_spec_len = draft_token_ids_tensor.size(1)
        if sampling_metadata.all_greedy:
            target_token_ids = target_probs.argmax(dim=-1).view(-1)
            target_token_ids = target_token_ids.split(sample_lens)
            target_token_ids = pad_sequence(target_token_ids,
                                            batch_first=True,
                                            padding_value=INVALID_TOKEN_ID)

            vocab_size = target_probs.size(-1)
            # NOTE: CPU <-> GPU synchronization happens here.
            draft_token_ids_tensor = draft_token_ids_tensor.to(
                target_probs.device)
            draft_probs = _create_greedy_token_probs(draft_token_ids_tensor,
                                                     vocab_size,
                                                     target_probs.device)
            target_probs = _create_greedy_token_probs(target_token_ids,
                                                      vocab_size,
                                                      target_probs.device)
            uniform_samples = torch.zeros(batch_size,
                                          max_spec_len + 1,
                                          device=target_probs.device)
        else:
            # NOTE: CPU <-> GPU synchronization happens here.
            draft_token_ids_tensor = draft_token_ids_tensor.to(
                target_probs.device)
            uniform_samples = _create_uniform_samples(
                sampling_metadata.generators, batch_size, max_spec_len + 1,
                target_probs.device)

        sampled_token_ids, _, _ = fs.chain_speculative_sampling(
            draft_probs,
            draft_token_ids_tensor,
            uniform_samples,
            target_probs,
        )
        return SamplerOutput(sampled_token_ids=sampled_token_ids,
                             logprobs_tensors=None)

    # TODO: The following method can be optimized for better performance.
    def forward_native(
        self,
        draft_token_ids: List[List[int]],
        draft_probs: Optional[
            torch.Tensor],  # [batch_size, max_spec_len, vocab_size]
        target_probs: torch.
        Tensor,  # [batch_size, max_spec_len + 1, vocab_size]
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        sample_lens = [len(x) + 1 for x in draft_token_ids]
        # Convert draft token IDs to a tensor, split by sample_lens, then pad.
        draft_token_ids = [
            torch.tensor(x, dtype=int, device='cpu') for x in draft_token_ids
        ]
        draft_token_ids_tensor = pad_sequence(draft_token_ids,
                                              batch_first=True,
                                              padding_value=INVALID_TOKEN_ID)
        draft_token_ids_tensor = draft_token_ids_tensor.to(target_probs.device)
        # Add 1 to include the 'bonus' token.
        if sampling_metadata.all_greedy:
            output_token_ids = target_probs.argmax(dim=-1).view(-1)
            output_token_ids = output_token_ids.split(sample_lens)
            output_token_ids = pad_sequence(output_token_ids,
                                            batch_first=True,
                                            padding_value=INVALID_TOKEN_ID)
            # Produce a mask that remains 1 (True) until the first
            # mismatch (cumprod turns 0 after a mismatch).
            accept_mask = (
                output_token_ids[:, :-1] == draft_token_ids_tensor).cumprod(
                    dim=1)

            # Identify valid positions (non-padding).
            valid_mask = output_token_ids != INVALID_TOKEN_ID
            # Generate mask with bonus token.
            generate_mask = torch.cat([
                accept_mask,
                torch.zeros(accept_mask.size(0), 1, device=accept_mask.device)
            ],
                                      dim=1).to(torch.bool) & valid_mask
            zeros_mask = (generate_mask == 0)
            first_zero_idx = zeros_mask.float().argmax(dim=1)
            # Figure out which rows actually contain at least one zero.
            rows_with_zero = zeros_mask.any(dim=1)
            # Use indexing to set the first zero in each of those rows to 1.
            generate_mask[rows_with_zero, first_zero_idx[rows_with_zero]] = 1

            output_token_ids[~generate_mask] = INVALID_TOKEN_ID

        else:
            # Reference: https://arxiv.org/pdf/2211.17192
            # 1. Extract the probabilities of the draft tokens.
            # [batch_size, max_spec_len]
            assert draft_probs is not None
            batch_size = draft_token_ids_tensor.size(0)
            max_spec_len = draft_token_ids_tensor.size(1)
            draft_token_probs = draft_probs.gather(
                dim=-1, index=draft_token_ids_tensor.unsqueeze(-1)).squeeze(-1)
            target_token_probs = target_probs.gather(
                dim=-1, index=draft_token_ids_tensor.unsqueeze(-1)).squeeze(-1)

            # 2. Generate uniform samples.
            # [batch_size, max_spec_len + 1]
            uniform_samples = _create_uniform_samples(
                sampling_metadata.generators, batch_size, max_spec_len,
                target_probs.device)

            # 3. Accept or reject the samples.
            # [batch_size, max_spec_len]
            accepted = uniform_samples <= target_token_probs / draft_token_probs
            accept_mask = accepted.cumprod(dim=1)
            # Set the token ids to the draft token ids if accepted, otherwise
            # set them to INVALID_TOKEN_ID.
            accepted_token_ids = (draft_token_ids_tensor * accept_mask +
                                  INVALID_TOKEN_ID * (1 - accept_mask))

            # 4. Adjust the distribution for the bonus token.
            bonus_prob = torch.clamp(target_probs[:, :-1, :] - draft_probs,
                                     min=1e-5)
            normalized_bonus_prob = bonus_prob / bonus_prob.sum(dim=-1,
                                                                keepdim=True)
            # Concatenate normalized prob with the prob of last target token.
            sample_prob = torch.cat(
                [normalized_bonus_prob, target_probs[:, -1, :].unsqueeze(1)],
                dim=1)

            # 5. Sample bonus token.
            # [batch_size, max_spec_len + 1]
            bonus_token_ids = random_sample(
                sample_prob,
                sampling_metadata.generators).reshape(batch_size,
                                                      max_spec_len + 1)

            # 6. Concatenate the bonus tokens with accepted tokens to get the
            # output token ids.
            # Generate mask with bonus token.
            generate_mask = torch.cat([
                accept_mask,
                torch.zeros(batch_size, 1, device=accept_mask.device)
            ],
                                      dim=1).to(torch.bool)
            zeros_mask = (generate_mask == 0)
            first_zero_idx = zeros_mask.float().argmax(dim=1)
            output_token_ids = torch.cat([
                accepted_token_ids,
                torch.full((batch_size, 1),
                           fill_value=INVALID_TOKEN_ID,
                           device=accept_mask.device)
            ],
                                         dim=1)
            output_token_ids[torch.arange(batch_size),
                             first_zero_idx] = bonus_token_ids[
                                 torch.arange(batch_size), first_zero_idx]

        return SamplerOutput(sampled_token_ids=output_token_ids,
                             logprobs_tensors=None)


def _create_greedy_token_probs(
    token_ids: torch.Tensor,
    vocab_size: int,
    out_device: torch.device,
) -> torch.Tensor:
    batch_size, num_tokens = token_ids.shape

    token_probs = torch.zeros(batch_size,
                              num_tokens,
                              vocab_size,
                              dtype=torch.float,
                              device=out_device)

    # Ignore INVALID_TOKEN_ID.
    valid_mask = (token_ids != INVALID_TOKEN_ID)
    valid_indices = token_ids.clone()
    valid_indices[~valid_mask] = 0

    token_probs.scatter_(dim=2,
                         index=valid_indices.unsqueeze(-1),
                         src=valid_mask.unsqueeze(-1).float())

    return token_probs


def _create_uniform_samples(seeded_seqs: Optional[Dict[int, torch.Generator]],
                            batch_size: int, k: int,
                            device: torch.device) -> torch.Tensor:
    """
        Generates a batch of uniform random samples, with optional seeding 
        for specific sequences.

        This method creates a tensor of shape `(batch_size, k)` filled 
        with uniform random values in the range [0, 1). If `seeded_seqs` 
        is provided, the sequences corresponding to specific indices 
        will be generated using the provided `torch.Generator` for 
        reproducibility. The other sequences will be generated without 
        a seed.

        Args:
            seeded_seqs : Optional[Dict[int, torch.Generator]]
                A dictionary mapping indices in the batch to 
                `torch.Generator` objects. If `None`, all samples are 
                generated without a seed.
            batch_size : int
                The number of sequences to generate.
            k : int
                The number of random samples per sequence.
            device : torch.device
                The device on which to allocate the tensor.

        Returns:
            uniform_rand : torch.Tensor
                A tensor of shape `(batch_size, k)` containing uniform 
                random values in the range [0, 1).
        """
    if not seeded_seqs:
        return torch.rand(batch_size, k, device=device)

    uniform_rand = torch.empty(batch_size, k, device=device)

    non_seeded_indices = []
    for idx in range(batch_size):
        generator = seeded_seqs.get(idx)
        if generator is None:
            non_seeded_indices.append(idx)
        else:
            uniform_rand[idx, :] = torch.rand(1,
                                              k,
                                              dtype=torch.float,
                                              device=device,
                                              generator=generator)
    if non_seeded_indices:
        uniform_rand[non_seeded_indices, :] = torch.rand(
            len(non_seeded_indices), k, dtype=torch.float, device=device)
    return uniform_rand
