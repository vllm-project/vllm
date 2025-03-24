# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.utils import random_sample

logger = init_logger(__name__)
INVALID_TOKEN_ID = -1


class RejectionSampler(nn.Module):
    """
    The implementation strictly follows the algorithm described in 
        https://arxiv.org/abs/2211.17192.
    However, we want to clarify the terminology used in the implementation:
    accepted tokens: tokens that are accepted based on the relationship 
            between the "raw" draft and target probabilities.
    recovered tokens: tokens that are sampled based on the adjusted probability
        distribution, which is derived from both the draft and target 
        probabilities.
    bonus tokens:
        If all proposed tokens are accepted, the bonus token is added to the
        end of the sequence. The bonus token is only sampled from the target
        probabilities. We pass in the bonus tokens instead of sampling them
        in the rejection sampler to allow for more flexibility in the
        sampling process. For example, we can use top_p, top_k sampling for
        bonus tokens, while spec decode does not support these sampling
        strategies.
    output tokens: 
        Tokens are finally generated with the rejection sampler. 
        output tokens = accepted tokens + recovered tokens + bonus tokens
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        draft_token_ids: list[list[int]],
        draft_probs: Optional[torch.Tensor],
        bonus_token_ids_tensor: torch.Tensor,  # [batch_size, 1]
        target_probs: torch.Tensor,  # [num_total_tokens, vocab_size]
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        '''
        Args:
            draft_token_ids (List[List[int]]):
                A 2D list of token IDs for each request in the batch. 
                Each request might have different number of draft tokens. 
                It may also contain empty lists for requests that have 
                no draft tokens.
            draft_probs (Optional[torch.Tensor]):
                Probability distribution for the draft tokens. Shape is
                [batch_size, max_spec_len, vocab_size]. Can be None if 
                probabilities are not provided, which is the case for
                ngram spec decode.
            bonus_token_ids_tensor (torch.Tensor):
                A tensor containing bonus tokens. Shape is [batch_size, 1]. 
                Bonus tokens are added to the end of the sequence if all 
                proposed tokens are accepted. We generate the bonus tokens 
                outside of the rejection sampler with the default sampling 
                strategy. It allows for more flexibility in the sampling 
                process such as top_p, top_k sampling.
            target_probs (torch.Tensor):
                Target model probability distribution.
                Shape is [num_total_tokens, vocab_size]. num_total_tokens 
                is the total number of tokens from all requests. Here, 
                probabilities from different requests are flattened into
                a single tensor because this is the shape of the output 
                logits.
            sampling_metadata (SamplingMetadata):
                Additional metadata needed for sampling, such as temperature,
                top-k/top-p parameters, or other relevant information.
        Returns:
            output_token_ids (torch.Tensor):
                A tensor containing the final output token IDs.
        '''

        # NOTE: The following input preparationg can be moved
        # to the model runner with a persistent manner for better
        # performance.
        # Convert draft token IDs to a tensor, split by sample_lens, then pad.
        draft_token_ids = [
            torch.tensor(x, dtype=int, device='cpu') for x in draft_token_ids
        ]
        draft_token_ids_tensor = pad_sequence(draft_token_ids,
                                              batch_first=True,
                                              padding_value=INVALID_TOKEN_ID)

        # NOTE: CPU <-> GPU synchronization happens here.
        draft_token_ids_tensor = draft_token_ids_tensor.to(target_probs.device)

        # Create one-hot tensor for draft token ids.
        # This is used for ngram where we don't have draft_probs.
        if draft_probs is None and not sampling_metadata.all_greedy:
            vocab_size = target_probs.size(-1)
            draft_probs = _create_greedy_token_probs(draft_token_ids_tensor,
                                                     vocab_size,
                                                     target_probs.device)
        sample_lens = [len(x) + 1 for x in draft_token_ids]
        target_probs = _convert_2d_probs(target_probs, sample_lens)

        return self.forward_native(draft_token_ids_tensor, draft_probs,
                                   bonus_token_ids_tensor, target_probs,
                                   sampling_metadata)

    # TODO: The following method can be optimized for better performance.
    def forward_native(
        self,
        draft_token_ids_tensor: torch.Tensor,
        # [batch_size, max_spec_len, vocab_size]
        draft_probs: Optional[torch.Tensor],
        bonus_token_ids_tensor: torch.Tensor,
        # [batch_size, max_spec_len + 1, vocab_size]
        target_probs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        # Add 1 to include the 'bonus' token.
        if sampling_metadata.all_greedy:
            # Produce a mask that remains 1 (True) until the first
            # mismatch (cumprod turns 0 after a mismatch).
            target_token_ids_tensor = target_probs.argmax(dim=-1)
            accept_mask = (target_token_ids_tensor[:, :-1] ==
                           draft_token_ids_tensor).cumprod(dim=1)

            # Identify valid positions (non-padding).
            valid_mask = target_token_ids_tensor != INVALID_TOKEN_ID
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

            output_token_ids = target_token_ids_tensor
            output_token_ids[~generate_mask] = INVALID_TOKEN_ID
        else:
            # Reference: https://arxiv.org/pdf/2211.17192
            # 1. Extract the probabilities of the draft tokens.
            # [batch_size, max_spec_len]
            batch_size = draft_token_ids_tensor.size(0)
            max_spec_len = draft_token_ids_tensor.size(1)
            invalid_idx = draft_token_ids_tensor == INVALID_TOKEN_ID
            draft_token_ids_tensor[invalid_idx] = 0
            assert draft_probs is not None
            draft_token_probs = draft_probs.gather(
                dim=-1, index=draft_token_ids_tensor.unsqueeze(-1)).squeeze(-1)
            target_token_probs = target_probs.gather(
                dim=-1, index=draft_token_ids_tensor.unsqueeze(-1)).squeeze(-1)
            # Force the probabilities of invalid tokens to inf
            # so that they are not accepted.
            draft_token_probs[invalid_idx] = float('inf')

            # 2. Generate uniform samples.
            # [batch_size, max_spec_len + 1]
            uniform_samples = _create_uniform_samples(
                sampling_metadata.generators, batch_size, max_spec_len,
                target_probs.device)

            # 3. Accept or reject the samples.
            # [batch_size, max_spec_len]
            # If the draft token probabilities are 0, set them to the smallest
            # positive normal value representable by float32.
            safe_draft_probs = torch.where(draft_token_probs > 0,
                                           draft_token_probs,
                                           torch.finfo(torch.float32).tiny)
            accepted = uniform_samples <= target_token_probs / safe_draft_probs
            accept_mask = accepted.cumprod(dim=1)
            # Set the token ids to the draft token ids if accepted, otherwise
            # set them to INVALID_TOKEN_ID.
            accepted_token_ids = (draft_token_ids_tensor * accept_mask +
                                  INVALID_TOKEN_ID * (1 - accept_mask))

            # 4. Adjust the distribution for the recovered tokens.
            # Clamp the bonus probabilities to the smallest positive normal
            # value representable by float32.
            bonus_prob = torch.clamp(target_probs[:, :-1, :] - draft_probs,
                                     min=torch.finfo(torch.float32).tiny)
            normalized_bonus_prob = bonus_prob / bonus_prob.sum(dim=-1,
                                                                keepdim=True)

            # 5. Sample recovered token ids.
            recovered_token_ids = random_sample(
                normalized_bonus_prob,
                sampling_metadata.generators).reshape(batch_size, max_spec_len)

            # 6. Get the final output token ids.
            # output_token_ids = accepted_token_ids +
            #                    recovered_token_ids +
            #                    bonus_token_id
            recovered_bonus_token_ids = torch.cat(
                [recovered_token_ids, bonus_token_ids_tensor], dim=1)
            # Generate mask with bonus tokens.
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
                             first_zero_idx] = recovered_bonus_token_ids[
                                 torch.arange(batch_size), first_zero_idx]

        return output_token_ids

    def compute_probs(self, logits: torch.Tensor,
                      sampling_metadata: SamplingMetadata,
                      sample_lens: list[int]) -> torch.Tensor:
        """
        Compute probability distribution from logits based on sampling metadata.
    
        This function applies temperature scaling to the logits and converts 
        them to probabilities using softmax. Note that division by 
        temperature is not performed inplace to preserve the original logits 
        tensor, which will be used by the original sampler to get bonus tokens.
        
        Args:
            logits: Input logits tensor to be converted to probabilities
            sampling_metadata: Metadata containing sampling parameters such 
                    as temperature and whether greedy sampling is used
            sample_lens: List of sample lengths used for repeating 
                    temperature values
            
        Returns:
            torch.Tensor: Probability distribution (softmax of scaled logits) 
                    if non-greedy sampling is used, otherwise returns the 
                    original logits
        """
        if sampling_metadata.all_greedy:
            return logits
        assert sampling_metadata.temperature is not None
        # We should optimize the following code as
        # it will cause CPU -> GPU synchronization.
        temperature = torch.repeat_interleave(
            sampling_metadata.temperature,
            torch.tensor(sample_lens,
                         device=sampling_metadata.temperature.device))
        temperature = temperature.unsqueeze(dim=1)
        logits = logits / temperature
        return logits.softmax(dim=-1, dtype=torch.float32)


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


def _convert_2d_probs(
        probs: torch.Tensor,  # [num_total_tokens, vocab_size]
        sample_lens: list[int]) -> torch.Tensor:
    """
        Converts a 2D tensor of probabilities to a 3D tensor with padding.
        [num_total_tokens, vocab_size] -> 
            [batch_size, max_spec_len + 1, vocab_size]
    """
    cumulative_lens = torch.cumsum(torch.tensor(sample_lens,
                                                device=probs.device),
                                   dim=0)
    split_indices = cumulative_lens[:-1].tolist()  # Exclude last index

    # Split into chunks without loops
    chunks = torch.tensor_split(probs, split_indices, dim=0)

    # Pad all sequences to maximum length
    padded_probs = pad_sequence(chunks, batch_first=True, padding_value=0.0)
    return padded_probs


def _create_uniform_samples(seeded_seqs: dict[int, torch.Generator],
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
                `torch.Generator` objects.
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

    uniform_rand = torch.rand(batch_size,
                              k,
                              dtype=torch.float32,
                              device=device)
    # Apply seeded generators only where needed
    if seeded_seqs:
        for idx, generator in seeded_seqs.items():
            uniform_rand[idx].uniform_(0, 1, generator=generator)
    return uniform_rand
