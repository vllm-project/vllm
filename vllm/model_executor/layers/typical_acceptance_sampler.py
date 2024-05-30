from functools import cached_property
from typing import Optional, Tuple

import torch
import torch.jit
import torch.nn as nn

from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler)


class TypicalAcceptanceSampler(SpecDecodeBaseSampler, nn.Module):
    """Apply modified rejection sampling as described in "Accelerating Large
        Language Model Decoding with Speculative Sampling"
        https://arxiv.org/pdf/2302.01318.pdf.
    """

    def __init__(self, disable_bonus_tokens: bool = False, strict_mode: bool = False):
        """Create a rejection sampler.

        Args:
            strict_mode: Whether or not to perform shape/device/dtype checks
                during sampling. This catches correctness issues but adds
                nontrivial latency.
        """
        super().__init__()
        SpecDecodeBaseSampler.__init__(
            self, disable_bonus_tokens=disable_bonus_tokens, strict_mode=strict_mode)
        nn.Module.__init__(self)
    
    def forward(
        self,
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sample token ids using rejection sampling. This accepts or rejects
        tokens proposed by the draft model using the probability of each token
        according to the draft and target models.

        In the worst case where all draft tokens are rejected, it is guaranteed
        one correct token will be emitted.

        In the case where all draft tokens are accepted, a bonus token will be
        accepted as its cheap to have the target model score this speculative
        sequence.

        Args:
            target_probs: The probability distribution over token ids given
                context according to the target model.
            shape = [batch_size, num_speculative_tokens, vocab_size]

            bonus_token_ids: The "bonus" token ids that are accepted iff all
                speculative tokens in a sequence are accepted.
            shape = [batch_size, num_bonus_tokens]

            draft_probs: The probability distribution over token ids given
                context according to the draft model.
            shape = [batch_size, num_speculative_tokens, vocab_size]

            draft_token_ids: The token ids that were sampled from the draft
                probabilities.
            shape = [batch_size, num_speculative_tokens]

        Returns:
            output_token_ids: The token ids sampled via rejection sampling,
                or -1 if unable to sample a token because the previous token
                was rejected.
            shape = [batch_size, num_speculative_tokens + num_bonus_tokens]
        """
        # Only perform shape/dtype/device checking in strict mode, as it adds
        # overhead.
        if self._strict_mode:
            self._raise_if_incorrect_input(
                target_probs, draft_token_ids, bonus_token_ids)
        accepted = self._evaluate_posterior(target_probs, draft_token_ids)
        recovered_token_ids = self._replacement_token_ids(target_probs)
        output_token_ids = self._create_output(
            accepted,
            recovered_token_ids,
            draft_token_ids, bonus_token_ids
        )
        print('----test input----')
        print('target_probs ' + str(target_probs))
        print('draft_token_ids ' + str(draft_token_ids))
        print('recovered_token_ids ' + str(recovered_token_ids))
        print(output_token_ids)
        return output_token_ids

    def _evaluate_posterior(
        self, target_probs, draft_token_ids,
        posterior_threshold=0.3, posterior_alpha = 0.09):
        
        """Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.
        Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
        probabilities to select the best candidate.

        Args:
        - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
        - candidates (torch.Tensor): Candidate token sequences.
        - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
        - posterior_threshold (float): Threshold for posterior probability.
        - posterior_alpha (float): Scaling factor for the threshold.
        - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
        - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
        - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
        - best_candidate (torch.Tensor): Index of the chosen best candidate.
        - accept_length (int): Length of the accepted candidate sequence.
        """
        candidates_prob = torch.gather(
            target_probs, dim=-1, index=draft_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        posterior_entropy = -torch.sum(
            target_probs * torch.log(target_probs + 1e-5), dim=-1
        )  # torch.sum(torch.log(*)) is faster than torch.prod
        threshold = torch.minimum(
            torch.ones_like(posterior_entropy) * posterior_threshold,
            torch.exp(-posterior_entropy) * posterior_alpha,
        )
        posterior_mask = candidates_prob > threshold
        return posterior_mask

    def _replacement_token_ids(self, target_probs):
        max_indices = torch.argmax(target_probs[:, 0, :], dim=1)
        output = -torch.ones((target_probs.shape[0], target_probs.shape[1]), dtype=self.token_id_dtype)
        output[:, 0] = max_indices
        return output
