from functools import cached_property
from typing import Dict, List, Optional, Tuple

import torch
import torch.jit

from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeStochasticBaseSampler)


class RejectionSampler(SpecDecodeStochasticBaseSampler):
    """Apply modified rejection sampling as described in "Accelerating Large
        Language Model Decoding with Speculative Sampling"
        https://arxiv.org/pdf/2302.01318.pdf.
    """

    def __init__(self,
                 disable_bonus_tokens: bool = True,
                 strict_mode: bool = False):
        """Create a rejection sampler.

        Args:
            disable_bonus_tokens: Whether or not to disable the bonus token.
            Require when bonus tokens will cause corrupt KV cache for
            proposal methods that require KV cache.
            strict_mode: Whether or not to perform shape/device/dtype checks
            during sampling. This catches correctness issues but adds
            nontrivial latency.
        """
        super().__init__(disable_bonus_tokens=disable_bonus_tokens,
                         strict_mode=strict_mode)

    def forward(
        self,
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        seeded_seqs: Optional[Dict[int, torch.Generator]] = None,
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

            seeded_seqs: Dict of batch row index to torch generator, for
                sequences using seeded generation.

        Returns:
            output_token_ids: The token ids sampled via rejection sampling,
                or -1 if unable to sample a token because the previous token
                was rejected.
            shape = [batch_size, num_speculative_tokens + num_bonus_tokens]
        """
        # Only perform shape/dtype/device checking in strict mode, as it adds
        # overhead.
        if self._strict_mode:
            self._raise_if_incorrect_input(target_probs, bonus_token_ids,
                                           draft_probs, draft_token_ids)

        accepted, recovered_token_ids = (
            self._batch_modified_rejection_sampling(
                target_probs,
                draft_probs,
                draft_token_ids,
                seeded_seqs,
            ))

        output_token_ids = self._create_output(
            accepted,
            recovered_token_ids,
            draft_token_ids,
            bonus_token_ids,
        )

        return output_token_ids

    def _batch_modified_rejection_sampling(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform modified rejection sampling on each sequence.

        Returns:
            A tuple of two tensors:
            0: A bool tensor of which tokens in each sequence is accepted.
                shape = [batch_size, k]
            1: Token ids sampled from a recovered distribution, to be used
                when a token is rejected.
                shape = [batch_size, k]
        """

        batch_size, k, vocab_size = draft_probs.shape

        # shape [batch_size, k]
        accepted = self._get_accepted(target_probs, draft_probs,
                                      draft_token_ids, seeded_seqs)

        recovered_probs = self._get_recovered_probs(
            target_probs, draft_probs).reshape(batch_size * k, vocab_size)

        # NOTE: the recovered_probs are overwritten by this method.
        recovered_token_ids = _multinomial(
            recovered_probs,
            num_samples=1,
            k=k,
            seeded_seqs=seeded_seqs or {},
        ).reshape(batch_size, k)

        return accepted, recovered_token_ids

    def _get_accepted(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> torch.Tensor:
        r"""Create bool matrix over the proposed draft tokens. If
        True, then a token can be accepted, else it should be
        rejected.

        Given :math:`q(\hat{x}_{n+1}|x_1, \dots, x_n)`, the probability of
        :math:`\hat{x}_{n+1}` given context :math:`x_1, \dots, x_n` according
        to the target model, and :math:`p(\hat{x}_{n+1}|x_1, \dots, x_n)`, the
        same conditional probability according to the draft model, the token
        is accepted with probability:

        .. math::
            \min\left(1, \frac{q(\hat{x}_{n+1}|x_1, \dots, x_n)}
                           {p(\hat{x}_{n+1}|x_1, \dots, x_n)}\right)

        This implementation does not apply causality. When using the output,
        if a token is rejected, subsequent tokens should not be used.

        Returns a bool tensor of shape [batch_size, k] specifying which tokens
        are accepted.
        """
        batch_size, k, _ = draft_probs.shape
        batch_indices = torch.arange(batch_size,
                                     device=target_probs.device)[:, None]
        probs_indicies = torch.arange(k, device=target_probs.device)

        # shape [batch_size, k]
        selected_draft_probs = draft_probs[batch_indices, probs_indicies,
                                           draft_token_ids]

        # shape [batch_size, k]
        selected_target_probs = target_probs[batch_indices, probs_indicies,
                                             draft_token_ids]

        if not seeded_seqs:
            uniform_rand = torch.rand_like(selected_target_probs)
        else:
            uniform_rand = torch.empty_like(selected_target_probs)

            non_seeded_indices = []
            for idx in range(batch_size):
                generator = seeded_seqs.get(idx)
                if generator is None:
                    non_seeded_indices.append(idx)
                else:
                    uniform_rand[idx, :] = torch.rand(
                        1,
                        k,
                        dtype=self.probs_dtype,
                        device=target_probs.device,
                        generator=generator)
            if non_seeded_indices:
                uniform_rand[non_seeded_indices, :] = torch.rand(
                    len(non_seeded_indices),
                    k,
                    dtype=self.probs_dtype,
                    device=target_probs.device)

        capped_ratio = torch.minimum(
            selected_target_probs / selected_draft_probs,
            torch.full((1, ), 1, device=target_probs.device))
        accepted = uniform_rand < capped_ratio

        return accepted

    def _get_recovered_probs(
            self,
            target_probs: torch.Tensor,  # [k, vocab_size]
            draft_probs: torch.Tensor,  # [k, vocab_size]
    ) -> torch.Tensor:
        r"""Create a probability distribution for each proposed token which can
        be sampled if the proposed token is rejected.

        When this routine is applied sequentially, the true distribution of the
        target model is recovered (within hardware numerics).

        The probability distribution used in this rejection case is constructed
        as follows. Given :math:`q(x|x_1, \dots, x_n)`, the probability of
        :math:`x` given context :math:`x_1, \dots, x_n` according to the target
        model and :math:`p(x|x_1, \dots, x_n)`, the same conditional probability
        according to the draft model:

        .. math::
            x_{n+1} \sim (q(x|x_1, \dots, x_n) - p(x|x_1, \dots, x_n))_+

        where :math:`(f(x))_+` is defined as:

        .. math::
            (f(x))_+ = \frac{\max(0, f(x))}{\sum_x \max(0, f(x))}

        See https://github.com/vllm-project/vllm/pull/2336 for a visualization
        of the draft, target, and recovered probability distributions.

        Returns a tensor of shape [batch_size, k, vocab_size].

        Note: This batches operations on GPU and thus constructs the recovered
        distribution for all tokens, even if they are accepted. This causes
        division-by-zero errors, so we use self._smallest_positive_value to
        avoid that. This introduces some drift to the distribution.
        """
        _, k, _ = draft_probs.shape

        # shape [batch_size, k, vocab_size]
        difference = target_probs - draft_probs

        # TODO(cade): Can we use logprobs instead of probs, and avoid the
        # division-by-zero errors without introducing distribution drift?

        # shape [batch_size, k, vocab_size]
        f = torch.clamp(difference, min=self._smallest_positive_value)

        # shape [batch_size, k, vocab_size]
        recovered_probs = f / torch.sum(f, dim=-1).reshape(-1, k, 1)

        return recovered_probs

    @cached_property
    def _smallest_positive_value(self) -> float:
        """Return the smallest positive value representable by the probs dtype.
        This value is used when constructing a distribution from which to sample
        recovered tokens in the first rejection case.

        See _get_recovered_probs for more details

        Note that this isn't actually the smallest positive value representable
        by float32, but the smallest positive normal value.
        See https://en.wikipedia.org/wiki/Subnormal_number for more information.
        """
        return torch.finfo(self.probs_dtype).tiny


# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead that skips the sync.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
@torch.jit.script
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
    k: int,
    seeded_seqs: Dict[int, torch.Generator],
) -> torch.Tensor:

    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
                                             -1, probs.shape[1])
    q = torch.empty_like(probs)
    if not seeded_seqs:
        q.exponential_(1.0)
    else:
        non_seeded_indices: List[int] = []
        start = 0
        for idx in range(len(q) // k):
            end = start + k
            generator = seeded_seqs.get(idx)
            if generator is None:
                non_seeded_indices.extend(list(range(start, end)))
            else:
                q[start:end].exponential_(1.0, generator=generator)
            start = end
        q[non_seeded_indices].exponential_(1.0)

    return probs.div_(q).argmax(dim=1).view(-1, num_samples)
