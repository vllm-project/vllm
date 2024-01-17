from typing import Tuple, Optional
from functools import cached_property

import torch
import torch.nn as nn
import torch.jit


class RejectionSampler(nn.Module):
    """Apply modified rejection sampling as described in "Accelerating Large
        Language Model Decoding with Speculative Sampling"
        https://arxiv.org/pdf/2302.01318.pdf.
    """

    def __init__(self, strict_mode: bool = False):
        """Create a rejection sampler.

        Args:
            strict_mode: Whether or not to perform shape/device/dtype checks
                during sampling. This catches correctness issues but adds
                nontrivial latency.
        """
        super().__init__()
        self.probs_dtype = torch.float32
        self.token_id_dtype = torch.int64
        self._strict_mode = strict_mode

        # NOTE: A "bonus token" is accepted iff all proposal tokens are
        # accepted. There is always only one possible bonus token. We store this
        # value in a variable for readability.
        self._num_bonus_tokens = 1

        self.num_accepted_tokens: Optional[torch.Tensor] = None
        self.num_emitted_tokens: Optional[torch.Tensor] = None
        self.num_draft_tokens: int = 0

    def init_gpu_tensors(self, rank: int) -> None:
        assert self.num_accepted_tokens is None
        device = f"cuda:{rank}"
        self.num_accepted_tokens = torch.tensor(0,
                                                dtype=torch.long,
                                                device=device)
        self.num_emitted_tokens = torch.tensor(0,
                                               dtype=torch.long,
                                               device=device)

    def forward(
        self,
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
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
            self._raise_if_incorrect_shape(target_probs, bonus_token_ids,
                                           draft_probs, draft_token_ids)
            self._raise_if_incorrect_dtype(target_probs, bonus_token_ids,
                                           draft_probs, draft_token_ids)
            self._raise_if_inconsistent_device(target_probs, bonus_token_ids,
                                               draft_probs, draft_token_ids)
            self._raise_if_out_of_bounds_vocab(target_probs.shape[-1],
                                               bonus_token_ids,
                                               draft_token_ids)

        accepted, recovered_token_ids = self._batch_modified_rejection_sampling(
            target_probs,
            draft_probs,
            draft_token_ids,
        )

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
                                      draft_token_ids)

        recovered_probs = self._get_recovered_probs(
            target_probs, draft_probs).reshape(batch_size * k, vocab_size)

        recovered_token_ids = _multinomial(recovered_probs,
                                           num_samples=1).reshape(
                                               batch_size, k)
        return accepted, recovered_token_ids

    def _get_accepted(
            self,
            target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
            draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
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

        uniform_rand = torch.rand(batch_size,
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

    def _create_output(
            self,
            accepted: torch.Tensor,  # [batch_size, k]
            recovered_token_ids: torch.Tensor,  # [batch_size, k]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
            bonus_token_ids: torch.Tensor,  # [batch_size]
    ) -> torch.Tensor:
        """Format output. Returns a matrix of token ids. When
        a token is rejected via rejection sampling, all subsequent
        token ids are set to -1 for the sequence.

        shape = [batch_size, k + num_bonus_tokens]
        """
        bonus_token_ids = bonus_token_ids.squeeze()
        batch_size, k = recovered_token_ids.shape

        # Determine the index of the first False value for each row.
        limits = (accepted == 0).max(1).indices
        limits[~(accepted == 0).any(1)] = k

        # Create masks using the indices.
        indices = torch.arange(k, device=accepted.device).unsqueeze(0)
        accepted_mask = indices < limits.unsqueeze(1)
        after_false_mask = indices == limits.unsqueeze(1)

        # Create an extended output tensor
        output_with_bonus_tokens = -torch.ones(
            (batch_size, k + self._num_bonus_tokens),
            dtype=self.token_id_dtype,
            device=accepted.device)
        output = output_with_bonus_tokens[:, :k]

        # Fill in the first k columns of the output tensor using masks and data
        # tensors.
        output[:, :k] = torch.where(accepted_mask, draft_token_ids,
                                    -torch.ones_like(draft_token_ids))

        # Fill the last column.
        # We check output directly as accepted may have True values inconsistent
        # with causal acceptance.
        output_with_bonus_tokens[:, -1] = torch.where(output[:, -1] != -1,
                                                      bonus_token_ids, -1)

        # Fill the recovered token ids.
        output.mul_(~after_false_mask).add_(
            recovered_token_ids.mul(after_false_mask))

        self.num_accepted_tokens += accepted.sum()
        self.num_emitted_tokens += (output_with_bonus_tokens != -1).sum()
        self.num_draft_tokens += batch_size * k

        return output_with_bonus_tokens

    def _raise_if_incorrect_shape(
        self,
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> None:
        (target_batch_size, num_target_probs,
         target_vocab_size) = target_probs.shape
        bonus_batch_size, num_bonus_tokens = bonus_token_ids.shape
        draft_batch_size, num_draft_probs, draft_vocab_size = draft_probs.shape
        draft_token_ids_batch_size, num_draft_token_ids = draft_token_ids.shape

        assert draft_batch_size == target_batch_size
        assert num_draft_probs == num_target_probs
        assert (draft_vocab_size == target_vocab_size
                ), f"{draft_vocab_size=} {target_vocab_size=}"

        assert draft_token_ids_batch_size == draft_batch_size
        assert num_draft_token_ids == num_draft_probs

        assert bonus_batch_size == target_batch_size
        assert num_bonus_tokens == self._num_bonus_tokens

    def _raise_if_incorrect_dtype(
        self,
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> None:
        assert all(probs.dtype == self.probs_dtype
                   for probs in [target_probs, draft_probs])
        assert all(token_ids.dtype == self.token_id_dtype
                   for token_ids in [bonus_token_ids, draft_token_ids])

    def _raise_if_inconsistent_device(
        self,
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> None:
        devices = [
            t.device for t in
            [target_probs, bonus_token_ids, draft_probs, draft_token_ids]
        ]
        assert all([devices[0] == device for device in devices])

    def _raise_if_out_of_bounds_vocab(
        self,
        vocab_size: int,
        bonus_token_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> None:
        assert torch.all(bonus_token_ids < vocab_size)
        assert torch.all(bonus_token_ids >= 0)
        assert torch.all(draft_token_ids < vocab_size)
        assert torch.all(draft_token_ids >= 0)


# torch.multinomial forces a GPU<->CPU sync.
# Therefore, we use an optimized implementation instead that skips the sync.
# Note that we always sample with replacement.
# probs will be modified in place, but this is fine, as we pass
# in a copy already.
@torch.jit.script
def _multinomial(
    probs: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    if num_samples > 1:
        # This is equivalent to torch.repeat_interleaved (which also
        # forces a GPU<->CPU sync).
        probs = probs[:, None, :].expand(probs.shape[0], num_samples,
                                         probs.shape[1]).contiguous().view(
                                             -1, probs.shape[1])
    q = torch.empty_like(probs).exponential_(1.0)
    return probs.div_(q).argmax(dim=1).view(-1, num_samples)
