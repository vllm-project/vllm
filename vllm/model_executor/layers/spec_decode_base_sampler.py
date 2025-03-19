# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Dict, Optional, Union

import torch
import torch.jit
import torch.nn as nn

from vllm.platforms import current_platform


class SpecDecodeBaseSampler(nn.Module):
    """Base class for samplers used for Speculative Decoding verification
        step.
    """

    def __init__(self, strict_mode: bool = False):
        """Base class constructor.
        Args:
            strict_mode: Whether or not to perform shape/device/dtype checks
                during sampling. This catches correctness issues but adds
                nontrivial latency.
        """
        super().__init__()
        self._strict_mode = strict_mode

        # NOTE: A "bonus token" is accepted iff all proposal tokens are
        # accepted. There is always only one possible bonus token. We store this
        # value in a variable for readability.
        self._num_bonus_tokens = 1

        self.num_accepted_tokens: Optional[torch.Tensor] = None
        self.num_emitted_tokens: Optional[torch.Tensor] = None
        self.num_draft_tokens: int = 0

    def init_gpu_tensors(self, device: Union[int, str]) -> None:
        assert self.num_accepted_tokens is None
        if isinstance(device, int):
            device = f"{current_platform.device_type}:{device}"
        elif not isinstance(device, str):
            raise ValueError(f"Device must be int or str, get {type(device)}")
        self.num_accepted_tokens = torch.tensor(0,
                                                dtype=torch.long,
                                                device=device)
        self.num_emitted_tokens = torch.tensor(0,
                                               dtype=torch.long,
                                               device=device)

    def init_tensors(self,
                     device: Union[int, str],
                     device_type: Union[torch.device, str] = 'cuda') -> None:
        assert self.num_accepted_tokens is None
        if isinstance(device_type, torch.device):
            device_type = device_type.type
        if isinstance(device, int):
            device = f"{device_type}:{device}"
        self.num_accepted_tokens = torch.tensor(0,
                                                dtype=torch.long,
                                                device=device)
        self.num_emitted_tokens = torch.tensor(0,
                                               dtype=torch.long,
                                               device=device)

    @property
    def probs_dtype(self):
        return torch.float32

    @property
    def token_id_dtype(self):
        return torch.int64

    def _create_output(
            self,
            accepted: torch.Tensor,  # [batch_size, k]
            substitute_token_ids: torch.Tensor,  # [batch_size, k]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
            bonus_token_ids: torch.Tensor,  # [batch_size]
    ) -> torch.Tensor:
        """Format output. Returns a matrix of token ids. When
        a token is rejected via sampling, all subsequent token ids are 
        set to -1 for the sequence.

        Args:
            accepted: A boolean tensor indicating if the corresponding
            draft token in draft_token_ids should be accepted or not.
            substitute_token_ids: A tensor of token_ids that can be used
            as substitutes for the draft token ids if the proposed token
            is rejected.
            draft_token_ids: A tensor of token ids speculated by the 
            draft model.
            bonus_token_ids: Token ids to use as the bonus token if
            all the draft tokens are accepted.
        Returns:
            A tensor containing the accepted token ids. The shape of the 
            tensor is [batch_size, k + num_bonus_tokens]
        """
        batch_size, k = substitute_token_ids.shape
        bonus_token_ids = bonus_token_ids.squeeze(-1)
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
            substitute_token_ids.mul(after_false_mask))

        self.num_accepted_tokens += accepted.sum()
        self.num_emitted_tokens += (output_with_bonus_tokens != -1).sum()
        self.num_draft_tokens += batch_size * k

        return output_with_bonus_tokens

    def _raise_if_incorrect_input(
        self,
        target_with_bonus_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: Optional[torch.Tensor] = None,
    ) -> None:
        self._raise_if_incorrect_shape(target_with_bonus_probs,
                                       draft_token_ids, bonus_token_ids,
                                       draft_probs)
        self._raise_if_incorrect_dtype(target_with_bonus_probs,
                                       draft_token_ids, bonus_token_ids,
                                       draft_probs)
        self._raise_if_inconsistent_device(target_with_bonus_probs,
                                           draft_token_ids, bonus_token_ids,
                                           draft_probs)
        self._raise_if_out_of_bounds_vocab(target_with_bonus_probs.shape[-1],
                                           draft_token_ids, bonus_token_ids)

    def _raise_if_incorrect_shape(
        self,
        target_with_bonus_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: Optional[torch.Tensor] = None,
    ) -> None:
        (target_batch_size, num_target_probs,
         target_vocab_size) = target_with_bonus_probs.shape

        # Does not count the extra token
        num_target_probs -= 1

        # validate the shape of draft token ids.
        draft_token_ids_batch_size, num_draft_token_ids = draft_token_ids.shape
        assert draft_token_ids_batch_size == target_batch_size
        assert num_draft_token_ids == num_target_probs

        # validate the shape of bonus token ids
        bonus_batch_size, num_bonus_tokens = bonus_token_ids.shape
        assert bonus_batch_size == target_batch_size
        assert num_bonus_tokens == self._num_bonus_tokens

        # validate the shape of draft probs if it is set
        if draft_probs is not None:
            (draft_batch_size, num_draft_probs,
             draft_vocab_size) = draft_probs.shape
            assert draft_batch_size == target_batch_size
            assert num_draft_probs == num_target_probs
            assert (draft_vocab_size == target_vocab_size
                    ), f"{draft_vocab_size=} {target_vocab_size=}"

    def _raise_if_incorrect_dtype(
        self,
        target_with_bonus_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: Optional[torch.Tensor] = None,
    ) -> None:
        assert target_with_bonus_probs.dtype == self.probs_dtype
        assert draft_token_ids.dtype == self.token_id_dtype
        assert bonus_token_ids.dtype == self.token_id_dtype
        if draft_probs is not None:
            assert draft_probs.dtype == self.probs_dtype

    def _raise_if_inconsistent_device(
        self,
        target_with_bonus_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: Optional[torch.Tensor] = None,
    ) -> None:
        devices = [
            t.device for t in [
                target_with_bonus_probs, bonus_token_ids, draft_probs,
                draft_token_ids
            ] if t is not None
        ]
        assert all([devices[0] == device for device in devices])

    def _raise_if_out_of_bounds_vocab(
        self,
        vocab_size: int,
        draft_token_ids: torch.Tensor,
        bonus_token_ids: torch.Tensor,
    ) -> None:
        assert torch.all(bonus_token_ids < vocab_size)
        assert torch.all(bonus_token_ids >= 0)
        assert torch.all(draft_token_ids < vocab_size)
        assert torch.all(draft_token_ids >= 0)


class SpecDecodeDeterministicBaseSampler(SpecDecodeBaseSampler):
    """Base class for samplers used for Speculative Decoding verification
       step which are deterministic.
    """

    @abstractmethod
    def forward(
        self,
        target_with_bonus_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class SpecDecodeStochasticBaseSampler(SpecDecodeBaseSampler):
    """Base class for samplers used for Speculative Decoding verification
       step which are stochastic
    """

    @abstractmethod
    def forward(
        self,
        target_with_bonus_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        seeded_seqs: Optional[Dict[int, torch.Generator]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
