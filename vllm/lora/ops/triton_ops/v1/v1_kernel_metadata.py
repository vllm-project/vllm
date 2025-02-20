# SPDX-License-Identifier: Apache-2.0
"""
V1 LoRA kernels metadata preparation utilities.
"""

from dataclasses import dataclass
from typing import Tuple, Union

import torch


@dataclass
class V1KernelMeta:
    token_lora_mapping: torch.Tensor
    token_indices_sorted_by_lora_ids: torch.Tensor
    active_lora_ids: torch.Tensor
    num_tokens_per_lora: torch.Tensor
    lora_token_start_loc: torch.Tensor

    @staticmethod
    def make(max_loras: int, max_num_tokens: int,
             device: Union[torch.device, str]) -> "V1KernelMeta":

        token_lora_mapping = torch.empty(max_num_tokens,
                                         dtype=torch.int32,
                                         device=device)

        token_indices_sorted_by_lora_ids = torch.empty(max_num_tokens,
                                                       dtype=torch.int32,
                                                       device=device)

        # +1 because "no-lora" is also a possibility
        # example: let max_loras be 3, active_lora_ids of [-1, 0, 2, 1]
        # is a possibility.
        active_lora_ids = torch.empty(max_loras + 1,
                                      dtype=torch.int32,
                                      device=device)

        # using running example, [3, 10, 5, 2] is a possibility.
        num_tokens_per_lora = torch.zeros(max_loras + 1,
                                          dtype=torch.int32,
                                          device=device)

        # +2 for this because, the first index is always 0.
        # using running example, lora_token_start_loc
        # is [0, 3, 13, 18, 20].
        lora_token_start_loc = torch.zeros(max_loras + 2,
                                           dtype=torch.int32,
                                           device=device)
        return V1KernelMeta(
            token_lora_mapping=token_lora_mapping,
            token_indices_sorted_by_lora_ids=token_indices_sorted_by_lora_ids,
            active_lora_ids=active_lora_ids,
            num_tokens_per_lora=num_tokens_per_lora,
            lora_token_start_loc=lora_token_start_loc)

    def _reset(self):
        self.active_lora_ids.fill_(-1)
        self.num_tokens_per_lora.fill_(0)
        self.lora_token_start_loc.fill_(0)

    def prepare_tensors(self, token_lora_mapping: torch.Tensor) -> None:
        """
        Prepare kernel metadata tensors for the current forward pass.

        Args:
            token_lora_tensor (torch.Tensor): Tensor containing lora indices
            for each input token.
        """

        self._reset()

        num_tokens = token_lora_mapping.size(0)

        # copy token lora mapping
        self.token_lora_mapping[:num_tokens].copy_(token_lora_mapping,
                                                   non_blocking=True)

        # token_indices_sorted_by_lora_ids
        _, token_indices_sorted_by_lora_ids = torch.sort(token_lora_mapping,
                                                         stable=True)
        # start gpu transfer
        self.token_indices_sorted_by_lora_ids[:num_tokens].copy_(
            token_indices_sorted_by_lora_ids, non_blocking=True)

        # active_lora_ids, num_tokens_per_lora
        lora_ids, num_tokens_per_lora = torch.unique(token_lora_mapping,
                                                     sorted=False,
                                                     return_counts=True)
        self.active_lora_ids[:lora_ids.size(0)].copy_(lora_ids,
                                                      non_blocking=True)
        self.num_tokens_per_lora[:num_tokens_per_lora.size(0)].copy_(
            num_tokens_per_lora, non_blocking=True)

        # lora_token_start_loc
        lora_token_start_loc = torch.cumsum(num_tokens_per_lora, dim=0)
        self.lora_token_start_loc[1:1 + lora_token_start_loc.size(0)].copy_(
            lora_token_start_loc, non_blocking=True)

    def meta_args(
        self, token_nums: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """
        This function returns the kernel metadata required for the current
        forward pass execution of the kernel. The function returns all the
        metadata required by the kernel, in order, as a tuple, so it can be
        unpacked directly during the v1_shrink/v1_expand function call.

        Args:
            token_nums (int): Number of input tokens in the current forward
            pass. 
        """
        return (self.token_lora_mapping[:token_nums],
                self.token_indices_sorted_by_lora_ids[:token_nums],
                self.num_tokens_per_lora, self.lora_token_start_loc,
                self.active_lora_ids)
