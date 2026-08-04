# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING
import abc

# Third Party
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.compute.attention.metadata import LMCAttnMetadata


class AttentionInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCAttnMetadata",
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform forward pass of the attention mechanism.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_attn_metadata(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> "LMCAttnMetadata":
        """
        Initialize attention metadata.
        """
        raise NotImplementedError
