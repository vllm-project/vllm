# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecoverSSMPostprocessMetadata:
    """Metadata used during postprocessing for align-mode prefix caching."""

    num_spec_decodes: int
    request_indices: torch.Tensor | None
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int


class RecoverSSMMetadata(abc.ABC):
    @abc.abstractmethod
    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        raise NotImplementedError
