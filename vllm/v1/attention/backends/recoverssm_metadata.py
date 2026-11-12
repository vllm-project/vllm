# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


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


class RecoverSSMMetadataBuilder:
    """A builder of `RecoverSSMMetadata`. With `use_recoverssm`, the runner binds
    its `RecoverSSMState` before every build, capture builds included: the
    pending replays per request slot and the step's request slots."""

    use_recoverssm: bool
    recoverssm_state: "RecoverSSMState | None" = None

    def fold_pending(self, slots: torch.Tensor, pending: torch.Tensor) -> None:
        """Fold the pending replays of request slots `slots` (requests leaving the
        runner) into their checkpoints."""
        raise NotImplementedError
