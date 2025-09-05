# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.sequence import PromptLogprobs
from vllm.v1.engine import EngineCoreOutput

logger = init_logger(__name__)

NONES = itertools.repeat(None)


@dataclass
class HiddenStatesProcessor:
    prompt_hidden_states: Optional[torch.Tensor]

    @classmethod
    def from_new_request(cls, ) -> "HiddenStatesProcessor":
        return cls(prompt_hidden_states=None)

    def _set_prompt_hidden_states(
        self,
        prompt_hidden_states_tensor: torch.Tensor,
    ) -> None:
        # We only need to set the prompt hidden states once.
        assert self.prompt_hidden_states is None

        self.prompt_hidden_states = prompt_hidden_states_tensor

    def pop_prompt_hidden_states(self) -> Optional[PromptLogprobs]:
        """Pop and return all request prompt hidden states

        The hidden states processor aggregates prompt chunk hidden states
        over one or more prefill chunks. This method returns
        all prompt hidden states at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt hidden states are returned at once at
        the end of prefill.

        Returns:
          None if prompt hidden states are disabled for this request.
          List of all prompt hidden states, otherwise.
        """
        plp = self.prompt_hidden_states
        if plp:
            self.prompt_hidden_states = None
        return plp

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.prompt_hidden_states is not None:
            self._set_prompt_hidden_states(output.prompt_hidden_states)
