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
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """

        # We only need to set the prompt hidden states once.
        # TODO: check logprobs
        assert self.prompt_hidden_states is None

        self.prompt_hidden_states = prompt_hidden_states_tensor

    def pop_prompt_hidden_states(self) -> Optional[PromptLogprobs]:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_hidden_states
        if plp:
            self.prompt_hidden_states = None
        return plp

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.prompt_hidden_states is not None:
            print("lxy update_from_output")
            self._set_prompt_hidden_states(output.prompt_hidden_states)
