# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.plugins.hidden_states_processors.interface import (
    HiddenStatesProcessor)


class IdentityHiddenStatesProcessor(HiddenStatesProcessor):

    def apply(self, data: torch.Tensor) -> Any:
        """
        This is the default identity hidden states processor
        that returns the hidden_states data as is
        """
        return data
