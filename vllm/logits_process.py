# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import TypeAlias

import torch

LogitsProcessor: TypeAlias = (
    Callable[[list[int], torch.Tensor], torch.Tensor]
    | Callable[[list[int], list[int], torch.Tensor], torch.Tensor]
)
"""LogitsProcessor is a function that takes a list
of previously generated tokens, the logits tensor
for the next token and, optionally, prompt tokens as a
first argument, and returns a modified tensor of logits
to sample from."""
