# SPDX-License-Identifier: Apache-2.0
from typing import Union

import torch


def compiled_softmax(
    logits: torch.Tensor,
    temperature: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """Faster softmax kernel generated by torch.compile.

    Args:
        logits: [n, vocab_size]
        temperature: [n] or float
    """
    # NOTE(woosuk): Avoid recompilation by marking the first dim as dynamic.
    torch._dynamo.mark_dynamic(logits, index=0)
    if isinstance(temperature, torch.Tensor):
        torch._dynamo.mark_dynamic(temperature, index=0)
    return _softmax(logits, temperature)


@torch.compile
def _softmax(
    logits: torch.Tensor,
    temperature: Union[float, torch.Tensor],
) -> torch.Tensor:
    logits = logits / temperature
    return torch.softmax(logits, dim=-1, dtype=torch.float32)
