# SPDX-License-Identifier: Apache-2.0
import torch


# NOTE(woosuk): torch.compile generates faster softmax kernels.
def compiled_softmax(logits: torch.Tensor) -> torch.Tensor:
    torch._dynamo.mark_dynamic(logits, index=0)
    return _softmax(logits)


@torch.compile
def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1, dtype=torch.float32)
