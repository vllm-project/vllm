# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import torch


class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()

    def uva(self, n: int | None = None):
        return self.cpu[:n] if n is not None else self.cpu
