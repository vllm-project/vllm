# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import torch

from vllm.utils.platform_utils import is_uva_available


class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError(
                "UVA (unified virtual addressing) is not available: it "
                "requires pinned host memory, which this platform does not "
                "support (e.g. vLLM disables pin_memory on WSL). The V2 "
                "model runner depends on UVA; set VLLM_USE_V2_MODEL_RUNNER=0 "
                "to fall back to the V1 model runner."
            )
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        self.uva = self.cpu
