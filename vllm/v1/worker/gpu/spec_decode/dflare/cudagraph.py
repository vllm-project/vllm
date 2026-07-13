# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlare CUDA graph manager.

Reuses DFlash's CUDA graph manager since the draft model forward
(query-only pass) is identical between DFlash and DFlare.
"""

from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    DFlashCudaGraphManager as DFlareCudaGraphManager,
)

__all__ = ["DFlareCudaGraphManager"]
