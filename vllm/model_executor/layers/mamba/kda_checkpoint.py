# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointExporter,
)
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

# The Triton chunk kernel sees the recurrent state only at chunk boundaries.
_KDA_PREFILL_CHECKPOINT_ALIGNMENT = {"flashkda": 16, "triton": FLA_CHUNK_SIZE}


def kda_prefill_checkpoint_alignment(backend: str) -> int | None:
    return _KDA_PREFILL_CHECKPOINT_ALIGNMENT.get(backend)


class FlashKDAPrefillCheckpointExporter(MambaPrefillCheckpointExporter):
    """Store KDA (FlashKDA or Triton) recurrent and convolution checkpoint states."""
