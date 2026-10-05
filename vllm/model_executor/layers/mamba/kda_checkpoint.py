# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.mamba.checkpoint import (
    ConvRecurrentCheckpointExporter,
)


def kda_prefill_checkpoint_alignment(backend: str) -> int | None:
    return 16 if backend == "flashkda" else None


class FlashKDAPrefillCheckpointExporter(ConvRecurrentCheckpointExporter):
    """Store FlashKDA recurrent and convolution checkpoint states."""
