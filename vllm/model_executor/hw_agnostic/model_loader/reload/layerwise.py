# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export of ``initialize_online_processing``.

A separate copy here would shadow the ``LAYERWISE_INFO`` registry that
``finalize_layerwise_processing`` reads at the end of weight loading —
the BF16->FP8 online quant pass would silently never run.
"""

from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)

__all__ = ["initialize_online_processing"]
