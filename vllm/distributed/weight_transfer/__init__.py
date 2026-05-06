# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Weight transfer engines for syncing model weights from trainers
to inference workers.
"""

from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

__all__ = [
    "WeightTransferEngineFactory",
]
