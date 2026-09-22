# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export the weight-transfer backend maintained by ModelExpress."""

from modelexpress import configure_vllm_logging
from modelexpress_rl.inference.engines.vllm.weight_transfer_engine import (
    ModelExpressWeightTransferEngine,
    ModelExpressWeightTransferInitInfo,
    ModelExpressWeightTransferUpdateInfo,
)

configure_vllm_logging()

__all__ = [
    "ModelExpressWeightTransferEngine",
    "ModelExpressWeightTransferInitInfo",
    "ModelExpressWeightTransferUpdateInfo",
]
