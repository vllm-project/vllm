# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export the weight-transfer backend maintained by ModelExpress.

Install ModelExpress from the main branch's modelexpress_client/python
subdirectory.
"""

try:
    from modelexpress import configure_vllm_logging
    from modelexpress_rl.inference.engines.vllm.weight_transfer_engine import (
        ModelExpressWeightTransferEngine,
        ModelExpressWeightTransferInitInfo,
        ModelExpressWeightTransferUpdateInfo,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {
        "modelexpress",
        "modelexpress_rl",
        "modelexpress_rl.inference",
        "modelexpress_rl.inference.engines",
        "modelexpress_rl.inference.engines.vllm",
        "modelexpress_rl.inference.engines.vllm.weight_transfer_engine",
    }:
        raise
    raise ImportError(
        "The 'modelexpress' weight transfer backend requires ModelExpress. "
        "Install it with `uv pip install "
        "'git+https://github.com/ai-dynamo/modelexpress@main"
        "#subdirectory=modelexpress_client/python'`."
    ) from exc

configure_vllm_logging()

__all__ = [
    "ModelExpressWeightTransferEngine",
    "ModelExpressWeightTransferInitInfo",
    "ModelExpressWeightTransferUpdateInfo",
]
