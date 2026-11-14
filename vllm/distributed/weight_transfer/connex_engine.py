# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Re-export the weight-transfer backend maintained by Connex.

Install Connex from https://github.com/HeterInfra/connex.
"""

try:
    from connex.rl.vllm_backend import (
        ConnexTrainerInitInfo,
        ConnexTrainerWeightTransferEngine,
        ConnexWeightTransferEngine,
        ConnexWeightTransferInitInfo,
        ConnexWeightTransferUpdateInfo,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"connex", "connex.rl", "connex.rl.vllm_backend"}:
        raise
    raise ImportError(
        "The 'connex' weight transfer backend requires Connex. "
        "Install it with `uv pip install 'connex-ai[rs]'`."
    ) from exc

__all__ = [
    "ConnexTrainerInitInfo",
    "ConnexTrainerWeightTransferEngine",
    "ConnexWeightTransferEngine",
    "ConnexWeightTransferInitInfo",
    "ConnexWeightTransferUpdateInfo",
]
