# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The `moe` stage of the MiniMax-M3 specialized decode: the fused NVFP4 routed +
MXFP8 shared-expert engine
(CuTe DSL, `kernel.py`), its host (`host.py`), the weight views and mode-Q prep
(`weights.py`) and the stage
wrapper (`stage.py`). Importing this package does not import CuTe DSL; the kernel
module loads on first build."""

from .stage import FusedMoeConfig, FusedMoeStage, default_n_shared, engine_config
from .weights import FC2_SCALE_Q, MoeWeightError, MoeWeights, from_vllm_moe, prep_mode_q

__all__ = [
    "FC2_SCALE_Q",
    "FusedMoeConfig",
    "FusedMoeStage",
    "MoeWeightError",
    "MoeWeights",
    "default_n_shared",
    "engine_config",
    "from_vllm_moe",
    "prep_mode_q",
]
