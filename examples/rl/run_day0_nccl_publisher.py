# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the day0-kit publisher with the current NCCL compatibility adapter."""

import runpy
import sys
from pathlib import Path

from day0_nccl_compat import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)

import vllm.distributed.weight_transfer.nccl_engine as nccl_engine

nccl_engine.NCCLTrainerSendWeightsArgs = NCCLTrainerSendWeightsArgs
nccl_engine.NCCLWeightTransferEngine = NCCLWeightTransferEngine
nccl_engine.NCCLWeightTransferUpdateInfo = NCCLWeightTransferUpdateInfo

kit_script = Path(sys.argv[1])
sys.path.insert(0, str(kit_script.parent))
sys.argv = [str(kit_script), *sys.argv[2:]]
runpy.run_path(str(kit_script), run_name="__main__")
