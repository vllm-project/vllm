# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.ir import ops
from vllm.platforms import current_platform

from .common import LoweringTestConfig

torch.set_default_device(current_platform.device_type)

PER_OP_LOWERING_TEST_CONFIGS = {
    "rms_norm": LoweringTestConfig(
        op=ops.rms_norm,
        inputs=[ops.rms_norm.generate_inputs(num_tokens=10, hidden_size=128)],
        unbacked_idx={"num_tokens": [0]},
    )
}
