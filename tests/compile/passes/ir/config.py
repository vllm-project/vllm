# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.ir import ops

from .common import LoweringTestConfig

CONFIGS = {
    "rms_norm": LoweringTestConfig(
        op=ops.rms_norm,
        inputs=[[torch.randn((2, 16)), torch.randn((1, 16)), 1e-5, None]],
        batched_args=["x"],
    )
}
