# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)

from .modular_kernel_tools.common import Config
from .modular_kernel_tools.profile_modular_kernel import run


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="profile_modular_kernel requires a CUDA device",
)
def test_profile_modular_kernel_smoke(tmp_path):
    config = Config(
        Ms=[16],
        K=128,
        N=256,
        E=4,
        topks=[2],
        dtype=torch.bfloat16,
        quant_config=None,
        prepare_finalize_type=MoEPrepareAndFinalizeNoDPEPModular,
        fused_experts_type=TritonExperts,
        world_size=1,
        torch_trace_dir_path=str(tmp_path),
    )

    run(config)

    traces = list(tmp_path.glob("m*_*_trace.json"))
    assert traces, "profile_modular_kernel.run did not emit any chrome traces"
