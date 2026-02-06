# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

import vllm.ir.ops
import vllm.kernels  # noqa: F401 to register kernels
from vllm.compilation.passes.ir.lowering_pass import (
    VllmIRLoweringPass,
)
from vllm.config import get_current_vllm_config

from ...backend import TestBackend


class Model(nn.Module):
    def __init__(self, hidden_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = torch.ones(hidden_size)

    def forward(self, x):
        y = x + 2.0
        z = vllm.ir.ops.rms_norm(y, self.weight, 1e-5, variance_size=4)
        return z + 3.0


@pytest.mark.parametrize("rms_provider", vllm.ir.ops.rms_norm.supported_providers())
def test_lowering_rms_norm(rms_provider, default_vllm_config):
    torch.set_default_device("cuda")

    lowering_pass = VllmIRLoweringPass(get_current_vllm_config())
    backend = TestBackend(lowering_pass)

    model = Model()
    x = torch.randn(8, 16)
    with vllm.ir.ops.rms_norm.set_priority([rms_provider, "native"]):
        output_eager = model(x)
        compiled_model = torch.compile(model, backend=backend, fullgraph=True)
        output_compiled = compiled_model(x)

    # TODO remove print
    backend.print_graphs()

    output_compiled2 = compiled_model(x)
    torch.testing.assert_close(output_eager, output_compiled)
    torch.testing.assert_close(output_eager, output_compiled2)
