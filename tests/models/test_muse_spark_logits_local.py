# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.models.muse_spark.amd.model import MuseSpark1xForCausalLM
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xNormPreservingResidualConnection,
)


def test_compute_logits_local_applies_output_transform() -> None:
    model = MuseSpark1xForCausalLM.__new__(MuseSpark1xForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
    model.logits_processor = lambda lm_head, h, skip_gather=False: lm_head(h)
    model.metap_output_multiplier = 0.5
    model.output_soft_cap_temp = 20.0
    model.mul_tanh_soft_cap = None
    hidden = torch.randn(3, 4, dtype=torch.bfloat16) * 50

    torch.testing.assert_close(
        model.compute_logits_local(hidden), model.compute_logits(hidden)
    )


def test_residual_coefficients_refilled_in_place() -> None:
    """Every load rebuilds the gated-residual coefficients in place."""
    connection = MuseSpark1xNormPreservingResidualConnection(4)
    buffers = []
    for gate in (0.1, 0.5):  # cold start, reload
        with torch.no_grad():
            connection.gate.fill_(gate)
        connection.materialize_coefficients()
        buffers.append((connection.alpha, connection.beta))
        alpha, beta = connection.coefficients(connection.gate / connection.temperature)
        assert torch.equal(connection.alpha, alpha)
        assert torch.equal(connection.beta, beta)
    assert buffers[0][0] is buffers[1][0] and buffers[0][1] is buffers[1][1]
