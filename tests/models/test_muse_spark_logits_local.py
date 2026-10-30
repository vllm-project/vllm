# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.models.muse_spark.amd.model import MuseSpark1xForCausalLM


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
