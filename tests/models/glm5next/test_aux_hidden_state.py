# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash aux hidden-state capture for DFlash2 / EAGLE3 drafters."""

from types import SimpleNamespace

import torch
from transformers import Glm5NextTextConfig

from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
from vllm.models.glm5next.common.model import Glm5NextModel

HIDDEN, STREAMS, TOKENS = 8, 4, 3


def test_aux_hidden_state_completes_and_contracts_mhc_streams():
    """The capture reads the stream count from the upstream config."""
    model = SimpleNamespace(
        config=Glm5NextTextConfig(hidden_size=HIDDEN, hc_mult=STREAMS),
        _aux_post_op=mhc_post_torch,
    )
    x = torch.randn(TOKENS, HIDDEN)
    residual = torch.randn(TOKENS, STREAMS, HIDDEN)
    post = torch.rand(TOKENS, STREAMS, 1)
    comb = torch.rand(TOKENS, STREAMS, STREAMS)

    aux = Glm5NextModel._aux_hidden_state(model, x, residual, post, comb)

    expected = mhc_post_torch(x, residual, post, comb).mean(dim=1)
    torch.testing.assert_close(aux, expected)
