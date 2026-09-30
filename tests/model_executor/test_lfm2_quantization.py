# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import torch


def test_silu_mul_quant_fusion_uses_registry():
    from vllm.model_executor.models.lfm2 import Lfm2MLP

    projected = torch.empty((1, 1), dtype=torch.bfloat16)
    fused = Mock()
    act_fn = Mock()
    w2 = Mock(side_effect=lambda x: (x, None))

    mlp = Lfm2MLP.__new__(Lfm2MLP)
    torch.nn.Module.__init__(mlp)
    mlp.w13 = Mock(return_value=(projected, None))
    mlp.w2 = w2
    mlp.act_fn = act_fn

    with patch(
        "vllm.model_executor.models.lfm2.maybe_fused_act_quant",
        return_value=fused,
    ) as maybe_fused:
        result = mlp(Mock())

    maybe_fused.assert_called_once_with(act_fn, projected, w2)
    act_fn.assert_not_called()
    assert result is fused
