# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.v1.worker.gpu.mm import rope


class _MRoPEModel(SupportsMRoPE):
    def get_mrope_input_positions(self, input_tokens, mm_features):
        raise NotImplementedError


def test_get_rope_state_uses_configured_mrope_dimensions():
    model_config = SimpleNamespace(
        uses_mrope=True,
        mrope_num_dims=4,
        uses_xdrope_dim=0,
    )

    with patch.object(rope, "RopeState") as rope_state_cls:
        rope.get_rope_state(
            model_config,
            _MRoPEModel(),
            max_num_reqs=2,
            max_num_tokens=8,
            max_model_len=16,
            device=torch.device("cpu"),
        )

    rope_state_cls.assert_called_once_with(
        num_dims=4,
        has_delta=True,
        max_num_reqs=2,
        max_num_tokens=8,
        max_model_len=16,
        device=torch.device("cpu"),
    )
