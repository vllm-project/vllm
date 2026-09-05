# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.transformers.multimodal import MultiModalMixin


class _ImageOnlyMRoPEModel:
    def get_rope_index(self, input_ids, image_grid_thw):
        seq_len = input_ids.shape[-1]
        positions = torch.arange(seq_len).view(1, 1, -1).expand(4, 1, -1)
        return positions, torch.tensor([0])


def test_get_mrope_input_positions_filters_unsupported_grid_kwargs():
    mixin = SimpleNamespace(model=_ImageOnlyMRoPEModel())

    positions, delta = MultiModalMixin.get_mrope_input_positions(mixin, [1, 2, 3], [])

    assert positions.shape == (4, 3)
    assert delta == 0
