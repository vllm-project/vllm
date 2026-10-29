# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.distributed import parallel_state
from vllm.model_executor.models import utils as model_utils
from vllm.models.deepseek_v32 import attention as deepseek_v32_attention
from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
from vllm.sequence import IntermediateTensors

# GLM-5.3: each group of 4 layers from layer 3 reuses its first layer's top-k.
GLM5_TOPK = dict(index_topk_freq=4, index_skip_topk_offset=3)


def _set_pp_size(monkeypatch, pp_size: int) -> None:
    monkeypatch.setattr(
        parallel_state, "get_pp_group", lambda: SimpleNamespace(world_size=pp_size)
    )


class TestPPSplitsLayerSharing:
    @pytest.mark.parametrize(
        "num_layers,producers,consumers,pp_size,expected",
        [
            (8, {0, 4}, {1, 2, 3, 5, 6, 7}, 2, False),
            (8, {0, 3}, {1, 2, 4, 5, 6, 7}, 2, True),
            (8, {0, 3}, {1, 2, 4, 5, 6, 7}, 1, False),
            # Layers 4-5 neither produce nor consume, so stage 1 starts at 6.
            (8, {0, 6}, {1, 2, 3, 7}, 2, False),
            (8, {0}, {1, 2, 3, 6, 7}, 2, True),
            (12, {0, 4, 8}, {1, 2, 3, 5, 6, 7, 9, 10, 11}, 3, False),
            (12, {0, 5, 8}, {1, 2, 3, 4, 6, 7, 9, 10, 11}, 3, True),
        ],
        ids=[
            "split-on-producer",
            "split-on-consumer",
            "no-pp",
            "skip-unrelated-layers",
            "unrelated-then-consumer",
            "pp3-aligned",
            "pp3-middle-split",
        ],
    )
    def test_stage_boundaries(
        self, monkeypatch, num_layers, producers, consumers, pp_size, expected
    ):
        _set_pp_size(monkeypatch, pp_size)
        assert (
            model_utils.pp_splits_layer_sharing(num_layers, producers, consumers)
            == expected
        )

    @pytest.mark.parametrize(
        "num_layers,expected",
        [(78, True), (76, False)],
        ids=["glm5-default-split", "split-on-group-start"],
    )
    def test_dsa_topk_buffers(self, monkeypatch, num_layers, expected):
        _set_pp_size(monkeypatch, 2)
        config = SimpleNamespace(
            num_hidden_layers=num_layers, index_topk=8, **GLM5_TOPK
        )
        buffer = torch.empty(4, 8, dtype=torch.int32)
        buffers = deepseek_v32_attention.pp_topk_buffers(config, buffer)
        assert buffers == ({"topk_indices": buffer} if expected else {})


def test_empty_intermediate_tensors_follow_buffers():
    buffer = torch.empty(16, 8, dtype=torch.int32)
    make_empty = model_utils.make_empty_intermediate_tensors_factory(
        ["hidden_states", "residual"], 4, {"topk_indices": buffer}
    )
    tensors = make_empty(3, torch.bfloat16, torch.device("cpu"))

    assert tensors["residual"].shape == (3, 4)
    assert tensors["topk_indices"].shape == (3, 8)
    assert tensors["topk_indices"].dtype == torch.int32


class _TopkReusingLayer(nn.Module):
    """Reads the shared top-k like a reusing layer, then indexes like the next."""

    def __init__(self, topk_indices_buffer: torch.Tensor) -> None:
        super().__init__()
        self.topk_indices_buffer = topk_indices_buffer
        self.seen_topk: torch.Tensor | None = None

    def forward(self, positions, hidden_states, residual, attn_in=None):
        num_tokens = hidden_states.shape[0]
        self.seen_topk = self.topk_indices_buffer[:num_tokens].clone()
        self.topk_indices_buffer[:num_tokens] += 10
        return hidden_states, residual


def test_middle_pp_stage_receives_and_forwards_topk(monkeypatch):
    model = object.__new__(deepseek_v32_model.DeepseekV32Model)
    nn.Module.__init__(model)
    model.topk_indices_buffer = torch.full((8, 2), -1, dtype=torch.int32)
    layer = _TopkReusingLayer(model.topk_indices_buffer)
    model.layers = nn.ModuleList([layer])
    model.start_layer, model.end_layer = 0, 1
    model.use_sequence_parallel = False
    model.aux_hidden_state_layers = ()
    model.pp_buffers = {"topk_indices": model.topk_indices_buffer}
    monkeypatch.setattr(
        deepseek_v32_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )

    received_topk = torch.arange(6, dtype=torch.int32).view(3, 2)
    output = model(
        None,
        torch.arange(3),
        IntermediateTensors(
            {
                "hidden_states": torch.zeros(3, 4),
                "residual": torch.zeros(3, 4),
                "topk_indices": received_topk,
            }
        ),
    )

    torch.testing.assert_close(layer.seen_topk, received_topk)
    torch.testing.assert_close(output["topk_indices"], received_topk + 10)
