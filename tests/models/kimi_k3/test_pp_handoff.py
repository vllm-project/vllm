# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AttnRes delta handling at a Kimi-K3 pipeline-stage boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia.model import KimiLinearModel


def _run_first_layer(monkeypatch, *, is_first_rank, is_block_write_layer):
    start_layer = 0 if is_first_rank else 4
    prefix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    layer = Mock(
        return_value=(
            torch.zeros_like(prefix),
            prefix,
            torch.zeros(prefix.size(0), 1, prefix.size(1)),
        )
    )
    layer.is_block_write_layer = is_block_write_layer

    model = object.__new__(KimiLinearModel)
    object.__setattr__(model, "use_sequence_parallel", False)
    object.__setattr__(model, "use_attn_res", True)
    object.__setattr__(model, "num_attn_res_blocks", 1)
    object.__setattr__(model, "aux_hidden_state_layers", ())
    object.__setattr__(model, "start_layer", start_layer)
    object.__setattr__(model, "end_layer", start_layer + 1)
    object.__setattr__(model, "layers", [None] * start_layer + [layer])
    monkeypatch.setattr(
        kimi_model,
        "get_pp_group",
        lambda: SimpleNamespace(
            is_first_rank=is_first_rank,
            is_last_rank=False,
        ),
    )

    model.forward(
        input_ids=None,
        positions=torch.tensor([0, 1]),
        intermediate_tensors={"hidden_states": prefix, "residual": None},
        inputs_embeds=prefix if is_first_rank else None,
    )
    return layer.call_args.kwargs["hidden_states"], prefix


def test_mid_block_stage_start_zeroes_pending_delta(monkeypatch):
    delta, prefix = _run_first_layer(
        monkeypatch,
        is_first_rank=False,
        is_block_write_layer=False,
    )

    assert delta is not None
    assert delta.shape == prefix.shape
    assert delta.dtype == prefix.dtype
    assert not delta.any()


@pytest.mark.parametrize(
    "is_first_rank,is_block_write_layer",
    [
        (False, True),
        (True, True),
    ],
)
def test_block_boundary_does_not_create_delta(
    monkeypatch,
    is_first_rank,
    is_block_write_layer,
):
    delta, _ = _run_first_layer(
        monkeypatch,
        is_first_rank=is_first_rank,
        is_block_write_layer=is_block_write_layer,
    )

    assert delta is None
