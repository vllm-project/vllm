# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 MTP: the gathered lm_head weight that captured CUDA graphs read keeps
its address across weight reloads."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.models import gemma4_mtp


def test_gathered_lm_head_is_refilled_in_place(monkeypatch):
    monkeypatch.setattr(gemma4_mtp, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        gemma4_mtp,
        "tensor_model_parallel_all_gather",
        lambda t, dim: torch.cat([t, t], dim=dim),
    )
    model = gemma4_mtp.Gemma4MTP.__new__(gemma4_mtp.Gemma4MTP)
    nn.Module.__init__(model)
    model.lm_head = nn.Linear(4, 3, bias=False)
    model.masked_embedding = SimpleNamespace(vocab_size=5)
    model._stable_full_lm_head_weight = None
    model._suppress_token_ids = [1]
    model._suppress_idx = None

    with torch.no_grad():
        model.lm_head.weight.fill_(1.0)
        model.process_weights_after_loading()  # cold start
    gathered = model._stable_full_lm_head_weight
    suppress_idx = model._suppress_idx
    assert gathered is not None and gathered.shape == (5, 4)
    with torch.inference_mode():  # graph capture and serving read it
        assert model._get_full_lm_head_weight() is gathered

    with torch.no_grad():  # weight reload
        model.lm_head.weight.fill_(2.0)
        model.process_weights_after_loading()
    assert model._stable_full_lm_head_weight is gathered
    assert model._suppress_idx is suppress_idx
    torch.testing.assert_close(gathered, torch.full((5, 4), 2.0))
