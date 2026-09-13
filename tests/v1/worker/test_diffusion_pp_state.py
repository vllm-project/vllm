# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP support for diffusion models: the need-sampled mask must be rank-agnostic
when forced, and canvas init must be deterministic so all PP ranks agree."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.pp_utils import PPHandler, compute_need_sampled_mask


def test_need_sampled_mask_always():
    """always=True must mark every request without reading worker-local state,
    which diverges across PP ranks for diffusion models."""
    batch = SimpleNamespace(num_reqs=3)
    mask = compute_need_sampled_mask(batch, always=True)
    assert mask is not None and mask.all() and len(mask) == 3


@pytest.mark.parametrize("always", [False, True])
def test_broadcast_drafts_nonfinal_prefill(monkeypatch, always):
    handler = object.__new__(PPHandler)
    handler.is_last_rank = True
    handler.always_need_sampled = always
    handler.last_rank = 1
    handler.broadcast_group = Mock()
    handler.broadcast_stream = Mock()
    handler.main_stream = Mock()
    batch = SimpleNamespace(
        num_reqs=1,
        num_computed_tokens_np=np.array([0]),
        num_scheduled_tokens=np.array([2]),
        prefill_len_np=np.array([8]),
        idx_mapping=torch.tensor([1]),
    )
    broadcast = Mock()
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    drafts = torch.tensor([[11, 12], [21, 22]])

    handler.broadcast_drafts(drafts, batch)

    assert broadcast.call_count == int(always)
    if always:
        torch.testing.assert_close(broadcast.call_args.args[0], drafts[[1]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_init_canvas_deterministic_per_seed():
    """Same per-slot seed must produce the identical canvas on repeated init
    (the cross-rank agreement contract); different seeds must differ."""
    from vllm.model_executor.models.diffusion_gemma import (
        DiffusionGemmaRequestStates,
    )

    states = DiffusionGemmaRequestStates(
        max_num_reqs=2,
        canvas_length=16,
        vocab_size=1024,
        max_denoising_steps=8,
        device=torch.device("cuda"),
        hidden_size=8,
        stability_threshold=2,
    )
    states.add_request(0, seed=1234)
    states.add_request(1, seed=5678)
    first = states.canvas[:2].clone()
    states.init_canvas(np.array([0, 1]))
    torch.testing.assert_close(states.canvas[:2], first)
    assert not torch.equal(states.canvas[0], states.canvas[1])
