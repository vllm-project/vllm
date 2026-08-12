# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP support for diffusion models: the need-sampled mask must be rank-agnostic
when forced, and canvas init must be deterministic so all PP ranks agree."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.pp_utils import compute_need_sampled_mask


def test_need_sampled_mask_always():
    """always=True must mark every request without reading worker-local state,
    which diverges across PP ranks for diffusion models."""
    batch = SimpleNamespace(num_reqs=3)
    mask = compute_need_sampled_mask(batch, always=True)
    assert mask is not None and mask.all() and len(mask) == 3


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
