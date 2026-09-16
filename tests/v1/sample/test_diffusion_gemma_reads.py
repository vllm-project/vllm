# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request DiffusionGemma state behind structured reads: seed canvases,
read-only slots and the per-slot step cap."""

import numpy as np
import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import DiffusionGemmaRequestStates
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the sampler state lives on the GPU"
)

CL = 8
VOCAB = 64
MAX_REQS = 4
MAX_STEPS = 48


def _states() -> DiffusionGemmaRequestStates:
    return DiffusionGemmaRequestStates(
        max_num_reqs=MAX_REQS,
        canvas_length=CL,
        vocab_size=VOCAB,
        max_denoising_steps=MAX_STEPS,
        device=torch.device("cuda"),
        hidden_size=4,
        stability_threshold=2,
    )


def _slots(*idx: int) -> tuple[np.ndarray, torch.Tensor]:
    slots = np.array(idx, dtype=np.int64)
    return slots, torch.tensor(slots, device="cuda")


def test_seed_canvas_replaces_only_seeded_slots():
    states = _states()
    for slot in range(3):
        states.add_request(slot)
    seed = list(range(CL))
    states.set_seed_canvas(1, seed)
    slots, slots_gpu = _slots(0, 1, 2)
    states.init_canvas(slots_gpu)
    before = states.canvas[slots_gpu].clone()

    states.apply_seed_canvases(slots, slots_gpu)

    after = states.canvas[slots_gpu]
    assert after[1].tolist() == seed
    assert torch.equal(after[0], before[0])
    assert torch.equal(after[2], before[2])


def test_apply_seed_canvases_leaves_unseeded_batches_alone():
    states = _states()
    states.add_request(0)
    slots, slots_gpu = _slots(0)
    states.init_canvas(slots_gpu)
    before = states.canvas[0].clone()

    states.apply_seed_canvases(slots, slots_gpu)

    assert torch.equal(states.canvas[0], before)


def test_add_request_clears_seed_and_read_only():
    states = _states()
    states.add_request(0)
    states.set_seed_canvas(0, [1] * CL)
    states.set_read_only(0)
    assert states.seeded_slots == {0}
    assert states.read_only_slots == {0}

    states.add_request(0)

    assert not states.seeded_slots
    assert not states.read_only_slots
    assert not bool(states.has_seed[0])
    assert not bool(states.read_only[0])


def test_remove_request_forgets_the_slot():
    states = _states()
    states.add_request(0)
    states.set_seed_canvas(0, [1] * CL)
    states.set_read_only(0)

    states.remove_request(0)

    assert not states.seeded_slots
    assert not states.read_only_slots
