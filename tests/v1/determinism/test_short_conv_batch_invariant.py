# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShortConv kernels must not depend on the rest of the batch.

SHORT_CONV_ATTN declares batch invariance on this basis, so a sequence has to
produce bitwise-identical outputs and conv states alone and inside any batch.
"""

import pytest
import torch
from utils import skip_if_not_cuda_alike

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.utils.torch_utils import set_random_seed

DEVICE = "cuda"
DIM, WIDTH, NUM_SLOTS = 2048, 3, 16
TARGET_SLOT = 5


def _inputs():
    set_random_seed(0)
    weight = torch.randn(DIM, WIDTH, device=DEVICE, dtype=torch.bfloat16)
    # (slots, dim, width - 1) with dim contiguous, like the ShortConv cache.
    conv_states = torch.randn(
        NUM_SLOTS, WIDTH - 1, DIM, device=DEVICE, dtype=torch.bfloat16
    ).transpose(1, 2)
    return weight, conv_states


def _prefill(x, weight, conv_states, seq_lens, slots, has_initial_state):
    query_start_loc = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )
    return causal_conv1d_fn(
        x.transpose(0, 1),
        weight,
        None,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
        has_initial_state=torch.tensor(has_initial_state, device=DEVICE),
        activation=None,
    ).transpose(0, 1)


@skip_if_not_cuda_alike
@pytest.mark.parametrize("other_lens", [[5], [64, 1, 130], [1] * 7])
@pytest.mark.parametrize("has_initial_state", [False, True])
def test_short_conv_prefill_is_batch_invariant(other_lens, has_initial_state):
    weight, conv_states = _inputs()
    target_len = 37
    target = torch.randn(target_len, DIM, device=DEVICE, dtype=torch.bfloat16)

    solo_states = conv_states.clone()
    solo = _prefill(
        target, weight, solo_states, [target_len], [TARGET_SLOT], [has_initial_state]
    )

    # Put the target second so its tokens sit at a different offset.
    lens = [other_lens[0], target_len, *other_lens[1:]]
    others = [
        torch.randn(n, DIM, device=DEVICE, dtype=torch.bfloat16) for n in other_lens
    ]
    x = torch.cat([others[0], target, *others[1:]])
    slots = [s for s in range(1, NUM_SLOTS) if s != TARGET_SLOT][: len(lens)]
    slots[1] = TARGET_SLOT
    init = [True] * len(lens)
    init[1] = has_initial_state
    batch_states = conv_states.clone()
    batched = _prefill(x, weight, batch_states, lens, slots, init)

    start = other_lens[0]
    assert torch.equal(solo, batched[start : start + target_len])
    assert torch.equal(solo_states[TARGET_SLOT], batch_states[TARGET_SLOT])


@skip_if_not_cuda_alike
@pytest.mark.parametrize("batch_size", [2, 8, 15])
def test_short_conv_decode_is_batch_invariant(batch_size):
    weight, conv_states = _inputs()
    x = torch.randn(batch_size, DIM, device=DEVICE, dtype=torch.bfloat16)
    slots = torch.arange(1, batch_size + 1, dtype=torch.int32, device=DEVICE)
    row = batch_size // 2

    # causal_conv1d_update writes its output into x, so give each call a copy.
    solo_states = conv_states.clone()
    solo = causal_conv1d_update(
        x[row : row + 1].clone(),
        solo_states,
        weight,
        None,
        activation=None,
        conv_state_indices=slots[row : row + 1],
    )
    batch_states = conv_states.clone()
    batched = causal_conv1d_update(
        x.clone(), batch_states, weight, None, activation=None, conv_state_indices=slots
    )

    slot = slots[row].item()
    assert torch.equal(solo[0], batched[row])
    assert torch.equal(solo_states[slot], batch_states[slot])
