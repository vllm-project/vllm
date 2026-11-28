# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mid-sequence checkpoints of the FLA chunked gated delta rule.

Align-mode prefix caching stores these snapshots as the state of a cache block
inside the prefill, so a wrong one is served silently on a later prefix hit.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="needs a GPU",
)

NUM_K_HEADS, NUM_V_HEADS, HEAD_DIM = 2, 4, 128


def _run(inputs, seq_lens, **kwargs):
    q, k, v, g, beta = inputs
    cu_seqlens = torch.tensor([0, *seq_lens]).cumsum(0).to(torch.int32)
    return chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        **kwargs,
    )


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_checkpoints_are_states_at_their_offsets(
    state_dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row n of the buffer is sequence n's final state cut at its offset.

    Only offsets on the 64-token chunk grid strictly inside a sequence are
    stored: 0 means none, 128 is the whole 128-token sequence and 100 is off
    the grid, and their rows are left untouched. Storing checkpoints leaves the
    outputs and final states unchanged.
    """
    torch.set_default_device(current_platform.device_type)
    set_random_seed(0)
    seq_lens = [300, 50, 257, 128, 200, 192]
    offsets = [128, 0, 256, 128, 100, 64]
    stored = [0, 2, 5]
    num_tokens = sum(seq_lens)
    q, k = (
        F.normalize(
            torch.randn(1, num_tokens, NUM_K_HEADS, HEAD_DIM), dim=-1
        ).bfloat16()
        for _ in range(2)
    )
    v = torch.randn(1, num_tokens, NUM_V_HEADS, HEAD_DIM).bfloat16()
    g = F.logsigmoid(torch.randn(1, num_tokens, NUM_V_HEADS))
    beta = torch.sigmoid(torch.randn(1, num_tokens, NUM_V_HEADS))
    inputs = (q, k, v, g, beta)
    initial_state = 0.1 * torch.randn(len(seq_lens), NUM_V_HEADS, HEAD_DIM, HEAD_DIM)
    initial_state[2] = 0

    snapshots = torch.full_like(initial_state, 7.0, dtype=state_dtype)
    o, final_state = _run(
        inputs,
        seq_lens,
        initial_state=initial_state,
        checkpoint_state=snapshots,
        checkpoint_offsets=torch.tensor(offsets, dtype=torch.int32),
    )

    autotuner = chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
    monkeypatch.setattr(autotuner, "configs", [autotuner.best_config])
    monkeypatch.setattr(autotuner, "cache", {})
    monkeypatch.setattr(autotuner, "cache_results", False)
    o_ref, final_ref = _run(inputs, seq_lens, initial_state=initial_state)
    torch.testing.assert_close(o, o_ref, atol=0, rtol=0)
    torch.testing.assert_close(final_state, final_ref, atol=0, rtol=0)
    skipped = [i for i in range(len(seq_lens)) if i not in stored]
    assert (snapshots[skipped] == 7.0).all()
    starts = [0, *torch.tensor(seq_lens).cumsum(0).tolist()]
    for i in stored:
        start, offset = starts[i], offsets[i]
        _, expected = _run(
            tuple(x[:, start : start + offset] for x in inputs),
            [offset],
            initial_state=initial_state[i : i + 1],
        )
        torch.testing.assert_close(
            snapshots[i], expected[0].to(state_dtype), atol=0, rtol=0
        )


def test_checkpoints_reject_a_noncontiguous_output_buffer():
    """input_guard must not redirect checkpoint writes into a temporary copy."""
    torch.set_default_device(current_platform.device_type)
    q = torch.zeros(1, 128, NUM_K_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.zeros(1, 128, NUM_V_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    g = torch.zeros(1, 128, NUM_V_HEADS)
    snapshots = torch.empty(1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM + 8)[..., :HEAD_DIM]
    with pytest.raises(ValueError, match="contiguous"):
        _run(
            (q, q, v, g, g),
            [128],
            checkpoint_state=snapshots,
            checkpoint_offsets=torch.tensor([64], dtype=torch.int32),
        )


def test_offset_checkpoint_metadata_reuses_warmed_kernel(monkeypatch):
    """A decode prefix makes the prefill metadata an offset view."""
    torch.set_default_device(current_platform.device_type)
    q = torch.zeros(1, 257, NUM_K_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.zeros(1, 257, NUM_V_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    g = torch.zeros(1, 257, NUM_V_HEADS)
    metadata = {
        "cu_seqlens": torch.tensor([0, 257], dtype=torch.int32),
        "checkpoint_offsets": torch.tensor([128], dtype=torch.int32),
    }
    snapshots = torch.empty(1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM)

    def run():
        return chunk_gated_delta_rule(
            q, q, v, g, g, checkpoint_state=snapshots, **metadata
        )

    expected, _ = run()
    expected_snapshot = snapshots.clone()
    for name, tensor in metadata.items():
        view = tensor.new_empty(tensor.numel() + 1)[1:]
        view.copy_(tensor)
        metadata[name] = view
    compiled = []

    def record_compile(**kwargs):
        compiled.append(getattr(kwargs.get("fn"), "name", None))

    monkeypatch.setattr(triton.knobs.runtime, "jit_post_compile_hook", record_compile)
    actual, _ = run()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(snapshots, expected_snapshot, atol=0, rtol=0)
    assert "chunk_gated_delta_rule_fwd_kernel_h_blockdim64" not in compiled
