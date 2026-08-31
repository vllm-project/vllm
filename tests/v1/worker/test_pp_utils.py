# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PPHandler sampled-token / draft-token relay under PP."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.pp_utils import PPHandler

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PPHandler drives a side CUDA stream"
)


def make_handler(
    monkeypatch,
    *,
    is_last_rank: bool,
    num_speculative_steps: int,
    relay_draft_tokens: bool,
    world_size: int = 2,
) -> PPHandler:
    """Build a real PPHandler with the PP group stubbed out."""
    pp_group = SimpleNamespace(
        is_last_rank=is_last_rank,
        last_rank=world_size - 1,
        world_size=world_size,
        make_sibling_device_group=lambda group_desc: object(),
    )
    monkeypatch.setattr(pp_utils, "get_pp_group", lambda: pp_group)
    return PPHandler(
        max_num_reqs=8,
        num_speculative_steps=num_speculative_steps,
        device=torch.device("cuda"),
        relay_draft_tokens=relay_draft_tokens,
    )


def record_broadcasts(monkeypatch) -> list[torch.Tensor]:
    """Capture every tensor handed to the collective, in call order."""
    calls: list[torch.Tensor] = []
    monkeypatch.setattr(
        torch.distributed, "broadcast", lambda t, src, group: calls.append(t)
    )
    return calls


def make_input_batch(num_reqs: int = 3, *, needs_sample: bool = True):
    # compute_need_sampled_mask only reads these fields. With needs_sample=False
    # every request is already at max_seq_len, so no sample is needed next step.
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        prefill_len_np=np.full(num_reqs, 4, dtype=np.int32),
        num_scheduled_tokens=np.full(num_reqs, 4, dtype=np.int32),
        max_seq_len_np=np.full(num_reqs, 100 if needs_sample else 1, dtype=np.int32),
        idx_mapping=torch.arange(num_reqs, device="cuda"),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
    )


def send_step(handler, input_batch, *, width: int, with_draft: bool):
    num_reqs = input_batch.num_reqs
    sampled = torch.zeros(num_reqs, width, dtype=torch.int64, device="cuda")
    counts = torch.zeros(num_reqs, dtype=torch.int32, device="cuda")
    handler.broadcast(sampled, counts, counts, input_batch)
    if with_draft:
        draft = torch.zeros(
            num_reqs, handler.max_sample_len - 1, dtype=torch.int64, device="cuda"
        )
        handler.broadcast_draft(draft, input_batch)


# ---------------------------------------------------------------------------
# broadcast() pads so send/recv element counts match on every step
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("width,num_spec", [(1, 1), (1, 3), (2, 3)])
def test_broadcast_pads_sampled_tokens_to_max_sample_len(monkeypatch, width, num_spec):
    """The sampler emits width 1 on steps with no draft tokens (prefill, first
    decode) and num_spec+1 only after rejection sampling. The receiver always
    posts a max_sample_len buffer, so an unpadded send is a count mismatch."""
    handler = make_handler(
        monkeypatch,
        is_last_rank=True,
        num_speculative_steps=num_spec,
        relay_draft_tokens=True,
    )
    calls = record_broadcasts(monkeypatch)
    input_batch = make_input_batch()

    send_step(handler, input_batch, width=width, with_draft=False)

    sent_sampled = calls[0]
    assert sent_sampled.shape == (input_batch.num_reqs, handler.max_sample_len)
    # Placeholder columns are ignored by post_update, which advances each
    # request by its own num_sampled count.
    assert (sent_sampled[:, width:] == -1).all()
    assert (sent_sampled[:, :width] == 0).all()


# ---------------------------------------------------------------------------
# Sender and receiver must post the same number of collectives per step
# ---------------------------------------------------------------------------


@requires_cuda
def test_send_and_recv_op_counts_match_with_speculator(monkeypatch):
    """With a speculator the step is three broadcasts: sampled, combined, draft."""
    sender = make_handler(
        monkeypatch, is_last_rank=True, num_speculative_steps=3, relay_draft_tokens=True
    )
    calls = record_broadcasts(monkeypatch)
    send_step(sender, make_input_batch(), width=1, with_draft=True)
    assert len(calls) == 3

    receiver = make_handler(
        monkeypatch,
        is_last_rank=False,
        num_speculative_steps=3,
        relay_draft_tokens=True,
    )
    calls.clear()
    assert receiver.receive(make_input_batch())
    assert len(calls) == 3
    assert calls[2].shape == (3, sender.max_sample_len - 1)


@requires_cuda
def test_send_and_recv_op_counts_match_without_speculator(monkeypatch):
    """Diffusion LLMs set num_speculative_steps > 0 but have no speculator, so
    the last rank never relays draft tokens. Gating the receiver's third recv on
    num_speculative_steps instead of on the speculator hangs the non-last ranks
    waiting for a broadcast that is never issued."""
    sender = make_handler(
        monkeypatch,
        is_last_rank=True,
        num_speculative_steps=3,
        relay_draft_tokens=False,
    )
    calls = record_broadcasts(monkeypatch)
    send_step(sender, make_input_batch(), width=1, with_draft=False)
    assert len(calls) == 2

    receiver = make_handler(
        monkeypatch,
        is_last_rank=False,
        num_speculative_steps=3,
        relay_draft_tokens=False,
    )
    calls.clear()
    assert receiver.receive(make_input_batch())
    assert len(calls) == 2
    assert receiver.queue[-1].draft_tokens is None


@requires_cuda
def test_both_ranks_skip_when_no_request_needs_sampling(monkeypatch):
    """The skip gate must be symmetric, or the ranks desynchronize."""
    sender = make_handler(
        monkeypatch, is_last_rank=True, num_speculative_steps=3, relay_draft_tokens=True
    )
    calls = record_broadcasts(monkeypatch)
    send_step(sender, make_input_batch(needs_sample=False), width=1, with_draft=True)
    assert calls == []

    receiver = make_handler(
        monkeypatch,
        is_last_rank=False,
        num_speculative_steps=3,
        relay_draft_tokens=True,
    )
    calls.clear()
    assert not receiver.receive(make_input_batch(needs_sample=False))
    assert calls == []


# ---------------------------------------------------------------------------
# Relayed draft tokens survive the deferred consume
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DeepSeekMTP under pipeline parallelism
# ---------------------------------------------------------------------------


def test_deepseek_mtp_passes_supports_pp_gate():
    """DeepSeekMTP must pass the supports_pp() gate used at model resolution;
    otherwise the engine refuses to build it under pipeline parallelism. The
    gate covers both the SupportsPP MRO entry and the forward() signature."""
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
    from vllm.model_executor.models.interfaces import supports_pp

    assert supports_pp(DeepSeekMTP)
