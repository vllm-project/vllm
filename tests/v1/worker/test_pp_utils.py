# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import model_runner, pp_utils


def _cuda_handler(max_sample_len=6):
    handler = object.__new__(pp_utils.PPHandler)
    handler.is_last_rank = True
    handler.disabled = False
    handler.max_sample_len = max_sample_len
    handler.last_rank = 1
    handler.broadcast_group = Mock()
    handler.device = torch.device("cuda")
    handler.main_stream = torch.cuda.current_stream()
    handler.broadcast_stream = torch.cuda.Stream()
    return handler


def _batch(num_computed, prefill_len, num_scheduled, idx_mapping=None):
    num_reqs = len(num_computed)
    if idx_mapping is None:
        idx_mapping = list(range(num_reqs))
    return Mock(
        num_reqs=num_reqs,
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
        idx_mapping=torch.tensor(idx_mapping, dtype=torch.int64),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


def test_disabled_handler_skips_broadcast_and_receive(monkeypatch):
    """While disabled (warmup), neither side enqueues a broadcast op."""
    sent = []
    monkeypatch.setattr(
        pp_utils.torch.distributed,
        "broadcast",
        lambda *args, **kwargs: sent.append((args, kwargs)),
    )

    handler = object.__new__(pp_utils.PPHandler)
    handler.set_disabled(True)

    handler.is_last_rank = False
    assert handler.receive(Mock()) is False

    handler.is_last_rank = True
    assert handler.broadcast(Mock(), Mock(), Mock(), Mock()) is None

    assert sent == []

    handler.set_disabled(False)
    assert handler.disabled is False


def test_alloc_combined_keeps_unbind_views_16_byte_aligned():
    """Triton specializes on pointer alignment: an unaligned `num_rejected`
    would compile a second `_post_update_kernel` variant at serving time,
    where the in-flight broadcast NCCL kernel can block the module load."""
    for num_reqs in range(1, 9):
        combined = pp_utils._alloc_combined(num_reqs, torch.device("cpu"))
        num_sampled, num_rejected = combined.unbind(dim=0)
        assert num_sampled.data_ptr() % 16 == 0
        assert num_rejected.data_ptr() % 16 == 0
        assert combined.shape[1] >= num_reqs


def test_warmup_pp_decode_update_matches_serving_specialization(monkeypatch):
    """The warmup launch must hit the same triton specialization as serving.

    A mismatch means the first real ``update_pp_decode_requests`` recompiles
    mid-serving, where the in-flight broadcast NCCL kernel blocks the CUDA
    module load and deadlocks the pipeline.
    """
    calls = []
    monkeypatch.setattr(model_runner, "post_update", lambda *args: calls.append(args))

    runner = object.__new__(model_runner.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.pp_handler = Mock(max_sample_len=3)
    runner.req_states = Mock()

    runner.warmup_pp_decode_update()

    assert len(calls) == 1
    args = calls[0]
    idx_mapping, _, _, output_bin_counts = args[:4]
    sampled_tokens, num_sampled, num_rejected, query_start_loc = args[4:8]
    broadcast_drafts, draft_tokens_out = args[10:12]
    assert idx_mapping.tolist() == [-1] and idx_mapping.dtype == torch.int64
    assert output_bin_counts is None
    assert query_start_loc is None
    assert sampled_tokens.shape == (1, 3) and sampled_tokens.dtype == torch.int64
    assert num_sampled.dtype == torch.int32
    assert num_rejected.dtype == torch.int32
    # Spec-enabled PP handlers receive drafts over the broadcast; the warmup
    # must compile that specialization (non-None draft pointers) too.
    assert broadcast_drafts.shape == (1, 2) and broadcast_drafts.dtype == torch.int64
    assert draft_tokens_out is runner.req_states.draft_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA stream")
def test_broadcast_pads_plain_sampler_rows_to_max_sample_len(monkeypatch):
    """The wire shape must not depend on whether the batch carried drafts:
    the receiver always allocates [num_reqs, max_sample_len], and a NCCL
    broadcast with mismatched counts hangs the receiver."""
    sent = []
    monkeypatch.setattr(
        pp_utils.torch.distributed,
        "broadcast",
        lambda tensor, **kw: sent.append(tensor),
    )
    handler = _cuda_handler()
    batch = _batch(num_computed=[10], prefill_len=[8], num_scheduled=[1])

    handler.broadcast(
        torch.zeros(1, 1, dtype=torch.int64, device="cuda"),  # plain sampler
        torch.ones(1, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        batch,
    )

    assert sent[0].shape == (1, 6)
    assert sent[1].shape == (2, 4)
    torch.accelerator.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA stream")
def test_broadcast_drafts_gathers_fresh_rows_from_the_table(monkeypatch):
    sent = []
    monkeypatch.setattr(
        pp_utils.torch.distributed,
        "broadcast",
        lambda tensor, **kw: sent.append(tensor),
    )
    handler = _cuda_handler()
    # The batch's single row maps to request-state row 2.
    batch = _batch(num_computed=[10], prefill_len=[8], num_scheduled=[1])
    batch.idx_mapping = torch.tensor([2], dtype=torch.int64, device="cuda")
    table = torch.arange(20, dtype=torch.int64, device="cuda").view(4, 5)

    handler.broadcast_drafts(table, batch)

    assert sent[0].shape == (1, 5)
    # The payload is a gather into a fresh tensor: propose() overwrites its
    # persistent buffer on the next step, possibly before this send completes.
    assert sent[0].data_ptr() != table.data_ptr()
    assert sent[0].cpu().tolist() == [table[2].cpu().tolist()]

    # An all-prefill batch sends nothing (receive() enqueues nothing either).
    sent.clear()
    prefill_batch = _batch(num_computed=[0], prefill_len=[4096], num_scheduled=[448])
    prefill_batch.idx_mapping = prefill_batch.idx_mapping.cuda()
    handler.broadcast_drafts(table, prefill_batch)
    assert sent == []
    torch.accelerator.synchronize()
