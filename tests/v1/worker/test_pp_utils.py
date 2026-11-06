# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from unittest.mock import Mock

import numpy as np
import torch

from vllm.v1.worker.gpu import model_runner, pp_utils, warmup


def _batch(
    num_computed,
    prefill_len,
    num_scheduled,
    is_prefilling=None,
    max_seq_len=None,
    idx_mapping=None,
):
    num_reqs = len(num_computed)
    if idx_mapping is None:
        idx_mapping = list(range(num_reqs))
    return Mock(
        num_reqs=num_reqs,
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
        is_prefilling_np=(
            np.array(is_prefilling, dtype=np.bool_)
            if is_prefilling is not None
            else None
        ),
        max_seq_len_np=(
            np.array(max_seq_len, dtype=np.int32) if max_seq_len is not None else None
        ),
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


def test_excludes_final_prefill_chunk_reaching_its_length_cap():
    """A max_tokens=1 final chunk hands its token off and leaves the engine.

    Disaggregated prefill submits max_tokens=1, so the request finishes with
    the final chunk's single sample and no follow-up step on this engine
    reads the broadcast payload.
    """
    batch = _batch(
        num_computed=[1000],
        prefill_len=[1004],
        num_scheduled=[4],
        is_prefilling=[True],
        max_seq_len=[1005],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_final_prefill_chunk_with_generation_budget_left():
    """The same final chunk, but the request keeps decoding here."""
    batch = _batch(
        num_computed=[1000],
        prefill_len=[1004],
        num_scheduled=[4],
        is_prefilling=[True],
        max_seq_len=[1004 + 256],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_length_cap_overrun_decode_row_is_kept():
    """Decoding rows are never excluded, even past their length cap.

    Speculative decoding can transiently overrun prompt_len + max_tokens
    while the scheduler still runs the request; dropping the row would
    desynchronize the PP stages.
    """
    batch = _batch(
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
        is_prefilling=[False],
        max_seq_len=[12176],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_finishing_prefill_row_does_not_drop_a_sibling_row():
    """Only finishing rows are excluded; a decoding sibling keeps its row."""
    batch = _batch(
        num_computed=[1000, 10],
        prefill_len=[1004, 8],
        num_scheduled=[4, 1],
        is_prefilling=[True, False],
        max_seq_len=[1005, 1024],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


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
    assert handler.receive(Mock(), recv_drafts=True) is False

    # A disabled broadcast must also clear the flag a previous step left set,
    # or broadcast_drafts would send drafts that no rank receives.
    handler.is_last_rank = True
    handler.sampled_tokens_broadcast = True
    handler.broadcast(Mock(), Mock(), Mock(), Mock())
    assert handler.sampled_tokens_broadcast is False
    handler.broadcast_drafts(Mock())

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
    """The warmup launches must hit every triton specialization serving uses.

    A miss means the first real ``update_pp_decode_requests`` recompiles
    mid-serving, where the in-flight broadcast NCCL kernel blocks the CUDA
    module load and deadlocks the pipeline.
    """
    calls = []
    monkeypatch.setattr(warmup, "post_update", lambda *args: calls.append(args))

    runner = object.__new__(model_runner.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.pp_handler = Mock(max_sample_len=3)
    runner.req_states = Mock()

    warmup.warmup_pp_decode_update(runner)

    variants = set()
    for args in calls:
        idx_mapping, _, _, output_bin_counts = args[:4]
        sampled_tokens, num_sampled, num_rejected, query_start_loc = args[4:8]
        draft_tokens, draft_tokens_out = args[10:12]
        # Serving passes an int64 idx_mapping on both the unfiltered and the
        # freed-request filtered path.
        assert idx_mapping.tolist() == [-1] and idx_mapping.dtype == torch.int64
        assert output_bin_counts is None
        assert query_start_loc is None
        assert sampled_tokens.dtype == torch.int64
        assert num_sampled.dtype == torch.int32
        assert num_rejected.dtype == torch.int32
        assert draft_tokens_out is runner.req_states.draft_tokens
        if draft_tokens is not None:
            assert draft_tokens.shape == (1, 2)
            assert draft_tokens.dtype == torch.int64
        variants.add((sampled_tokens.shape[1], draft_tokens is not None))
    # The sampled-token width is 1 for a batch without drafts (a stride of 1
    # specializes), and drafts arrive only in steps that proposed some.
    assert variants == {(1, False), (1, True), (3, False), (3, True)}


def test_warmup_pp_decode_update_without_spec_decode(monkeypatch):
    calls = []
    monkeypatch.setattr(warmup, "post_update", lambda *args: calls.append(args))

    runner = object.__new__(model_runner.GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.pp_handler = Mock(max_sample_len=1)
    runner.req_states = Mock()

    warmup.warmup_pp_decode_update(runner)

    assert len(calls) == 1
    assert calls[0][4].shape == (1, 1)
    assert calls[0][10] is None
