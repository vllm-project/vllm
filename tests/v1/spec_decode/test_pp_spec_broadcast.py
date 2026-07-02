# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PP sampled-token broadcast under speculative decoding.

Under PP+async scheduling the last rank broadcasts its sampled token ids to the
other ranks so they can advance positions / build the next input. Without spec,
the broadcast carries shape ``[num_reqs, 1]``. With MTP/EAGLE spec the sampler
emits ``[num_reqs, num_spec + 1]`` (accepted drafts + bonus, ``-1``-padded), so
the transport must carry the full width and the receiver must advance each
request by its per-request *valid* count, not by 1.

These tests pin that contract WITHOUT a GPUModelRunner: the pure per-request
count is tested directly, and the variable-width transport is tested over a real
2-rank gloo group on CPU (no CUDA).
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port

# The sampled-token grid the last rank "produces": req0 has 3 valid tokens
# (2 accepted drafts + bonus), req1 has 2, req2 has 1. -1 pads the rejected tail.
_SAMPLED = [[10, 11, 12], [20, 21, -1], [30, -1, -1]]
_EXPECTED_VALID = [3, 2, 1]


def test_count_valid_sampled_tokens_per_req_counts_non_sentinel():
    from vllm.v1.worker.pp_spec_broadcast import count_valid_sampled_tokens_per_req

    t = torch.tensor(_SAMPLED, dtype=torch.int32)
    assert count_valid_sampled_tokens_per_req(t).tolist() == _EXPECTED_VALID


def test_count_valid_sampled_tokens_per_req_non_spec_width_one():
    from vllm.v1.worker.pp_spec_broadcast import count_valid_sampled_tokens_per_req

    # Non-spec: width 1, every row a real token -> each advances by 1.
    t = torch.tensor([[10], [20], [30]], dtype=torch.int32)
    assert count_valid_sampled_tokens_per_req(t).tolist() == [1, 1, 1]


# C4: the non-last rank must feed the REAL latest sampled token as its next input,
# never the -1 placeholder. The latest valid token for a request that advanced by
# v positions is the last valid column recv[i, v-1] (reject v=1 -> col 0; accept +
# bonus v=2 -> col 1). This is the value-back-write the receiver must persist.
def test_select_latest_sampled_token_per_req_picks_last_valid_column():
    from vllm.v1.worker.pp_spec_broadcast import select_latest_sampled_token_per_req

    # _SAMPLED: req0 valid=3 -> col2=12; req1 valid=2 -> col1=21; req2 valid=1 -> col0.
    t = torch.tensor(_SAMPLED, dtype=torch.int32)
    assert select_latest_sampled_token_per_req(t).tolist() == [12, 21, 30]


def test_select_latest_sampled_token_per_req_non_spec_width_one():
    from vllm.v1.worker.pp_spec_broadcast import select_latest_sampled_token_per_req

    # Non-spec width-1: the single column is the latest token.
    t = torch.tensor([[10], [20], [30]], dtype=torch.int32)
    assert select_latest_sampled_token_per_req(t).tolist() == [10, 20, 30]


def test_select_latest_sampled_token_per_req_never_returns_sentinel():
    from vllm.v1.worker.pp_spec_broadcast import select_latest_sampled_token_per_req

    # Every confirmed request has >=1 valid token, so the result is never -1 even
    # when the rejected tail is -1-padded (this is exactly what break #2 needs).
    t = torch.tensor([[42, -1, -1], [7, 8, -1]], dtype=torch.int32)
    out = select_latest_sampled_token_per_req(t).tolist()
    assert out == [42, 8]
    assert -1 not in out


# Holistic C4: select_latest only persists the single LATEST token, but under
# MTP the non-last rank must backfill ALL v confirmed tokens (accepted drafts +
# bonus) into its token_ids_cpu so the next step's input read (indexed by
# num_computed_tokens, which includes the spec tokens) finds real ids at EVERY
# position, not just the last one. recv[i, 0:v] is exactly that ordered run.
def test_gather_valid_sampled_tokens_per_req_returns_all_valid_in_order():
    from vllm.v1.worker.pp_spec_broadcast import gather_valid_sampled_tokens_per_req

    # _SAMPLED: req0 valid=3 -> [10,11,12]; req1 valid=2 -> [20,21]; req2 valid=1 [30].
    t = torch.tensor(_SAMPLED, dtype=torch.int32)
    assert gather_valid_sampled_tokens_per_req(t) == [[10, 11, 12], [20, 21], [30]]


def test_gather_valid_sampled_tokens_per_req_non_spec_width_one():
    from vllm.v1.worker.pp_spec_broadcast import gather_valid_sampled_tokens_per_req

    # Non-spec width-1: each request advances by exactly one real token.
    t = torch.tensor([[10], [20], [30]], dtype=torch.int32)
    assert gather_valid_sampled_tokens_per_req(t) == [[10], [20], [30]]


def test_gather_valid_sampled_tokens_per_req_all_rejected_row_is_empty():
    from vllm.v1.worker.pp_spec_broadcast import gather_valid_sampled_tokens_per_req

    # A fully -1-padded row (no valid token, e.g. a discarded/chunked req) yields
    # an empty list so the receiver writes nothing and advances the cursor by 0.
    t = torch.tensor([[5, 6, -1], [-1, -1, -1]], dtype=torch.int32)
    assert gather_valid_sampled_tokens_per_req(t) == [[5, 6], []]


def test_gather_valid_sampled_tokens_per_req_never_contains_sentinel():
    from vllm.v1.worker.pp_spec_broadcast import gather_valid_sampled_tokens_per_req

    # Whatever the padding, the gathered runs never carry a -1 (break #2 needs
    # every backfilled position to be a real embedding index).
    t = torch.tensor(_SAMPLED, dtype=torch.int32)
    flat = [tok for row in gather_valid_sampled_tokens_per_req(t) for tok in row]
    assert -1 not in flat


def test_num_computed_tokens_drift_correction_reject_subtracts_one():
    from vllm.v1.worker.pp_spec_broadcast import (
        num_computed_tokens_drift_correction,
    )

    # 1 draft proposed, draft rejected -> only the bonus is valid (v=1). The
    # optimistic advance counted the draft as accepted, so subtract 1.
    assert num_computed_tokens_drift_correction(1, 1) == 1


def test_num_computed_tokens_drift_correction_accept_subtracts_zero():
    from vllm.v1.worker.pp_spec_broadcast import (
        num_computed_tokens_drift_correction,
    )

    # 1 draft proposed, draft accepted -> draft + bonus valid (v=2). The optimistic
    # advance was right, so no correction.
    assert num_computed_tokens_drift_correction(1, 2) == 0


def test_num_computed_tokens_drift_correction_no_drafts_is_zero():
    from vllm.v1.worker.pp_spec_broadcast import (
        num_computed_tokens_drift_correction,
    )

    # First decode after prefill: no drafts last step (prev_num_draft_len=0), the
    # single bonus is valid (v=1). Nothing optimistic was counted -> 0.
    assert num_computed_tokens_drift_correction(0, 1) == 0


def test_num_computed_tokens_drift_correction_partial_accept():
    from vllm.v1.worker.pp_spec_broadcast import (
        num_computed_tokens_drift_correction,
    )

    # 3 drafts proposed, 1 accepted + bonus valid (v=2) -> 2 drafts rejected.
    assert num_computed_tokens_drift_correction(3, 2) == 2
    # all 3 accepted + bonus (v=4) -> no correction.
    assert num_computed_tokens_drift_correction(3, 4) == 0


def _broadcast_worker(rank: int, world_size: int, port: int, width: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from vllm.v1.worker.pp_spec_broadcast import (
            broadcast_sampled_token_ids,
            count_valid_sampled_tokens_per_req,
            receive_sampled_token_ids,
        )

        src = world_size - 1  # the last rank is the sender
        num_reqs = len(_SAMPLED)
        if rank == src:
            sampled = torch.tensor(_SAMPLED, dtype=torch.int32)
            broadcast_sampled_token_ids(sampled, dist.group.WORLD, src)
        else:
            recv = receive_sampled_token_ids(
                num_reqs, width, dist.group.WORLD, src, device="cpu"
            )
            expected = torch.tensor(_SAMPLED, dtype=torch.int32)
            assert torch.equal(recv, expected), f"recv mismatch: {recv}"
            counts = count_valid_sampled_tokens_per_req(recv)
            assert counts.tolist() == _EXPECTED_VALID, f"counts: {counts}"
    finally:
        dist.destroy_process_group()


def test_variable_width_broadcast_roundtrip():
    """Last rank broadcasts [num_reqs, num_spec+1]; non-last receives same width.

    The current code hardcodes a [num_reqs, 1] receive buffer, which cannot carry
    the spec width. This pins the variable-width transport over a real gloo group.
    """
    port = get_open_port()
    width = len(_SAMPLED[0])  # num_spec + 1 == 3
    mp.spawn(_broadcast_worker, args=(2, port, width), nprocs=2, join=True)
