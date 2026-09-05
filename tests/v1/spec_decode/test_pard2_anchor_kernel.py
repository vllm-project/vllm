# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout tests for the PARD-2 anchor path of copy_and_expand_eagle_inputs_kernel.

PARD-2's draft is trained on the full sequence: token 0 paired with a zero
target feature. The EAGLE-style shift drops that row, so the kernel grows an
ANCHOR mode that puts it back. These tests pin the resulting layout and check
the Triton kernel and the CPU-backend torch implementation against each other.
"""

import pytest
import torch

from vllm.utils.cpu_triton_utils import _copy_and_expand_eagle_inputs_anchored

PAD_ID = 0
PARALLEL_DRAFTING_TOKEN_ID = 999


def _run_torch(
    query_start_loc,
    query_end_loc,
    target_token_ids,
    target_positions,
    next_token_ids,
    anchor_flags,
    num_padding_slots,
    shift_input_ids,
    device="cpu",
):
    num_reqs = query_start_loc.numel() - 1
    net_new_slots = num_padding_slots - (1 if shift_input_ids else 0) + 1
    total_out = int(query_start_loc[-1]) + num_reqs * net_new_slots
    out = dict(
        input_ids=torch.zeros(total_out, dtype=torch.int32, device=device),
        positions=torch.zeros(total_out, dtype=torch.int32, device=device),
        rejected=torch.zeros(total_out, dtype=torch.bool, device=device),
        masked=torch.zeros(total_out, dtype=torch.bool, device=device),
        anchor=torch.zeros(total_out, dtype=torch.bool, device=device),
        new_idx=torch.zeros(
            num_reqs * num_padding_slots, dtype=torch.int32, device=device
        ),
        hidden_map=torch.zeros(
            int(query_start_loc[-1]), dtype=torch.int32, device=device
        ),
    )
    _copy_and_expand_eagle_inputs_anchored(
        target_token_ids,
        target_positions,
        next_token_ids,
        out["input_ids"],
        out["positions"],
        out["rejected"],
        out["masked"],
        out["new_idx"],
        out["hidden_map"],
        query_start_loc,
        query_end_loc,
        PAD_ID,
        PARALLEL_DRAFTING_TOKEN_ID,
        num_padding_slots,
        shift_input_ids,
        anchor_flags,
        out["anchor"],
        net_new_slots,
    )
    return out


def test_prefill_request_gets_leading_anchor_row():
    """A prefilling request keeps its whole sequence, preceded by token 0."""
    tokens = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    out = _run_torch(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        query_end_loc=torch.tensor([3], dtype=torch.int32),
        target_token_ids=tokens,
        target_positions=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        next_token_ids=torch.tensor([77], dtype=torch.int32),
        anchor_flags=torch.tensor([1], dtype=torch.int32),
        num_padding_slots=3,
        shift_input_ids=True,
    )
    # anchor row, then the shifted sequence, then bonus + parallel slots.
    assert out["input_ids"].tolist()[:5] == [11, 12, 13, 14, 77]
    assert out["positions"].tolist()[:5] == [0, 1, 2, 3, 4]
    assert out["anchor"].tolist()[:5] == [True, False, False, False, False]
    # Without the anchor the draft would have started at token 12 / position 1.


def test_decode_request_gets_trailing_junk_row():
    """A decode request has the anchor in its KV already; it only pads a row."""
    out = _run_torch(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_end_loc=torch.tensor([1], dtype=torch.int32),
        target_token_ids=torch.tensor([21, 22], dtype=torch.int32),
        target_positions=torch.tensor([9, 10], dtype=torch.int32),
        next_token_ids=torch.tensor([88], dtype=torch.int32),
        anchor_flags=torch.tensor([0], dtype=torch.int32),
        num_padding_slots=2,
        shift_input_ids=True,
    )
    assert not out["anchor"].any(), "decode requests must not get an anchor row"
    assert out["rejected"].any(), "the extra row must be marked rejected"
    # Real rows keep true positions (target position + 1).
    assert out["positions"][0].item() == 10


def test_every_request_adds_the_same_number_of_rows():
    """Uniform row growth is what lets the caller use a single stride."""
    num_padding_slots, shift = 4, True
    qsl = torch.tensor([0, 5, 7], dtype=torch.int32)
    out = _run_torch(
        query_start_loc=qsl,
        query_end_loc=torch.tensor([4, 6], dtype=torch.int32),
        target_token_ids=torch.arange(1, 8, dtype=torch.int32),
        target_positions=torch.tensor([0, 1, 2, 3, 4, 30, 31], dtype=torch.int32),
        next_token_ids=torch.tensor([61, 62], dtype=torch.int32),
        anchor_flags=torch.tensor([1, 0], dtype=torch.int32),  # prefill + decode
        num_padding_slots=num_padding_slots,
        shift_input_ids=shift,
    )
    net_new_slots = num_padding_slots - 1 + 1
    assert out["input_ids"].numel() == int(qsl[-1]) + 2 * net_new_slots
    assert out["anchor"].sum().item() == 1  # only the prefilling request


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shift_input_ids", [True, False])
@pytest.mark.parametrize("anchor_pattern", [[1], [0], [1, 0], [0, 1], [1, 1, 0]])
def test_triton_matches_torch(shift_input_ids, anchor_pattern):
    """The Triton ANCHOR path and the CPU torch path must agree exactly."""
    from vllm.v1.spec_decode.utils import copy_and_expand_eagle_inputs_kernel

    torch.manual_seed(0)
    num_reqs = len(anchor_pattern)
    query_lens = [5, 3, 4][:num_reqs]
    num_padding_slots = 3
    qsl = torch.tensor(
        [0] + list(torch.tensor(query_lens).cumsum(0)), dtype=torch.int32
    )
    qel = qsl[1:] - 1
    total_in = int(qsl[-1])
    tokens = torch.randint(1, 1000, (total_in,), dtype=torch.int32)
    positions = torch.cat(
        [
            torch.arange(n, dtype=torch.int32)
            if a
            else torch.arange(50, 50 + n, dtype=torch.int32)
            for n, a in zip(query_lens, anchor_pattern)
        ]
    )
    next_tokens = torch.randint(1, 1000, (num_reqs,), dtype=torch.int32)
    flags = torch.tensor(anchor_pattern, dtype=torch.int32)

    cpu = _run_torch(
        qsl,
        qel,
        tokens,
        positions,
        next_tokens,
        flags,
        num_padding_slots,
        shift_input_ids,
    )

    dev = "cuda"
    net_new_slots = num_padding_slots - (1 if shift_input_ids else 0) + 1
    total_out = total_in + num_reqs * net_new_slots
    g = lambda t: t.to(dev)  # noqa: E731
    out_ids = torch.zeros(total_out, dtype=torch.int32, device=dev)
    out_pos = torch.zeros(total_out, dtype=torch.int32, device=dev)
    out_rej = torch.zeros(total_out, dtype=torch.bool, device=dev)
    out_msk = torch.zeros(total_out, dtype=torch.bool, device=dev)
    out_anc = torch.zeros(total_out, dtype=torch.bool, device=dev)
    out_new = torch.zeros(num_reqs * num_padding_slots, dtype=torch.int32, device=dev)
    out_map = torch.zeros(total_in, dtype=torch.int32, device=dev)

    block = 32
    copy_and_expand_eagle_inputs_kernel[(num_reqs, 1)](
        target_token_ids_ptr=g(tokens),
        target_positions_ptr=g(positions),
        next_token_ids_ptr=g(next_tokens),
        out_input_ids_ptr=out_ids,
        out_positions_ptr=out_pos,
        out_is_rejected_token_mask_ptr=out_rej,
        out_is_masked_token_mask_ptr=out_msk,
        out_new_token_indices_ptr=out_new,
        out_hidden_state_mapping_ptr=out_map,
        query_start_loc_ptr=g(qsl),
        query_end_loc_ptr=g(qel),
        padding_token_id=PAD_ID,
        parallel_drafting_token_id=PARALLEL_DRAFTING_TOKEN_ID,
        total_input_tokens=total_in,
        num_padding_slots_per_request=num_padding_slots,
        shift_input_ids=shift_input_ids,
        anchor_flags_ptr=g(flags),
        out_is_anchor_mask_ptr=out_anc,
        net_new_slots_per_request=net_new_slots,
        BLOCK_SIZE_TOKENS=block,
        ANCHOR=True,
    )
    torch.testing.assert_close(out_ids.cpu(), cpu["input_ids"])
    torch.testing.assert_close(out_pos.cpu(), cpu["positions"])
    torch.testing.assert_close(out_anc.cpu(), cpu["anchor"])
    torch.testing.assert_close(out_msk.cpu(), cpu["masked"])
    torch.testing.assert_close(out_rej.cpu(), cpu["rejected"])
    torch.testing.assert_close(out_new.cpu(), cpu["new_idx"])
    if shift_input_ids:
        torch.testing.assert_close(out_map.cpu(), cpu["hidden_map"])
