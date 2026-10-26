# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the decode segment at an MTP depth K whose 1 + K tokens do not fill the
kernels' row groups of four.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm53.nvidia import decode as glm53_decode
from vllm.models.glm53.nvidia import rows as glm53_rows
from vllm.models.glm53.nvidia import switch
from vllm.models.glm53.nvidia.decode import TierState, ledger_stop, null_padding_block
from vllm.models.glm53.nvidia.ops.cute_dsl import row_table
from vllm.models.glm53.nvidia.ops.cute_dsl.row_table import RowTableLayout

BLOCK = 64


def _tier(rows: int, tokens: int, tokens_buffer: torch.Tensor, hidden=None, **kw):
    segment = SimpleNamespace(geometry=SimpleNamespace(rows=rows))
    tensors = {"token_ids": tokens_buffer}
    if hidden is not None:
        tensors["segment.hidden"] = hidden
    return TierState(segment, tensors, None, None, 1, 2, tokens=tokens, **kw)


@pytest.mark.parametrize(
    ("live", "padded"),
    [(1, [7, 7, 7, 7]), (2, [7, 8, 8, 8]), (3, [7, 8, 9, 9]), (4, [7, 8, 9, 10])],
)
def test_a_lone_request_repeats_its_last_token_in_its_dead_rows(live, padded):
    buffer = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
    hidden = torch.arange(4 * 2, dtype=torch.bfloat16).view(4, 2)
    tier = _tier(4, live, buffer, hidden)
    tier.stage_inputs()
    assert buffer.tolist() == padded
    assert torch.equal(tier.output(), hidden[:live])


def test_a_packed_tier_stages_nothing_and_returns_vllms_rows():
    buffer = torch.arange(7, 23, dtype=torch.int32)
    hidden = torch.arange(16 * 2, dtype=torch.bfloat16).view(16, 2)
    tier = _tier(16, 6, buffer, hidden, packed=True)  # three requests of K = 1
    tier.stage_inputs()
    assert buffer.tolist() == list(range(7, 23))
    assert torch.equal(tier.output(), hidden[:6])


@pytest.mark.parametrize(
    ("position", "rows", "live", "zeroed"),
    [
        (BLOCK - 1, 4, 1, True),  # MTP off: rows 1..3 are in the next block
        (BLOCK - 1, 4, 2, False),  # K = 1: row 1 is real and needs the next block
        (BLOCK - 3, 4, 2, True),  # K = 1: the dead row at p+3 starts the next block
        (BLOCK - 2, 8, 5, False),  # K = 4: the live rows already end in it
        (2 * BLOCK - 6, 8, 5, True),  # K = 4: the dead rows p+5..p+7 start a block
    ],
)
def test_only_the_block_past_the_live_rows_is_nulled(position, rows, live, zeroed):
    table = torch.arange(1, 9, dtype=torch.int32) * 100
    null_padding_block(table, position, rows, BLOCK, live)
    nulled = [i for i, entry in enumerate(table.tolist()) if entry == 0]
    expected = [(position + rows - 1) // BLOCK] if zeroed else []
    assert nulled == expected


@pytest.mark.parametrize(
    ("tier", "per_request", "max_requests"),
    [
        (16, 1, 16),
        (16, 2, 8),
        (16, 3, 5),
        (16, 5, 3),
        (8, 1, 8),
        (8, 3, 2),
    ],
)
def test_the_kernels_and_the_host_agree_on_the_packed_row_table(
    geometry, tier, per_request, max_requests
):
    shaped = geometry.for_tier(tier, per_request - 1, packed=True)
    assert shaped.table_requests == row_table.PACKED_SLOTS[tier] == tier
    assert shaped.max_requests == max_requests
    layout = RowTableLayout(shaped.rows, shaped.table_requests)
    assert layout.words == 2 + tier + 3 * tier
    assert layout.table_words == layout.step_word + 1 == layout.words + 1
    assert shaped.selector_split == "request8" and shaped.selector_ranks == 8


def test_an_aligned_tier_keeps_the_layout_the_registry_builds(geometry):
    # The registry's REQUESTS knob is rows // 4 for the unpacked tiers.
    assert geometry.for_tier(8, 3).table_requests == 2
    assert geometry.for_tier(16, 3).table_requests == 4
    assert geometry.for_tier(16, 3).max_requests == 4


@pytest.mark.parametrize(
    ("tier", "per_request", "tag", "selector", "slots", "buffer_rows"),
    [
        (16, 2, "r16p", "mla_selector_r16p_serve8r0_paged_vlin", 16, 16 + 3),
        (16, 5, "r16p", "mla_selector_r16p_serve8r0_paged_vlin", 16, 16 + 3),
    ],
)
def test_a_packed_build_names_its_own_attention_entries(
    geometry, tier, per_request, tag, selector, slots, buffer_rows
):
    shaped = geometry.for_tier(tier, per_request - 1, packed=True)
    assert shaped.tag == tag and shaped.selector_kernel(0) == selector
    assert shaped.prologue_kernel(False) == f"prologue_vllm_{tag}"
    # The MoE and the dense FFN read only the row table's rows: the unpacked entries.
    assert shaped.row_tag == f"r{tier}"
    assert shaped.dense_kernel == f"dense_ffn_r{tier}_vllm"
    assert shaped.moe_tag == f"r{max(tier, 16)}"
    assert shaped.selector_slots == slots == row_table.PACKED_SLOTS[tier]
    # No push role in a packed build: the output slice is sized by the shared one.
    full = next(i for i in range(geometry.layers) if not geometry.is_shared(i))
    assert shaped.mailbox_kernel(full) == f"mla_worker_{tag}_shared_paged_vlin"
    assert geometry.for_tier(tier, 3).mailbox_kernel(full) == (
        f"mla_worker_r{tier}_shared_paged_vlin"
    )
    assert shaped.buffer_rows == buffer_rows and shaped.slack_rows == 3
    unpacked = geometry.for_tier(tier, 3)
    assert unpacked.selector_slots == tier // 4 and unpacked.slack_rows == 0
    assert unpacked.tag == f"r{tier}" and unpacked.buffer_rows == max(tier, 16)


def test_a_lone_request_of_five_or_more_tokens_selects_its_groups_in_parallel(
    geometry,
):
    # One request fills tier 8 at K >= 4: rank g selects its row group g.
    lone = geometry.for_tier(8, 7)
    assert lone.selector_split == "group8" and lone.selector_ranks == 8
    assert lone.selector_kernel(1) == "mla_selector_r8_gserve8r1_paged_vlin"
    assert lone.worker_kernel(0, 0) == "mla_worker_r8_full_solo_paged_vlin"
    assert geometry.for_tier(8, 3).selector_split == "request8"
    assert geometry.for_tier(16, 7).selector_split == "request8"
    assert geometry.for_tier(4, 3).selector_split == "key"
    assert geometry.for_tier(16, 0, packed=True).selector_split == "request8"


def test_a_packed_build_is_request_parallel_and_exists_for_tiers_8_16(geometry):
    with pytest.raises(ValueError, match="packed builds exist for the tiers"):
        geometry.for_tier(4, 0, packed=True)
    with pytest.raises(ValueError, match="request8"):
        geometry.for_tier(16, 2, packed=True).with_split("key")


def test_steps_of_unclaimed_sizes_are_counted(geometry):
    claims = switch.tier_claims(8, 2)
    assert claims == {
        2: switch.Build(4),
        4: switch.Build(8, w8a16_rows=4),
        8: switch.Build(8, w8a16_rows=8),
        16: switch.Build(16),
    }
    decode = glm53_decode.SpecializedDecode(
        SimpleNamespace(),
        SimpleNamespace(geometry=geometry),
        claims,
        rows_per_request=2,
        row_tiers=True,
    )
    assert decode.lone_rows == 4
    decode.step(["a", "b"], [10, 10], [2, 2], 4)  # B = 2 at K = 1: packed, q4 MoE
    decode.step(["a", "b"], [12, 12], [2, 1], 4)  # not a uniform decode step
    decode.step(["a", "b", "c", "d"], [10] * 4, [2] * 4, 16)  # claimed (no tier built)
    assert decode.unclaimed_steps == 0 and decode.uniform_steps == 2
    decode.step(["a", "b", "c"], [10] * 3, [2] * 3, 6)  # a size no graph holds
    assert decode.unclaimed_steps == 1


def test_a_stock_step_completes_the_pending_rows(geometry, monkeypatch):
    decode = glm53_decode.SpecializedDecode(
        SimpleNamespace(),
        SimpleNamespace(geometry=geometry),
        switch.tier_claims(16, 4),  # K = 3: B <= 4 claimed, B = 5-16 stock
        rows_per_request=4,
        row_tiers=True,
    )
    groups = []
    decode._kv = SimpleNamespace(
        pending=True, complete_pending=lambda blocks, group: groups.append(group)
    )
    monkeypatch.setattr("vllm.distributed.get_tp_group", lambda: "tp")
    decode._batch = ("a",)
    decode.step(["a"], [10], [4], 4)  # claimed (no tier built): nothing to complete
    assert groups == [] and decode._batch == ("a",)
    ids = [str(i) for i in range(5)]
    decode.step(ids, [10] * 5, [4] * 5, 20)  # R = 20: vLLM's stock forward
    assert groups == ["tp"] and decode._batch == ()
    assert decode.unclaimed_steps == 1 and decode.uniform_steps == 2


def test_the_host_writes_the_true_rows_and_leaves_the_replay_counter(geometry):
    shaped = geometry.for_tier(16, 4, packed=True)
    layout = RowTableLayout(shaped.rows, shaped.table_requests)
    table = glm53_rows.RowTable(layout, torch.device("cpu"), packed=True)
    assert table.tensor.numel() == layout.words + 1
    table.tensor[layout.step_word] = 41  # the prologue's counter
    table.set_uniform_step(3, 5)
    assert table.tensor[layout.step_word] == 41
    table.fill([5, 5, 5], [100, 200, 300])
    assert table.tensor[layout.step_word] == 41
    assert [table.tensor[layout.context_word(b)].item() for b in range(3)] == [
        100,
        200,
        300,
    ]
    with pytest.raises(ValueError, match="packed row table"):
        glm53_rows.RowTable(RowTableLayout(16, 4), torch.device("cpu"), packed=True)
    # An aligned tier's table has no counter.
    aligned = glm53_rows.RowTable(RowTableLayout(16, 4), torch.device("cpu"))
    assert aligned.tensor.numel() == RowTableLayout(16, 4).words


@pytest.mark.parametrize(
    ("per_request", "packed"),
    [(5, True), (5, False), (7, False), (2, False), (4, False)],
)
def test_the_ledger_covers_what_a_step_writes(per_request, packed):
    # A packed step writes its tokens; a lone aligned request also its dead rows.
    lone = switch.lone_rows(per_request)
    stop = ledger_stop(100, per_request, 0 if packed else lone - per_request)
    assert stop == 100 + (per_request if packed else lone)
