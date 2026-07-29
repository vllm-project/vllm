# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU unit tests for the MoRIIO hybrid (attention + mamba/KDA) scheduler
and the KDA offset-template cache.

These exercise the highest-risk, otherwise-untested scheduler logic without a
GPU or the ``mori`` runtime: per-group block-id splitting, the READ/WRITE
``N-1`` token accounting, P-side prompt truncation, and the C2 offset-cache
equivalence (cached offsets must be byte-identical to freshly computed ones).

Like ``test_moriio_kv_layout.py`` the whole module is skipped unless it is
running on ROCm with ``mori`` installed (importing the connector pulls in
``mori``). The authoritative run happens on the MIA recipe image.
"""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

mori_available = importlib.util.find_spec("mori") is not None

if not (current_platform.is_rocm() and mori_available):
    pytest.skip(
        "MoRIIOs are only available on ROCm with mori package installed",
        allow_module_level=True,
    )

moriio_connector = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector"
)
moriio_layout = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout"
)
ssm_conv_transfer_utils = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils"
)
MambaConvSplitInfo = ssm_conv_transfer_utils.MambaConvSplitInfo
MoRIIOMode = moriio_connector.MoRIIOMode
moriio_common = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common"
)
as_attn_mamba = moriio_common.as_attn_mamba


class _FakeScheduler(moriio_connector.MoRIIOConnectorScheduler):
    """Constructs a scheduler without the heavy real ``__init__``; only the
    attributes read by the method under test are set."""

    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class _FakeConnector(moriio_connector.MoRIIOConnector):
    def __init__(self, connector_scheduler):
        self.connector_scheduler = connector_scheduler


class _FakeWorker(moriio_connector.MoRIIOConnectorWorker):
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def _slot_strided(num_slots, per_slot_shape, slot_stride, dtype=torch.bfloat16):
    inner = 1
    for d in per_slot_shape:
        inner *= d
    assert slot_stride >= inner
    backing = torch.zeros(num_slots * slot_stride, dtype=dtype)
    inner_strides = []
    acc = 1
    for d in reversed(per_slot_shape):
        inner_strides.insert(0, acc)
        acc *= d
    return backing.as_strided((num_slots, *per_slot_shape), (slot_stride, *inner_strides))


def _gdn_split_info(conv_rows=3, key_dim=4, value_dim=8, dtype_size=2):
    conv_dim = 2 * key_dim + value_dim
    conv_state_bytes = conv_dim * conv_rows * dtype_size
    ssm_state_bytes = 64
    return MambaConvSplitInfo(
        conv_rows=conv_rows,
        local_proj_dims=(key_dim, key_dim, value_dim),
        conv_dtype_size=dtype_size,
        ssm_sizes=(conv_state_bytes, ssm_state_bytes),
    )


# --------------------------------------------------------------------------
# split_block_groups
# --------------------------------------------------------------------------
def test_split_block_groups_separates_attention_and_mamba():
    sched = _FakeScheduler(_has_mamba=True, _mamba_group_ids={2})
    # groups 0,1 = attention; group 2 = mamba recurrent-state slot.
    block_ids = ([1, 2], [3, 4], [99])
    attn, mamba = sched.split_block_groups(block_ids)
    assert attn == [1, 2, 3, 4]
    assert mamba == [99]


def test_split_block_groups_no_mamba_reduces_to_prior_behavior():
    sched = _FakeScheduler(_has_mamba=False, _mamba_group_ids=set())
    block_ids = ([5, 6, 7],)
    attn, mamba = sched.split_block_groups(block_ids)
    # Pure-attention model: mamba empty, attention == block_ids[0].
    assert attn == [5, 6, 7]
    assert mamba == []
    # Empty input is tolerated.
    assert _FakeScheduler(_has_mamba=False, _mamba_group_ids=set()).split_block_groups(
        ()
    ) == ([], [])


# --------------------------------------------------------------------------
# request_finished_all_groups
# --------------------------------------------------------------------------
def test_request_finished_all_groups_carries_attn_and_mamba_in_one_field():
    seen = {}

    def _fake_request_finished(request, attn_block_ids):
        seen["attn"] = list(attn_block_ids)
        return True, {"do_remote_prefill": True, "remote_block_ids": attn_block_ids}

    sched = _FakeScheduler(_has_mamba=True, _mamba_group_ids={1})
    sched.request_finished = _fake_request_finished
    conn = _FakeConnector(sched)

    request = SimpleNamespace(request_id="r0")
    block_ids = ([10, 11], [42])
    delay_free, params = conn.request_finished_all_groups(request, block_ids)

    assert delay_free is True
    # Attention ids drive request_finished; the mamba slot rides the SAME
    # remote_block_ids channel as [attn, mamba] -- no separate wire field, so
    # the proxy/router need no KDA-specific field.
    assert seen["attn"] == [10, 11]
    assert params["remote_block_ids"] == [[10, 11], [42]]
    assert "remote_mamba_block_ids" not in params
    # Round-trips through the consumer-side unpacker.
    assert as_attn_mamba(params["remote_block_ids"]) == ([10, 11], [42])


def test_request_finished_all_groups_pure_attention_stays_flat():
    def _fake_request_finished(request, attn_block_ids):
        return True, {"remote_block_ids": attn_block_ids}

    sched = _FakeScheduler(_has_mamba=False, _mamba_group_ids=set())
    sched.request_finished = _fake_request_finished
    conn = _FakeConnector(sched)

    delay_free, params = conn.request_finished_all_groups(
        SimpleNamespace(request_id="r1"), ([7, 8],)
    )
    # Pure attention: remote_block_ids stays a flat list; no KDA field added.
    assert params["remote_block_ids"] == [7, 8]
    assert "remote_mamba_block_ids" not in params
    assert as_attn_mamba(params["remote_block_ids"]) == ([7, 8], [])


# --------------------------------------------------------------------------
# as_attn_mamba (carried block-ids unpacker)
# --------------------------------------------------------------------------
def test_as_attn_mamba_unpacks_flat_and_paired():
    # Empty / None -> no blocks.
    assert as_attn_mamba(None) == ([], [])
    assert as_attn_mamba([]) == ([], [])
    # Flat list (attention-only / legacy) -> all attention, no mamba.
    assert as_attn_mamba([1, 2, 3]) == ([1, 2, 3], [])
    # [attn, mamba] pair (hybrid) unpacks both halves.
    assert as_attn_mamba([[1, 2], [9]]) == ([1, 2], [9])
    # Tuple form with empty mamba half.
    assert as_attn_mamba(([4, 5], [])) == ([4, 5], [])


# --------------------------------------------------------------------------
# _truncate_mamba_request_for_prefill
# --------------------------------------------------------------------------
def _mk_request(prompt, max_tokens=64, params=None):
    return SimpleNamespace(
        kv_transfer_params={} if params is None else params,
        num_prompt_tokens=len(prompt),
        prompt_token_ids=list(prompt),
        prompt_embeds=None,
        _all_token_ids=list(prompt),
        max_tokens=max_tokens,
    )


def test_truncate_mamba_request_pops_last_token_and_caps_tokens():
    sched = _FakeScheduler()
    req = _mk_request([10, 11, 12, 13, 14])
    sched._truncate_mamba_request_for_prefill(req)

    assert req.prompt_token_ids == [10, 11, 12, 13]  # last token popped
    assert req._all_token_ids == [10, 11, 12, 13]
    assert req.num_prompt_tokens == 4  # N-1 accounting
    assert req.max_tokens == 1
    assert req.kv_transfer_params["_p_side_truncated"] is True


def test_truncate_mamba_request_is_idempotent_across_reschedule():
    sched = _FakeScheduler()
    req = _mk_request([10, 11, 12])
    sched._truncate_mamba_request_for_prefill(req)
    assert req.num_prompt_tokens == 2
    # A second call (e.g. after a preemption) must not truncate again.
    sched._truncate_mamba_request_for_prefill(req)
    assert req.num_prompt_tokens == 2
    assert req.prompt_token_ids == [10, 11]


def test_truncate_mamba_request_noop_for_single_token_prompt():
    sched = _FakeScheduler()
    req = _mk_request([10])
    sched._truncate_mamba_request_for_prefill(req)
    assert req.num_prompt_tokens == 1
    assert req.prompt_token_ids == [10]
    assert "_p_side_truncated" not in req.kv_transfer_params


# --------------------------------------------------------------------------
# get_num_new_matched_tokens  (hybrid N-1 accounting)
# --------------------------------------------------------------------------
def test_get_num_new_matched_tokens_read_recomputes_last_token():
    sched = _FakeScheduler(is_producer=False, mode=MoRIIOMode.READ, _has_mamba=True)
    req = SimpleNamespace(prompt_token_ids=list(range(10)), kv_transfer_params=None)
    n, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    # READ always recomputes the final token locally: N-1 - computed.
    assert n == 9
    assert is_async is False


def test_get_num_new_matched_tokens_write_hybrid_drops_last_token():
    sched = _FakeScheduler(is_producer=False, mode=MoRIIOMode.WRITE, _has_mamba=True)
    req = SimpleNamespace(prompt_token_ids=list(range(10)), kv_transfer_params=None)
    n, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    # Hybrid WRITE: decoder recomputes final token from pushed KDA state -> N-1.
    assert n == 9
    assert is_async is True


def test_get_num_new_matched_tokens_write_plain_keeps_all_tokens():
    sched = _FakeScheduler(is_producer=False, mode=MoRIIOMode.WRITE, _has_mamba=False)
    req = SimpleNamespace(prompt_token_ids=list(range(10)), kv_transfer_params=None)
    n, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=2)
    # Pure-attention WRITE: no N-1 drop; full length minus already-computed.
    assert n == 8
    assert is_async is True


def test_get_num_new_matched_tokens_producer_truncates_and_returns_zero():
    sched = _FakeScheduler(is_producer=True, mode=MoRIIOMode.WRITE, _has_mamba=True)
    req = _mk_request(list(range(5)), params={"do_remote_decode": True})
    n, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert (n, is_async) == (0, False)
    # Producer stops at h(N-1): the last prompt token was dropped.
    assert req.num_prompt_tokens == 4
    assert req.max_tokens == 1
    assert req.kv_transfer_params["_p_side_truncated"] is True


# --------------------------------------------------------------------------
# C2: cached offset template == freshly recomputed (byte-exactness)
# --------------------------------------------------------------------------
_SAMPLE_SLOT_SETS = [
    ([1, 2], [1, 2]),
    ([0], [3]),
    ([3, 0, 2], [3, 0, 2]),
    ([2, 2], [5, 5]),
    ([], []),
]


@pytest.mark.parametrize("local_slots,remote_slots", _SAMPLE_SLOT_SETS)
def test_offset_template_apply_matches_direct_compute(local_slots, remote_slots):
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(8, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(8, (2, 4, 4), slot_stride=64)

    template = moriio_layout.build_mamba_offset_template(
        "kda.0", conv, ssm, {}, split, tp_ratio=1, tp_rank=0, world_size=1
    )
    cached = moriio_layout.apply_mamba_offset_template(
        template, local_slots, remote_slots
    )
    fresh = moriio_layout.compute_mamba_conv_ssm_offsets(
        "kda.0", conv, ssm, {}, local_slots, remote_slots, split,
        tp_ratio=1, tp_rank=0, world_size=1,
    )
    assert cached == fresh


def test_worker_compute_mamba_offsets_caches_and_matches_recompute():
    split = _gdn_split_info()
    conv_dim = sum(split.local_proj_dims)
    conv = _slot_strided(8, (conv_dim, split.conv_rows), slot_stride=200)
    ssm = _slot_strided(8, (2, 4, 4), slot_stride=64)

    worker = _FakeWorker(
        kv_caches={"kda.0": (conv, ssm)},
        layer_to_spec={},
        _conv_decomp=split,
        tp_rank=0,
        world_size=1,
    )

    for local_slots, remote_slots in _SAMPLE_SLOT_SETS:
        lo, ro, sz, n_conv = worker._compute_mamba_transfer_offsets(
            "kda.0", local_slots, remote_slots
        )
        fresh = moriio_layout.compute_mamba_conv_ssm_offsets(
            "kda.0", conv, ssm, {}, local_slots, remote_slots, split,
            tp_ratio=1, tp_rank=0, world_size=1,
        )
        assert (lo, ro, sz) == fresh
        assert n_conv == moriio_layout.compute_mamba_conv_split_count(
            local_slots, split
        )

    # The per-(layer, tp_ratio) template is cached after the first call.
    assert ("kda.0", 1) in worker._mamba_offset_templates
