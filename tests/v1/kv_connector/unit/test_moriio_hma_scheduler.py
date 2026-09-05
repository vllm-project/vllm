# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-CPU unit tests for the MoRIIO hybrid (attention + mamba/KDA) scheduler
and the KDA offset-template cache.

These exercise the highest-risk, otherwise-untested scheduler logic without a
GPU or the ``mori`` runtime: per-group block-id splitting, the READ/WRITE
``N-1`` token accounting, P-side prompt truncation, and offset-template cache
wiring.

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
split_attn_mamba_block_ids = moriio_common.split_attn_mamba_block_ids


class _FakeScheduler(moriio_connector.MoRIIOConnectorScheduler):  # type: ignore[name-defined]
    """Constructs a scheduler without the heavy real ``__init__``; only the
    attributes read by the method under test are set."""

    def __init__(self, **attrs):
        # Mamba slot-clipping inputs default to "no scratch slots, single
        # running state", which is what a non-spec-decode deployment has.
        self._mamba_group_ids: set[int] = set()
        self._attn_group_ids: set[int] = set()
        self._ssm_spec_blocks: list = []
        self._ssm_state_slots_are_positional = False
        for k, v in attrs.items():
            setattr(self, k, v)


class _FakeConnector(moriio_connector.MoRIIOConnector):  # type: ignore[name-defined]
    def __init__(self, connector_scheduler):
        self.connector_scheduler = connector_scheduler


class _FakeWorker(moriio_connector.MoRIIOConnectorWorker):  # type: ignore[name-defined]
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


class _FakeBlocks:
    def __init__(self, all_groups):
        self._all_groups = all_groups

    def get_block_ids(self):
        return self._all_groups


def _slot_strided(num_slots, per_slot_shape, slot_stride, dtype=torch.bfloat16):
    inner = 1
    for d in per_slot_shape:
        inner *= d
    assert slot_stride >= inner
    backing = torch.zeros(num_slots * slot_stride, dtype=dtype)
    inner_strides: list[int] = []
    acc = 1
    for d in reversed(per_slot_shape):
        inner_strides.insert(0, acc)
        acc *= d
    return backing.as_strided(
        (num_slots, *per_slot_shape), (slot_stride, *inner_strides)
    )


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
    sched = _FakeScheduler(
        _has_mamba=True, _attn_group_ids={0, 1}, _mamba_group_ids={2}
    )
    # groups 0,1 = attention; group 2 = mamba recurrent-state slot.
    block_ids = ([1, 2], [3, 4], [99])
    attn, mamba = sched.split_block_groups(block_ids)
    assert attn == [1, 2, 3, 4]
    assert mamba == [99]


def test_split_block_groups_ignores_transfer_disabled_group():
    mamba_spec = moriio_connector.MambaSpec(
        block_size=16,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="all",
    )
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=object()),
            SimpleNamespace(enable_kv_transfer=False, kv_cache_spec=object()),
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=mamba_spec),
        ]
    )
    attn_groups, mamba_groups = moriio_connector._split_kv_cache_group_kinds(config)
    assert attn_groups == [0]
    assert mamba_groups == [2]

    sched = _FakeScheduler(
        _has_mamba=True,
        _attn_group_ids=set(attn_groups),
        _mamba_group_ids=set(mamba_groups),
    )
    assert sched.split_block_groups(([1, 2], [70, 71], [99])) == ([1, 2], [99])


def test_scheduler_rejects_multiple_attention_groups():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=object()),
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=object()),
        ]
    )

    with pytest.raises(moriio_common.MoRIIOError, match="single attention"):
        moriio_connector.MoRIIOConnectorScheduler(SimpleNamespace(), "engine", config)


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


@pytest.mark.parametrize(
    ("attn_group_ids", "expected"),
    [({1}, [20, 21]), (set(), [])],
)
def test_split_block_groups_no_mamba_honors_transfer_groups(attn_group_ids, expected):
    sched = _FakeScheduler(
        kv_cache_config=object(),
        _has_mamba=False,
        _attn_group_ids=attn_group_ids,
        _mamba_group_ids=set(),
    )

    assert sched.split_block_groups(([10, 11], [20, 21])) == (expected, [])


def test_split_block_groups_drops_speculative_scratch_slots():
    # Group 1 is mamba with 2 trailing speculative scratch slots; only the
    # running state (the last non-speculative slot) is transferable.
    sched = _FakeScheduler(
        _has_mamba=True,
        _attn_group_ids={0},
        _mamba_group_ids={1},
        _ssm_spec_blocks=[None, 2],
    )
    attn, mamba = sched.split_block_groups(([1, 2], [40, 41, 42]))
    assert attn == [1, 2]
    assert mamba == [40]


def test_split_block_groups_keeps_positional_slots_in_all_mode():
    # mamba_cache_mode="all" keeps a state per block position, so only the
    # speculative tail is stripped.
    sched = _FakeScheduler(
        _has_mamba=True,
        _attn_group_ids={0},
        _mamba_group_ids={1},
        _ssm_spec_blocks=[None, 1],
        _ssm_state_slots_are_positional=True,
    )
    attn, mamba = sched.split_block_groups(([1], [40, 41, 42]))
    assert attn == [1]
    assert mamba == [40, 41]


def test_split_block_groups_never_strips_the_only_slot():
    sched = _FakeScheduler(
        _has_mamba=True,
        _attn_group_ids={0},
        _mamba_group_ids={1},
        _ssm_spec_blocks=[None, 4],
    )
    # Fewer slots than the speculative reservation: the running state stays.
    assert sched.split_block_groups(([1], [40])) == ([1], [40])


# --------------------------------------------------------------------------
# request_finished_all_groups
# --------------------------------------------------------------------------
def test_request_finished_all_groups_carries_attn_and_mamba_in_one_field():
    seen = {}

    def _fake_request_finished(request, attn_block_ids, mamba_block_ids):
        seen["attn"] = list(attn_block_ids)
        seen["mamba"] = list(mamba_block_ids)
        return True, {"do_remote_prefill": True, "remote_block_ids": attn_block_ids}

    sched = _FakeScheduler(_has_mamba=True, _attn_group_ids={0}, _mamba_group_ids={1})
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
    assert seen["mamba"] == [42]
    assert params["remote_block_ids"] == [[10, 11], [42]]
    assert "remote_mamba_block_ids" not in params
    # Round-trips through the consumer-side unpacker.
    assert split_attn_mamba_block_ids(params["remote_block_ids"]) == ([10, 11], [42])


def test_request_finished_all_groups_pure_attention_stays_flat():
    def _fake_request_finished(request, attn_block_ids, mamba_block_ids):
        assert mamba_block_ids == []
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
    assert split_attn_mamba_block_ids(params["remote_block_ids"]) == ([7, 8], [])


def test_connector_supports_divergent_local_hybrid_hits():
    connector = _FakeConnector(None)
    connector.mode = MoRIIOMode.READ
    assert connector.supports_divergent_local_hybrid_hits is True
    connector.mode = MoRIIOMode.WRITE
    assert connector.supports_divergent_local_hybrid_hits is False


def _make_read_scheduler():
    return _FakeScheduler(
        is_producer=False,
        mode=MoRIIOMode.READ,
        _has_mamba=True,
        _attn_group_ids={0},
        _mamba_group_ids={1},
        _ssm_spec_blocks=[None, None],
        _ssm_state_slots_are_positional=True,
        request_id_to_transfer_id={},
        transfer_id_to_request_id={},
        _reqs_need_recv={},
        _req_kv_params={},
        _max_decode_tail_blocks=1,
    )


def _make_read_request(remote_block_ids):
    return SimpleNamespace(
        request_id="req",
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": "tx",
            "remote_engine_id": "prefill",
            "remote_block_ids": remote_block_ids,
        },
    )


def test_update_state_drops_decode_recompute_tail_block():
    sched = _make_read_scheduler()
    request = _make_read_request([[10, 11], [90, 91]])
    blocks = _FakeBlocks(
        all_groups=([100, 101, 102], [200, 201, 202]),
    )

    sched.update_state_after_alloc(request, blocks, num_external_tokens=256)

    assert sched._reqs_need_recv["req"][1] == [[100, 101], [200, 201]]
    assert sched._req_kv_params["req"]["remote_block_ids"] == [
        [10, 11],
        [90, 91],
    ]


def test_update_state_pairs_shorter_local_blocks_with_remote_suffix():
    sched = _make_read_scheduler()
    request = _make_read_request([[10, 11, 12], [92]])
    blocks = _FakeBlocks(
        all_groups=([102], [202]),
    )

    sched.update_state_after_alloc(request, blocks, num_external_tokens=64)

    assert sched._reqs_need_recv["req"][1] == [[102], [202]]
    assert sched._req_kv_params["req"]["remote_block_ids"] == [[12], [92]]


def test_update_state_full_attention_hit_still_carries_mamba_state():
    sched = _make_read_scheduler()
    request = _make_read_request([[10], [90]])

    sched.update_state_after_alloc(
        request,
        _FakeBlocks(all_groups=([100], [200])),
        num_external_tokens=0,
    )

    assert sched._reqs_need_recv["req"][1] == [[], [200]]


def test_align_read_blocks_allows_configured_lookahead_tail():
    assert moriio_connector.MoRIIOConnectorScheduler._align_read_blocks(
        [100, 101, 102], [10], max_decode_tail_blocks=2
    ) == ([100], [10])
    with pytest.raises(ValueError, match="allowed local tail"):
        moriio_connector.MoRIIOConnectorScheduler._align_read_blocks(
            [100, 101, 102], [10], max_decode_tail_blocks=1
        )


def test_update_state_aligns_attention_only_decode_tail():
    sched = _make_read_scheduler()
    sched._has_mamba = False
    sched._mamba_group_ids = set()
    request = _make_read_request([10, 11])

    sched.update_state_after_alloc(
        request,
        _FakeBlocks(all_groups=([100, 101, 102],)),
        num_external_tokens=256,
    )

    assert sched._reqs_need_recv["req"][1] == [[100, 101], []]
    assert sched._req_kv_params["req"]["remote_block_ids"] == [10, 11]


def test_read_producer_does_not_queue_write_only_save_state():
    sched = _FakeScheduler(
        is_producer=True,
        mode=MoRIIOMode.READ,
        _has_mamba=True,
        _attn_group_ids={0},
        _mamba_group_ids={1},
        request_id_to_transfer_id={},
        transfer_id_to_request_id={},
        _reqs_need_save={},
        _req_kv_params={},
    )
    request = SimpleNamespace(
        request_id="req",
        kv_transfer_params={"do_remote_decode": True, "transfer_id": "tx"},
    )

    sched.update_state_after_alloc(
        request,
        _FakeBlocks(all_groups=([100, 101], [200])),
        num_external_tokens=0,
    )

    assert sched._reqs_need_save == {}
    assert sched._req_kv_params == {}


def test_write_producer_queues_flat_attention_blocks():
    sched = _FakeScheduler(
        is_producer=True,
        mode=MoRIIOMode.WRITE,
        _has_mamba=False,
        _attn_group_ids={0},
        _mamba_group_ids=set(),
        request_id_to_transfer_id={},
        transfer_id_to_request_id={},
        _reqs_need_save={},
        _req_kv_params={},
    )
    request = SimpleNamespace(
        request_id="req",
        kv_transfer_params={"do_remote_decode": True, "transfer_id": "tx"},
    )

    sched.update_state_after_alloc(
        request,
        _FakeBlocks(all_groups=([100, 101],)),
        num_external_tokens=0,
    )

    assert sched._reqs_need_save["req"][1] == [100, 101]
    assert sched._req_kv_params["req"] == request.kv_transfer_params


def test_session_build_rejects_per_layer_region_count_mismatch():
    worker = _FakeWorker(
        built_write_session={},
        layer_name_to_local_kv_cache_metadata={"kda.0": ["conv", "ssm"]},
        layer_name_to_remote_kv_cache_metadata={"prefill": {"kda.0": ["conv"]}},
    )

    with pytest.raises(moriio_common.MoRIIOError, match="registered 1 region"):
        worker._get_built_session("prefill")


def test_register_kv_caches_rejects_hybrid_write_before_registration():
    worker = _FakeWorker(mode=MoRIIOMode.WRITE)
    worker._is_mamba_layer = lambda _layer_name: True

    with pytest.raises(moriio_common.MoRIIOError, match="READ mode only"):
        worker.register_kv_caches({"kda.0": object()})


# --------------------------------------------------------------------------
# split_attn_mamba_block_ids (carried block-ids unpacker)
# --------------------------------------------------------------------------
def test_split_attn_mamba_block_ids_unpacks_flat_and_paired():
    # Empty / None -> no blocks.
    assert split_attn_mamba_block_ids(None) == ([], [])
    assert split_attn_mamba_block_ids([]) == ([], [])
    # Flat list (attention-only / legacy) -> all attention, no mamba.
    assert split_attn_mamba_block_ids([1, 2, 3]) == ([1, 2, 3], [])
    # [attn, mamba] pair (hybrid) unpacks both halves.
    assert split_attn_mamba_block_ids([[1, 2], [9]]) == ([1, 2], [9])
    # Tuple form with empty mamba half.
    assert split_attn_mamba_block_ids(([4, 5], [])) == ([4, 5], [])


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
    req = SimpleNamespace(
        num_prompt_tokens=10,
        prompt_token_ids=list(range(10)),
        kv_transfer_params=None,
    )
    n, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    # READ always recomputes the final token locally: N-1 - computed.
    assert n == 9
    assert is_async is False


def test_get_num_new_matched_tokens_write_is_unchanged():
    sched = _FakeScheduler(is_producer=False, mode=MoRIIOMode.WRITE, _has_mamba=True)
    req = SimpleNamespace(
        num_prompt_tokens=10,
        prompt_token_ids=list(range(10)),
        kv_transfer_params=None,
    )
    n, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    # This PR adds hybrid READ support and does not change WRITE accounting.
    assert n == 10
    assert is_async is True


def test_get_num_new_matched_tokens_write_plain_keeps_all_tokens():
    sched = _FakeScheduler(is_producer=False, mode=MoRIIOMode.WRITE, _has_mamba=False)
    req = SimpleNamespace(
        num_prompt_tokens=10,
        prompt_token_ids=list(range(10)),
        kv_transfer_params=None,
    )
    n, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=2)
    # Pure-attention WRITE: no N-1 drop; full length minus already-computed.
    assert n == 8
    assert is_async is True


@pytest.mark.parametrize(
    ("mode", "num_computed_tokens", "expected", "is_async"),
    [
        (MoRIIOMode.READ, 0, 9, False),
        (MoRIIOMode.READ, 10, 0, False),
        (MoRIIOMode.WRITE, 2, 8, True),
    ],
)
def test_get_num_new_matched_tokens_supports_embeds_only_prompts(
    mode, num_computed_tokens, expected, is_async
):
    sched = _FakeScheduler(is_producer=False, mode=mode, _has_mamba=True)
    req = SimpleNamespace(
        num_prompt_tokens=10,
        prompt_token_ids=None,
        prompt_embeds=object(),
        kv_transfer_params=None,
    )

    assert sched.get_num_new_matched_tokens(req, num_computed_tokens) == (
        expected,
        is_async,
    )


def test_get_num_new_matched_tokens_producer_returns_zero():
    sched = _FakeScheduler(is_producer=True, mode=MoRIIOMode.WRITE, _has_mamba=True)
    req = _mk_request(list(range(5)), params={"do_remote_decode": True})
    n, is_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert (n, is_async) == (0, False)
    # The producer never truncates here: the scheduler has already measured the
    # prefix-cache hit against the full prompt by this point.
    assert req.num_prompt_tokens == 5


def test_on_new_request_truncates_producer_prompt():
    sched = _FakeScheduler(is_producer=True, mode=MoRIIOMode.WRITE, _has_mamba=True)
    req = _mk_request(list(range(5)), params={"do_remote_decode": True})
    sched.on_new_request(req)
    # Producer stops at h(N-1): the last prompt token was dropped.
    assert req.num_prompt_tokens == 4
    assert req.max_tokens == 1
    assert req.kv_transfer_params["_p_side_truncated"] is True


def test_on_new_request_is_noop_without_mamba_or_on_decode():
    plain = _FakeScheduler(is_producer=True, mode=MoRIIOMode.WRITE, _has_mamba=False)
    req = _mk_request(list(range(5)), params={"do_remote_decode": True})
    plain.on_new_request(req)
    assert req.num_prompt_tokens == 5

    decode_side = _FakeScheduler(
        is_producer=False, mode=MoRIIOMode.WRITE, _has_mamba=True
    )
    req2 = _mk_request(list(range(5)), params={"do_remote_prefill": True})
    decode_side.on_new_request(req2)
    assert req2.num_prompt_tokens == 5


# --------------------------------------------------------------------------
# Offset-template cache
# --------------------------------------------------------------------------
_SAMPLE_SLOT_SETS = [
    ([1, 2], [1, 2]),
    ([0], [3]),
    ([3, 0, 2], [3, 0, 2]),
    ([2, 2], [5, 5]),
    ([], []),
]


def test_worker_compute_mamba_offsets_caches_and_matches_template():
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

    cached_template = None
    for local_slots, remote_slots in _SAMPLE_SLOT_SETS:
        lo, ro, sz, n_conv = worker._compute_mamba_transfer_offsets(
            "kda.0", local_slots, remote_slots
        )
        template = moriio_layout.build_mamba_offset_template(
            conv, ssm, split, tp_ratio=1
        )
        expected = moriio_layout.apply_mamba_offset_template(
            template, local_slots, remote_slots
        )
        assert (lo, ro, sz) == expected
        assert n_conv == moriio_layout.compute_mamba_conv_split_count(
            local_slots, split
        )

        current_template = worker._mamba_offset_templates[("kda.0", 1)]
        if cached_template is None:
            cached_template = current_template
        else:
            assert current_template is cached_template

    # The per-(layer, tp_ratio) template is cached after the first call.
    assert ("kda.0", 1) in worker._mamba_offset_templates
