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


class _FakeScheduler(moriio_connector.MoRIIOConnectorScheduler):  # type: ignore[name-defined]
    """Constructs a scheduler without the heavy real ``__init__``; only the
    attributes read by the method under test are set."""

    def __init__(self, **attrs):
        self._mamba_group_ids: list[int] = []
        self._attn_group_ids: list[int] = [0]
        self._ssm_state_slots_are_positional = False
        self._is_hma_required = False
        self.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[None, None],
            select_transfer_block_ids=lambda block_ids: tuple(block_ids),
        )
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


def _mamba_spec(num_states: int = 2):
    shapes = tuple((1, 1) for _ in range(num_states))
    return moriio_connector.MambaSpec(
        block_size=16,
        shapes=shapes,
        dtypes=(torch.float32,) * num_states,
        mamba_cache_mode="all",
    )


def _gate_vllm_config(*, read_mode: bool = True, speculative_config=None):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"read_mode": read_mode}
        ),
        speculative_config=speculative_config,
    )


# --------------------------------------------------------------------------
# split_block_groups
# --------------------------------------------------------------------------
def test_split_block_groups_separates_attention_and_mamba():
    sched = _FakeScheduler(_has_mamba=True, _attn_group_ids=[0], _mamba_group_ids=[1])
    block_ids = ([1, 2], [99])
    attn, mamba = sched.split_block_groups(block_ids)
    assert attn == [1, 2]
    assert mamba == [[99]]


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
        ],
    )
    config.transfer_groups = tuple(
        group for group in config.kv_cache_groups if group.enable_kv_transfer
    )
    config.select_transfer_block_ids = lambda block_ids: tuple(
        block_ids[group_id]
        for group_id, group in enumerate(config.kv_cache_groups)
        if group.enable_kv_transfer
    )
    attn_groups, mamba_groups = moriio_connector._split_kv_cache_group_kinds(config)
    assert attn_groups == [0]
    assert mamba_groups == [1]

    sched = _FakeScheduler(
        _has_mamba=True,
        _attn_group_ids=attn_groups,
        _mamba_group_ids=mamba_groups,
        kv_cache_config=config,
    )
    assert sched.split_block_groups(([1, 2], [70, 71], [99])) == (
        [1, 2],
        [[99]],
    )


def test_exchange_blocks_ignore_transfer_disabled_group():
    config = SimpleNamespace(
        kv_cache_groups=[None, None, None],
        select_transfer_block_ids=lambda block_ids: (block_ids[0], block_ids[2]),
    )
    sched = _FakeScheduler(
        kv_cache_config=config,
        _is_hma_required=True,
        blocks_per_sw=[0, 2],
    )

    assert sched.get_exchange_clipped_blocks(([1, 2], [70, 71], [90, 91, 92])) == [
        [1, 2],
        [91, 92],
    ]


def test_scheduler_rejects_multiple_attention_groups_with_mamba():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=object()),
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=object()),
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=_mamba_spec()),
        ],
    )
    config.transfer_groups = tuple(config.kv_cache_groups)

    with pytest.raises(moriio_common.MoRIIOError, match="exactly one transferable"):
        moriio_connector.MoRIIOConnectorScheduler(_gate_vllm_config(), "engine", config)


def test_split_block_groups_preserves_multiple_mamba_groups():
    sched = _FakeScheduler(
        _has_mamba=True,
        _attn_group_ids=[0],
        _mamba_group_ids=[1, 2],
    )

    assert sched.split_block_groups(([1, 2], [40], [70])) == (
        [1, 2],
        [[40], [70]],
    )


def test_scheduler_rejects_incompatible_mamba_groups():
    incompatible = _mamba_spec()
    object.__setattr__(incompatible, "page_size_padded", 4096)
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=object()),
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=_mamba_spec()),
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=incompatible),
        ]
    )
    config.transfer_groups = tuple(config.kv_cache_groups)

    with pytest.raises(moriio_common.MoRIIOError, match="same cache spec"):
        moriio_connector.MoRIIOConnectorScheduler(_gate_vllm_config(), "engine", config)


def test_scheduler_rejects_mamba_group_without_two_states():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=object()),
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=_mamba_spec(1)),
        ],
    )
    config.transfer_groups = tuple(config.kv_cache_groups)

    with pytest.raises(moriio_common.MoRIIOError, match="exactly two Mamba states"):
        moriio_connector.MoRIIOConnectorScheduler(_gate_vllm_config(), "engine", config)


def test_scheduler_rejects_speculative_hybrid_read():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=object()),
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=_mamba_spec()),
        ],
    )
    config.transfer_groups = tuple(config.kv_cache_groups)

    with pytest.raises(moriio_common.MoRIIOError, match="speculative decoding"):
        moriio_connector.MoRIIOConnectorScheduler(
            _gate_vllm_config(speculative_config=object()), "engine", config
        )


def test_scheduler_rejects_hybrid_write():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=object()),
            SimpleNamespace(enable_kv_transfer=True, kv_cache_spec=_mamba_spec()),
        ],
    )
    config.transfer_groups = tuple(config.kv_cache_groups)

    with pytest.raises(moriio_common.MoRIIOError, match="READ mode only"):
        moriio_connector.MoRIIOConnectorScheduler(
            _gate_vllm_config(read_mode=False), "engine", config
        )


def test_split_block_groups_rejects_non_hybrid_use():
    sched = _FakeScheduler(_has_mamba=False)

    with pytest.raises(moriio_common.MoRIIOError, match="non-hybrid"):
        sched.split_block_groups(([5, 6, 7],))


def test_split_block_groups_accepts_empty_abort_payload():
    sched = _FakeScheduler(_has_mamba=True)

    assert sched.split_block_groups(()) == ([], [])


def test_split_block_groups_keeps_positional_slots_in_all_mode():
    # mamba_cache_mode="all" keeps a state per block position.
    sched = _FakeScheduler(
        _has_mamba=True,
        _attn_group_ids=[0],
        _mamba_group_ids=[1],
        _ssm_state_slots_are_positional=True,
    )
    attn, mamba = sched.split_block_groups(([1], [40, 41, 42]))
    assert attn == [1]
    assert mamba == [[40, 41, 42]]


def test_split_block_groups_keeps_only_running_state_outside_all_mode():
    sched = _FakeScheduler(
        _has_mamba=True,
        _attn_group_ids=[0],
        _mamba_group_ids=[1],
    )
    assert sched.split_block_groups(([1], [40, 41])) == ([1], [[41]])


# --------------------------------------------------------------------------
# request_finished_all_groups
# --------------------------------------------------------------------------
def test_request_finished_all_groups_carries_attn_and_mamba_in_one_field():
    seen = {}

    def _fake_request_finished(request, attn_block_ids, mamba_block_groups):
        seen["attn"] = list(attn_block_ids)
        seen["mamba"] = [list(group) for group in mamba_block_groups]
        return True, {
            "do_remote_prefill": True,
            "remote_block_ids": [attn_block_ids, *mamba_block_groups],
        }

    sched = _FakeScheduler(_has_mamba=True, _attn_group_ids=[0], _mamba_group_ids=[1])
    sched.request_finished = _fake_request_finished
    conn = _FakeConnector(sched)

    request = SimpleNamespace(request_id="r0")
    block_ids = ([10, 11], [42])
    delay_free, params = conn.request_finished_all_groups(request, block_ids)

    assert delay_free is True
    # Attention ids drive request_finished; the mamba slot rides the SAME
    # remote_block_ids channel as [attn, *mamba_groups], so the proxy/router
    # needs no KDA-specific field.
    assert seen["attn"] == [10, 11]
    assert seen["mamba"] == [[42]]
    assert params["remote_block_ids"] == [[10, 11], [42]]
    assert "remote_mamba_block_ids" not in params


def test_request_finished_all_groups_pure_attention_stays_grouped():
    def _fake_request_finished(request, block_ids):
        return True, {"remote_block_ids": block_ids}

    sched = _FakeScheduler(_has_mamba=False, _mamba_group_ids=[])
    sched.request_finished = _fake_request_finished
    conn = _FakeConnector(sched)

    delay_free, params = conn.request_finished_all_groups(
        SimpleNamespace(request_id="r1"), ([7, 8],)
    )
    # Main's HMA contract keeps pure-attention block ids grouped.
    assert params["remote_block_ids"] == ([7, 8],)
    assert "remote_mamba_block_ids" not in params


def test_connector_supports_divergent_local_hybrid_hits():
    connector = _FakeConnector(None)
    connector._transfers_mamba_state = True
    connector.mode = MoRIIOMode.READ
    assert connector.supports_divergent_local_hybrid_hits is True
    connector.mode = MoRIIOMode.WRITE
    assert connector.supports_divergent_local_hybrid_hits is False

    connector.mode = MoRIIOMode.READ
    connector._transfers_mamba_state = False
    assert connector.supports_divergent_local_hybrid_hits is False


def _make_read_scheduler():
    return _FakeScheduler(
        is_producer=False,
        mode=MoRIIOMode.READ,
        _has_mamba=True,
        _attn_group_ids=[0],
        _mamba_group_ids=[1],
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


def test_update_state_preserves_each_mamba_group():
    sched = _make_read_scheduler()
    sched._mamba_group_ids = [1, 2]
    request = _make_read_request([[10], [90], [80]])

    sched.update_state_after_alloc(
        request,
        _FakeBlocks(all_groups=([100], [200], [300])),
        num_external_tokens=64,
    )

    assert sched._reqs_need_recv["req"][1] == [[100], [200], [300]]
    assert sched._req_kv_params["req"]["remote_block_ids"] == [
        [10],
        [90],
        [80],
    ]


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
    sched._mamba_group_ids = []
    request = _make_read_request([[10, 11]])

    sched.update_state_after_alloc(
        request,
        _FakeBlocks(all_groups=([100, 101, 102],)),
        num_external_tokens=256,
    )

    assert sched._reqs_need_recv["req"][1] == [[100, 101]]
    assert sched._req_kv_params["req"]["remote_block_ids"] == [[10, 11]]


def test_read_producer_does_not_queue_write_only_save_state():
    sched = _FakeScheduler(
        is_producer=True,
        mode=MoRIIOMode.READ,
        _has_mamba=True,
        _attn_group_ids=[0],
        _mamba_group_ids=[1],
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


def test_write_producer_queues_grouped_attention_blocks():
    sched = _FakeScheduler(
        is_producer=True,
        mode=MoRIIOMode.WRITE,
        _has_mamba=False,
        _attn_group_ids=[0],
        _mamba_group_ids=[],
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

    assert sched._reqs_need_save["req"][1] == ([100, 101],)
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
    worker = _FakeWorker(mode=MoRIIOMode.WRITE, _transfer_layer_names={"kda.0"})
    worker._is_mamba_layer = lambda _layer_name: True

    with pytest.raises(moriio_common.MoRIIOError, match="READ mode only"):
        worker.register_kv_caches({"kda.0": object()})


def test_register_kv_caches_rejects_no_transfer_enabled_layers():
    worker = _FakeWorker(mode=MoRIIOMode.READ, _transfer_layer_names=set())

    with pytest.raises(moriio_common.MoRIIOError, match="no transfer-enabled"):
        worker.register_kv_caches({"disabled.0": object()})


def test_worker_rejects_incompatible_mamba_specs_before_registration():
    incompatible = _mamba_spec()
    object.__setattr__(incompatible, "page_size_padded", 4096)
    worker = _FakeWorker(
        mode=MoRIIOMode.READ,
        world_size=1,
        _transfer_layer_names={"kda.0", "kda.1"},
        layer_to_spec={"kda.0": _mamba_spec(), "kda.1": incompatible},
    )

    with pytest.raises(moriio_common.MoRIIOError, match="same cache spec"):
        worker.register_kv_caches({"kda.0": object(), "kda.1": object()})


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
    conv_shape = (conv_dim, split.conv_rows)
    ssm_shape = (2, 4, 4)
    page_bytes = 256
    pages = torch.zeros((8, 1, 1, page_bytes), dtype=torch.uint8)
    spec = moriio_connector.MambaSpec(
        block_size=1,
        shapes=(conv_shape, ssm_shape),
        dtypes=(torch.bfloat16, torch.bfloat16),
        page_size_padded=page_bytes,
    )
    conv, ssm = moriio_layout.kda_conv_ssm(pages, spec)

    worker = _FakeWorker(
        kv_caches={"kda.0": pages},
        layer_to_spec={"kda.0": spec},
        _conv_decomp=split,
        _mamba_offset_templates={},
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

        current_template = worker._mamba_offset_templates["kda.0"]
        if cached_template is None:
            cached_template = current_template
        else:
            assert current_template is cached_template

    assert "kda.0" in worker._mamba_offset_templates


@pytest.mark.parametrize("remote_tp_size", [1, None])
def test_mamba_tp_ratio_accepts_only_equal_tp(remote_tp_size):
    worker = _FakeWorker(world_size=1)
    assert worker._mamba_tp_ratio(remote_tp_size) == 1


@pytest.mark.parametrize("remote_tp_size", [2, 3])
def test_mamba_tp_ratio_rejects_heterogeneous_tp(remote_tp_size):
    worker = _FakeWorker(world_size=1)
    with pytest.raises(NotImplementedError, match="heterogeneous-TP"):
        worker._mamba_tp_ratio(remote_tp_size)


def test_worker_selects_each_mamba_layers_own_block_group():
    worker = _FakeWorker(_mamba_payload_index_by_layer={"kda.0": 1, "kda.1": 2})
    groups = ([10], [40], [70])

    assert worker._mamba_blocks_for_layer("kda.0", groups) == [40]
    assert worker._mamba_blocks_for_layer("kda.1", groups) == [70]


def test_worker_rejects_missing_mamba_block_group():
    worker = _FakeWorker(_mamba_payload_index_by_layer={"kda.0": 2})

    with pytest.raises(moriio_common.MoRIIOError, match="no block group"):
        worker._mamba_blocks_for_layer("kda.0", ([10], [40]))
