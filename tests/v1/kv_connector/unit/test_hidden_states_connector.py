# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for ExampleHiddenStatesConnector KV-cache-group logic."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
    PendingSave,
)
from vllm.v1.core.kv_cache_planning import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


def _full(block_size: int) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=128, dtype=torch.bfloat16
    )


def _hidden(block_size: int) -> HiddenStateCacheSpec:
    return HiddenStateCacheSpec(
        block_size=block_size, num_kv_heads=6, head_size=2048, dtype=torch.bfloat16
    )


def _config(*specs):
    """Minimal stand-in exposing only `kv_cache_groups` (all the helpers read)."""
    return SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[f"layer.{i}"], kv_cache_spec=spec)
            for i, spec in enumerate(specs)
        ]
    )


# ---- _find_cache_kv_group_id ------------------------------------------------


def test_find_group_id_none_config_returns_zero():
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(None) == 0


def test_find_group_id_single_non_hidden_group_returns_zero():
    # Uniform (dense) model: one group, no HiddenStateCacheSpec -> group 0.
    cfg = _config(_full(16))
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == 0


def test_find_group_id_locates_hidden_group_when_not_first():
    # Hybrid layout: the hidden-states group is not group 0.
    cfg = _config(_full(528), _hidden(22), _full(528))
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == 1


def test_find_group_id_locates_hidden_group_last():
    cfg = _config(_full(528), _full(528), _hidden(22))
    assert ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == 2


def test_find_group_id_raises_when_no_hidden_group_and_multiple_groups():
    cfg = _config(_full(16), _full(16))
    with pytest.raises(ValueError, match="Could not uniquely identify"):
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg)


def test_find_group_id_raises_when_multiple_hidden_groups():
    cfg = _config(_hidden(22), _hidden(22))
    with pytest.raises(ValueError, match="Could not uniquely identify"):
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg)


# ---- _get_cache_block_size --------------------------------------------------


def test_get_block_size_reads_hidden_group_spec_not_global():
    # Hidden group keeps block size 22; the global is bumped to 528 for hybrids.
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=528))
    cfg = _config(_full(528), _hidden(22))
    block_size = ExampleHiddenStatesConnector._get_cache_block_size(
        vllm_config, cfg, cache_kv_group_id=1
    )
    assert block_size == 22


def test_get_block_size_falls_back_to_cache_config_when_no_kv_cache_config():
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    block_size = ExampleHiddenStatesConnector._get_cache_block_size(
        vllm_config, None, cache_kv_group_id=0
    )
    assert block_size == 16


# ---- Packed MLA grouping ----------------------------------------------------


def test_hidden_state_group_isolated_from_packed_mla_groups():
    # HiddenStateCacheSpec subclasses MLAAttentionSpec, but grouping must pull
    # it out before packing compatible MLA cache specs.
    dt = torch.bfloat16
    spec = {
        "layers.0.mla": MLAAttentionSpec(
            block_size=64, num_kv_heads=1, head_size=576, dtype=dt
        ),
        "layers.1.swa": SlidingWindowMLASpec(
            block_size=64, num_kv_heads=1, head_size=576, dtype=dt, sliding_window=512
        ),
        "cache_only_layers.61": _hidden(64),
    }
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
    )
    groups = get_kv_cache_groups(vllm_config, spec)
    hidden_group_ids = [
        i
        for i, group in enumerate(groups)
        if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
    ]
    assert len(hidden_group_ids) == 1
    cfg = SimpleNamespace(kv_cache_groups=groups)
    assert (
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == hidden_group_ids[0]
    )


def test_hidden_state_group_isolated_from_packed_mixed_page_groups():
    # Packed grouping keeps groups with unequal page sizes (blocks are strided
    # by the widest group), so the hidden group is appended without padding.
    hidden = _hidden(16)
    spec = {
        "layers.0.attn": _full(16),
        "layers.1.mamba": MambaSpec(
            block_size=16, shapes=((1024,),), dtypes=(torch.float32,)
        ),
        "cache_only_layers.61": hidden,
    }
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
    )
    groups = get_kv_cache_groups(vllm_config, spec)
    page_sizes = {g.kv_cache_spec.page_size_bytes for g in groups}
    assert len(page_sizes) > 1, "spec must exercise the packed grouping path"
    hidden_group_ids = [
        i
        for i, group in enumerate(groups)
        if isinstance(group.kv_cache_spec, HiddenStateCacheSpec)
    ]
    assert len(hidden_group_ids) == 1
    assert groups[hidden_group_ids[0]].kv_cache_spec == hidden
    cfg = SimpleNamespace(kv_cache_groups=groups)
    assert (
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == hidden_group_ids[0]
    )


# ---- abort-path robustness --------------------------------------------------


def _bare_connector() -> ExampleHiddenStatesConnector:
    """Instance with scheduler-side state but bypassing ``__init__`` (no engine)."""
    conn = ExampleHiddenStatesConnector.__new__(ExampleHiddenStatesConnector)
    conn._request_filenames = {}
    conn._pending_saves = {}
    conn._lock_fds = {}
    conn._cache_kv_group_id = 1
    conn._connector_metadata = None
    conn._req_copy_events = {}
    conn._accumulated_finished_req_ids = set()
    return conn


def test_get_finished_count_is_one():
    # Only TP rank 0 writes, so KVOutputAggregator must expect a single
    # finished_sending notification per request (not the TP world size).
    assert _bare_connector().get_finished_count() == 1


def test_request_finished_is_noop_for_never_scheduled_request():
    # A request aborted while still queued never reaches build_connector_meta,
    # so no filename was recorded. request_finished must not raise KeyError.
    conn = _bare_connector()
    request = SimpleNamespace(request_id="cmpl-aborted", kv_transfer_params=None)
    assert conn.request_finished(request, []) == (False, None)


def test_request_finished_all_groups_handles_missing_group():
    # Guard against indexing a nonexistent per-group block table.
    conn = _bare_connector()
    conn._cache_kv_group_id = 2
    request = SimpleNamespace(request_id="cmpl-aborted", kv_transfer_params=None)
    assert conn.request_finished_all_groups(request, ([], [])) == (False, None)


def test_get_finished_does_not_report_untracked_request():
    # A never-scheduled aborted request has no copy event. get_finished must not
    # report it as done_sending, or the scheduler asserts it is still tracked.
    conn = _bare_connector()
    assert conn.get_finished({"cmpl-aborted"}) == (None, None)
    assert conn._accumulated_finished_req_ids == set()


def test_get_finished_reports_tracked_completed_request():
    conn = _bare_connector()

    class _DoneEvent:
        def query(self) -> bool:
            return True

    conn._req_copy_events["cmpl-done"] = _DoneEvent()
    done_sending, done_recving = conn.get_finished({"cmpl-done"})
    assert done_sending == {"cmpl-done"}
    assert done_recving is None


# ---- CPU-pages streaming (hs_store_device=cpu) --------------------------------


def _cpu_pages_connector(**extra_config) -> ExampleHiddenStatesConnector:
    """Worker-side instance with CPU-pages state, bypassing `__init__`."""
    conn = ExampleHiddenStatesConnector.__new__(ExampleHiddenStatesConnector)
    conn._hs_store_cpu = True
    conn._is_tp_rank_zero = True
    conn._block_size = 4
    conn._cpu_pages = {}
    conn._hs_staging = None
    conn._logged_cpu_pages = False
    conn._hs_prefix_index_on = True
    conn._hs_hash_blocks = OrderedDict()
    conn._hs_hash_tokens = 0
    conn._hs_prefix_max_tokens = 131072
    conn._req_block_hashes = {}
    cfg = {"hs_gather_chunk_tokens": 8, **extra_config}
    conn._kv_transfer_config = SimpleNamespace(
        get_from_extra_config=lambda key, default=None: cfg.get(key, default)
    )
    conn._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(prefix_caching_hash_algo="sha256")
    )
    return conn


def _cam(num_reqs: int, qsl: list[int], seq_lens: list[int], num_actual: int):
    """Duck-typed CommonAttentionMetadata for save_hs_batch."""
    return SimpleNamespace(
        num_reqs=num_reqs,
        query_start_loc_cpu=np.array(qsl, dtype=np.int64),
        seq_lens=np.array(seq_lens, dtype=np.int64),
        num_actual_tokens=num_actual,
    )


def _hs(rows: int, start: float = 0) -> torch.Tensor:
    """Deterministic [rows, 1, 2] hidden states, values distinct per call."""
    t = torch.arange(start, start + rows * 2, dtype=torch.float32)
    return t.reshape(rows, 1, 2)


def _scope(prompt_tokens: list[int], **overrides) -> SimpleNamespace:
    attrs = dict(
        num_prompt_tokens=len(prompt_tokens),
        mm_features=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        lora_request=None,
        prompt_token_ids=list(prompt_tokens),
        output_token_ids=[],
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def test_save_hs_batch_noop_off_tp_rank_zero():
    conn = _cpu_pages_connector()
    conn._is_tp_rank_zero = False
    conn.save_hs_batch(_hs(2), _cam(1, [0, 2], [2], 2), ["A"], {"A": _scope([1, 2])})
    assert conn._cpu_pages == {}


def test_save_hs_batch_noop_in_device_mode():
    conn = _cpu_pages_connector()
    conn._hs_store_cpu = False
    conn.save_hs_batch(_hs(2), _cam(1, [0, 2], [2], 2), ["A"], {"A": _scope([1, 2])})
    assert conn._cpu_pages == {}


def test_save_hs_batch_splices_rows_at_num_computed_offsets():
    conn = _cpu_pages_connector()
    hs = _hs(5)
    # A prefills 4 rows at offset 0; B's first streamed step is a decode at
    # num_computed_tokens=5 (its prompt hit the engine prefix cache).
    conn.save_hs_batch(
        hs,
        _cam(2, [0, 4, 5], [4, 6], 5),
        ["A", "B"],
        {"A": _scope([11, 12, 13, 14]), "B": _scope([21, 22, 23, 24, 25])},
    )
    assert torch.equal(conn._cpu_pages["A"], hs[:4])
    page_b = conn._cpu_pages["B"]
    assert page_b.shape[0] == 6  # capacity from num_prompt_tokens
    assert torch.equal(page_b[5:6], hs[4:5])
    # B's prefix-hit rows could not be proven against the (empty) index:
    # they keep the NaN sentinel instead of stale uninitialized memory.
    assert torch.isnan(page_b[:5]).all()


def test_save_hs_batch_appends_chunked_prefill_at_offsets():
    conn = _cpu_pages_connector()
    scopes = {"A": _scope(list(range(10)))}
    c1 = _hs(3, start=100)
    conn.save_hs_batch(c1, _cam(1, [0, 3], [3], 3), ["A"], scopes)
    c2 = _hs(2, start=200)
    conn.save_hs_batch(c2, _cam(1, [0, 2], [5], 2), ["A"], scopes)
    c3 = _hs(1, start=300)
    conn.save_hs_batch(c3, _cam(1, [0, 1], [6], 1), ["A"], scopes)
    page = conn._cpu_pages["A"]
    assert torch.equal(page[:3], c1[:3])
    assert torch.equal(page[3:5], c2[:2])
    assert torch.equal(page[5:6], c3[:1])
    assert page.shape[0] == 10  # page grew to the prompt capacity
    assert not torch.isnan(page[:6]).any()


def test_save_hs_batch_prefers_host_num_computed_tokens():
    # With request state carrying num_computed_tokens (the scheduler-fed
    # host value), it is the page write offset and seq_lens is never
    # consulted — the two disagree here and the host value wins.
    conn = _cpu_pages_connector()
    hs = _hs(3)
    scopes = {"A": _scope([1, 2, 3, 4, 5], num_computed_tokens=2)}
    # The seq_lens derivation would place the rows at offset 9 - 3 = 6.
    conn.save_hs_batch(hs, _cam(1, [0, 3], [9], 3), ["A"], scopes)
    page = conn._cpu_pages["A"]
    assert torch.equal(page[2:5], hs)
    assert torch.isnan(page[:2]).all()


def test_save_hs_batch_falls_back_to_seq_lens_without_state():
    # No request state: the offsets fall back to the seq_lens derivation.
    conn = _cpu_pages_connector()
    hs = _hs(3)
    conn.save_hs_batch(hs, _cam(1, [0, 3], [8], 3), ["A"], None)
    page = conn._cpu_pages["A"]
    # offset = 8 - 3 = 5, capacity grows to num_prompt_tokens (unknown) ->
    # exactly need.
    assert torch.equal(page[5:8], hs)
    assert torch.isnan(page[:5]).all()


def test_publish_skips_blocks_with_nan_holes():
    # A prefix-hit start that is not block-aligned leaves unproven rows in
    # the first candidate block. Publishing it would spread the NaN holes
    # to every future request hitting the same hash, so it is skipped.
    conn = _cpu_pages_connector()
    hs = _hs(3)
    scopes = {"A": _scope(list(range(100, 108)), num_computed_tokens=2)}
    conn.save_hs_batch(hs, _cam(1, [0, 3], [9], 3), ["A"], scopes)
    page = conn._cpu_pages["A"]
    assert torch.equal(page[2:5], hs)  # streamed rows
    assert torch.isnan(page[:2]).all()  # unproven prefix rows
    assert len(conn._hs_hash_blocks) == 0  # no polluted block published


def test_page_growth_nan_fills_gap_from_dropped_step():
    conn = _cpu_pages_connector()
    scopes = {"A": _scope([1, 2])}  # small capacity: 2 rows
    c1 = _hs(2, start=10)
    conn.save_hs_batch(c1, _cam(1, [0, 2], [2], 2), ["A"], scopes)
    # Simulate a dropped streaming step: the next write starts at
    # num_computed_tokens=4 while the page only holds 2 rows.
    c2 = _hs(2, start=20)
    conn.save_hs_batch(c2, _cam(1, [0, 2], [6], 2), ["A"], scopes)
    page = conn._cpu_pages["A"]
    assert page.shape[0] == 6
    assert torch.equal(page[:2], c1)
    assert torch.isnan(page[2:4]).all()  # the gap rows keep the sentinel
    assert torch.equal(page[4:6], c2)


def test_prefix_index_roundtrip_backfills_prefix_hit():
    conn = _cpu_pages_connector()
    prompt = list(range(100, 108))
    a = _hs(8, start=1)
    conn.save_hs_batch(a, _cam(1, [0, 8], [8], 8), ["A"], {"A": _scope(prompt)})
    assert len(conn._hs_hash_blocks) == 2  # both full blocks published

    # B shares A's prompt and starts decoding with that prefix cached.
    b_row = _hs(1, start=50)
    conn.save_hs_batch(b_row, _cam(1, [0, 1], [9], 1), ["B"], {"B": _scope(prompt)})
    page_b = conn._cpu_pages["B"]
    assert torch.equal(page_b[:8], a[:8])  # backfilled from the index
    assert torch.equal(page_b[8:9], b_row)
    assert not torch.isnan(page_b[:9]).any()


def test_prefix_index_lru_eviction_keeps_newest_block(caplog):
    conn = _cpu_pages_connector(hs_prefix_cache_max_tokens=4)
    prompt = list(range(100, 108))
    a = _hs(8, start=1)
    conn.save_hs_batch(a, _cam(1, [0, 8], [8], 8), ["A"], {"A": _scope(prompt)})
    # One block of budget: publishing the second block evicted the first.
    assert len(conn._hs_hash_blocks) == 1

    b_row = _hs(1, start=50)
    with caplog.at_level("WARNING"):
        conn.save_hs_batch(b_row, _cam(1, [0, 1], [9], 1), ["B"], {"B": _scope(prompt)})
    page_b = conn._cpu_pages["B"]
    assert torch.isnan(page_b[:4]).all()  # evicted block stays NaN
    assert torch.equal(page_b[4:8], a[4:8])  # newest block backfilled
    assert any("backfilled 1/2" in rec.message for rec in caplog.records)


def test_mm_request_never_publishes_or_backfills(caplog):
    conn = _cpu_pages_connector()
    prompt = list(range(100, 108))
    a = _hs(8, start=1)
    conn.save_hs_batch(
        a,
        _cam(1, [0, 8], [8], 8),
        ["A"],
        {"A": _scope(prompt, mm_features=object())},
    )
    assert len(conn._hs_hash_blocks) == 0  # never published

    b_row = _hs(1, start=50)
    with caplog.at_level("WARNING"):
        conn.save_hs_batch(
            b_row,
            _cam(1, [0, 1], [9], 1),
            ["B"],
            {"B": _scope(prompt, prompt_embeds=torch.zeros(1))},
        )
    page_b = conn._cpu_pages["B"]
    assert torch.isnan(page_b[:8]).all()  # never backfilled
    assert torch.equal(page_b[8:9], b_row)
    assert any("skipped backfill" in rec.message for rec in caplog.records)


def test_prefix_index_off_leaves_prefix_rows_nan():
    conn = _cpu_pages_connector()
    conn._hs_prefix_index_on = False
    prompt = list(range(100, 108))
    a = _hs(8, start=1)
    conn.save_hs_batch(a, _cam(1, [0, 8], [8], 8), ["A"], {"A": _scope(prompt)})
    assert len(conn._hs_hash_blocks) == 0  # index disabled: no publish

    b_row = _hs(1, start=50)
    conn.save_hs_batch(b_row, _cam(1, [0, 1], [9], 1), ["B"], {"B": _scope(prompt)})
    assert torch.isnan(conn._cpu_pages["B"][:8]).all()  # no backfill attempted


def test_abort_discard_drops_cpu_pages_without_write():
    conn = _cpu_pages_connector()
    conn._cpu_pages["A"] = _hs(4)
    conn._executor = None  # would raise if a disk write were submitted
    pending = PendingSave(
        req_id="A",
        filename="/unused",
        token_ids=torch.zeros(4, dtype=torch.long),
        block_ids=[],
        discard=True,
    )
    conn._submit_async_write(pending)
    assert "A" not in conn._cpu_pages
    assert "A" not in conn._req_block_hashes


def test_finalize_cpu_page_keeps_full_page_without_truncation():
    conn = _cpu_pages_connector()
    conn._cpu_pages["A"] = _hs(6)
    hs, ids = conn._finalize_cpu_page("A", torch.arange(6, dtype=torch.long))
    assert hs.shape[0] == ids.shape[0] == 6
    assert torch.equal(hs, _hs(6))
    assert "A" not in conn._cpu_pages


def test_finalize_cpu_page_trims_token_ids_to_page_length(caplog):
    conn = _cpu_pages_connector()
    conn._cpu_pages["A"] = _hs(4)
    with caplog.at_level("ERROR"):
        hs, ids = conn._finalize_cpu_page("A", torch.arange(6, dtype=torch.long))
    # Both tensors keep row i aligned with token i: the file is a valid
    # prefix of the sequence instead of a length-mismatched pair.
    assert hs.shape[0] == ids.shape[0] == 4
    assert torch.equal(ids, torch.arange(4))
    assert "page too short" in caplog.text


def test_finalize_cpu_page_missing_page_returns_none(caplog):
    conn = _cpu_pages_connector()
    with caplog.at_level("ERROR"):
        assert conn._finalize_cpu_page("A", torch.zeros(1, dtype=torch.long)) is None
    assert "No CPU hidden states" in caplog.text


def test_get_finished_stays_silent_in_cpu_pages_mode():
    # request_finished frees without delay in CPU-pages mode, so the
    # scheduler has already dropped the req_id; reporting finished_sending
    # would trip its tracking assertion. The channel must stay silent.
    conn = _cpu_pages_connector()
    conn._connector_metadata = None
    conn._accumulated_finished_req_ids = set()
    conn._lock_fds = {}

    class _DoneEvent:
        def query(self) -> bool:
            return True

    conn._req_copy_events["cmpl-done"] = _DoneEvent()
    assert conn.get_finished({"cmpl-done"}) == (None, None)
