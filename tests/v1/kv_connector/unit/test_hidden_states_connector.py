# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for ExampleHiddenStatesConnector KV-cache-group logic."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
    extract_from_kv_cache,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    create_kv_cache_views,
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
    """Minimal stand-in exposing only ``kv_cache_groups`` (all the helpers read)."""
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


@pytest.mark.parametrize("head_size, expected_block_size", [(2048, 2), (65536, 1)])
def test_hidden_state_group_isolated_from_packed_mixed_page_groups(
    head_size, expected_block_size
):
    # Packed grouping keeps groups with unequal page sizes (blocks are strided
    # by the widest group), so the hidden group is appended without padding.
    hidden = replace(_hidden(16), head_size=head_size)
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
    hidden_spec = groups[hidden_group_ids[0]].kv_cache_spec
    assert hidden_spec.block_size == expected_block_size
    assert hidden_spec.page_size_bytes <= max(
        _full(16).page_size_bytes, hidden.page_size_bytes // hidden.block_size
    )
    cfg = SimpleNamespace(kv_cache_groups=groups)
    assert (
        ExampleHiddenStatesConnector._find_cache_kv_group_id(cfg) == hidden_group_ids[0]
    )


@pytest.mark.parametrize("layout", [KVCacheLayout.LBNHC, KVCacheLayout.BLHNC])
def test_hidden_state_writes_preserve_native_pages(layout):
    """The extractor and connector must respect packed offsets and strides."""
    from vllm.model_executor.models.extract_hidden_states import basic_cache

    spec = HiddenStateCacheSpec(
        block_size=4, num_kv_heads=3, head_size=8, dtype=torch.float32
    )
    num_blocks = 7
    native_bytes = 64
    stride = native_bytes + spec.page_size_bytes
    raw = torch.full((num_blocks * stride,), 42, dtype=torch.int8)
    tensor = KVCacheTensor(
        size=raw.numel(),
        layers=["hidden"],
        offset=native_bytes,
        block_stride=stride,
        layer_stride=spec.page_size_bytes,
    )
    (cache,) = create_kv_cache_views(raw, spec, num_blocks, layout, tensor)
    # Noncontiguous, interleaved blocks from two requests, including a partial
    # final block. Block zero is reserved for padding writes.
    slots = torch.tensor([20, 21, 22, 23, 8, 9, 16, 17, 18])
    expected = torch.arange(9 * 3 * 8, dtype=torch.float32).reshape(9, 3, 8)
    basic_cache(expected, cache, slots)
    basic_cache(torch.zeros(1, 3, 8), cache, torch.tensor([-1]))
    torch.testing.assert_close(extract_from_kv_cache(cache, slots, 9), expected)
    assert (raw.view(num_blocks, stride)[:, :native_bytes] == 42).all()
    # Untouched pages must also survive the writes.
    assert (raw.view(num_blocks, stride)[[1, 3, 6]] == 42).all()


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
