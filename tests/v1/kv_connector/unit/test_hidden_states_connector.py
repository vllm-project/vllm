# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for ExampleHiddenStatesConnector KV-cache-group logic."""

import threading
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (  # noqa: E501
    ExampleHiddenStatesConnector,
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


@pytest.fixture
def window_connector():
    from vllm.distributed.kv_transfer.kv_connector.v1.hidden_state_window import (
        HiddenStateWindow,
    )

    connector = ExampleHiddenStatesConnector.__new__(ExampleHiddenStatesConnector)
    connector._request_filenames = {"r": "/tmp/r.safetensors"}
    connector._request_windows = {"r": HiddenStateWindow(1, 8192, 1)}
    connector._request_block_counts = {"r": 3}
    connector._request_prompt_lengths = {"r": 3}
    connector._cancelled_req_ids = set()
    connector._rejected_req_ids = set()
    connector._window_enabled = True
    connector._pending_saves = {}
    connector._layer_ids = [1, 2, -1]
    connector._layout = "auxiliary_plus_final"
    return connector


@pytest.mark.cpu_test
def test_window_completion_uses_accepted_tokens_and_delays_cache_free(window_connector):
    from vllm.v1.request import RequestStatus

    req = SimpleNamespace(
        request_id="r",
        num_prompt_tokens=3,
        output_token_ids=[20, 21, 22],
        all_token_ids=[1, 2, 3, 20, 21, 22],
        status=RequestStatus.FINISHED_STOPPED,
    )
    delay, params = window_connector.request_finished(req, [7, 8, 9])
    assert delay and params["hidden_state_capture_rows"] == 2
    pending = window_connector._pending_saves["r"]
    assert pending.positions.tolist() == [3, 4]
    assert pending.response_positions.tolist() == [1, 2]
    assert pending.token_ids.tolist() == [20, 21]
    assert pending.target_token_ids.tolist() == [21, 22]
    assert pending.layer_ids.tolist() == [1, 2, -1]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "status", ["FINISHED_ABORTED", "FINISHED_ERROR", "FINISHED_IGNORED"]
)
def test_window_cancellation_frees_blocks_without_export(window_connector, status):
    from vllm.v1.request import RequestStatus

    req = SimpleNamespace(request_id="r", status=getattr(RequestStatus, status))
    assert window_connector.request_finished(req, [7]) == (False, None)
    assert not window_connector._pending_saves
    assert window_connector._cancelled_req_ids == {"r"}
    assert not window_connector._request_windows


@pytest.mark.cpu_test
def test_window_before_response_end_returns_skip_and_releases_blocks(window_connector):
    from vllm.v1.request import RequestStatus

    req = SimpleNamespace(
        request_id="r",
        num_prompt_tokens=3,
        output_token_ids=[20],
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    delay, params = window_connector.request_finished(req, [7])
    assert not delay
    assert params == {"hidden_state_capture_skip_reason": "insufficient_rows"}
    assert not window_connector._pending_saves


@pytest.mark.cpu_test
def test_finished_sending_waits_for_copy_event_before_releasing_blocks(
    window_connector,
):
    from unittest.mock import Mock

    connector = window_connector
    connector.has_connector_metadata = lambda: False
    connector._accumulated_finished_req_ids = set()
    event = Mock()
    event.query.return_value = False
    connector._req_copy_events = {"r": event}
    connector._lock_fds = {}
    assert connector.get_finished({"r"}) == (None, None)
    event.query.return_value = True
    assert connector.get_finished(set()) == ({"r"}, None)
    assert not connector._req_copy_events


@pytest.mark.cpu_test
def test_finished_sending_ignores_requests_without_pending_export(
    window_connector,
):
    connector = window_connector
    connector.has_connector_metadata = lambda: False
    connector._accumulated_finished_req_ids = set()
    connector._req_copy_events = {}
    connector._lock_fds = {}

    assert connector.get_finished({"immediate"}) == (None, None)
    assert not connector._accumulated_finished_req_ids


@pytest.mark.cpu_test
def test_window_file_publication_waits_for_d2h_and_is_atomic(tmp_path):
    from unittest.mock import Mock

    from safetensors.torch import load_file

    event = Mock()
    path = str(tmp_path / "window.safetensors")
    values = {"hidden_states": torch.arange(8, dtype=torch.bfloat16).view(2, 1, 4)}
    event.synchronize.side_effect = lambda: values["hidden_states"].fill_(7)
    ExampleHiddenStatesConnector._write_tensors(values, event, path, None)
    event.synchronize.assert_called_once()
    assert torch.equal(load_file(path)["hidden_states"], values["hidden_states"])
    assert not (tmp_path / "window.safetensors.incomplete").exists()


@pytest.mark.cpu_test
def test_real_speculation_is_rejected_instead_of_exporting_unverified_rows():
    config = SimpleNamespace(speculative_config=SimpleNamespace(method="eagle3"))
    with pytest.raises(ValueError, match="independent capture hook"):
        ExampleHiddenStatesConnector._validate_window_config(config)


@pytest.mark.cpu_test
def test_export_budget_waits_for_previous_writer_before_buffer_reuse(window_connector):
    from collections import deque
    from concurrent.futures import Future

    connector = window_connector
    connector._export_budget = 16
    old = Future()
    connector._export_queue = deque([(old, 12)])
    timer = threading.Timer(0.1, old.set_result, args=(None,))
    timer.start()
    connector._reserve_export(8)
    assert old.done()
    assert not connector._export_queue


@pytest.mark.cpu_test
def test_failed_earlier_export_does_not_fail_next_reservation(window_connector):
    from collections import deque
    from concurrent.futures import Future

    connector = window_connector
    connector._export_budget = 16
    failed = Future()
    failed.set_exception(OSError("disk full"))
    blocked = Future()
    blocked.set_exception(OSError("EIO"))
    connector._export_queue = deque([(failed, 4), (blocked, 12)])
    connector._reserve_export(8)
    assert not connector._export_queue


@pytest.mark.cpu_test
def test_failed_write_removes_partial_file_and_releases_lock(tmp_path, monkeypatch):
    import os
    from unittest.mock import Mock

    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        example_hidden_states_connector as module,
    )

    path = str(tmp_path / "failed.safetensors")
    fd = os.open(path + ".lock", os.O_CREAT | os.O_WRONLY, 0o600)

    def fail(tensors, filename, **kwargs):
        from pathlib import Path

        Path(filename).write_bytes(b"incomplete")
        raise OSError("disk full")

    monkeypatch.setattr(module, "save_file", fail)
    with pytest.raises(OSError, match="disk full"):
        ExampleHiddenStatesConnector._write_tensors({}, Mock(), path, fd)
    with pytest.raises(OSError):
        os.fstat(fd)
    assert not (tmp_path / "failed.safetensors.incomplete").exists()
    assert not (tmp_path / "failed.safetensors").exists()
    assert (tmp_path / "failed.safetensors.error").read_text() == "disk full"


@pytest.mark.cpu_test
def test_invalid_window_is_skipped_by_scheduler_hooks_instead_of_raising(
    window_connector,
):
    connector = window_connector
    connector._window_enabled = False
    connector._max_window_rows = 4096
    request = SimpleNamespace(
        request_id="r",
        kv_transfer_params={"hidden_state_window": {"start": 0, "end": 2**63}},
    )
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector._rejected_req_ids == {"r"}
    finished = SimpleNamespace(request_id="r", kv_transfer_params=None)
    delay, params = connector.request_finished(finished, [7])
    assert not delay
    assert params == {"hidden_state_capture_skip_reason": "invalid_window"}
    assert connector._cancelled_req_ids == {"r"}
    assert not connector._rejected_req_ids


@pytest.mark.cpu_test
def test_window_disabled_server_skips_well_formed_window_request(window_connector):
    connector = window_connector
    connector._window_enabled = False
    connector._max_window_rows = 4096
    request = SimpleNamespace(
        request_id="r",
        kv_transfer_params={"hidden_state_window": {"start": 0, "end": 4}},
    )
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector._rejected_req_ids == {"r"}


@pytest.mark.cpu_test
def test_non_window_abort_keeps_delayed_save_so_lock_is_released():
    from vllm.v1.request import RequestStatus

    conn = _bare_connector()
    conn._layer_ids = [0]
    conn._request_filenames["r"] = "/tmp/r.safetensors"
    request = SimpleNamespace(
        request_id="r",
        status=RequestStatus.FINISHED_ABORTED,
        kv_transfer_params=None,
        prompt_token_ids=[1, 2, 3],
    )
    delay, _ = conn.request_finished(request, [7])
    assert delay
    assert "r" in conn._pending_saves
    assert not conn._cancelled_req_ids


@pytest.mark.cpu_test
def test_extraction_compile_hash_changes_with_layout_and_window_store():
    from vllm.config import KVTransferConfig, SpeculativeConfig

    hf = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=[1, 2], hidden_states_layout="auxiliary"
    )
    spec = SimpleNamespace(
        method="extract_hidden_states",
        draft_model_config=SimpleNamespace(hf_config=hf, compute_hash=lambda: "target"),
    )
    original = SpeculativeConfig.compute_hash(spec)
    hf.hidden_states_layout = "auxiliary_plus_final"
    assert SpeculativeConfig.compute_hash(spec) != original
    transfer = KVTransferConfig(
        kv_connector="ExampleHiddenStatesConnector", kv_role="kv_producer"
    )
    original = transfer.compute_hash()
    transfer.kv_connector_extra_config["enable_window_capture"] = True
    assert transfer.compute_hash() != original


@pytest.mark.cpu_test
def test_partial_window_below_min_rows_returns_no_payload(window_connector):
    from vllm.distributed.kv_transfer.kv_connector.v1.hidden_state_window import (
        HiddenStateWindow,
    )
    from vllm.v1.request import RequestStatus

    window_connector._request_windows["r"] = HiddenStateWindow(1, 8192, 3)
    request = SimpleNamespace(
        request_id="r",
        num_prompt_tokens=3,
        output_token_ids=[20, 21, 22],
        status=RequestStatus.FINISHED_STOPPED,
    )
    delay, params = window_connector.request_finished(request, [7])
    assert not delay
    assert params["hidden_state_capture_skip_reason"] == "insufficient_rows"
    assert not window_connector._pending_saves



# ---- abort-path robustness --------------------------------------------------


def _bare_connector() -> ExampleHiddenStatesConnector:
    """Instance with scheduler-side state but bypassing ``__init__`` (no engine)."""
    conn = ExampleHiddenStatesConnector.__new__(ExampleHiddenStatesConnector)
    conn._request_filenames = {}
    conn._request_windows = {}
    conn._request_block_counts = {}
    conn._request_prompt_lengths = {}
    conn._rejected_req_ids = set()
    conn._cancelled_req_ids = set()
    conn._window_enabled = False
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
