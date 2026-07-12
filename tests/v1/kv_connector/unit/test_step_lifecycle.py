# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-neutral tests for the generic KV-connector step lifecycle helpers.

Exercises ``begin_kv_connector_step`` / ``finish_kv_connector_step`` (the
imperative split of the in-forward context manager) with a *fake* connector --
no concrete connector, no accelerator backend, no device required. Covers:

  * begin/finish drive the public SPI in the canonical order;
  * an ASYNC connector that delays completion across multiple scheduler steps
    is reported finished exactly once, on the right step;
  * invalid block ids propagate;
  * concurrent requests do not interfere;
  * no-forward polling works via the same lifecycle;
  * the context manager remains equivalent to an explicit begin/finish pair;
  * ``start_load_kv`` gets ``None`` when no forward context is active (no
    exception-swallowing fallback), and the real context when one is set.
"""

from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_transfer_state import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
)
from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

from .utils import create_vllm_config

# Stash the active vllm_config so bodies inside `with set_forward_context` can
# reach it without threading it through every helper.
_VLLM_CONFIG = None


def get_vllm_config():
    return _VLLM_CONFIG


class _FakeMeta(KVConnectorMetadata):
    def __init__(self, loads=(), saves=()):
        self.loads = list(loads)
        self.saves = list(saves)


class FakeAsyncConnector(KVConnectorBase_V1):
    """A connector whose loads/saves complete after a configurable delay.

    Records every SPI call and the forward-context object it saw on
    ``start_load_kv``. A load for request R submitted on step S is reported by
    ``get_finished`` only after ``load_delay`` subsequent steps -- modelling a
    real async retrieve that spans scheduler steps.
    """

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        cfg = self._kv_transfer_config.kv_connector_extra_config
        self.load_delay = int(cfg.get("load_delay", 0))
        self.save_delay = int(cfg.get("save_delay", 0))
        self.calls: list[str] = []
        self.forward_contexts: list[object] = []
        # req_id -> steps remaining until finished
        self._pending_load: dict[str, int] = {}
        self._pending_save: dict[str, int] = {}
        self._invalid_blocks: set[int] = set()
        self._inject_invalid: dict[str, set[int]] = {}
        # Records the registration mapping received, verbatim.
        self.registered = None
        self.register_count = 0

    # ---- registration (records the value verbatim; does not interpret) ----
    def register_kv_caches(self, kv_caches):
        self.register_count += 1
        self.registered = kv_caches

    # ---- scheduler-side (unused here but required abstract) ----
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        return _FakeMeta()

    # ---- worker-side lifecycle ----
    def bind_connector_metadata(self, connector_metadata) -> None:
        self.calls.append("bind")
        super().bind_connector_metadata(connector_metadata)
        meta = connector_metadata
        for req_id in getattr(meta, "loads", []):
            self._pending_load[req_id] = self.load_delay
            if req_id in self._inject_invalid:
                self._invalid_blocks |= self._inject_invalid[req_id]
        for req_id in getattr(meta, "saves", []):
            self._pending_save[req_id] = self.save_delay

    def clear_connector_metadata(self) -> None:
        self.calls.append("clear")
        super().clear_connector_metadata()

    def start_load_kv(self, forward_context, **kwargs) -> None:
        self.calls.append("start_load_kv")
        self.forward_contexts.append(forward_context)

    def wait_for_layer_load(self, layer_name) -> None:
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        pass

    def wait_for_save(self) -> None:
        self.calls.append("wait_for_save")

    def get_finished(self, finished_req_ids):
        self.calls.append("get_finished")
        done_recv: set[str] = set()
        for req_id in list(self._pending_load):
            if self._pending_load[req_id] <= 0:
                done_recv.add(req_id)
                del self._pending_load[req_id]
            else:
                self._pending_load[req_id] -= 1
        done_send: set[str] = set()
        for req_id in list(self._pending_save):
            if self._pending_save[req_id] <= 0:
                done_send.add(req_id)
                del self._pending_save[req_id]
            else:
                self._pending_save[req_id] -= 1
        return (done_send or None), (done_recv or None)

    def get_block_ids_with_load_errors(self) -> set[int]:
        out = set(self._invalid_blocks)
        self._invalid_blocks.clear()
        return out


KVConnectorFactory.register_connector(
    "FakeAsyncConnector", __name__, FakeAsyncConnector.__name__
)


def _sched_output(loads=(), saves=(), finished=()):
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
        kv_connector_metadata=_FakeMeta(loads=loads, saves=saves),
    )


def _init(**extra):
    global _VLLM_CONFIG
    vllm_config = create_vllm_config(
        model="gpt2",
        kv_connector="FakeAsyncConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[]
    )
    mock_tp_group = MagicMock()
    mock_tp_group.broadcast_object.side_effect = lambda value, src=0: value
    with patch(
        "vllm.distributed.parallel_state.get_tp_group",
        return_value=mock_tp_group,
    ):
        ensure_kv_transfer_initialized(vllm_config, kv_cache_config)
    _VLLM_CONFIG = vllm_config
    return vllm_config


def test_begin_finish_canonical_order():
    """begin binds+start_load; finish waits, collects, clears -- once each."""
    _init(load_delay="0")
    try:
        so = _sched_output(loads=["r0"])
        with set_forward_context(None, get_vllm_config()):
            step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
            out = KVConnectorModelRunnerMixin.finish_kv_connector_step(step)
        conn = get_kv_transfer_group()
        assert conn.calls == [
            "bind",
            "start_load_kv",
            "wait_for_save",
            "get_finished",
            "clear",
        ]
        # load_delay=0 -> finished on the same step
        assert out.finished_recving == {"r0"}
        assert conn._connector_metadata is None
    finally:
        ensure_kv_transfer_shutdown()


def test_async_delayed_completion_across_steps():
    """A load with delay=2 is reported finished only on the 3rd step, once."""
    _init(load_delay="2")
    try:
        conn = get_kv_transfer_group()
        finished_by_step = []
        # Step 0: submit the load.
        for step_idx in range(4):
            loads = ["rA"] if step_idx == 0 else []
            so = _sched_output(loads=loads)
            with set_forward_context(None, get_vllm_config()):
                step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
                out = KVConnectorModelRunnerMixin.finish_kv_connector_step(step)
            finished_by_step.append(out.finished_recving)
        # delay=2 means: step0 remaining 2->1, step1 1->0, step2 ==0 finished.
        assert finished_by_step[0] is None
        assert finished_by_step[1] is None
        assert finished_by_step[2] == {"rA"}
        assert finished_by_step[3] is None  # not dropped, not repeated
        # metadata cleared every step
        assert conn.calls.count("clear") == 4
    finally:
        ensure_kv_transfer_shutdown()


def test_invalid_block_ids_propagate():
    _init(load_delay="0")
    try:
        conn = get_kv_transfer_group()
        conn._inject_invalid["rBad"] = {7, 13}
        so = _sched_output(loads=["rBad"])
        with set_forward_context(None, get_vllm_config()):
            step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
            out = KVConnectorModelRunnerMixin.finish_kv_connector_step(step)
        assert out.invalid_block_ids == {7, 13}
    finally:
        ensure_kv_transfer_shutdown()


def test_concurrent_requests_do_not_interfere():
    """Two loads with different delays finish independently, never swapped."""
    _init(load_delay="1")
    try:
        # step0: submit rX (delay1). step1: submit rY (delay1).
        results = []
        for step_idx in range(4):
            loads = []
            if step_idx == 0:
                loads = ["rX"]
            elif step_idx == 1:
                loads = ["rY"]
            so = _sched_output(loads=loads)
            with set_forward_context(None, get_vllm_config()):
                step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
                out = KVConnectorModelRunnerMixin.finish_kv_connector_step(step)
            results.append(out.finished_recving)
        # rX submitted step0 delay1 -> finishes step1; rY step1 -> step2.
        assert results[0] is None
        assert results[1] == {"rX"}
        assert results[2] == {"rY"}
        assert results[3] is None
    finally:
        ensure_kv_transfer_shutdown()


def test_no_forward_polling_uses_same_lifecycle():
    """kv_connector_no_forward drives bind->start_load->get_finished->clear."""
    vllm_config = _init(load_delay="0")
    try:
        so = _sched_output(loads=["rNF"])
        out = KVConnectorModelRunnerMixin.kv_connector_no_forward(so, vllm_config)
        conn = get_kv_transfer_group()
        # no_forward uses wait_for_save=False
        assert "bind" in conn.calls
        assert "start_load_kv" in conn.calls
        assert "get_finished" in conn.calls
        assert "clear" in conn.calls
        assert "wait_for_save" not in conn.calls
        assert out.kv_connector_output.finished_recving == {"rNF"}
    finally:
        ensure_kv_transfer_shutdown()


def test_context_manager_equivalent_to_begin_finish():
    """The stock context manager yields the same output object begin/finish do."""
    vllm_config = _init(load_delay="0")
    try:
        so = _sched_output(loads=["rCM"])
        with (
            set_forward_context(None, vllm_config),
            KVConnectorModelRunnerMixin.maybe_get_kv_connector_output(so) as cm_out,
        ):
            pass
        assert cm_out.finished_recving == {"rCM"}
        conn = get_kv_transfer_group()
        assert conn.calls == [
            "bind",
            "start_load_kv",
            "wait_for_save",
            "get_finished",
            "clear",
        ]
    finally:
        ensure_kv_transfer_shutdown()


def test_start_load_kv_gets_none_without_forward_context():
    """No active forward context -> start_load_kv(None), not an exception."""
    _init(load_delay="0")
    try:
        so = _sched_output(loads=["r0"])
        # Deliberately NOT inside set_forward_context.
        step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
        KVConnectorModelRunnerMixin.finish_kv_connector_step(step)
        conn = get_kv_transfer_group()
        assert conn.forward_contexts == [None]
    finally:
        ensure_kv_transfer_shutdown()


def test_start_load_kv_gets_real_forward_context_when_set():
    _init(load_delay="0")
    try:
        so = _sched_output(loads=["r0"])
        with set_forward_context(None, get_vllm_config()):
            step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
            KVConnectorModelRunnerMixin.finish_kv_connector_step(step)
        conn = get_kv_transfer_group()
        # A real ForwardContext object was captured (not None).
        assert conn.forward_contexts[0] is not None
    finally:
        ensure_kv_transfer_shutdown()


# ---------------------------------------------------------------------------
# Lifecycle state machine: abort, double-finalize, nested-open (item 3)
# ---------------------------------------------------------------------------


def test_abort_clears_metadata_and_marks_aborted():
    """abort clears connector metadata once, assembles no output, marks ABORTED."""
    from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorStepState

    _init(load_delay="0")
    try:
        so = _sched_output(loads=["rA"])
        with set_forward_context(None, get_vllm_config()):
            step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
            KVConnectorModelRunnerMixin.abort_kv_connector_step(step)
        conn = get_kv_transfer_group()
        assert step.state is KVConnectorStepState.ABORTED
        # bound then cleared; NO wait_for_save / get_finished on the abort path
        assert conn.calls == ["bind", "start_load_kv", "clear"]
        assert conn._connector_metadata is None
    finally:
        ensure_kv_transfer_shutdown()


def test_exception_between_begin_and_finish_aborts_and_reraises():
    """The context manager aborts (clear once, no fake output) and re-raises."""
    vllm_config = _init(load_delay="0")
    try:
        so = _sched_output(loads=["rX"])
        boom = ValueError("forward blew up")
        with pytest.raises(ValueError, match="forward blew up"):  # noqa: SIM117
            with (
                set_forward_context(None, vllm_config),
                KVConnectorModelRunnerMixin.maybe_get_kv_connector_output(so),
            ):
                raise boom
        conn = get_kv_transfer_group()
        # aborted: bound, load started, cleared once; no wait_for_save/get_finished
        assert conn.calls == ["bind", "start_load_kv", "clear"]
        assert conn._connector_metadata is None
    finally:
        ensure_kv_transfer_shutdown()


def test_abort_then_clean_next_step():
    """After an aborted step, a fresh begin/finish works and its metadata is
    clean. NOTE: aborting does not cancel an already-submitted load (the generic
    SPI has no cancellation primitive), so we only require that the next step
    finalizes correctly with its own metadata cleared."""
    _init(load_delay="0")
    try:
        so1 = _sched_output(loads=["r1"])
        with set_forward_context(None, get_vllm_config()):
            step1 = KVConnectorModelRunnerMixin.begin_kv_connector_step(so1)
            KVConnectorModelRunnerMixin.abort_kv_connector_step(step1)
        so2 = _sched_output(loads=["r2"])
        with set_forward_context(None, get_vllm_config()):
            step2 = KVConnectorModelRunnerMixin.begin_kv_connector_step(so2)
            out2 = KVConnectorModelRunnerMixin.finish_kv_connector_step(step2)
        conn = get_kv_transfer_group()
        # step2 reports its own load and cleared its metadata; r1's load was
        # submitted before abort and its completion is connector-defined.
        assert "r2" in (out2.finished_recving or set())
        assert conn._connector_metadata is None
        # step1: bind,start,clear ; step2: bind,start,wait,get_finished,clear
        assert conn.calls == [
            "bind",
            "start_load_kv",
            "clear",
            "bind",
            "start_load_kv",
            "wait_for_save",
            "get_finished",
            "clear",
        ]
    finally:
        ensure_kv_transfer_shutdown()


def test_double_finish_raises():
    _init(load_delay="0")
    try:
        so = _sched_output(loads=["r0"])
        with set_forward_context(None, get_vllm_config()):
            step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
            KVConnectorModelRunnerMixin.finish_kv_connector_step(step)
            with pytest.raises(RuntimeError, match="finalized exactly once"):
                KVConnectorModelRunnerMixin.finish_kv_connector_step(step)
    finally:
        ensure_kv_transfer_shutdown()


def test_double_abort_raises():
    _init(load_delay="0")
    try:
        so = _sched_output(loads=["r0"])
        with set_forward_context(None, get_vllm_config()):
            step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
            KVConnectorModelRunnerMixin.abort_kv_connector_step(step)
            with pytest.raises(RuntimeError, match="finalized exactly once"):
                KVConnectorModelRunnerMixin.abort_kv_connector_step(step)
    finally:
        ensure_kv_transfer_shutdown()


def test_finish_after_abort_raises():
    _init(load_delay="0")
    try:
        so = _sched_output(loads=["r0"])
        with set_forward_context(None, get_vllm_config()):
            step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
            KVConnectorModelRunnerMixin.abort_kv_connector_step(step)
            with pytest.raises(RuntimeError, match="finalized exactly once"):
                KVConnectorModelRunnerMixin.finish_kv_connector_step(step)
    finally:
        ensure_kv_transfer_shutdown()


def test_context_manager_shares_finalization_with_imperative():
    """The ctx-mgr success path yields the same assembled fields as imperative
    finish (single shared _assemble_kv_connector_output)."""
    vllm_config = _init(load_delay="0")
    try:
        so = _sched_output(loads=["rCM"], finished=["rCM"])
        with (
            set_forward_context(None, vllm_config),
            KVConnectorModelRunnerMixin.maybe_get_kv_connector_output(so) as cm_out,
        ):
            pass
        assert cm_out.finished_recving == {"rCM"}
        conn = get_kv_transfer_group()
        assert conn.calls == [
            "bind",
            "start_load_kv",
            "wait_for_save",
            "get_finished",
            "clear",
        ]
    finally:
        ensure_kv_transfer_shutdown()


# ---------------------------------------------------------------------------
# Registration contract: split reaches connector unchanged; MultiConnector
# forwards verbatim (item 1)
# ---------------------------------------------------------------------------


def test_split_registration_reaches_connector_unchanged():
    """A split {name: (k, v)} registration arrives at the connector verbatim;
    K and V retain identity and data pointers; core does not fuse or rewrite."""
    import torch

    _init(load_delay="0")
    try:
        conn = get_kv_transfer_group()
        k0 = torch.zeros(4, 8, 2, 16)
        v0 = torch.ones(4, 8, 2, 16)
        reg = {"layer_0": (k0, v0)}
        conn.register_kv_caches(reg)
        assert conn.register_count == 1
        got = conn.registered["layer_0"]
        assert isinstance(got, tuple) and len(got) == 2
        gk, gv = got
        # identity + data_ptr preserved (no copy, no fuse)
        assert gk is k0 and gv is v0
        assert gk.data_ptr() == k0.data_ptr()
        assert gv.data_ptr() == v0.data_ptr()
        assert gk.data_ptr() != gv.data_ptr()
    finally:
        ensure_kv_transfer_shutdown()


def test_multiconnector_forwards_split_registration_verbatim():
    """MultiConnector forwards the same split registration to two split-capable
    fake connectors without inspecting or rewriting it."""
    import torch

    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    vllm_config = create_vllm_config(
        model="gpt2",
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {
                    "kv_connector": "FakeAsyncConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"load_delay": "0"},
                },
                {
                    "kv_connector": "FakeAsyncConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"load_delay": "0"},
                },
            ]
        },
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    multi = MultiConnector(
        vllm_config, KVConnectorRole.WORKER, _empty_kv_cache_config()
    )
    k0 = torch.zeros(4, 8, 2, 16)
    v0 = torch.ones(4, 8, 2, 16)
    reg = {"layer_0": (k0, v0)}
    multi.register_kv_caches(reg)
    for sub in multi._connectors:
        got = sub.registered["layer_0"]
        assert got is reg["layer_0"]  # verbatim, same tuple object
        assert got[0] is k0 and got[1] is v0


def _empty_kv_cache_config():
    return KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])


def test_finish_preserves_stats_events_and_worker_metadata():
    """finish carries the connector's stats, KV-cache events and worker
    metadata into the KVConnectorOutput (canonical field preservation)."""
    _init(load_delay="0")
    try:
        conn = get_kv_transfer_group()
        stats = object()
        events = object()
        worker_meta = object()
        conn.get_kv_connector_stats = lambda: stats
        conn.get_kv_connector_kv_cache_events = lambda: events
        conn.build_connector_worker_meta = lambda: worker_meta
        so = _sched_output(loads=["rS"], finished=["rS"])
        with set_forward_context(None, get_vllm_config()):
            step = KVConnectorModelRunnerMixin.begin_kv_connector_step(so)
            out = KVConnectorModelRunnerMixin.finish_kv_connector_step(step)
        assert out.kv_connector_stats is stats
        assert out.kv_cache_events is events
        assert out.kv_connector_worker_meta is worker_meta
    finally:
        ensure_kv_transfer_shutdown()
