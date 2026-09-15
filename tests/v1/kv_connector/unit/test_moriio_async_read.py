# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MoRIIO READ-mode async KV-load fix (PR #54301).

Background
----------
MoRIIO PD-disaggregation schedules decode requests whose ``request_id`` is the
router-embedded compound id
(``...___prefill_addr_...___decode_addr_..._<uuid>``). Previously READ mode
returned ``load_kv_async=False`` from ``get_num_new_matched_tokens``, so the
scheduler put the request straight into ``num_scheduled_tokens`` (RUNNING). For
a step where the worker produced no output row for it, that compound req_id was
absent from ``model_runner_output.req_id_to_index`` and the scheduler's
unconditional index lookup in ``update_from_output`` raised ``KeyError`` ->
``EngineDeadError`` (observed on the first inference request of MoRIIO 1P1D).

The fix (connector-side, no core ``scheduler.py`` change, no monkeypatch of
``Scheduler.update_from_output``): READ mode now returns ``load_kv_async=True``
so the request is held in ``WAITING_FOR_REMOTE_KVS`` -- OUT of
``num_scheduled_tokens`` -- until the worker reports the load done via
``get_finished()`` -> ``finished_recving``. It is therefore never
scheduled-but-unindexed and the KeyError cannot occur. This mirrors the
existing WRITE-mode consumer lifecycle.

These tests assert:
  * READ mode advertises the async path (and WRITE mode is unchanged);
  * the producer leg never loads KV;
  * the tiny-prompt / nothing-to-load edge keeps the sync path (the scheduler
    asserts ``num_external_computed_tokens > 0`` for async loads);
  * the old scheduler monkeypatch is gone: importing/using the connector must
    NOT wrap ``Scheduler.update_from_output``.

CPU-only: no GPU / no ``mori`` runtime required (the scheduler-side connector
object is built via ``__new__`` and only ``get_num_new_matched_tokens`` -- which
reads just ``self.is_producer`` and ``self.mode`` -- is exercised).
"""

import pytest

import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector as moriio
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import MoRIIOMode
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
)
from vllm.v1.core.sched.scheduler import Scheduler

from .utils import create_request

pytestmark = pytest.mark.cpu_test

# A representative MoRIIO PD compound req_id (host/port addresses + uuid), as
# seen in the original crash logs (decode EngineCore KeyError).
_MORIIO_COMPOUND_REQ = (
    "cmpl-___prefill_addr_host:10.158.215.209,handshake:8405,"
    "notify:61005___decode_addr_host:10.158.214.246,handshake:8405,"
    "notify:61005_16f41eb3df75420fa6b78acc63bafd9a-0-b8ee174a"
)


def _make_scheduler_conn(mode: MoRIIOMode, is_producer: bool):
    """Build a MoRIIOConnectorScheduler without running __init__.

    ``get_num_new_matched_tokens`` only reads ``self.is_producer`` and
    ``self.mode``, so this avoids needing the full VllmConfig / handshake ports
    / mori runtime. Mirrors the ``make_nixl_scheduler`` __new__ pattern in
    tests/v1/kv_connector/unit/utils.py.
    """
    conn = object.__new__(MoRIIOConnectorScheduler)
    conn.mode = mode
    conn.is_producer = is_producer
    return conn


def test_read_mode_uses_async_path():
    """READ-mode consumer decode requests must advertise load_kv_async=True so
    they are never scheduled-but-unindexed (the req_id_to_index KeyError)."""
    conn = _make_scheduler_conn(MoRIIOMode.READ, is_producer=False)
    request = create_request(
        request_id=1, num_tokens=8, do_remote_prefill=True, max_tokens=16
    )
    request.request_id = _MORIIO_COMPOUND_REQ

    num_external, load_kv_async = conn.get_num_new_matched_tokens(request, 0)

    # len(prompt) - 1 - num_computed = 8 - 1 - 0 = 7 tokens loaded remotely.
    assert num_external == 7
    # The crux of the fix: async path (WAITING_FOR_REMOTE_KVS), not sync.
    assert load_kv_async is True


def test_read_mode_async_requires_positive_external_tokens():
    """When there is nothing to load remotely, READ mode must keep the sync
    path: the scheduler asserts num_external_computed_tokens > 0 for async
    loads, so returning (<=0, True) would trip that assert."""
    conn = _make_scheduler_conn(MoRIIOMode.READ, is_producer=False)

    # Prompt of length 1 => len - 1 - 0 = 0 external tokens.
    request = create_request(
        request_id=2, num_tokens=1, do_remote_prefill=True, max_tokens=16
    )
    num_external, load_kv_async = conn.get_num_new_matched_tokens(request, 0)
    assert num_external == 0
    assert load_kv_async is False


def test_write_mode_async_unchanged():
    """WRITE mode (the pre-existing async consumer path) is left intact."""
    conn = _make_scheduler_conn(MoRIIOMode.WRITE, is_producer=False)
    request = create_request(
        request_id=3, num_tokens=8, do_remote_prefill=True, max_tokens=16
    )
    num_external, load_kv_async = conn.get_num_new_matched_tokens(request, 0)
    # WRITE mode loads the full prompt: len - num_computed = 8 - 0 = 8.
    assert num_external == 8
    assert load_kv_async is True


def test_producer_never_loads_kv():
    """The prefill/producer leg never pulls external KV, in either mode."""
    for mode in (MoRIIOMode.READ, MoRIIOMode.WRITE):
        conn = _make_scheduler_conn(mode, is_producer=True)
        request = create_request(request_id=4, num_tokens=8, max_tokens=16)
        assert conn.get_num_new_matched_tokens(request, 0) == (0, False)


def test_scheduler_update_from_output_is_not_monkeypatched():
    """Regression guard for PR #54301: the connector must fix this on its own
    side, NOT by wrapping the core scheduler. The removed monkeypatch tagged
    the wrapped function with ``_moriio_reqid_guarded``; that must never appear,
    and the helper that installed it must no longer exist."""
    assert not hasattr(Scheduler.update_from_output, "_moriio_reqid_guarded")
    assert not hasattr(moriio, "_install_moriio_scheduler_reqid_guard")
