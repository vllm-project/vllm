# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the offloader selection/diagnostics in
vllm.model_executor.offloader.base.

Covers https://github.com/vllm-project/vllm/issues/56283: a NoopOffloader
being silently selected while a non-zero cpu_offload_gb / offload_group_size
was requested should be loud, not silent, since that combination can only
happen if the offload configuration was lost before reaching
create_offloader()/set_offloader().
"""

import pytest

from vllm.config.offload import OffloadConfig, PrefetchOffloadConfig, UVAOffloadConfig
from vllm.model_executor.offloader import base
from vllm.model_executor.offloader.base import (
    NoopOffloader,
    create_offloader,
    set_offloader,
)

pytestmark = pytest.mark.skip_global_cleanup


def _capture_log_calls(monkeypatch):
    calls: dict[str, list] = {"debug_once": [], "info_once": [], "warning_once": []}
    for level in calls:
        monkeypatch.setattr(
            base.logger,
            level,
            lambda *args, _level=level, **kwargs: calls[_level].append(args),
        )
    return calls


# --------------------------------------------------------------------------
# create_offloader(): warn when an explicitly requested (non-"auto") backend
# is handed a zero budget for that backend.
# --------------------------------------------------------------------------


def test_create_offloader_auto_zero_budget_is_silent(monkeypatch):
    """cpu_offload_gb=0 with the default offload_backend='auto' is the
    ordinary no-offload configuration and must not warn."""
    calls = _capture_log_calls(monkeypatch)

    offloader = create_offloader(OffloadConfig())

    assert isinstance(offloader, NoopOffloader)
    assert calls["warning_once"] == []


def test_create_offloader_auto_nonzero_selects_uva_silently(monkeypatch):
    calls = _capture_log_calls(monkeypatch)

    offloader = create_offloader(
        OffloadConfig(uva=UVAOffloadConfig(cpu_offload_gb=15.1))
    )

    assert type(offloader).__name__ == "UVAOffloader"
    assert calls["warning_once"] == []


def test_create_offloader_explicit_uva_backend_zero_budget_warns(monkeypatch):
    """offload_backend='uva' explicitly requested but cpu_offload_gb=0: this
    combination can only happen if the value was lost upstream."""
    calls = _capture_log_calls(monkeypatch)

    offloader = create_offloader(
        OffloadConfig(offload_backend="uva", uva=UVAOffloadConfig(cpu_offload_gb=0))
    )

    # Still constructs a (zero-budget) UVAOffloader, matching prior behavior
    # for an explicit backend selection.
    assert type(offloader).__name__ == "UVAOffloader"
    assert len(calls["warning_once"]) == 1
    msg, *args = calls["warning_once"][0]
    assert "uva" in (msg % tuple(args))
    assert "0" in (msg % tuple(args))


def test_offload_backend_prefetch_zero_group_size_is_rejected_at_construction():
    """offload_backend='prefetch' with offload_group_size=0 can never reach
    create_offloader() in the first place: validate_offload_config() rejects
    it (offload_num_in_group > offload_group_size) before the OffloadConfig
    object even exists. create_offloader's matching check for this backend
    is therefore a defense against a corrupted/reconstructed config object,
    not a reachable user-facing state -- this test documents that
    invariant."""
    with pytest.raises(ValueError):
        OffloadConfig(
            offload_backend="prefetch",
            prefetch=PrefetchOffloadConfig(offload_group_size=0),
        )


# --------------------------------------------------------------------------
# set_offloader(): warn when a NoopOffloader is selected despite a non-zero
# offload budget having been requested. This is the exact signature of
# issue #56283: cpu_offload_gb echoed in non-default args, NoopOffloader
# selected, weights never offloaded.
# --------------------------------------------------------------------------


def test_set_offloader_warns_on_noop_with_nonzero_uva_request(monkeypatch):
    calls = _capture_log_calls(monkeypatch)
    cfg = OffloadConfig(uva=UVAOffloadConfig(cpu_offload_gb=15.1))

    set_offloader(NoopOffloader(), cfg)

    assert len(calls["warning_once"]) == 1
    msg, *args = calls["warning_once"][0]
    rendered = msg % tuple(args)
    assert "15.1" in rendered
    assert "NoopOffloader" in rendered
    assert calls["debug_once"] == []


def test_set_offloader_warns_on_noop_with_nonzero_prefetch_request(monkeypatch):
    calls = _capture_log_calls(monkeypatch)
    cfg = OffloadConfig(
        prefetch=PrefetchOffloadConfig(offload_group_size=8, offload_num_in_group=7)
    )

    set_offloader(NoopOffloader(), cfg)

    assert len(calls["warning_once"]) == 1
    msg, *args = calls["warning_once"][0]
    assert "8" in (msg % tuple(args))


def test_set_offloader_noop_with_legitimate_zero_default_does_not_warn(monkeypatch):
    """cpu_offload_gb=0 (the default, i.e. offloading genuinely not
    requested) must keep logging at debug level, not warn."""
    calls = _capture_log_calls(monkeypatch)

    set_offloader(NoopOffloader(), OffloadConfig())

    assert calls["warning_once"] == []
    assert len(calls["debug_once"]) == 1


def test_set_offloader_without_offload_config_arg_is_backward_compatible(monkeypatch):
    calls = _capture_log_calls(monkeypatch)

    set_offloader(NoopOffloader())

    assert calls["warning_once"] == []
    assert len(calls["debug_once"]) == 1


def test_set_offloader_info_once_preserved_for_real_offloader(monkeypatch):
    calls = _capture_log_calls(monkeypatch)
    cfg = OffloadConfig(uva=UVAOffloadConfig(cpu_offload_gb=8.0))

    offloader = create_offloader(cfg)
    calls["warning_once"].clear()  # only inspect set_offloader's own logging
    set_offloader(offloader, cfg)

    assert calls["warning_once"] == []
    assert len(calls["info_once"]) == 1
