# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms.interface import Platform
from vllm.v1.worker import tpsp_profile
from vllm.v1.worker.tpsp_profile import (
    ChunkConfig,
    SPProfile,
    TPSPBackend,
    TPSPProfileSession,
    TPSPProjectionContext,
    TPSPShape,
    get_tpsp_backend,
)


class FakeBackend(TPSPBackend):
    requires_projection_context = True
    supports_projection_bias = True
    last_opened = None
    disable_hidden_size = None

    @classmethod
    def create(cls, group_name, device):
        backend = cls(SimpleNamespace(), group_name, device)
        cls.last_opened = backend
        return backend

    def open(self, *, group_name, device, **kwargs):
        assert group_name == self.group_name
        assert device == self.device
        context = object()
        self.contexts.append(context)
        return context

    def __init__(self, ops, group_name, device):
        super().__init__(ops, group_name, device)
        self.contexts = []
        self.closed_contexts = []

    def close(self, context=None):
        if context is not None:
            self.closed_contexts.append(context)
        else:
            super().close()

    def profile(self, *, context, tp_size, hidden_size, max_batched_tokens, **kwargs):
        enabled = hidden_size != self.disable_hidden_size
        return SPProfile(
            tp_size,
            hidden_size,
            max_batched_tokens,
            "enabled" if enabled else "unsupported",
            "",
            threshold_tokens=1 if enabled else None,
            config=TPSPProjectionContext(context, ChunkConfig(64)) if enabled else None,
        )

    def run(
        self,
        a,
        b,
        weight,
        residual,
        eps,
        config,
        *,
        norm_type,
        projection_bias,
        norm_bias,
        context,
    ):
        assert context in self.contexts
        assert config == ChunkConfig(64)
        return a, b, residual


def test_unsupported_platform_does_not_open_backend(monkeypatch):
    monkeypatch.setattr("vllm.platforms.current_platform", Platform(), raising=False)
    assert (
        TPSPBackend.create("test", torch.device("cpu")).open(
            dtype=torch.bfloat16,
            tp_size=2,
            hidden_size=4,
            max_batched_tokens=8,
            group_name="test",
            device=torch.device("cpu"),
        )
        is None
    )
    assert get_tpsp_backend("test", torch.device("cpu")) is None


@pytest.mark.parametrize("has_context", (False, True))
def test_base_profile_forwards_context_to_generic_scanner(monkeypatch, has_context):
    backend = TPSPBackend.create("test", torch.device("cpu"))
    expected = SPProfile(2, 4, 8, "disabled", "test")
    calls = []
    context = object() if has_context else None

    def scan(selected_backend, **kwargs):
        calls.append((selected_backend, kwargs))
        return expected

    monkeypatch.setattr(tpsp_profile, "profile_sp_config", scan)
    result = backend.profile(
        tp_size=2,
        hidden_size=4,
        input_width=2,
        max_batched_tokens=8,
        norm_eps=1e-5,
        sharded_residual=False,
        time_budget_s=1,
        context=context,
    )
    assert result is expected
    assert calls[0][0] is backend
    assert calls[0][1]["context"] is context


def test_native_profile_binds_opaque_config_to_context(monkeypatch):
    context = object()
    opaque_config = object()
    result = object()

    def fused(*args, **kwargs):
        return result

    ops = SimpleNamespace(
        _C=SimpleNamespace(
            profile_tpsp_config=lambda **kwargs: SimpleNamespace(
                threshold_tokens=1, config=opaque_config
            ),
            fused_matmul_reduce_scatter_norm_all_gather_profiled=fused,
        )
    )
    monkeypatch.setattr(tpsp_profile.c10d, "_resolve_process_group", lambda name: None)
    monkeypatch.setattr(tpsp_profile.dist, "all_reduce", lambda *args, **kwargs: None)
    backend = TPSPBackend(ops, "test", torch.device("cpu"))
    profile = backend.profile(
        tp_size=2,
        hidden_size=4,
        input_width=2,
        max_batched_tokens=8,
        norm_eps=1e-5,
        sharded_residual=False,
        time_budget_s=1,
        context=context,
    )
    assert profile.config == TPSPProjectionContext(context, opaque_config)
    assert backend.fused(1, 2, 3, 4, 1e-5, profile.config, synchronize=False) is result


def test_profile_session_dispatches_and_closes_projection_contexts(monkeypatch):
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(get_tpsp_backend_cls=lambda: f"{__name__}.FakeBackend"),
        raising=False,
    )
    shapes = {
        "first": TPSPShape(2, 4, 1e-5, False),
        "second": TPSPShape(2, 6, 1e-5, False),
    }
    session = TPSPProfileSession(shapes, 2, "test")
    session.profile(8, torch.nn.Parameter(torch.ones(4)))
    backend = FakeBackend.last_opened
    assert session.backend is backend
    assert len(backend.contexts) == 2
    assert session.profiles is not None
    assert all(
        profile.config.transport is backend.contexts[index]
        for index, profile in enumerate(session.profiles.values())
    )
    for profile in session.profiles.values():
        assert backend.fused(1, 2, 3, 4, 1e-5, profile.config) == (1, 2, 4)
    session.close()
    assert backend.closed_contexts == backend.contexts
    assert backend._closed
    assert session.backend is None


def test_mixed_plans_release_contexts_without_discarding_profiles(monkeypatch, caplog):
    monkeypatch.setattr(FakeBackend, "disable_hidden_size", 6)
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(get_tpsp_backend_cls=lambda: f"{__name__}.FakeBackend"),
        raising=False,
    )
    session = TPSPProfileSession(
        {"first": TPSPShape(2, 4, 1e-5, False), "second": TPSPShape(2, 6, 1e-5, False)},
        2,
        "test",
    )
    session.profile(8, torch.nn.Parameter(torch.ones(4)))
    backend = FakeBackend.last_opened
    assert backend is not None
    assert session.backend is None
    assert session.contexts == {}
    assert backend.closed_contexts == backend.contexts
    assert backend._closed
    assert session.profiles is not None
    assert session.profiles["first"].enabled
    assert session.profiles["second"].status == "unsupported"
    assert "TPSP using standard forward" in caplog.text
    assert "first=enabled" in caplog.text
    assert "second=unsupported" in caplog.text


def test_unsupported_platform_logs_standard_forward(monkeypatch, caplog):
    monkeypatch.setattr("vllm.platforms.current_platform", Platform(), raising=False)
    session = TPSPProfileSession({"first": TPSPShape(2, 4, 1e-5, False)}, 2, "test")
    session.profile(8, torch.nn.Parameter(torch.ones(4)))
    assert session.backend is None
    assert session.profiles is not None
    assert session.profiles["first"].status == "unsupported"
    assert "first=unsupported (no fused backend on this device)" in caplog.text
    assert "TPSP using standard forward" in caplog.text
