# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.platforms.interface import Platform
from vllm.utils import mem_utils
from vllm.v1.worker import tpsp_profile
from vllm.v1.worker.tpsp_profile import (
    ChunkConfig,
    SPProfile,
    TPSPBackend,
    TPSPProjection,
    TPSPProjectionContext,
    TPSPShape,
    close_tpsp_projections,
    get_tpsp_backend,
    profile_tpsp_projections,
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


def make_model(shapes):
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(4))
    model.projections = nn.ModuleDict(
        {name: TPSPProjection(shape, 2, "test") for name, shape in shapes.items()}
    )
    return model


def test_projection_modules_dispatch_and_close_contexts(monkeypatch):
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(get_tpsp_backend_cls=lambda: f"{__name__}.FakeBackend"),
        raising=False,
    )
    shapes = {
        "first": TPSPShape(2, 4, 1e-5, False),
        "second": TPSPShape(2, 6, 1e-5, False),
    }
    model = make_model(shapes)
    assert profile_tpsp_projections(model, 8)
    backend = FakeBackend.last_opened
    assert backend is not None
    assert len(backend.contexts) == 2
    assert all(
        projection.profile.config.transport is backend.contexts[index]
        for index, projection in enumerate(model.projections.values())
    )
    for projection in model.projections.values():
        assert backend.fused(1, 2, 3, 4, 1e-5, projection.profile.config) == (1, 2, 4)
    assert profile_tpsp_projections(model, 8)
    assert len(backend.contexts) == 2
    close_tpsp_projections(model)
    assert backend.closed_contexts == backend.contexts
    assert backend._closed
    assert all(projection.backend is None for projection in model.projections.values())


def test_live_tpsp_contexts_count_toward_non_kv_memory(monkeypatch):
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(get_tpsp_backend_cls=lambda: f"{__name__}.FakeBackend"),
        raising=False,
    )
    monkeypatch.setattr(
        mem_utils.current_platform, "is_integrated_gpu", lambda _: False
    )
    monkeypatch.setattr(torch.accelerator, "memory_stats", lambda device: {})
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "reset_peak_memory_stats", lambda device: None
    )
    baseline = mem_utils.MemorySnapshot(device="cpu", auto_measure=False)
    baseline.free_memory = 1024
    baseline.total_memory = 1024

    model = make_model(
        {"o": TPSPShape(2, 4, 1e-5, True), "down": TPSPShape(2, 6, 1e-5, True)}
    )
    profile_tpsp_projections(model, 8)
    backend = FakeBackend.last_opened
    assert backend is not None
    monkeypatch.setattr(
        torch.accelerator,
        "get_memory_info",
        lambda device: (
            1024 - 128 * (len(backend.contexts) - len(backend.closed_contexts)),
            1024,
        ),
    )
    with mem_utils.memory_profiling(baseline) as result:
        pass
    assert result.non_kv_cache_memory == 256
    close_tpsp_projections(model)
    assert torch.accelerator.get_memory_info("cpu")[0] == 1024


def test_profile_failure_releases_opened_contexts(monkeypatch):
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(get_tpsp_backend_cls=lambda: f"{__name__}.FakeBackend"),
        raising=False,
    )
    original_profile = FakeBackend.profile

    def fail_second(self, *, hidden_size, **kwargs):
        if hidden_size == 6:
            raise RuntimeError("profiling failed")
        return original_profile(self, hidden_size=hidden_size, **kwargs)

    monkeypatch.setattr(FakeBackend, "profile", fail_second)
    model = make_model(
        {"o": TPSPShape(2, 4, 1e-5, True), "down": TPSPShape(2, 6, 1e-5, True)}
    )
    with pytest.raises(RuntimeError, match="profiling failed"):
        profile_tpsp_projections(model, 8)
    backend = FakeBackend.last_opened
    assert backend is not None
    assert backend.closed_contexts == backend.contexts
    assert backend._closed
    assert all(not projection.active for projection in model.projections.values())


@pytest.mark.parametrize("disabled_size", (4, 6))
def test_mixed_plans_keep_only_enabled_context(monkeypatch, caplog, disabled_size):
    monkeypatch.setattr(FakeBackend, "disable_hidden_size", disabled_size)
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(get_tpsp_backend_cls=lambda: f"{__name__}.FakeBackend"),
        raising=False,
    )
    model = make_model(
        {"first": TPSPShape(2, 4, 1e-5, False), "second": TPSPShape(2, 6, 1e-5, False)}
    )
    profile_tpsp_projections(model, 8)
    backend = FakeBackend.last_opened
    assert backend is not None
    enabled_name = "first" if disabled_size == 6 else "second"
    disabled_name = "second" if disabled_size == 6 else "first"
    assert (
        model.projections[enabled_name].context is backend.contexts[disabled_size == 4]
    )
    assert model.projections[enabled_name].backend is backend
    assert model.projections[disabled_name].context is None
    assert backend.closed_contexts == [backend.contexts[disabled_size == 6]]
    assert not backend._closed
    assert model.projections[enabled_name].active
    assert model.projections[disabled_name].profile.status == "unsupported"
    assert "TPSP using regular projection for inactive plans" in caplog.text
    close_tpsp_projections(model)
    assert len(backend.closed_contexts) == 2
    assert set(backend.closed_contexts) == set(backend.contexts)
    assert backend._closed


def test_no_enabled_plans_release_backend(monkeypatch, caplog):
    monkeypatch.setattr(
        "vllm.platforms.current_platform",
        SimpleNamespace(get_tpsp_backend_cls=lambda: f"{__name__}.FakeBackend"),
        raising=False,
    )
    monkeypatch.setattr(
        FakeBackend,
        "profile",
        lambda self, *, tp_size, hidden_size, max_batched_tokens, **kwargs: SPProfile(
            tp_size, hidden_size, max_batched_tokens, "disabled", "no benefit"
        ),
    )
    model = make_model(
        {"first": TPSPShape(2, 4, 1e-5, False), "second": TPSPShape(2, 6, 1e-5, False)}
    )
    profile_tpsp_projections(model, 8)
    backend = FakeBackend.last_opened
    assert backend is not None
    assert all(projection.backend is None for projection in model.projections.values())
    assert all(not projection.active for projection in model.projections.values())
    assert backend.closed_contexts == backend.contexts
    assert backend._closed
    assert "TPSP using standard forward" in caplog.text


def test_unsupported_platform_logs_standard_forward(monkeypatch, caplog):
    monkeypatch.setattr("vllm.platforms.current_platform", Platform(), raising=False)
    model = make_model({"first": TPSPShape(2, 4, 1e-5, False)})
    profile_tpsp_projections(model, 8)
    assert model.projections["first"].backend is None
    assert model.projections["first"].profile.status == "unsupported"
    assert "first=unsupported (no fused backend on this device)" in caplog.text
    assert "TPSP using standard forward" in caplog.text
