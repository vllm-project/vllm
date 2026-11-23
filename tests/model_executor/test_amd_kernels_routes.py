# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`AmdKernelsConfig` and the shared `amd_kernels` route switches (host only)."""

import pytest

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
from vllm.config.kernel import AmdKernelsConfig
from vllm.model_executor import amd_kernels_routes as akr


@pytest.fixture
def routes():
    r = akr.AmdKernelsRoutes(
        "test_model",
        ("a", "b", "c", "d"),
        needs={"b": ("a",), "c": ("b",)},
    )
    yield r
    akr._REGISTRY.pop("test_model", None)


def test_resolve_drops_disabled_routes_and_their_dependents(routes):
    assert routes.resolve(True, (), True) == {"a", "b", "c", "d"}
    assert routes.resolve(True, ("a",), True) == {"d"}
    assert routes.resolve(True, ("c",), True) == {"a", "b", "d"}
    assert routes.resolve(False, (), True) == frozenset()
    assert routes.resolve(True, (), False) == frozenset()


def test_unknown_prerequisite_is_an_error():
    with pytest.raises(ValueError, match="unknown routes"):
        akr.AmdKernelsRoutes("bad", ("a",), needs={"a": ("x",)})


def test_routes_follow_the_config(routes, monkeypatch):
    monkeypatch.setattr(akr, "available", lambda: True)
    cfg = VllmConfig(
        kernel_config=KernelConfig(
            amd_kernels=AmdKernelsConfig(disabled_routes=["b", "other_model_route"])
        )
    )
    with set_current_vllm_config(cfg):
        assert routes.routes() == {"a", "d"}
    assert routes.on("a") and not routes.on("c")
    with pytest.raises(AssertionError):
        routes.on("unknown")


def test_off_without_the_package(routes, monkeypatch):
    monkeypatch.setattr(akr, "available", lambda: False)
    with set_current_vllm_config(VllmConfig()):
        assert routes.routes() == frozenset()


def test_fail_takes_out_the_route_and_its_dependents(routes, monkeypatch):
    monkeypatch.setattr(akr, "available", lambda: True)
    with set_current_vllm_config(VllmConfig()):
        routes.routes()
    routes.fail("b", "post_load failed")
    assert routes.on("a") and not routes.on("b") and not routes.on("c")
    assert routes.failed() == {"b": "post_load failed"}
    routes.reset()
    assert routes.failed() == {}


def test_covered():
    assert akr.covered("op", None)
    assert not akr.covered("op", "op: no kernel for n=1")


def test_config_parses_and_hashes():
    assert AmdKernelsConfig(disabled_routes="a, b").disabled_routes == ["a", "b"]
    one = KernelConfig(amd_kernels=AmdKernelsConfig(disabled_routes=["b", "a"]))
    two = KernelConfig(amd_kernels=AmdKernelsConfig(disabled_routes=["a", "b", "a"]))
    off = KernelConfig(amd_kernels=AmdKernelsConfig(enabled=False))
    assert one.compile_factors() == two.compile_factors()
    assert one.compile_factors() != off.compile_factors()
    assert (
        KernelConfig(**{"amd_kernels": {"enabled": False}}).amd_kernels.enabled is False
    )
