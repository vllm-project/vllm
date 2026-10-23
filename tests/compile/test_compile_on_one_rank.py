# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for VLLM_COMPILE_ON_ONE_RANK.

A worker's local rank no longer implies its device (``--device-ids``), so the
compile cache, which is keyed by rank, can be reused on a different device.
Inductor bakes the compile-time device index into the generated code unless
torch's compile-on-one-rank mode is on, which resolves it at run time.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.config.utils import hash_factors
from vllm.env_override import _apply_compile_on_one_rank
from vllm.platforms import current_platform

compiler_config = pytest.importorskip("torch.compiler.config")
has_compile_on_one_rank = hasattr(compiler_config, "compile_on_one_rank")


@pytest.mark.parametrize("torch_has_mode", [True, False])
def test_default(monkeypatch: pytest.MonkeyPatch, torch_has_mode: bool):
    monkeypatch.delenv("VLLM_COMPILE_ON_ONE_RANK", raising=False)
    monkeypatch.setattr(
        "vllm.utils.torch_utils.is_torch_equal_or_newer",
        lambda version: torch_has_mode,
    )
    platform_supported = current_platform.is_cuda_alike() or current_platform.is_xpu()
    assert envs.VLLM_COMPILE_ON_ONE_RANK is (torch_has_mode and platform_supported)


def test_flag_is_compile_cache_factor(monkeypatch: pytest.MonkeyPatch):
    # Code compiled with the mode off bakes in a device index, so a cache
    # built with it off must never be reused with it on, and vice versa.
    monkeypatch.setenv("VLLM_COMPILE_ON_ONE_RANK", "1")
    factors_on = envs.compile_factors()
    monkeypatch.setenv("VLLM_COMPILE_ON_ONE_RANK", "0")
    factors_off = envs.compile_factors()

    assert factors_on["VLLM_COMPILE_ON_ONE_RANK"] is True
    assert factors_off["VLLM_COMPILE_ON_ONE_RANK"] is False
    assert hash_factors(factors_on) != hash_factors(factors_off)


@pytest.mark.skipif(
    not has_compile_on_one_rank, reason="torch has no compile_on_one_rank"
)
@pytest.mark.parametrize("value", ["1", "0"])
def test_env_override_applies_flag_to_torch(
    monkeypatch: pytest.MonkeyPatch, value: str
):
    monkeypatch.setenv("VLLM_COMPILE_ON_ONE_RANK", value)
    with compiler_config.patch(compile_on_one_rank=value != "1"):
        _apply_compile_on_one_rank()
        assert compiler_config.compile_on_one_rank is (value == "1")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    not has_compile_on_one_rank, reason="torch has no compile_on_one_rank"
)
@pytest.mark.parametrize("enabled", [True, False])
def test_generated_code_device_index(enabled: bool):
    from torch._inductor.utils import run_and_get_code

    def fn(x):
        return torch.sin(x) + 1

    device_idx = torch.accelerator.current_device_index()
    x = torch.randn(64, device=f"cuda:{device_idx}")
    torch._dynamo.reset()
    with (
        compiler_config.patch(compile_on_one_rank=enabled),
        torch._inductor.config.patch(fx_graph_cache=False),
    ):
        out, codes = run_and_get_code(torch.compile(fn), x)
    torch._dynamo.reset()

    torch.testing.assert_close(out, fn(x))
    code = "\n".join(codes)
    baked = (
        f"get_raw_stream({device_idx})" in code or f"_DeviceGuard({device_idx})" in code
    )
    # With the mode on, the device is resolved at run time, so the compiled
    # code (and the compile cache holding it) runs on any device index.
    assert baked is not enabled
    assert ("current_device()" in code) is enabled
