# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for selective_state_update tuning config loading."""

import json

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    get_ssu_config_file_name,
    get_ssu_configs,
    get_ssu_default_config,
    get_ssu_kernel_config,
    selective_state_update,
)
from vllm.utils.torch_utils import set_random_seed


def selective_state_update_ref(
    state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """Reference implementation for selective_state_update."""
    import torch.nn.functional as F
    from einops import rearrange

    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    if dt_bias is not None:
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, "b h d -> b h d 1") * A)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")
    state.copy_(state * dA + dB * rearrange(x, "b h d -> b h d 1"))
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


class TestGetSSUConfigFileName:
    """Tests for get_ssu_config_file_name."""

    def test_basic_format(self):
        filename = get_ssu_config_file_name(2048, 16)
        assert filename.startswith("dim=2048,dstate=16,device_name=")
        assert filename.endswith(".json")

    def test_different_params(self):
        f1 = get_ssu_config_file_name(2048, 16)
        f2 = get_ssu_config_file_name(4096, 64)
        assert f1 != f2


class TestGetSSUDefaultConfig:
    """Tests for get_ssu_default_config."""

    def test_small_dstate(self):
        config = get_ssu_default_config(16)
        assert config["BLOCK_SIZE_M"] == 32
        assert config["num_warps"] == 4

    def test_medium_dstate(self):
        config = get_ssu_default_config(32)
        assert config["BLOCK_SIZE_M"] == 16
        assert config["num_warps"] == 4

    def test_large_dstate(self):
        config = get_ssu_default_config(64)
        assert config["BLOCK_SIZE_M"] == 8
        assert config["num_warps"] == 4

    def test_very_large_dstate(self):
        config = get_ssu_default_config(128)
        assert config["BLOCK_SIZE_M"] == 4
        assert config["num_warps"] == 4

    def test_blackwell_large_dstate(self):
        config = get_ssu_default_config(128, is_blackwell=True)
        assert config["BLOCK_SIZE_M"] == 32
        assert config["num_warps"] == 8

    def test_default_fallback(self):
        config = get_ssu_default_config(256)
        assert config["BLOCK_SIZE_M"] == 4
        assert config["num_warps"] == 8


class TestGetSSUConfigs:
    """Tests for get_ssu_configs with file-based config loading."""

    def test_returns_none_without_config_file(self):
        get_ssu_configs.cache_clear()
        result = get_ssu_configs(99999, 99999)
        assert result is None

    def test_loads_from_custom_folder(self, monkeypatch, tmp_path):
        get_ssu_configs.cache_clear()
        config_data = {
            "16": {"BLOCK_SIZE_M": 64, "num_warps": 2},
        }
        config_file = tmp_path / get_ssu_config_file_name(2048, 16)
        config_file.write_text(json.dumps(config_data))

        monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        result = get_ssu_configs(2048, 16)
        assert result is not None
        assert result["16"]["BLOCK_SIZE_M"] == 64
        assert result["16"]["num_warps"] == 2

        get_ssu_configs.cache_clear()

    def test_strips_triton_version(self, tmp_path, monkeypatch):
        get_ssu_configs.cache_clear()
        config_data = {
            "triton_version": "3.0.0",
            "16": {"BLOCK_SIZE_M": 32, "num_warps": 4},
        }
        config_file = tmp_path / get_ssu_config_file_name(2048, 16)
        config_file.write_text(json.dumps(config_data))

        monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        result = get_ssu_configs(2048, 16)
        assert result is not None
        assert "triton_version" not in result

        get_ssu_configs.cache_clear()


class TestGetSSUKernelConfig:
    """Tests for get_ssu_kernel_config."""

    def test_falls_back_to_defaults(self):
        get_ssu_configs.cache_clear()
        config = get_ssu_kernel_config(99999, 16)
        expected = get_ssu_default_config(16)
        assert config == expected

    def test_returns_valid_config_keys(self):
        get_ssu_configs.cache_clear()
        config = get_ssu_kernel_config(2048, 64)
        assert "BLOCK_SIZE_M" in config
        assert "num_warps" in config
        assert isinstance(config["BLOCK_SIZE_M"], int)
        assert isinstance(config["num_warps"], int)


@pytest.mark.parametrize("dstate", [16, 64])
@pytest.mark.parametrize("dim", [2048, 4096])
def test_selective_state_update_with_config_loading(dim, dstate):
    """Verify that config loading does not change kernel correctness."""
    device = "cuda"
    itype = torch.float32
    rtol, atol = 3e-4, 1e-3

    set_random_seed(0)
    batch_size = 1
    state = torch.randn(batch_size, dim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    z = torch.randn_like(x)
    state_ref = state.detach().clone()

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
    )
    out_ref = selective_state_update_ref(
        state_ref,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
    )

    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
