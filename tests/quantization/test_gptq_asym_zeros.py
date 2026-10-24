# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for converting asymmetric GPTQ qzeros to the WNA16 zero-point layout.

Run `pytest tests/quantization/test_gptq_asym_zeros.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    gptq_qzeros_to_standard_format,
)


def _pack_along_last(vals: torch.Tensor, bits: int) -> torch.Tensor:
    """(..., M) small ints -> (..., M // pack) int32, element j at bits*j."""
    pack = 32 // bits
    v = vals.to(torch.int64).reshape(*vals.shape[:-1], -1, pack) & ((1 << bits) - 1)
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)
    packed = (v << shifts).sum(-1)
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


def _unpack_dim0(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """(M // pack, G) int32 packed along dim 0 -> (M, G)."""
    pack = 32 // bits
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    vals = (packed.unsqueeze(1) >> shifts[None, :, None]) & ((1 << bits) - 1)
    return vals.reshape(packed.shape[0] * pack, packed.shape[1])


@pytest.mark.parametrize("v1", [True, False])
def test_qzeros_layout_and_offset(v1: bool):
    torch.manual_seed(0)
    groups, n = 5, 64
    z = torch.randint(0, 16, (groups, n))
    z[0, :8] = torch.tensor([0, 15, 8, 1, 14, 0, 7, 15])  # edge values
    stored = (z - 1) if v1 else z  # v1 stores z - 1 (per element, mod 16)
    qzeros = _pack_along_last(stored, 4)  # GPTQ layout (G, N // 8)

    out = gptq_qzeros_to_standard_format(qzeros, 4, stores_zero_minus_one=v1)

    assert out.shape == (n // 8, groups) and out.dtype == torch.int32
    assert out.is_contiguous()
    torch.testing.assert_close(_unpack_dim0(out, 4), z.t().to(torch.int32))


def test_v1_minus_one_does_not_carry():
    # z = 0 is stored as 0xF in v1; +1 must wrap to 0 without touching the next nibble.
    z = torch.zeros(1, 8, dtype=torch.int64)
    z[0, 1] = 5
    qzeros = _pack_along_last(z - 1, 4)
    out = gptq_qzeros_to_standard_format(qzeros, 4, stores_zero_minus_one=True)
    assert _unpack_dim0(out, 4)[:, 0].tolist() == [0, 5, 0, 0, 0, 0, 0, 0]


def _asym_config(dynamic: dict | None = None):
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    cfg = {"bits": 4, "group_size": 128, "desc_act": False, "sym": False}
    if dynamic is not None:
        cfg["dynamic"] = dynamic
    return AutoGPTQConfig.from_config(cfg)


@pytest.mark.parametrize(
    "dynamic", [None, {"-:.*mtp.*": {}}, {"+:.*mlp.*": {"group_size": 64}}]
)
def test_asym_config_and_dynamic_override_on_xpu(monkeypatch, dynamic):
    # Per-module dynamic rules go through override_config, which must accept
    # the same (bits, sym) as the base config (GPTQModel writes "-:.*mtp.*").
    from vllm.model_executor.layers.quantization import auto_gptq
    from vllm.model_executor.layers.quantization.utils.gptq_utils import (
        override_config,
    )
    from vllm.scalar_type import scalar_types

    monkeypatch.setattr(auto_gptq.current_platform, "is_xpu", lambda: True)
    config = _asym_config(dynamic)
    assert config.quant_type == scalar_types.uint4
    override_config(config, prefix="model.layers.0.mlp.gate_up_proj")
    assert config.quant_type == scalar_types.uint4


def test_asym_config_rejected_off_xpu(monkeypatch):
    from vllm.model_executor.layers.quantization import auto_gptq

    monkeypatch.setattr(auto_gptq.current_platform, "is_xpu", lambda: False)
    with pytest.raises(ValueError, match="sym=False"):
        _asym_config()
