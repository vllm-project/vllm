# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for vllm#64456.

The NIXL receive buffer has a fixed capacity per expert, but an expert may
receive fewer tokens than that capacity; rows past ``expert_num_tokens[e]``
are stale leftovers from earlier calls. Dynamic per-tensor quantization used
to compute its single scale over the whole buffer, so stale rows inflated the
scale and corrupted the quantized valid rows. These tests pin the fix: the
scale passed to the quantizer must come from valid rows only, while
per-token, block, and static-scale paths are left untouched.

The tests stub the optional ``nixl_ep`` package (unavailable on CPU test
hosts) and the CUDA-only quant kernel, so they run on CPU: what is under
test is the prepare/finalize wiring (which rows feed the scale), not the
kernel itself.
"""

import sys
import types

import pytest
import torch

if "nixl_ep" not in sys.modules:
    _nixl_ep_stub = types.ModuleType("nixl_ep")
    _nixl_ep_stub.Buffer = object
    sys.modules["nixl_ep"] = _nixl_ep_stub

import vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep as nixl_ep_mod
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
    NixlEPPrepareAndFinalize,
)

FP8_MAX = 448.0
INT8_MAX = 127.0


def _make_buffer() -> tuple[torch.Tensor, torch.Tensor]:
    """Two experts, capacity 4, with huge stale rows past the valid counts."""
    buf = torch.zeros(2, 4, 8)
    buf[0, :2] = torch.tensor([[1.0] * 8, [-0.5] * 8])
    buf[1, :3] = torch.tensor([[0.5] * 8, [-0.25] * 8, [0.125] * 8])
    buf[0, 2:] = 1000.0
    buf[1, 3:] = -1000.0
    return buf, torch.tensor([2, 3])


def _make_prepare() -> NixlEPPrepareAndFinalize:
    # The buffer is never touched by _do_quant, so None suffices here.
    return NixlEPPrepareAndFinalize(None, 8, 1, 8)


class _FakeQuantizer:
    """Reference quantizer standing in for the CUDA-only kernel.

    Records every call and emulates static-scale quantization when a scale
    is provided, so tests can assert on the exact scale _do_quant chose.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, A, A_scale, q_dtype, per_act_token, block_shape):
        self.calls.append(
            {
                "A_scale": A_scale,
                "q_dtype": q_dtype,
                "per_act_token": per_act_token,
                "block_shape": block_shape,
            }
        )
        if A_scale is not None:
            scale = A_scale
        elif per_act_token:
            scale = A.abs().amax(dim=-1, keepdim=True) / FP8_MAX
            scale = torch.clamp(scale, min=1e-10)
        elif block_shape is not None:
            return A.clone(), torch.ones(A.size(0), 1)
        else:
            scale = torch.tensor([A.abs().max().item() / FP8_MAX])
        if q_dtype == torch.int8:
            q = torch.clamp(A / scale, -INT8_MAX, INT8_MAX).to(torch.int8)
        else:
            q = torch.clamp(A / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q, scale


def test_valid_rows_scale_ignores_stale_rows():
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
        _dynamic_per_tensor_scale_for_valid_rows,
    )

    buf, counts = _make_buffer()
    scale = _dynamic_per_tensor_scale_for_valid_rows(buf, counts, FP8_MAX)
    assert scale is not None
    assert scale.item() == pytest.approx(1.0 / FP8_MAX)
    # The polluted scale over the whole buffer would be 1000 / 448.
    assert scale.item() != pytest.approx(1000.0 / FP8_MAX)


def test_valid_rows_scale_mismatched_counts_falls_back():
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
        _dynamic_per_tensor_scale_for_valid_rows,
    )

    buf, _ = _make_buffer()
    assert (
        _dynamic_per_tensor_scale_for_valid_rows(buf, torch.tensor([2]), FP8_MAX)
        is None
    )


def test_valid_rows_scale_all_zero_token_experts_is_clamped():
    from vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep import (
        _dynamic_per_tensor_scale_for_valid_rows,
    )

    buf, _ = _make_buffer()
    scale = _dynamic_per_tensor_scale_for_valid_rows(buf, torch.tensor([0, 0]), FP8_MAX)
    assert scale is not None
    assert scale.item() == pytest.approx(1e-10)


@pytest.mark.parametrize(
    ("q_dtype", "quant_max"),
    [(torch.float8_e4m3fn, FP8_MAX), (torch.int8, INT8_MAX)],
)
def test_do_quant_uses_valid_rows_scale(monkeypatch, q_dtype, quant_max):
    buf, counts = _make_buffer()
    prepare = _make_prepare()
    fake = _FakeQuantizer()
    monkeypatch.setattr(nixl_ep_mod, "moe_kernel_quantize_input", fake)
    quant_config = FusedMoEQuantConfig.make(q_dtype)
    out, out_scales = prepare._do_quant(buf, torch.bfloat16, quant_config, counts)

    assert len(fake.calls) == 1
    scale = fake.calls[0]["A_scale"]
    assert scale is not None
    assert scale.item() == pytest.approx(1.0 / quant_max)
    assert out.shape == buf.shape
    assert out_scales.shape == (2, 1, 1)
    assert out_scales[0, 0, 0].item() == pytest.approx(1.0 / quant_max)
    # Valid rows must match quantization under the valid-only scale, not the
    # polluted one.
    valid = torch.cat([buf[e, :n] for e, n in enumerate(counts.tolist())])
    out_valid = torch.cat([out[e, :n] for e, n in enumerate(counts.tolist())])
    if q_dtype == torch.int8:
        expected = torch.clamp(valid / (1.0 / quant_max), -INT8_MAX, INT8_MAX).to(
            torch.int8
        )
    else:
        expected = torch.clamp(valid / (1.0 / quant_max), -FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn
        )
    assert torch.equal(out_valid, expected)


def test_do_quant_zero_token_expert(monkeypatch):
    buf, _ = _make_buffer()
    counts = torch.tensor([0, 2])
    prepare = _make_prepare()
    fake = _FakeQuantizer()
    monkeypatch.setattr(nixl_ep_mod, "moe_kernel_quantize_input", fake)
    quant_config = FusedMoEQuantConfig.make(torch.float8_e4m3fn)
    out, _ = prepare._do_quant(buf, torch.bfloat16, quant_config, counts)
    # Only expert 1's valid rows (max 0.5) feed the scale.
    assert fake.calls[0]["A_scale"].item() == pytest.approx(0.5 / FP8_MAX)
    assert out.shape == buf.shape


def test_receiver_passes_counts_to_quant(monkeypatch):
    # The reported bug: _receiver had the counts but did not pass them on.
    buf, counts = _make_buffer()
    prepare = _make_prepare()
    fake = _FakeQuantizer()
    monkeypatch.setattr(nixl_ep_mod, "moe_kernel_quantize_input", fake)
    quant_config = FusedMoEQuantConfig.make(torch.float8_e4m3fn)
    out, out_scales, meta, _, _ = prepare._receiver(
        buf, counts, None, torch.bfloat16, quant_config
    )
    assert fake.calls[0]["A_scale"].item() == pytest.approx(1.0 / FP8_MAX)
    assert out.shape == buf.shape
    assert out_scales.shape == (2, 1, 1)
    assert torch.equal(meta.expert_num_tokens, counts)


def test_do_quant_without_counts_keeps_dynamic_behavior(monkeypatch):
    buf, _ = _make_buffer()
    prepare = _make_prepare()
    fake = _FakeQuantizer()
    monkeypatch.setattr(nixl_ep_mod, "moe_kernel_quantize_input", fake)
    quant_config = FusedMoEQuantConfig.make(torch.float8_e4m3fn)
    prepare._do_quant(buf, torch.bfloat16, quant_config)
    assert fake.calls[0]["A_scale"] is None


@pytest.mark.parametrize(
    "quant_config",
    [
        FusedMoEQuantConfig.make(torch.float8_e4m3fn, per_act_token_quant=True),
        FusedMoEQuantConfig.make(torch.float8_e4m3fn, a1_scale=torch.tensor([0.25])),
        FusedMoEQuantConfig.make(torch.float8_e4m3fn, block_shape=[1, 128]),
    ],
)
def test_do_quant_leaves_other_quant_paths_untouched(monkeypatch, quant_config):
    buf, counts = _make_buffer()
    prepare = _make_prepare()
    fake = _FakeQuantizer()
    monkeypatch.setattr(nixl_ep_mod, "moe_kernel_quantize_input", fake)
    out, _ = prepare._do_quant(buf, torch.bfloat16, quant_config, counts)
    assert out.shape == buf.shape
    if quant_config.a1_scale is not None:
        assert fake.calls[0]["A_scale"].item() == pytest.approx(0.25)
    else:
        assert fake.calls[0]["A_scale"] is None


def test_do_quant_fp8_dispatch_block_path_early_returns(monkeypatch):
    prepare = NixlEPPrepareAndFinalize(None, 8, 1, 8, use_fp8_dispatch=True)
    fake = _FakeQuantizer()
    monkeypatch.setattr(nixl_ep_mod, "moe_kernel_quantize_input", fake)
    quant_config = FusedMoEQuantConfig.make(torch.float8_e4m3fn, block_shape=[128, 128])
    x_fp8 = torch.zeros(2, 4, 8, dtype=torch.float8_e4m3fn)
    x_scales = torch.ones(2, 4, 1)
    out, out_scales = prepare._do_quant(
        (x_fp8, x_scales),
        torch.bfloat16,
        quant_config,
        torch.tensor([2, 3]),
    )
    assert fake.calls == []
    # Early return hands the dispatched tensors back untouched.
    assert out is x_fp8
    assert out_scales is x_scales
