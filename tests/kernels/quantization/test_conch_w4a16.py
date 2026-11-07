#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Conch W4A16 mixed-precision kernel (ROCm).

Covers can_implement's tile preconditions, the raise for an explicit
--linear-backend that cannot implement a shape, auto-selection, and numerics
with the padded-N weight cache.
"""

import pytest
import torch

import vllm.model_executor.kernels.linear as linear_kernels_pkg
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.conch import (
    ConchLinearKernel,
    _conch_padded_width,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

# This test module is ROCm/Conch specific. Avoid import-time failures on
# non-ROCm or environments without conch by skipping early.
if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("conch")

device = "cuda"


def _make_config(
    K: int,
    N: int,
    group_size: int,
    weight_type=scalar_types.uint4b8,
    act_type=torch.bfloat16,
    zero_points: bool = False,
) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=weight_type,
        act_type=act_type,
        group_size=group_size,
        zero_points=zero_points,
    )


@pytest.mark.parametrize(
    "K,N,group_size,expected,reason_substr",
    [
        (5120, 16384, 128, True, ""),
        # N=96 (issue-60581): padded up in process_weights_after_loading, so accept it.
        (5120, 96, 128, True, ""),
        # A partial packing group cannot be padded (the unpacked weight is ceil-packed).
        (5120, 100, 128, False, "packing"),
        # K-side violations cannot be padded away.
        (192, 1024, 128, False, "group size"),  # K % gs != 0
        (196, 1024, 128, False, "block_k"),  # K % block_k != 0
        # Channelwise: gs resolves to K, which is tile-aligned here.
        (5120, 16384, -1, True, ""),
    ],
)
def test_conch_can_implement_matrix(K, N, group_size, expected, reason_substr):
    config = _make_config(K, N, group_size)
    can_implement, reason = ConchLinearKernel.can_implement(config)
    assert can_implement is expected
    if not expected:
        assert reason is not None and reason_substr in reason


def test_explicit_backend_raises_for_unimplementable_shape(monkeypatch):
    """--linear-backend conch + K=196 (rejected by conch) raises ValueError with
    the rejection reason instead of silently running another backend."""
    monkeypatch.setattr(linear_kernels_pkg, "_get_linear_backend", lambda **_: "conch")
    config = _make_config(196, 96, 128)

    with pytest.raises(ValueError) as excinfo:
        choose_mp_linear_kernel(config)

    message = str(excinfo.value)
    assert "WNA16 linear layer" in message
    assert "ConchLinearKernel" in message
    assert "must be divisible" in message


def test_auto_backend_still_raises_when_nothing_implements(monkeypatch):
    """With backend "auto" nothing falls back: every real kernel rejects K=196,
    so selection still raises ValueError listing the per-kernel reasons."""
    monkeypatch.setattr(linear_kernels_pkg, "_get_linear_backend", lambda **_: "auto")
    config = _make_config(196, 96, 128)

    with pytest.raises(ValueError) as excinfo:
        choose_mp_linear_kernel(config)

    message = str(excinfo.value)
    assert "WNA16 linear layer" in message
    assert "TritonW4A16LinearKernel" in message
    assert "ConchLinearKernel" in message


def test_uint8b128_auto_selects_conch():
    # triton supports only 4-bit weights and exllama needs fp16 activations, so
    # Conch is the only candidate for uint8b128 + bf16 (N=96 pads up).
    config = _make_config(
        5120, 96, 128, weight_type=scalar_types.uint8b128, zero_points=False
    )
    assert choose_mp_linear_kernel(config) is ConchLinearKernel


def _build_quark_layer(
    monkeypatch, K, N, w_ckpt_kn8, scale_gn, zp_ckpt=None, process=True
):
    """A real QuarkW4A16Int4 layer (parameters + process sequence) wired to
    ConchLinearKernel with random packed checkpoint data; zp_ckpt, if given, is
    the asymmetric zero point in Quark's packed layout."""
    from vllm.model_executor.layers.quantization.quark.schemes import (
        quark_w4a16_int4 as quark_w4a16_int4_module,
    )
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w4a16_int4 import (
        QuarkW4A16Int4,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt4Static,
        kInt4StaticAsym,
    )

    monkeypatch.setattr(
        quark_w4a16_int4_module,
        "choose_mp_linear_kernel",
        lambda config, compute_capability=None: ConchLinearKernel,
    )

    scheme = QuarkW4A16Int4(
        weight_quant_key=kInt4Static if zp_ckpt is None else kInt4StaticAsym,
        activation_quant_key=None,
        pack_method="reorder",
        weight_config={"group_size": 128, "symmetric": zp_ckpt is None},
    )
    layer = torch.nn.Module()
    # The scheme locks in the kernel config before weights exist.
    assert scheme.group_size == 128
    assert scheme.is_symmetric == (zp_ckpt is None)
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[N],
        input_size_per_partition=K,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
        input_size=K,
        output_size=N,
    )
    layer.weight.data = w_ckpt_kn8
    layer.weight_scale.data = scale_gn
    if zp_ckpt is not None:
        layer.weight_zero_point.data = zp_ckpt
    if process:
        scheme.process_weights_after_loading(layer)
    return scheme, layer


def _quark_reference(x, w_q_kn8, scale_gn, *, group_size, weight_bias, zp_gn=None):
    """fp32 dequant reference reading the POST-transform [K/8, N] nibbles:
    nibble at shift 4*(k%8), minus weight_bias (or the per-group zero points),
    times scale[k//gs, n]."""
    K, N = x.shape[1], w_q_kn8.shape[1]
    shifts = torch.arange(8, device=w_q_kn8.device, dtype=torch.int32) * 4
    nibbles = ((w_q_kn8.unsqueeze(-1) >> shifts) & 0xF).permute(0, 2, 1)
    w_fp = nibbles.reshape(K, N).to(torch.float32)
    if zp_gn is not None:
        w_fp = w_fp - zp_gn.to(torch.float32).repeat_interleave(group_size, dim=0)
    else:
        w_fp = w_fp - weight_bias
    w_fp = w_fp * scale_gn.to(torch.float32).repeat_interleave(group_size, dim=0)
    return (x.to(torch.float32) @ w_fp).to(x.dtype)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def _init_test_distributed() -> None:
    """PackedvLLMParameter captures the TP rank at construction time."""
    if torch.distributed.is_initialized():
        return
    from tests.utils import init_test_distributed_environment

    init_test_distributed_environment(1, 1, 0, distributed_init_port="0")


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_conch_padded_n96_numerics_and_cross_config_invariance(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    _init_test_distributed()
    set_random_seed(0)

    K, N, G = 5120, 96, 128
    w_ckpt = torch.randint(0, 16, (K, N // 8), device=device, dtype=torch.int32)
    scale = (0.05 * torch.rand((K // G, N), device=device, dtype=torch.float32)).to(
        torch.bfloat16
    )

    scheme96, layer96 = _build_quark_layer(monkeypatch, K, N, w_ckpt, scale)
    pad_to = N + _conch_padded_width(N)

    # Registered parameters keep their original shapes; the pads do not.
    assert tuple(layer96.weight.shape) == (K // 8, N)
    assert tuple(layer96.weight_scale.shape) == (K // G, N)
    w_q_pad = layer96._conch_padded_w_q
    w_s_pad = layer96._conch_padded_w_s
    w_zp_pad = layer96._conch_padded_w_zp
    assert w_q_pad is not None and w_s_pad is not None
    assert tuple(w_q_pad.shape) == (K // 8, pad_to)
    assert tuple(w_s_pad.shape) == (K // G, pad_to)
    assert w_zp_pad is None  # symmetric uint4b8: no zero points
    # non-persistent buffers: in layer._buffers, absent from state_dict()
    assert not any(k.startswith("_conch_padded") for k in layer96.state_dict())

    M = 17
    x = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.bfloat16
    )
    out = scheme96.apply_weights(layer96, x)
    assert out.shape == (M, N)

    ref = _quark_reference(
        x,
        layer96.weight.data,
        layer96.weight_scale.data,
        group_size=G,
        weight_bias=scalar_types.uint4b8.bias,
    )
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    # Cross-config invariance: the same weights configured as N=64 (no padding)
    # must produce bit-identical first 64 columns - tile-identical math.
    scheme64, layer64 = _build_quark_layer(
        monkeypatch, K, 64, w_ckpt[:, : 64 // 8], scale[:, :64]
    )
    assert layer64._conch_padded_w_q is None
    out64 = scheme64.apply_weights(layer64, x)
    assert out64.shape == (M, 64)
    assert torch.equal(out[:, :64], out64), (
        "padded N=96 and tile-aligned N=64 must be bit-identical for cols 0..63"
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_conch_asymmetric_padded_n96_numerics(monkeypatch):
    """Asymmetric layers pad their zero points too: the padded zeros need the
    scales' shape and strides, and the numerics must match unpadded zeros."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    _init_test_distributed()
    set_random_seed(0)

    K, N, G = 5120, 96, 128
    w_ckpt = torch.randint(0, 16, (K, N // 8), device=device, dtype=torch.int32)
    scale = (0.05 * torch.rand((K // G, N), device=device, dtype=torch.float32)).to(
        torch.bfloat16
    )
    # Quark's checkpoint layout: (num_groups, N // 8) packed int32.
    zp_ckpt = torch.randint(0, 16, (K // G, N // 8), device=device, dtype=torch.int32)

    scheme, layer = _build_quark_layer(
        monkeypatch, K, N, w_ckpt, scale, zp_ckpt=zp_ckpt
    )

    w_q_pad = layer._conch_padded_w_q
    w_s_pad = layer._conch_padded_w_s
    w_zp_pad = layer._conch_padded_w_zp
    pad_to = N + _conch_padded_width(N)
    assert w_q_pad is not None and w_s_pad is not None and w_zp_pad is not None
    assert tuple(w_q_pad.shape) == (K // 8, pad_to)
    assert tuple(w_s_pad.shape) == (K // G, pad_to)
    assert tuple(w_zp_pad.shape) == (K // G, pad_to)
    # The kernel reads the zeros with the scales' strides, so they must match.
    assert w_zp_pad.stride(0) == w_s_pad.stride(0) == pad_to
    assert w_zp_pad.is_contiguous()
    # Head is the unpadded, unpacked zero points; the pad columns are zero.
    assert torch.equal(w_zp_pad[:, :N], layer.weight_zero_point.data)
    assert not w_zp_pad[:, N:].any()
    # Pads stay out of state_dict().
    assert not any(k.startswith("_conch_padded") for k in layer.state_dict())

    M = 17
    x = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.bfloat16
    )
    out = scheme.apply_weights(layer, x)
    assert out.shape == (M, N)

    # Elementwise check: with x = I_K the GEMM returns the dequantized weights, so
    # the padded asymmetric path can be compared tightly (no cancellation noise).
    x_id = torch.eye(K, device=device, dtype=torch.bfloat16)
    deq = scheme.apply_weights(layer, x_id)
    ref_deq = _quark_reference(
        x_id,
        layer.weight.data,
        layer.weight_scale.data,
        group_size=G,
        weight_bias=scalar_types.uint4.bias,
        zp_gn=layer.weight_zero_point.data,
    )
    torch.testing.assert_close(deq, ref_deq, rtol=2e-2, atol=5e-3)

    # And a plain activation; tolerance is peak-relative because the output is
    # bf16 and this dequantized weight matrix is centred on zero.
    ref = _quark_reference(
        x,
        layer.weight.data,
        layer.weight_scale.data,
        group_size=G,
        weight_bias=scalar_types.uint4.bias,
        zp_gn=layer.weight_zero_point.data,
    )
    peak = ref.abs().max().to(torch.float32)
    torch.testing.assert_close(
        out.float(), ref.float(), rtol=2e-2, atol=0.02 * peak.item()
    )


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_conch_channelwise_group_size_is_passed_as_k(monkeypatch):
    """A channelwise layer (group_size=-1) must produce the same numbers as an
    explicit group size of K: conch derives stride_mul = block_k / group_size,
    so -1 would make the group offset negative and read outside the scale
    buffer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    K, N = 5120, 96
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=-1,
        zero_points=False,
    )
    kernel = ConchLinearKernel(config, "weight", "weight_scale", None)
    layer = torch.nn.Module()
    layer.weight = torch.randint(0, 16, (K // 8, N), dtype=torch.int32, device=device)
    # Channelwise: one scale per output column.
    layer.weight_scale = (0.05 * torch.rand((1, N), device=device)).to(torch.bfloat16)

    # x = I_K returns the dequantized weight matrix.
    x_id = torch.eye(K, device=device, dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x_id)
    assert out.shape == (K, N)

    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
    nibbles = ((layer.weight.unsqueeze(-1) >> shifts) & 0xF).permute(0, 2, 1)
    ref = (
        nibbles.reshape(K, N).to(torch.float32) - scalar_types.uint4b8.bias
    ) * layer.weight_scale.float().repeat_interleave(K, dim=0)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=5e-3)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_conch_padded_layer_refuses_weight_reload(monkeypatch):
    """A padded layer refuses a weight reload, driven through the real entry
    points (record metadata, first load, initialize_layerwise_reload). Scope: the
    loader/quant_method plumbing of _layerwise_process is not exercised."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    from vllm.model_executor.model_loader.reload import (
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    _init_test_distributed()
    set_random_seed(0)

    K, N, G = 5120, 96, 128
    w_ckpt = torch.randint(0, 16, (K, N // 8), device=device, dtype=torch.int32)
    scale = (0.05 * torch.rand((K // G, N), device=device, dtype=torch.float32)).to(
        torch.bfloat16
    )
    scheme, layer = _build_quark_layer(monkeypatch, K, N, w_ckpt, scale, process=False)
    model = torch.nn.Module()
    model.add_module("quant_linear", layer)

    # Construction-time metadata: no pads exist yet, exactly as in a real model.
    record_metadata_for_reloading(model)

    # The normal first load, which is what builds the pads.
    scheme.process_weights_after_loading(layer)
    assert layer._conch_padded_w_q is not None
    assert layer._conch_padded_once is True

    # The reload: deletes registered tensors, pads included (metadata predates them).
    initialize_layerwise_reload(model)
    assert "_conch_padded_w_q" not in layer._buffers
    assert not hasattr(layer, "_conch_padded_w_q")
    assert layer._conch_padded_once is True  # the marker survives

    # The reload's processing step (_layerwise_process): must refuse instead of
    # rebuilding a pad the copy-back would discard.
    pad_to = N + _conch_padded_width(N)
    with pytest.raises(NotImplementedError, match=rf"N={N} to {pad_to}"):
        scheme.process_weights_after_loading(layer)
