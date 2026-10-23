# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the AITER W4A16 linear kernel (gfx942, aiter gemm_a16w4).

Run `pytest tests/kernels/quantization/test_aiter_w4a16.py`.
"""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm._aiter_ops import (  # noqa: E402
    is_aiter_found_and_supported,
    rocm_aiter_ops,
)
from vllm.model_executor.kernels.linear.mixed_precision.aiter_w4a16 import (  # noqa: E402
    AiterW4A16LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.platforms.rocm import on_gfx942  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
from vllm.utils.torch_utils import set_random_seed  # noqa: E402

if not (
    is_aiter_found_and_supported()
    and rocm_aiter_ops._gemm_a16w4_importable()
    and on_gfx942()
):
    pytest.skip(
        "needs gfx942 and an AITER build with gemm_a16w4", allow_module_level=True
    )

device = "cuda"


@pytest.fixture(scope="module", autouse=True)
def _dist_init():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.distributed.parallel_state import model_parallel_is_initialized

    if not model_parallel_is_initialized():
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:0",
                local_rank=0,
            )
            ensure_model_parallel_initialized(1, 1)


def _pack8(v: torch.Tensor, dim: int) -> torch.Tensor:
    """Pack int values 0..15 along `dim` (8 per int32, value i at bit 4*i)."""
    v = v.movedim(dim, -1).to(torch.int64)
    v = v.reshape(*v.shape[:-1], v.shape[-1] // 8, 8)
    shifts = torch.arange(8, device=v.device, dtype=torch.int64) * 4
    return (v << shifts).sum(-1).to(torch.int32).movedim(-1, dim).contiguous()


def _config(K, N, G, weight_type, act_type, zero_points):
    return MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=weight_type,
        act_type=act_type,
        group_size=G,
        zero_points=zero_points,
    )


def _make_layer(fmt, q_kn, s_gn, z_gn):
    """Register checkpoint tensors the way the loaders hand them to the kernel.

    fmt="gptq": qweight [K//8, N] (packed_dim=0), scales [G, N] (GPTQ and AWQ
    after _convert_awq_to_standard_format). fmt="ct": weight_packed [N, K//8]
    (packed_dim=1), weight_scale [N, G], weight_zero_point [N//8, G].
    """
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedColumnParameter,
        PackedvLLMParameter,
    )

    loader = lambda *args, **kwargs: None  # noqa: E731
    layer = torch.nn.Module()
    if fmt == "gptq":
        layer.register_parameter(
            "w_q",
            PackedvLLMParameter(
                data=_pack8(q_kn, 0),
                weight_loader=loader,
                input_dim=0,
                output_dim=1,
                packed_dim=0,
                packed_factor=8,
            ),
        )
        layer.register_parameter(
            "w_s",
            GroupQuantScaleParameter(
                data=s_gn.contiguous(), weight_loader=loader, input_dim=0, output_dim=1
            ),
        )
    else:
        layer.register_parameter(
            "w_q",
            PackedvLLMParameter(
                data=_pack8(q_kn.t(), 1),
                weight_loader=loader,
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=8,
            ),
        )
        layer.register_parameter(
            "w_s",
            GroupQuantScaleParameter(
                data=s_gn.t().contiguous(),
                weight_loader=loader,
                input_dim=1,
                output_dim=0,
            ),
        )
    if z_gn is not None:
        layer.register_parameter(
            "w_zp",
            PackedColumnParameter(
                data=_pack8(z_gn.t(), 0),  # [N//8, G]
                weight_loader=loader,
                output_dim=0,
                packed_dim=0,
                packed_factor=8,
            ),
        )
    return layer


def test_gemm_a16w4_fake_impl():
    with FakeTensorMode():
        x = torch.empty((4, 256), device=device, dtype=torch.bfloat16)
        w = torch.empty((512, 128), device=device, dtype=torch.uint8)
        s = torch.empty((512, 2), device=device, dtype=torch.bfloat16)
        out = rocm_aiter_ops.gemm_a16w4(x, w, s, None, None)
        assert out.shape == (4, 512) and out.dtype == x.dtype
        graph = make_fx(lambda x, w, s: rocm_aiter_ops.gemm_a16w4(x, w, s))(x, w, s)
    assert any(
        node.op == "call_function"
        and node.target == torch.ops.vllm.rocm_aiter_gemm_a16w4.default
        for node in graph.graph.nodes
    )


@pytest.mark.parametrize("fmt", ["gptq", "ct"])
@pytest.mark.parametrize("act_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_type", [scalar_types.uint4b8, scalar_types.uint4])
@pytest.mark.parametrize("group_size", [-1, 32, 128])
@pytest.mark.parametrize("M", [1, 16, 33, 256])
def test_aiter_w4a16_matches_reference(fmt, act_type, weight_type, group_size, M):
    set_random_seed(0)
    K, N = 512, 384
    G = K if group_size == -1 else group_size
    zero_points = weight_type == scalar_types.uint4
    q = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    s = (0.02 * torch.rand((K // G, N), device=device) + 0.002).to(act_type)
    z = (
        torch.randint(0, 16, (K // G, N), device=device, dtype=torch.int32)
        if zero_points
        else None
    )
    layer = _make_layer(fmt, q, s, z)
    kernel = AiterW4A16LinearKernel(
        _config(K, N, group_size, weight_type, act_type, zero_points),
        w_q_param_name="w_q",
        w_s_param_name="w_s",
        w_zp_param_name="w_zp" if zero_points else None,
    )
    kernel.process_weights_after_loading(layer)
    assert layer.w_q.dtype == torch.uint8 and tuple(layer.w_q.shape) == (N, K // 2)
    assert tuple(layer.w_s.shape) == (N, K // G) and layer.w_s.dtype == act_type

    x = (0.5 * torch.randn((M, K), device=device)).to(act_type)
    bias = torch.randn((N,), device=device).to(act_type)
    out = kernel.apply_weights(layer, x, bias)

    zf = 8.0 if z is None else z.float().repeat_interleave(G, 0)
    w_ref = (q.float() - zf) * s.float().repeat_interleave(G, 0)  # [K, N]
    ref = x.float() @ w_ref + bias.float()
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_aiter_w4a16_symmetric_ignores_qzeros():
    """GPTQ registers qzeros even for uint4b8 (zero_points=False)."""
    set_random_seed(0)
    K, N, G, M = 256, 256, 128, 8
    q = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    s = (0.02 * torch.rand((K // G, N), device=device)).to(torch.float16)
    z = torch.zeros((K // G, N), device=device, dtype=torch.int32)
    layer = _make_layer("gptq", q, s, z)
    kernel = AiterW4A16LinearKernel(
        _config(K, N, G, scalar_types.uint4b8, torch.float16, False),
        w_q_param_name="w_q",
        w_s_param_name="w_s",
        w_zp_param_name="w_zp",
    )
    kernel.process_weights_after_loading(layer)
    x = torch.randn((M, K), device=device, dtype=torch.float16)
    out = kernel.apply_weights(layer, x)
    ref = x.float() @ ((q.float() - 8) * s.float().repeat_interleave(G, 0))
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "group_size,weight_type,ok",
    [
        (128, scalar_types.uint4b8, True),
        (64, scalar_types.uint4, True),
        (-1, scalar_types.uint4b8, True),
        (48, scalar_types.uint4b8, False),
        (128, scalar_types.uint8b128, False),
    ],
)
def test_aiter_w4a16_can_implement(monkeypatch, group_size, weight_type, ok):
    monkeypatch.setattr(
        rocm_aiter_ops, "is_linear_w4a16_enabled", classmethod(lambda cls: True)
    )
    c = _config(1536, 512, group_size, weight_type, torch.bfloat16, False)
    assert AiterW4A16LinearKernel.can_implement(c)[0] == ok


def test_aiter_w4a16_warmup_m_values():
    from vllm.model_executor.warmup.aiter_w4a16_warmup import _warmup_m_values

    m = _warmup_m_values(300)
    assert m[0] == 1 and m[-1] == 300
    assert {2, 3, 4, 7, 8, 15, 16, 31, 32, 255, 256, 299} <= set(m)
    assert all(1 <= v <= 300 for v in m)


def test_aiter_w4a16_warmup_runs_each_shape_once():
    """Layers sharing weight shapes compile one set of M values; the kernel
    is found on quant_method (GPTQ/AWQ) and on scheme (compressed-tensors)."""
    from vllm.model_executor.warmup.aiter_w4a16_warmup import (
        _warmup_m_values,
        aiter_w4a16_warmup,
    )

    K, G = 256, 128
    calls = []

    class CountingKernel(AiterW4A16LinearKernel):
        def apply_weights(self, layer, x, bias=None):
            calls.append(x.shape[0])
            return super().apply_weights(layer, x, bias)

    model = torch.nn.Module()
    for i, (N, owner) in enumerate(
        [(128, "quant_method"), (128, "quant_method"), (256, "scheme")]
    ):
        q = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
        s = (0.02 * torch.rand((K // G, N), device=device)).to(torch.bfloat16)
        layer = _make_layer("gptq", q, s, None)
        kernel = CountingKernel(
            _config(K, N, G, scalar_types.uint4b8, torch.bfloat16, False),
            w_q_param_name="w_q",
            w_s_param_name="w_s",
        )
        kernel.process_weights_after_loading(layer)
        setattr(layer, owner, type("Owner", (), {"kernel": kernel})())
        model.add_module(f"layer{i}", layer)

    aiter_w4a16_warmup(model, max_tokens=64)
    assert calls == _warmup_m_values(64) * 2


def test_aiter_w4a16_rejects_batch_invariant(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(
        rocm_aiter_ops, "is_linear_w4a16_enabled", classmethod(lambda cls: True)
    )
    # setattr before setenv, so monkeypatch restores the unset value.
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    c = _config(1536, 512, 128, scalar_types.uint4b8, torch.bfloat16, False)
    ok, reason = AiterW4A16LinearKernel.can_implement(c)
    assert not ok
    assert reason is not None and "batch invariant" in reason
