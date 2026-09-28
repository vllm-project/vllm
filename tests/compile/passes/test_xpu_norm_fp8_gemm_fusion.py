# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator
import os

import pytest
import torch
from torch import fx

from tests.compile.backend import TestBackend
from vllm.compilation.passes.fusion.xpu_norm_fp8_gemm_fusion import (
    XpuNormFp8GemmFusionPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.utils import Range
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")

# Only config.json is read (GDN head geometry, rms_norm_eps).
MODEL = os.environ.get("VLLM_TEST_QWEN36_MOE_MODEL", "Qwen/Qwen3.6-35B-A3B")
GEMM = torch.ops._xpu_C.fp8_gemm_w8a16.default


def _config():
    return VllmConfig(
        model_config=ModelConfig(
            model=MODEL, dtype=torch.float16, trust_remote_code=True
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["none"],
            pass_config=PassConfig(fuse_xpu_norm_fp8_gemm=True, eliminate_noops=True),
        ),
    )


def _available():
    import vllm._xpu_ops  # noqa: F401

    return hasattr(torch.ops._xpu_C, "gated_rmsnorm_fp8_gemm")


class GdnOutput(torch.nn.Module):
    """RMSNormGated + out_proj, as QwenGatedDeltaNetAttention part 3 (TP1)."""

    def __init__(self, heads=32, head_dim=128):
        super().__init__()
        self.norm = RMSNormGated(head_dim, eps=1e-6, norm_before_gate=True)
        with torch.no_grad():
            self.norm.weight.normal_(1.0, 0.1)
        w = (torch.randn(2048, heads * head_dim) * 0.05).to(torch.float8_e4m3fn)
        self.register_buffer("w_t", w.t(), persistent=False)
        self.register_buffer("scale", torch.tensor([0.02]), persistent=False)

    def forward(self, core_attn_out, z):
        z_shape = z.shape
        x = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        y = self.norm(x, z.reshape(-1, z.shape[-1]))
        y = y.reshape(z_shape).flatten(-2)
        return GEMM(y, self.w_t, self.scale, None)


def test_gated_norm_fused():
    if not _available():
        pytest.skip("vllm-xpu-kernels without the norm + fp8 GEMV ops")
    torch.set_default_device("xpu")
    torch.set_default_dtype(torch.float16)
    torch.manual_seed(0)
    vllm_config = _config()
    # Inference graph (as in vLLM): no autograd decompositions.
    with set_current_vllm_config(vllm_config), torch.inference_mode():
        model = GdnOutput()
        fusion = XpuNormFp8GemmFusionPass(vllm_config)
        noop, cleanup = NoOpEliminationPass(vllm_config), PostCleanupPass(vllm_config)
        x = torch.randn(1, 32, 128)
        z = torch.randn(1, 32, 128)
        ref = torch.compile(model, backend=TestBackend(noop, cleanup))(x, z)
        out = torch.compile(model, backend=TestBackend(noop, fusion, cleanup))(x, z)
        assert fusion.gated_count == 1
        torch.testing.assert_close(out, ref, atol=3e-3, rtol=2e-2)


def _resadd_graph(pair=False, extra_user=False):
    g = fx.Graph()
    val = torch.empty(1, 2048, dtype=torch.float16, device="meta")
    x, res, w = (g.placeholder(n) for n in ("x", "res", "w"))
    x.meta["val"], res.meta["val"] = val, val
    w.meta["val"] = torch.empty(2048, dtype=torch.float16, device="meta")
    wf = g.call_function(
        torch.ops.prims.convert_element_type.default, (w, torch.float32)
    )
    w1 = g.call_function(torch.ops.aten.add.Tensor, (wf, 1.0))
    norm = g.call_function(
        torch.ops.vllm_ir.fused_add_rms_norm.default, (x, res, w1, 1e-6)
    )
    h = g.call_function(operator.getitem, (norm, 0))
    new_res = g.call_function(operator.getitem, (norm, 1))
    b1, s1, b2, s2 = (g.placeholder(n) for n in ("b1", "s1", "b2", "s2"))
    if pair:
        mm = g.call_function(
            torch.ops._xpu_C.fp8_gemm_w8a16_pair.default, (h, b1, s1, b2, s2)
        )
        outs = [g.call_function(operator.getitem, (mm, i)) for i in (0, 1)]
    else:
        mm = g.call_function(GEMM, (h, b1, s1, None))
        mm.meta["val"] = val
        outs = [mm]
    outs.append(new_res)
    if extra_user:
        outs.append(h)
    g.output(tuple(outs))
    return g


@pytest.mark.parametrize("pair", [False, True])
def test_resadd_norm_fused(pair):
    if not _available():
        pytest.skip("vllm-xpu-kernels without the norm + fp8 GEMV ops")
    fusion = XpuNormFp8GemmFusionPass(VllmConfig())
    g = _resadd_graph(pair)
    fusion(g)
    assert fusion.resadd_count == 1
    target = (
        torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm_pair.default
        if pair
        else torch.ops._xpu_C.resadd_rmsnorm_fp8_gemm.default
    )
    assert any(n.target is target for n in g.nodes)
    assert not any(
        n.target is torch.ops.vllm_ir.fused_add_rms_norm.default for n in g.nodes
    )


def test_resadd_norm_with_other_user_unchanged():
    if not _available():
        pytest.skip("vllm-xpu-kernels without the norm + fp8 GEMV ops")
    fusion = XpuNormFp8GemmFusionPass(VllmConfig())
    g = _resadd_graph(extra_user=True)
    fusion(g)
    assert fusion.resadd_count == 0


def test_decode_ranges_only():
    if not _available():
        pytest.skip("vllm-xpu-kernels without the norm + fp8 GEMV ops")
    fusion = XpuNormFp8GemmFusionPass(VllmConfig())
    assert fusion.is_applicable_for_range(Range(start=1, end=8))
    assert not fusion.is_applicable_for_range(Range(start=9, end=4096))
