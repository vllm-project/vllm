# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU experts stage the UVA-offloaded tensors of their own XpuFusedMoe
(--cpu-offload-stage-min-tokens): weights and per-group scales, into a shallow
copy used for the step, leaving the built kernel untouched."""

import types

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts import xpu_moe
from vllm.model_executor.offloader import base
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_uva_available


class _FakeOffloader:
    def __init__(self, min_tokens: int):
        self.min_tokens = min_tokens
        self.calls: list[str] = []

    def maybe_stage(self, weight, num_tokens, slot):
        self.calls.append(slot)
        if num_tokens < self.min_tokens or not getattr(
            weight, "_vllm_is_uva_offloaded", False
        ):
            return weight
        return weight.clone()


def _impl(offloaded: set[str]):
    impl = types.SimpleNamespace()
    for name in xpu_moe._STAGED_FIELDS:
        t = torch.randn(2, 3)
        if name in offloaded:
            t._vllm_is_uva_offloaded = True
        setattr(impl, name, t)
    return impl


def test_experts_do_not_take_staged_weight_args():
    assert xpu_moe.XPUExperts.reads_weight_args is False


@pytest.mark.parametrize("tokens", [16, 4096])
def test_staged_impl(monkeypatch, tokens):
    offloader = _FakeOffloader(min_tokens=512)
    monkeypatch.setattr(base, "get_offloader", lambda: offloader)
    impl = _impl({"w13", "w2", "gemm1_wei_scales"})
    out = xpu_moe.staged_impl(impl, tokens, "experts")
    assert offloader.calls == [f"experts.{n}" for n in xpu_moe._STAGED_FIELDS]
    if tokens < offloader.min_tokens:
        assert out is impl
        return
    assert out is not impl
    for name in ("w13", "w2", "gemm1_wei_scales"):
        assert getattr(out, name) is not getattr(impl, name)
        torch.testing.assert_close(getattr(out, name), getattr(impl, name))
    # not offloaded: read in place
    assert out.gemm2_wei_scales is impl.gemm2_wei_scales


requires_xpu_uva = pytest.mark.skipif(
    not current_platform.is_xpu() or not is_uva_available(),
    reason="XPU with UVA",
)


@requires_xpu_uva
@torch.inference_mode()
def test_xpu_int4_moe_reads_staged_copy():
    """One int4 layer at the Qwen3.6-35B-A3B shape with its weights on host
    views: at the threshold the staged copy computes what the VRAM layer
    computes and the buffers hold this step's weights; one token below it
    reads in place; under graph capture nothing is staged."""
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

    from vllm.model_executor.offloader import get_offloader, set_offloader
    from vllm.model_executor.offloader.uva import UVAOffloader
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    dev = torch.device("xpu")
    e, k, inter, group, topk, m = 32, 2048, 512, 128, 8, 64
    gen = torch.Generator().manual_seed(11)
    w13 = torch.randint(
        0, 256, (e, 2 * inter, k // 2), dtype=torch.uint8, generator=gen
    )
    w2 = torch.randint(0, 256, (e, k, inter // 2), dtype=torch.uint8, generator=gen)
    s13 = (torch.rand(e, 2 * inter, k // group, generator=gen) * 4e-3 + 2e-3).to(
        torch.bfloat16
    )
    s2 = (torch.rand(e, k, inter // group, generator=gen) * 4e-3 + 2e-3).to(
        torch.bfloat16
    )

    def build(host: bool):
        def param(t):
            if host:
                p = torch.nn.Parameter(
                    get_accelerator_view_from_cpu_tensor(t.clone().pin_memory()),
                    requires_grad=False,
                )
                p._vllm_is_uva_offloaded = True
                return p
            return torch.nn.Parameter(t.to(dev), requires_grad=False)

        return XpuFusedMoe(
            w13=param(w13),
            w13_scales=s13.to(dev),
            w13_bias=None,
            w2=param(w2),
            w2_scales=s2.to(dev),
            w2_bias=None,
            n_experts_per_token=topk,
            activation="silu",
            num_experts=e,
        )

    def run(impl, tokens):
        g = torch.Generator().manual_seed(tokens)
        ids = torch.stack(
            [torch.randperm(e, generator=g)[:topk] for _ in range(tokens)]
        )
        w = torch.softmax(torch.randn(tokens, topk, generator=g), dim=-1)
        x = (torch.randn(tokens, k, generator=g) * 0.5).to(torch.bfloat16).to(dev)
        out = torch.empty_like(x)
        impl.apply(out, x, w.to(dev), ids.to(dev, torch.int32))
        torch.xpu.synchronize()
        return out

    ref, host = build(False), build(True)
    previous = get_offloader()
    offloader = UVAOffloader(cpu_offload_max_bytes=0, stage_min_tokens=m)
    set_offloader(offloader)
    try:
        staged = xpu_moe.staged_impl(host, m, "experts")
        assert staged is not host
        assert staged.w13.data_ptr() != host.w13.data_ptr()
        torch.testing.assert_close(run(staged, m), run(ref, m), rtol=0, atol=0)
        assert torch.equal(staged.w13.cpu(), host.w13.cpu())
        assert xpu_moe.staged_impl(host, m - 1, "experts") is host

        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            assert xpu_moe.staged_impl(host, m, "experts") is host
    finally:
        set_offloader(previous)


def test_staged_impl_skips_missing_fields(monkeypatch):
    offloader = _FakeOffloader(min_tokens=1)
    monkeypatch.setattr(base, "get_offloader", lambda: offloader)
    impl = _impl({"w13"})
    impl.gemm1_wei_scales = None
    impl.gemm2_wei_scales = None
    out = xpu_moe.staged_impl(impl, 8, "cold")
    assert offloader.calls == ["cold.w13", "cold.w2"]
    assert out.w13 is not impl.w13 and out.w2 is impl.w2
