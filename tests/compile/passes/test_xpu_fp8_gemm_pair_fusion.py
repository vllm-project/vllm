# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import fx

from vllm.compilation.passes.fusion.xpu_fp8_gemm_pair_fusion import (
    XpuFp8GemmPairFusionPass,
)
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")


@pytest.fixture
def pair_pass():
    import vllm._xpu_ops  # noqa: F401

    p = XpuFp8GemmPairFusionPass(VllmConfig())
    if not p.enabled:
        pytest.skip("vllm-xpu-kernels without fp8_gemm_w8a16_pair")
    return p


def _meta(*shape, dtype=torch.float16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _graph(n_gemms=2, bias=False, block_scale=False):
    g = fx.Graph()
    a = g.placeholder("a")
    a.meta["val"] = _meta(1, 2048)
    sizes = (6144, 32, 128)[:n_gemms]
    params = []
    for i, n in enumerate(sizes):
        w, s = g.placeholder(f"w{i}"), g.placeholder(f"s{i}")
        w.meta["val"] = _meta(n, 2048, dtype=torch.float8_e4m3fn).t()
        s.meta["val"] = (
            _meta(16, 16, dtype=torch.float32)
            if block_scale
            else _meta(1, dtype=torch.float32)
        )
        params.append((w, s))
    outs = []
    for n, (w, s) in zip(sizes, params):
        args = (a, w, s, a) if bias else (a, w, s, None)
        mm = g.call_function(torch.ops._xpu_C.fp8_gemm_w8a16.default, args)
        mm.meta["val"] = _meta(1, n)
        outs.append(mm)
    g.output(tuple(outs))
    return g


def _count(g, target):
    return sum(1 for n in g.nodes if n.target is target)


def test_pairs_two_gemms_sharing_input(pair_pass):
    g = _graph()
    pair_pass(g)
    assert pair_pass.matched_count == 1
    assert _count(g, torch.ops._xpu_C.fp8_gemm_w8a16_pair.default) == 1
    assert _count(g, torch.ops._xpu_C.fp8_gemm_w8a16.default) == 0


@pytest.mark.parametrize("kw", [{"n_gemms": 3}, {"bias": True}, {"block_scale": True}])
def test_other_cases_unchanged(pair_pass, kw):
    g = _graph(**kw)
    pair_pass(g)
    assert pair_pass.matched_count == 0


def test_decode_ranges_only(pair_pass):
    assert pair_pass.is_applicable_for_range(Range(start=1, end=8))
    assert not pair_pass.is_applicable_for_range(Range(start=9, end=4096))
