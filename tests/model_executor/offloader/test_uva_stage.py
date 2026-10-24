# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Staging of UVA-offloaded fused-MoE expert weights for large steps."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import OffloadConfig, UVAOffloadConfig, VllmConfig
from vllm.model_executor.offloader.uva import UVAExpertStager, UVAOffloader
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_uva_available

pytestmark = pytest.mark.skip_global_cleanup


def _offloaded(t: torch.Tensor) -> nn.Parameter:
    """A parameter carrying the UVA offloader's marker."""
    p = nn.Parameter(t, requires_grad=False)
    p._vllm_is_uva_offloaded = True
    return p


@pytest.fixture
def cpu_stream(monkeypatch):
    """The copy path asks the current stream whether it is capturing; CPU-only
    runs have no device stream to ask. Capture itself is tested on CUDA."""
    stream = SimpleNamespace(is_capturing=lambda: False)
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda: stream)


@pytest.mark.parametrize(
    ("value", "max_capture_size", "expected"),
    [
        ("AUTO", 512, 513),  # every step that runs without CUDA graphs
        ("auto", 0, 512),  # no CUDA graphs: never below 512
        (0, 512, 0),  # disabled (the default)
        (2048, 512, 2048),
    ],
)
def test_resolve_stage_min_tokens(value, max_capture_size, expected):
    config = UVAOffloadConfig(cpu_offload_stage_min_tokens=value)
    assert config.resolve_stage_min_tokens(max_capture_size) == expected


def test_vllm_config_resolves_auto_and_hash_ignores_it():
    def offload(value):
        return OffloadConfig(
            uva=UVAOffloadConfig(cpu_offload_gb=1, cpu_offload_stage_min_tokens=value)
        )

    # No model, so no CUDA graphs: "auto" resolves to the floor.
    config = VllmConfig(offload_config=offload("auto"))
    assert config.offload_config.uva.cpu_offload_stage_min_tokens == 512
    assert offload(0).compute_hash() == offload(4096).compute_hash()


def test_passthrough():
    stager = UVAExpertStager(min_tokens=1024)
    w = _offloaded(torch.randn(4, 8, 16))
    assert stager.maybe_stage(w, 1023, "w1") is w  # below threshold

    plain = nn.Parameter(torch.randn(4, 8, 16), requires_grad=False)
    assert stager.maybe_stage(plain, 4096, "w1") is plain  # not offloaded

    assert UVAExpertStager(min_tokens=0).maybe_stage(w, 1 << 20, "w1") is w


def test_stage_copies_and_reuses_buffer(cpu_stream):
    stager = UVAExpertStager(min_tokens=1024)
    layer0 = _offloaded(torch.randn(4, 8, 16))
    layer1 = _offloaded(torch.randn(4, 8, 16))

    staged0 = stager.maybe_stage(layer0, 1024, "w1")
    assert staged0.data_ptr() != layer0.data_ptr()
    assert torch.equal(staged0, layer0)

    # Every layer of the same shape reuses one buffer...
    staged1 = stager.maybe_stage(layer1, 4096, "w1")
    assert staged1.data_ptr() == staged0.data_ptr()
    assert torch.equal(staged1, layer1)

    # ...but weights read by the same kernel call never share one.
    other = stager.maybe_stage(_offloaded(torch.randn(4, 8, 16)), 4096, "w2")
    assert other.data_ptr() != staged1.data_ptr()
    assert torch.equal(staged1, layer1)

    # Strides are kept.
    t = _offloaded(torch.randn(4, 16, 8).transpose(1, 2))
    staged_t = stager.maybe_stage(t, 4096, "w2")
    assert staged_t.stride() == t.stride() and torch.equal(staged_t, t)


requires_uva = pytest.mark.skipif(
    not current_platform.is_cuda() or not is_uva_available(),
    reason="staging is CUDA only; needs UVA",
)


def _uva_view(t: torch.Tensor) -> nn.Parameter:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    return _offloaded(get_accelerator_view_from_cpu_tensor(t.cpu().pin_memory()))


@requires_uva
def test_no_stage_inside_cuda_graph():
    stager = UVAExpertStager(min_tokens=1)
    w = _uva_view(torch.randn(8, 64, 128))
    x = torch.zeros(4, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = stager.maybe_stage(w, 4096, "w1")
        x.add_(1)
    assert out is w
    assert not stager._buffers


@requires_uva
@torch.inference_mode()
def test_modular_kernel_reads_staged_copy(monkeypatch, workspace_init):
    """At the threshold the experts kernel gets the staging buffers, holding
    this step's weights, and computes what the weights in VRAM compute; one
    token below it gets the UVA views."""
    from tests.kernels.moe.utils import make_dummy_moe_config, modular_triton_fused_moe
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.offloader import get_offloader, set_offloader

    E, N, K, topk, M = 8, 256, 128, 2, 256
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * N, K, device="cuda", dtype=torch.bfloat16) / 10
    w2 = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) / 10
    w1_uva, w2_uva = _uva_view(w1), _uva_view(w2)
    kernel = modular_triton_fused_moe(
        make_dummy_moe_config(), FUSED_MOE_UNQUANTIZED_CONFIG
    )

    seen: list[tuple[int, int]] = []
    experts = kernel.fused_experts
    original_apply = experts.apply

    def spy(*args, **kwargs):
        seen.append((kwargs["w1"].data_ptr(), kwargs["w2"].data_ptr()))
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(experts, "apply", spy)

    def run(a, w1, w2):
        torch.manual_seed(1)
        score = torch.randn(a.shape[0], E, device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids, _ = fused_topk(a, score.float(), topk, False)
        return kernel.apply(
            a,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
            global_num_experts=E,
        )

    previous = get_offloader()
    offloader = UVAOffloader(cpu_offload_max_bytes=0, stage_min_tokens=M)
    set_offloader(offloader)
    try:
        with set_current_vllm_config(VllmConfig()):
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 10
            ref = run(a, w1, w2)
            seen.clear()
            torch.testing.assert_close(run(a, w1_uva, w2_uva), ref)
            buffers = offloader.expert_stager._buffers
            assert len(buffers) == 2
            buf1, buf2 = (
                next(b for b in buffers.values() if b.shape == w.shape)
                for w in (w1, w2)
            )
            assert seen == [(buf1.data_ptr(), buf2.data_ptr())]
            assert torch.equal(buf1, w1) and torch.equal(buf2, w2)

            a = a[: M - 1]
            ref = run(a, w1, w2)
            seen.clear()
            torch.testing.assert_close(run(a, w1_uva, w2_uva), ref)
            assert seen == [(w1_uva.data_ptr(), w2_uva.data_ptr())]
            assert len(buffers) == 2
    finally:
        set_offloader(previous)
