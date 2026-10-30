# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova Loop mHC numerical and CUDA Graph regression tests."""

import pytest
import torch


@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("hidden_size", [257, 2048])
@pytest.mark.parametrize("fixed_identity", [False, True])
def test_loop_mhc_cuda_graph_replays_new_residuals(
    streams, expanded, hidden_size, fixed_identity
):
    from vllm.model_executor.layers.sensenova_loop_mhc import MHCRouter

    torch.manual_seed(0)
    router = MHCRouter(
        hidden_size,
        {
            "streams": streams,
            "iterations": 20,
            "fixed_identity": fixed_identity,
            "pre_scale": 0.7,
        },
    ).cuda()
    with torch.no_grad():
        router.weight.normal_(std=0.01)
        router.bias.normal_(std=0.01)
    storage = torch.randn(3, streams, hidden_size, device="cuda", dtype=torch.bfloat16)
    residual = storage[:, :1].expand_as(storage) if expanded else storage

    def route_and_merge():
        branch, post, mix = router(residual)
        return router.merge(residual, branch, post, mix)

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            route_and_merge()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = route_and_merge()
    for _ in range(2):
        storage.normal_()
        flat = residual.flatten(-2).float()
        normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        logits = torch.nn.functional.linear(normalized, router.weight, router.bias)
        pre = logits[:, :streams].sigmoid().to(residual.dtype)
        reference_branch = (pre.unsqueeze(-2) @ residual).squeeze(-2) * 0.7
        branch, _, _ = router(residual)
        torch.testing.assert_close(branch, reference_branch, rtol=0, atol=0)
        expected = route_and_merge()
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("tokens", [0, 1, 3, 128, 2048])
@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_loop_mhc_log_sinkhorn_matches_reference(streams, tokens, scale):
    from vllm.model_executor.kernels.mhc.sensenova_loop import mhc_log_sinkhorn_triton

    torch.manual_seed(0)
    logits = torch.randn(tokens, 2 * streams + streams**2, device="cuda") * scale
    mix = logits[:, 2 * streams :].unflatten(-1, (streams, streams))
    expected = mix
    for _ in range(20):
        expected = expected - expected.logsumexp(-1, keepdim=True)
        expected = expected - expected.logsumexp(-2, keepdim=True)
    expected = expected.exp()
    actual = mhc_log_sinkhorn_triton(mix, 20)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("tokens", [0, 1, 8, 32, 128])
def test_loop_mhc_projection_preserves_fp32_routing(streams, tokens):
    from vllm.model_executor.kernels.mhc.sensenova_loop import (
        mhc_project_triton,
        mhc_rmsnorm_triton,
    )

    torch.manual_seed(1)
    hidden = 4096
    storage = torch.randn(
        tokens, streams, hidden + 1, device="cuda", dtype=torch.bfloat16
    )
    x = storage[..., 1:]
    flat = x.flatten(-2).float()
    expected = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    actual = mhc_rmsnorm_triton(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    weight = (
        torch.randn(2 * streams + streams**2, streams * hidden, device="cuda") * 0.01
    )
    bias = torch.randn(weight.shape[0], device="cuda") * 0.01
    # FP64 avoids using cuBLAS FP32 accumulation error as the oracle.
    reference = torch.nn.functional.linear(
        expected.double(), weight.double(), bias.double()
    ).float()
    projected = mhc_project_triton(actual, weight, bias)
    torch.testing.assert_close(projected, reference, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("tokens", [0, 1, 8, 128])
@pytest.mark.parametrize("expanded", [False, True])
def test_loop_mhc_fused_merge_preserves_bf16_rounding(streams, tokens, expanded):
    from vllm.model_executor.kernels.mhc.sensenova_loop import mhc_merge_triton

    torch.manual_seed(2)
    storage = torch.randn(tokens, streams, 258, device="cuda", dtype=torch.bfloat16)
    residual = storage[..., 1:]
    if expanded:
        residual = residual[:, :1].expand_as(residual)
    branch = torch.randn(tokens, 257, device="cuda", dtype=torch.bfloat16)
    post = torch.randn(tokens, streams, device="cuda")
    mix = (
        torch.eye(streams, device="cuda")
        if expanded
        else torch.randn(tokens, streams, streams, device="cuda")
    )
    mixed = mix.to(residual.dtype) @ residual
    update = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(-2)
    expected = mixed + update
    actual = mhc_merge_triton(residual, branch, post, mix)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
