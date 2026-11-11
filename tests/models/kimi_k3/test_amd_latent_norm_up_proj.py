# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The folded RMSNorm + sharded up-projection must equal the unfused chain.

``torch.ops.vllm.k3_latent_norm_up_proj`` replaces ``rms_norm`` + ``addmm_`` in
the ROCm latent-MoE tail, so it is checked against exactly that chain at the
TP8 shard shape, across each kernel-config boundary and the fallback.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.models.kimi_k3.amd.ops import latent_norm_up_proj as op_mod
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="The folded latent tail is only wired up on ROCm",
)

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
TP = 8
SHARD = HIDDEN_SIZE // TP
EPS = 1e-5
DTYPE = torch.bfloat16
# One bf16 ulp at |out| ~ 4: the GEMM input is rounded before the row scale.
ATOL, RTOL = 4e-2, 2e-2


def _inputs(num_tokens: int, device: torch.device, seed: int = 0):
    gen = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(
        num_tokens, LATENT_SIZE, device=device, dtype=DTYPE, generator=gen
    )
    norm_weight = (1 + 0.1 * torch.randn(LATENT_SIZE, device=device, generator=gen)).to(
        DTYPE
    )
    up_weight = (
        torch.randn(SHARD, LATENT_SIZE, device=device, dtype=DTYPE, generator=gen)
        / LATENT_SIZE**0.5
    )
    shared = torch.randn(
        num_tokens, HIDDEN_SIZE, device=device, dtype=DTYPE, generator=gen
    )
    return latent, norm_weight, up_weight, shared


def _reference(latent, norm_weight, up_weight, shared, shard_start):
    """The unfused tail: rms_norm, then addmm into this rank's shard."""
    expected = shared.clone()
    normed = F.rms_norm(latent.float(), (LATENT_SIZE,), norm_weight.float(), EPS)
    expected.narrow(-1, shard_start, SHARD).addmm_(normed.to(DTYPE), up_weight.t())
    return expected


def _check(actual, expected, shard_start):
    end = shard_start + SHARD
    torch.testing.assert_close(
        actual[:, shard_start:end], expected[:, shard_start:end], atol=ATOL, rtol=RTOL
    )
    # Columns outside the shard belong to other ranks and must be untouched.
    torch.testing.assert_close(actual[:, :shard_start], expected[:, :shard_start])
    torch.testing.assert_close(actual[:, end:], expected[:, end:])


@pytest.mark.parametrize(
    "num_tokens", [1, 4, 16, 17, 48, 64, 65, 128, 129, 192, 193, 256]
)
@pytest.mark.parametrize("rank", [0, 3, 7])
def test_matches_unfused_chain(num_tokens: int, rank: int) -> None:
    device = torch.device("cuda:0")
    latent, norm_weight, up_weight, shared = _inputs(num_tokens, device, rank)
    shard_start = rank * SHARD
    expected = _reference(latent, norm_weight, up_weight, shared, shard_start)

    actual = shared.clone()
    torch.ops.vllm.k3_latent_norm_up_proj(
        latent, norm_weight, EPS, up_weight, actual, shard_start
    )

    _check(actual, expected, shard_start)


@pytest.mark.parametrize(
    "num_tokens,fused", [(1, True), (192, True), (193, False), (512, False)]
)
def test_token_gate(num_tokens: int, fused: bool) -> None:
    device = torch.device("cuda:0")
    latent, norm_weight, up_weight, shared = _inputs(num_tokens, device)
    out = shared.narrow(-1, 0, SHARD)

    assert (
        op_mod.latent_norm_up_proj_supported(latent, norm_weight, up_weight, out)
        is fused
    )


def test_rejects_shapes_the_kernel_does_not_tile() -> None:
    device = torch.device("cuda:0")
    latent, norm_weight, up_weight, shared = _inputs(4, device)
    out = shared.narrow(-1, 0, SHARD)
    odd_k = LATENT_SIZE - 128

    assert not op_mod.latent_norm_up_proj_supported(
        latent[:, :odd_k], norm_weight[:odd_k], up_weight[:, :odd_k], out
    )
    assert not op_mod.latent_norm_up_proj_supported(
        latent.float(), norm_weight, up_weight, out
    )


def test_cuda_graph_replay_tracks_new_inputs() -> None:
    """Captured once, replayed with new latents and a fresh shared partial."""
    device = torch.device("cuda:0")
    shard_start = 2 * SHARD
    latent, norm_weight, up_weight, shared = _inputs(4, device)
    static_latent, static_shared = latent.clone(), shared.clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops.vllm.k3_latent_norm_up_proj(
            static_latent, norm_weight, EPS, up_weight, static_shared, shard_start
        )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.vllm.k3_latent_norm_up_proj(
            static_latent, norm_weight, EPS, up_weight, static_shared, shard_start
        )

    for seed in (1, 2):
        new_latent, _, _, new_shared = _inputs(4, device, seed)
        static_latent.copy_(new_latent)
        static_shared.copy_(new_shared)
        graph.replay()
        torch.accelerator.synchronize()
        expected = _reference(
            new_latent, norm_weight, up_weight, new_shared, shard_start
        )
        _check(static_shared, expected, shard_start)


def test_token_gate_is_taken_at_run_time_under_compile() -> None:
    """The caller is traced with a symbolic batch size; one compiled graph must
    take the kernel at 4 tokens and the fallback at 256."""
    device = torch.device("cuda:0")
    shard_start = SHARD

    def tail(latent, norm_weight, up_weight, shared):
        torch.ops.vllm.k3_latent_norm_up_proj(
            latent, norm_weight, EPS, up_weight, shared, shard_start
        )
        return shared * 1.0

    compiled = torch.compile(tail, dynamic=True, fullgraph=True)
    for num_tokens in (4, 256, 64):
        latent, norm_weight, up_weight, shared = _inputs(num_tokens, device)
        expected = _reference(latent, norm_weight, up_weight, shared, shard_start)
        actual = compiled(latent, norm_weight, up_weight, shared.clone())
        _check(actual, expected, shard_start)
