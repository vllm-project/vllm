# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import pytest
import torch

from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# Qwen3.6 TP=1 attention geometry.
HEAD_DIM = 256
ROTARY_DIM = 64
RMS_NORM_EPS = 1e-6
MAX_POSITION_EMBEDDINGS = 262144
ROPE_THETA = 10000000.0
DTYPE = torch.bfloat16
SEED = 13
MROPE_SECTION = (11, 11, 10)
ROPE_CASES = [
    pytest.param(24, 4, None, id="rope"),
    pytest.param(16, 2, MROPE_SECTION, id="interleaved-mrope"),
]

TensorTriplet = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Runner = Callable[[], TensorTriplet]

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_qk_rmsnorm_rope_gate Triton kernel requires CUDA/ROCm",
)


def _reference(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    mrope_section: tuple[int, int, int] | None,
) -> TensorTriplet:
    num_tokens = q_gate.shape[0]
    q_gate = q_gate.view(num_tokens, num_q_heads, 2 * HEAD_DIM)
    q = q_gate[..., :HEAD_DIM]
    gate = q_gate[..., HEAD_DIM:].reshape(num_tokens, num_q_heads * HEAD_DIM)
    k = k.view(num_tokens, num_kv_heads, HEAD_DIM)

    def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        variance = x_float.square().mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + RMS_NORM_EPS)
        return (normalized * (weight.float() + 1.0)).to(x.dtype)

    q = rms_norm(q, q_weight)
    k = rms_norm(k, k_weight)

    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)
    if mrope_section is not None:
        cos = apply_interleaved_rope(cos, list(mrope_section))
        sin = apply_interleaved_rope(sin, list(mrope_section))
    cos = cos.float()[:, None, :]
    sin = sin.float()[:, None, :]

    def apply_rope(x: torch.Tensor) -> torch.Tensor:
        rotary, passthrough = x[..., :ROTARY_DIM], x[..., ROTARY_DIM:]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat(
            [
                first.float() * cos - second.float() * sin,
                second.float() * cos + first.float() * sin,
            ],
            dim=-1,
        ).to(x.dtype)
        return torch.cat([rotated, passthrough], dim=-1)

    return (
        apply_rope(q).reshape(num_tokens, num_q_heads * HEAD_DIM),
        apply_rope(k).reshape(num_tokens, num_kv_heads * HEAD_DIM),
        gate,
    )


def _make_runners(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    mrope_section: tuple[int, int, int] | None,
    device: torch.device,
) -> tuple[Runner, Runner]:
    q_gate = torch.randn(
        num_tokens, num_q_heads * 2 * HEAD_DIM, dtype=DTYPE, device=device
    )
    k = torch.randn(num_tokens, num_kv_heads * HEAD_DIM, dtype=DTYPE, device=device)
    q_weight = torch.empty(HEAD_DIM, dtype=DTYPE, device=device).normal_(std=0.1)
    k_weight = torch.empty(HEAD_DIM, dtype=DTYPE, device=device).normal_(std=0.1)

    if mrope_section is None:
        rope = RotaryEmbedding(
            HEAD_DIM,
            ROTARY_DIM,
            MAX_POSITION_EMBEDDINGS,
            ROPE_THETA,
            True,
            DTYPE,
        ).to(device)
        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
    else:
        rope = MRotaryEmbedding(
            HEAD_DIM,
            ROTARY_DIM,
            MAX_POSITION_EMBEDDINGS,
            ROPE_THETA,
            True,
            DTYPE,
            mrope_section=list(mrope_section),
            mrope_interleaved=True,
        ).to(device)
        storage = torch.arange(
            3 * num_tokens * 2, dtype=torch.long, device=device
        ).view(3, num_tokens, 2)
        positions = storage[..., 0]
        assert not positions.is_contiguous()
        assert positions.stride(-1) == 2
        assert torch.unique(positions[:, 0]).numel() == 3

    def reference() -> TensorTriplet:
        return _reference(
            q_gate,
            k,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            positions,
            num_q_heads,
            num_kv_heads,
            mrope_section,
        )

    def fused() -> TensorTriplet:
        return fused_qk_rmsnorm_rope_gate(
            q_gate,
            k,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            positions,
            RMS_NORM_EPS,
            num_q_heads,
            num_kv_heads,
            HEAD_DIM,
            ROTARY_DIM,
            mrope_section=mrope_section,
        )

    return reference, fused


def _assert_matches(actual: TensorTriplet, expected: TensorTriplet) -> None:
    for actual_tensor, expected_tensor in zip(actual[:2], expected[:2], strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=2e-3, rtol=5e-3)
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)


def _assert_bitwise(actual: TensorTriplet, expected: TensorTriplet) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)


def _assert_runner_matches(reference: Runner, actual: Runner) -> None:
    expected_outputs = reference()
    actual_outputs = actual()
    _assert_matches(actual_outputs, expected_outputs)


@requires_cuda
@pytest.mark.parametrize("num_q_heads,num_kv_heads,mrope_section", ROPE_CASES)
@pytest.mark.parametrize("num_tokens", [1, 4, 37])
@torch.inference_mode()
def test_fused_qk_norm_rope_gate_matches_reference(
    default_vllm_config,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    mrope_section: tuple[int, int, int] | None,
) -> None:
    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.set_default_device(device)
    set_random_seed(SEED)
    reference, fused = _make_runners(
        num_tokens, num_q_heads, num_kv_heads, mrope_section, device
    )
    _assert_runner_matches(reference, fused)


@requires_cuda
@torch.inference_mode()
def test_fused_qk_norm_interleaved_mrope_gate_torch_compile(
    default_vllm_config,
) -> None:
    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.set_default_device(device)
    set_random_seed(SEED)
    reference, fused = _make_runners(37, 16, 2, MROPE_SECTION, device)
    compiled = torch.compile(fused, backend="inductor", fullgraph=True, dynamic=False)
    _assert_runner_matches(reference, compiled)


@requires_cuda
@torch.inference_mode()
def test_fused_qk_norm_interleaved_mrope_gate_cudagraph_replay(
    default_vllm_config,
) -> None:
    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.set_default_device(device)
    set_random_seed(SEED)
    _, fused = _make_runners(37, 16, 2, MROPE_SECTION, device)

    eager = fused()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = fused()

    replays = []
    for _ in range(2):
        graph.replay()
        torch.accelerator.synchronize()
        replays.append(tuple(output.clone() for output in graph_outputs))

    _assert_bitwise(replays[0], eager)
    _assert_bitwise(replays[1], eager)
    _assert_bitwise(replays[0], replays[1])
