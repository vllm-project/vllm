# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3.1 DSA on ROCm: the fused q_a/kv_a RMSNorm and the fused output gate
(VLLM_K3_DSA_FUSED_NORM_GATE) against the separate AITER RMSNorms and the
torch sigmoid/mul they replace.

``python tests/models/kimi_k3/test_amd_dsa_fused_norm_gate.py`` also runs a
CUDA-graph microbenchmark of both paths at decode token counts.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm import ir
from vllm.model_executor.layers.mla import mla_output_gate
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3.1 DSA fused norm/gate path is ROCm only",
)

# Kimi-K3.1-Pro DSA layer, TP8: q_lora 1536, kv_lora 512 + rope 64 in the
# fused_qkv_a_proj output, 96 heads / 8 ranks x v_head_dim 128 for the gate.
Q_LORA = 1536
KV_LORA = 512
ROPE = 64
GATE_WIDTH = 96 // 8 * 128
EPS = 1e-5
# vllm/models/kimi_k3/amd/dsa.py _FUSED_QKV_NORM_MAX_TOKENS
MAX_TOKENS = 1024


def _aiter_rms_norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    from vllm.kernels.aiter_ops import _rms_norm_impl

    return _rms_norm_impl(x, w, EPS)


def _inputs(num_tokens: int, seed: int = 0):
    torch.manual_seed(seed)
    qkv_lora = torch.randn(
        num_tokens, Q_LORA + KV_LORA + ROPE, device="cuda", dtype=torch.bfloat16
    )
    q_c, kv_lora = qkv_lora.split([Q_LORA, KV_LORA + ROPE], dim=-1)
    kv_c, _ = kv_lora.split([KV_LORA, ROPE], dim=-1)
    q_w = (1 + 0.1 * torch.randn(Q_LORA, device="cuda")).to(torch.bfloat16)
    kv_w = (1 + 0.1 * torch.randn(KV_LORA, device="cuda")).to(torch.bfloat16)
    attn_out = torch.randn(num_tokens, GATE_WIDTH, device="cuda", dtype=torch.bfloat16)
    gate = 2 * torch.randn(num_tokens, GATE_WIDTH, device="cuda", dtype=torch.bfloat16)
    return q_c, kv_c, q_w, kv_w, attn_out, gate


def _ref_rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x.float(), (x.shape[-1],), w.float(), EPS).to(x.dtype)


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64, 192, 1000])
def test_fused_q_kv_rmsnorm_matches_separate_norms(num_tokens: int) -> None:
    q_c, kv_c, q_w, kv_w, _, _ = _inputs(num_tokens)
    assert q_c.stride(0) == kv_c.stride(0) == Q_LORA + KV_LORA + ROPE

    q_old = _aiter_rms_norm(q_c, q_w)
    kv_old = _aiter_rms_norm(kv_c, kv_w)
    q_new, kv_new = fused_q_kv_rmsnorm(q_c, kv_c, q_w, kv_w, EPS)

    assert q_new.shape == q_old.shape and kv_new.shape == kv_old.shape
    assert q_new.is_contiguous() and kv_new.is_contiguous()
    # Both are within one bf16 ulp of the fp32 reference.
    torch.testing.assert_close(q_new, _ref_rms(q_c, q_w), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(kv_new, _ref_rms(kv_c, kv_w), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(q_new, q_old, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(kv_new, kv_old, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [4, 192, MAX_TOKENS, MAX_TOKENS + 1, 4096])
def test_bounded_q_kv_rmsnorm_op(num_tokens: int) -> None:
    q_c, kv_c, q_w, kv_w, _, _ = _inputs(num_tokens)
    with ir.ops.rms_norm.set_priority(["aiter", "native"]):
        q, kv = torch.ops.vllm.mla_bounded_q_kv_rmsnorm(
            q_c, kv_c, q_w, kv_w, EPS, MAX_TOKENS
        )
    if num_tokens <= MAX_TOKENS:
        q_exp, kv_exp = fused_q_kv_rmsnorm(q_c, kv_c, q_w, kv_w, EPS)
    else:
        q_exp, kv_exp = _aiter_rms_norm(q_c, q_w), _aiter_rms_norm(kv_c, kv_w)
    torch.testing.assert_close(q, q_exp, atol=0, rtol=0)
    torch.testing.assert_close(kv, kv_exp, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64, 192, 1000])
def test_fused_output_gate_matches_torch(num_tokens: int) -> None:
    _, _, _, _, attn_out, gate = _inputs(num_tokens)
    old = mla_output_gate(attn_out, gate, fused=False)
    new = mla_output_gate(attn_out, gate, fused=True)
    # Same roundings as the eager expression: bitwise equal.
    torch.testing.assert_close(new, old, atol=0, rtol=0)
    # Row-strided inputs (e.g. split views of a merged projection).
    wide = torch.randn(num_tokens, 2 * GATE_WIDTH, device="cuda").bfloat16()
    strided = wide[:, GATE_WIDTH:]
    torch.testing.assert_close(
        mla_output_gate(attn_out, strided, fused=True),
        mla_output_gate(attn_out, strided, fused=False),
        atol=0,
        rtol=0,
    )


def _new_path(q_c, kv_c, q_w, kv_w, attn_out, gate):
    q, kv = torch.ops.vllm.mla_bounded_q_kv_rmsnorm(
        q_c, kv_c, q_w, kv_w, EPS, MAX_TOKENS
    )
    return q, kv, mla_output_gate(attn_out, gate, fused=True)


def test_fused_norm_gate_torch_compile_and_cuda_graph() -> None:
    # fullgraph: the vLLM piecewise backend does not tolerate graph breaks.
    # Inductor may specialize the Triton launches differently (symbolic
    # strides), so the norm sums can round one bf16 ulp apart from eager.
    torch._dynamo.reset()
    compiled = torch.compile(_new_path, fullgraph=True, dynamic=True)
    for num_tokens in (4, 192, 2048):
        args = _inputs(num_tokens, seed=num_tokens)
        expected = _new_path(*args)
        out = compiled(*args)
        for a, b in zip(out, expected):
            torch.testing.assert_close(a, b, atol=1e-2, rtol=1e-2)

        static = _inputs(num_tokens, seed=1)
        compiled(*static)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = compiled(*static)
        for dst, src in zip(static, _inputs(num_tokens, seed=2)):
            dst.copy_(src)
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(graph_out, _new_path(*static)):
            torch.testing.assert_close(a, b, atol=1e-2, rtol=1e-2)


def test_fused_norm_gate_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm import envs

    monkeypatch.setenv("VLLM_K3_DSA_FUSED_NORM_GATE", "0")
    assert envs.VLLM_K3_DSA_FUSED_NORM_GATE is False
    monkeypatch.setenv("VLLM_K3_DSA_FUSED_NORM_GATE", "1")
    assert envs.VLLM_K3_DSA_FUSED_NORM_GATE is True


def _old_norms(q_c, kv_c, q_w, kv_w):
    return _aiter_rms_norm(q_c, q_w), _aiter_rms_norm(kv_c, kv_w)


def _bench(fn, *args, iters: int = 200, reps: int = 5) -> float:
    """Median us per call, from CUDA-graph replays of ``iters`` calls."""
    fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn(*args)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters)
    return sorted(times)[len(times) // 2]


def main() -> None:
    print("tokens | norms old | norms new | gate old | gate new | total old | new")
    for num_tokens in (4, 16, 64, 192, 512, 1024, 2048, 32768):
        q_c, kv_c, q_w, kv_w, attn_out, gate = _inputs(num_tokens)
        norm_old = _bench(_old_norms, q_c, kv_c, q_w, kv_w)
        norm_new = _bench(
            torch.ops.vllm.mla_bounded_q_kv_rmsnorm,
            q_c,
            kv_c,
            q_w,
            kv_w,
            EPS,
            MAX_TOKENS,
        )
        gate_old = _bench(mla_output_gate, attn_out, gate, False)
        gate_new = _bench(mla_output_gate, attn_out, gate, True)
        print(
            f"{num_tokens:6d} | {norm_old:9.2f} | {norm_new:9.2f} | "
            f"{gate_old:8.2f} | {gate_new:8.2f} | {norm_old + gate_old:9.2f} | "
            f"{norm_new + gate_new:.2f}"
        )


if __name__ == "__main__":
    with ir.ops.rms_norm.set_priority(["aiter", "native"]):
        main()
