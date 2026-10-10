# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoERunner holding the gate of a latent MoE (routed input transform).

The runner applies the routed input transform inside the MoE op and runs the
gate, which reads the pre-transform input, on the aux stream concurrently with
it. The output must be bitwise identical to the model-held gate (router logits
computed outside the runner, transform applied in `forward`), eagerly and under
CUDA graph capture/replay, and the gate must actually run on the aux stream up
to VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD tokens. For NemotronH, the
output and router logits must also be bitwise identical to the stock path, in
which the model computed the gate and fc1_latent_proj in its compiled forward.
"""

import contextlib

import pytest
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    aux_stream,
    set_default_torch_dtype,
    set_random_seed,
)

HIDDEN = 256
LATENT = 128
NUM_EXPERTS = 16
TOP_K = 4
DTYPE = torch.bfloat16


class _Gate(nn.Module):
    """bf16 x bf16 -> fp32 router gate that records the stream it ran on."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", dtype=DTYPE) / 10
        )
        self.streams: list[torch.cuda.Stream] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert x.shape[-1] == HIDDEN, "gate must see the pre-transform input"
        self.streams.append(torch.cuda.current_stream())
        return torch.mm(x, self.weight.T, out_dtype=torch.float32), None


class _Linear(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(n_out, n_in, device="cuda", dtype=DTYPE) / 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight)


class _SharedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = _Linear(HIDDEN, 2 * HIDDEN)
        self.down = _Linear(HIDDEN, HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.up(x).chunk(2, dim=-1)
        return self.down(nn.functional.silu(gate) * up)


@pytest.fixture(autouse=True)
def setup_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.set_default_device("cuda")


def _make_moes(vllm_config: VllmConfig):
    set_random_seed(0)
    gate = _Gate()
    fc1 = _Linear(HIDDEN, LATENT)
    fc2 = _Linear(LATENT, HIDDEN)
    shared = _SharedExperts()
    common = dict(
        shared_experts=shared,
        routed_input_transform=fc1,
        routed_output_transform=fc2,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden_size=LATENT,
        intermediate_size=2 * LATENT,
        renormalize=True,
        params_dtype=DTYPE,
        router_logits_dtype=torch.float32,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
    )
    with set_current_vllm_config(vllm_config):
        # Runner-held gate (overlapped).
        moe = FusedMoEFactory(gate=gate, prefix="runner_gate", **common)
        # Model-held gate: logits computed outside, transform in `forward`.
        ref = FusedMoEFactory(gate=None, prefix="model_gate", **common)
    experts, ref_experts = moe.routed_experts, ref.routed_experts
    with torch.no_grad():
        experts.w13_weight.normal_().div_(10)
        experts.w2_weight.normal_().div_(10)
        ref_experts.w13_weight.copy_(experts.w13_weight)
        ref_experts.w2_weight.copy_(experts.w2_weight)
    experts.quant_method.process_weights_after_loading(experts)
    ref_experts.quant_method.process_weights_after_loading(ref_experts)
    return gate, moe, ref


def _run_moe(moe, x):
    # Placeholder router logits: the runner computes them with its gate.
    return moe(x, x)


def _run_ref(gate, ref, x):
    logits, _ = gate(x)
    return ref(x, logits)


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 300])
def test_runner_gate_overlap_eager_bitwise(num_tokens, dist_init, workspace_init):
    assert aux_stream() is not None
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()
    gate, moe, ref = _make_moes(vllm_config)
    assert moe._gate_stream is aux_stream()

    x = torch.randn(num_tokens, HIDDEN, dtype=DTYPE)
    with set_forward_context(None, vllm_config, num_tokens=num_tokens):
        gate.streams.clear()
        out = _run_moe(moe, x)
        moe_streams = list(gate.streams)
        expected = _run_ref(gate, ref, x)
    torch.cuda.synchronize()

    # The gate ran on the aux stream iff the batch is under the threshold.
    on_aux = num_tokens <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
    assert moe_streams == [aux_stream() if on_aux else torch.cuda.current_stream()]
    assert out.shape == (num_tokens, HIDDEN)
    assert torch.equal(out, expected)


def test_runner_gate_no_overlap_when_stream_disabled(
    dist_init, workspace_init, monkeypatch
):
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()
    gate, moe, ref = _make_moes(vllm_config)
    assert moe._gate_stream is None

    x = torch.randn(4, HIDDEN, dtype=DTYPE)
    with set_forward_context(None, vllm_config, num_tokens=4):
        gate.streams.clear()
        out = _run_moe(moe, x)
        assert gate.streams == [torch.cuda.current_stream()]
        expected = _run_ref(gate, ref, x)
    torch.cuda.synchronize()
    assert torch.equal(out, expected)


@pytest.mark.parametrize("num_tokens", [1, 8, 48])
def test_runner_gate_overlap_cuda_graph_bitwise(num_tokens, dist_init, workspace_init):
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()
    gate, moe, ref = _make_moes(vllm_config)
    x = torch.randn(num_tokens, HIDDEN, dtype=DTYPE)

    def capture(fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()  # warmup
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = fn()
        torch.cuda.current_stream().wait_stream(stream)
        return graph, out

    with set_forward_context(None, vllm_config, num_tokens=num_tokens):
        g_moe, out = capture(lambda: _run_moe(moe, x))
        g_ref, expected = capture(lambda: _run_ref(gate, ref, x))
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        g_moe.replay()
        g_ref.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)


# NemotronH (latent MoE) against the stock data flow, in which the model ran
# the gate and fc1_latent_proj in its compiled forward and passed the router
# logits to the runner.
NH_TOKENS = [1, 2, 4, 8, 16, 32, 256]


@contextlib.contextmanager
def _model_held_gate(runner):
    gate, runner.gate = runner.gate, None
    try:
        yield
    finally:
        runner.gate = gate


def _nemotron_h_moe(vllm_config: VllmConfig):
    from vllm.model_executor.models.nemotron_h import NemotronHMoE
    from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig

    config = NemotronHConfig(
        hidden_size=4096,
        n_routed_experts=512,
        n_shared_experts=1,
        moe_intermediate_size=512,
        moe_shared_expert_intermediate_size=1024,
        moe_latent_size=1024,
        num_experts_per_tok=8,
        routed_scaling_factor=2.5,
        mlp_hidden_act="relu2",
    )
    set_random_seed(0)
    with set_current_vllm_config(vllm_config), set_default_torch_dtype(DTYPE):
        moe = NemotronHMoE(
            config,
            parallel_config=vllm_config.parallel_config,
            prefix="model.layers.1.mixer",
        )
    with torch.no_grad():
        for param in moe.parameters():
            param.normal_(0, 0.02)
    experts = moe.experts.routed_experts
    experts.quant_method.process_weights_after_loading(experts)
    # Plain Parameters (same storage) for the layers traced below; Dynamo
    # outside vLLM's compile wrapper recurses on vLLM's Parameter subclasses.
    for module in (moe.gate, moe.fc1_latent_proj):
        module.weight = nn.Parameter(module.weight.data, requires_grad=False)
    return moe


def _compiled_stock_prologue(moe):
    """What the stock NemotronHMoE.forward ran before the MoE op, compiled
    the way vLLM compiles the model: dynamic batch, traced at a large batch,
    guards dropped."""

    def prologue(x):
        router_logits, _ = moe.gate(x)
        latent, _ = moe.fc1_latent_proj(x)
        return router_logits, latent

    compiled = torch.compile(
        prologue,
        dynamic=True,
        options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    )
    # Stock GateLinear allowed the low-latency tier; tracing at a large batch
    # took the cuBLAS branch for all batch sizes.
    moe.gate.allow_ll_bf16_gemm = _ll_bf16_available()
    x = torch.randn(8192, 4096, dtype=DTYPE)
    torch._dynamo.mark_dynamic(x, 0)
    compiled(x)
    moe.gate.allow_ll_bf16_gemm = False
    return compiled


def _ll_bf16_available() -> bool:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import is_available

    return current_platform.is_device_capability_family(100) and is_available()


def _stock_forward(moe, compiled, x):
    router_logits, latent = compiled(x)
    with _model_held_gate(moe.experts):
        out = moe.experts(
            hidden_states=latent, router_logits=router_logits, shared_experts_input=x
        )
    return out.view(x.shape), router_logits


@pytest.mark.parametrize("cuda_graph", [False, True])
def test_nemotron_h_runner_gate_matches_stock_compiled(
    cuda_graph, dist_init, workspace_init
):
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()
    moe = _nemotron_h_moe(vllm_config)
    assert moe.experts.gate is moe.gate
    assert moe.experts._gate_stream is aux_stream()
    assert not moe.gate.allow_ll_bf16_gemm
    compiled = _compiled_stock_prologue(moe)

    for num_tokens in NH_TOKENS:
        x = torch.randn(num_tokens, 4096, dtype=DTYPE)
        with set_forward_context(None, vllm_config, num_tokens=num_tokens):
            if cuda_graph:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    moe(x), _stock_forward(moe, compiled, x)  # warmup
                    g_out, g_ref = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_out, stream=stream):
                        out = moe(x)
                    with torch.cuda.graph(g_ref, stream=stream):
                        ref, ref_logits = _stock_forward(moe, compiled, x)
                torch.cuda.current_stream().wait_stream(stream)
                for _ in range(3):
                    x.copy_(torch.randn_like(x))
                    g_out.replay()
                    g_ref.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, ref), num_tokens
            else:
                out = moe(x)
                ref, ref_logits = _stock_forward(moe, compiled, x)
                torch.cuda.synchronize()
                assert torch.equal(out, ref), num_tokens
                # The gate the runner calls gives the stock logits bitwise.
                assert torch.equal(moe.gate(x)[0], ref_logits), num_tokens


def test_nemotron_h_low_latency_gate_tier_differs():
    """Why NemotronH pins the gate's cuBLAS tier: the low-latency tier, which
    the runner would otherwise pick at run time for small batches, does not
    round like the stock compiled path."""
    if not _ll_bf16_available():
        pytest.skip("low-latency bf16 gate tier not available")
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm

    set_random_seed(0)
    weight = torch.randn(512, 4096, dtype=DTYPE) * 0.02
    x = torch.randn(8, 4096, dtype=DTYPE)
    cublas = torch.mm(x, weight.T, out_dtype=torch.float32)
    assert not torch.equal(ll_bf16_gemm(x, weight), cublas)
