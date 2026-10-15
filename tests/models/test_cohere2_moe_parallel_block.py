# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Cohere2 MoE parallel block (attention || MLP).

Covers the single TP all-reduce of attention + MLP and the attention / MLP
dual stream. Decoder layers are built without ``__init__`` around stub
branches, so the tests need no checkpoint or GPU; the side-stream race test
needs CUDA and is skipped elsewhere.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.models import cohere2_moe
from vllm.platforms import current_platform

_NO_MOE_PARALLEL = FusedMoEParallelConfig.make_no_parallel()
_UNUSED_STREAM = object()


def _unused_dual_stream(token_threshold: int = 4) -> cohere2_moe._DualStreamState:
    """State for paths that never touch the stream or events, so CPU tests
    need no CUDA objects."""
    return cohere2_moe._DualStreamState(
        stream=_UNUSED_STREAM,
        token_threshold=token_threshold,
        fork_event=_UNUSED_STREAM,
        join_event=_UNUSED_STREAM,
    )


class _PassthroughNorm(torch.nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, residual


class _Branch(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output
        self.calls = 0

    def forward(self, *args, **kwargs) -> torch.Tensor:
        self.calls += 1
        return self.output


def _make_decoder(
    *,
    attention: torch.nn.Module,
    mlp: torch.nn.Module,
    mlp_output_is_reduced: bool = False,
    dual_stream: cohere2_moe._DualStreamState | None = None,
) -> cohere2_moe.Cohere2MoeDecoderLayer:
    layer = cohere2_moe.Cohere2MoeDecoderLayer.__new__(
        cohere2_moe.Cohere2MoeDecoderLayer
    )
    torch.nn.Module.__init__(layer)
    layer.layer_name = "model.layers.0"
    layer.input_layernorm = _PassthroughNorm()
    layer.self_attn = attention
    layer.mlp = mlp
    layer.mlp_output_is_reduced = mlp_output_is_reduced
    layer.dual_stream = dual_stream
    return layer


def _make_moe_stub(
    moe_parallel_config: FusedMoEParallelConfig = _NO_MOE_PARALLEL,
) -> cohere2_moe.Cohere2Moe:
    moe = cohere2_moe.Cohere2Moe.__new__(cohere2_moe.Cohere2Moe)
    torch.nn.Module.__init__(moe)
    moe.experts = SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=moe_parallel_config)
    )
    return moe


def _record_all_reduce(monkeypatch, tp_size: int = 4) -> list[torch.Tensor]:
    """Replace the TP all-reduce with a sum over ``tp_size`` identical ranks."""
    inputs: list[torch.Tensor] = []

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        inputs.append(tensor)
        return tensor * tp_size

    monkeypatch.setattr(cohere2_moe, "tensor_model_parallel_all_reduce", all_reduce)
    return inputs


@pytest.mark.parametrize(
    ("num_shared_experts", "tp_size", "skip_final_all_reduce", "expected"),
    [
        pytest.param(0, 4, True, (False, False), id="tp4-routed-only"),
        pytest.param(0, 4, False, (False, True), id="tp4-routed-only-all2all"),
        pytest.param(1, 4, False, (True, True), id="tp4-shared-experts"),
        pytest.param(0, 1, False, (True, True), id="tp1"),
    ],
)
def test_moe_defers_all_reduce_only_without_shared_experts(
    monkeypatch,
    num_shared_experts: int,
    tp_size: int,
    skip_final_all_reduce: bool,
    expected: tuple[bool, bool],
) -> None:
    """Only a routed-only TP MoE asks FusedMoE to skip its all-reduce, and it
    stays reduced when the runner resolves the request away (e.g. all2all)."""
    factory_kwargs: dict[str, object] = {}

    def make_experts(**kwargs):
        factory_kwargs.update(kwargs)
        return SimpleNamespace(
            moe_config=SimpleNamespace(skip_final_all_reduce=skip_final_all_reduce)
        )

    monkeypatch.setattr(
        cohere2_moe, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(
        cohere2_moe, "GateLinear", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(
        cohere2_moe, "Cohere2MoeMLP", lambda *args, **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(cohere2_moe, "FusedMoEFactory", make_experts)
    config = SimpleNamespace(
        hidden_size=4,
        num_experts=4,
        num_experts_per_tok=1,
        intermediate_size=4,
        num_shared_experts=num_shared_experts,
    )

    moe = cohere2_moe.Cohere2Moe(config)

    assert (factory_kwargs["reduce_results"], moe.output_is_reduced) == expected


@pytest.mark.parametrize("mlp_output_is_reduced", [False, True])
def test_parallel_block_reduces_attention_plus_mlp_once(
    monkeypatch, mlp_output_is_reduced: bool
) -> None:
    """Attention + MLP share one all-reduce unless the MLP output is already
    reduced, in which case only attention is reduced."""
    hidden_states = torch.ones(2, 4)
    attention_output = torch.full_like(hidden_states, 2)
    mlp_output = torch.full_like(hidden_states, 3)
    layer = _make_decoder(
        attention=_Branch(attention_output),
        mlp=_Branch(mlp_output),
        mlp_output_is_reduced=mlp_output_is_reduced,
    )
    all_reduce_inputs = _record_all_reduce(monkeypatch)

    output, residual = layer(
        positions=torch.arange(hidden_states.shape[0]),
        hidden_states=hidden_states,
        residual=None,
    )

    assert len(all_reduce_inputs) == 1
    if mlp_output_is_reduced:
        torch.testing.assert_close(all_reduce_inputs[0], attention_output)
        expected_block = 4 * attention_output + mlp_output
    else:
        torch.testing.assert_close(all_reduce_inputs[0], attention_output + mlp_output)
        expected_block = 4 * (attention_output + mlp_output)
    torch.testing.assert_close(output, hidden_states + expected_block)
    torch.testing.assert_close(residual, hidden_states)


@pytest.mark.parametrize(
    ("make_mlp", "mlp_output_is_reduced", "tp_size", "expected"),
    [
        pytest.param(_make_moe_stub, True, 1, True, id="tp1-moe"),
        pytest.param(lambda: torch.nn.Identity(), False, 4, True, id="tp4-dense"),
        pytest.param(_make_moe_stub, False, 4, True, id="tp4-moe-deferred"),
        pytest.param(_make_moe_stub, True, 4, False, id="tp4-moe-reduced"),
    ],
)
def test_dual_stream_only_where_tp_collectives_stay_serial(
    monkeypatch, make_mlp, mlp_output_is_reduced: bool, tp_size: int, expected: bool
) -> None:
    """Under TP, an MLP that reduces internally must stay single-stream."""
    monkeypatch.setattr(
        cohere2_moe, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    layer = _make_decoder(
        attention=torch.nn.Identity(),
        mlp=make_mlp(),
        mlp_output_is_reduced=mlp_output_is_reduced,
    )

    assert layer._allows_dual_stream() is expected


@pytest.mark.parametrize("tp_size", [1, 4], ids=["tp1", "tp4"])
def test_dual_stream_off_with_all2all_kernels(monkeypatch, tp_size: int) -> None:
    """EP all2all MoE (here DP2 + EP) stays single-stream at any TP size."""
    monkeypatch.setattr(
        cohere2_moe, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    moe_parallel_config = dataclasses.replace(_NO_MOE_PARALLEL, dp_size=2, use_ep=True)
    assert moe_parallel_config.use_all2all_kernels
    layer = _make_decoder(
        attention=torch.nn.Identity(), mlp=_make_moe_stub(moe_parallel_config)
    )

    assert layer._allows_dual_stream() is False


@pytest.mark.parametrize(
    ("runtime_mode", "num_tokens", "breakable_capture"),
    [
        (CUDAGraphMode.PIECEWISE, 2, False),
        (CUDAGraphMode.NONE, 5, False),
        (CUDAGraphMode.FULL, 2, True),
    ],
    ids=["piecewise-graph", "over-token-threshold", "breakable-capture"],
)
def test_fork_runs_mlp_inline_when_overlap_is_unsafe(
    monkeypatch,
    runtime_mode: CUDAGraphMode,
    num_tokens: int,
    breakable_capture: bool,
) -> None:
    """Piecewise graphs (prefill / mixed batches), oversized batches and
    breakable capture must not fork: the MLP runs inline and join only adds.

    The layer's stream is a plain sentinel, so any attempt to fork fails.
    """
    monkeypatch.setattr(
        cohere2_moe,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=runtime_mode),
    )
    monkeypatch.setattr(
        cohere2_moe.BreakableCUDAGraphCapture,
        "is_active",
        staticmethod(lambda: breakable_capture),
    )
    hidden_states = torch.ones(num_tokens, 4)
    mlp_output = torch.full_like(hidden_states, 3)
    attention_output = torch.full_like(hidden_states, 2)
    mlp = _Branch(mlp_output)
    layer = _make_decoder(
        attention=_Branch(attention_output),
        mlp=mlp,
        dual_stream=_unused_dual_stream(token_threshold=4),
    )

    forked = layer._fork_mlp(hidden_states)

    assert forked is mlp_output
    assert mlp.calls == 1
    assert layer.dual_stream.join_pending is False
    torch.testing.assert_close(
        layer._join_mlp(attention_output, forked), attention_output + mlp_output
    )


def test_dual_stream_forward_reduces_attention_plus_mlp_once(monkeypatch) -> None:
    """With the dual stream the layer goes through the fork/join ops and does
    a single TP all-reduce of attention + MLP after the join."""
    device = torch.device(current_platform.device_type)
    hidden_states = torch.ones(2, 4, device=device)
    attention_output = torch.full_like(hidden_states, 2)
    mlp_output = torch.full_like(hidden_states, 3)
    layer = _make_decoder(
        attention=_Branch(attention_output),
        mlp=_Branch(mlp_output),
        dual_stream=_unused_dual_stream(),
    )
    monkeypatch.setattr(
        cohere2_moe,
        "get_forward_context",
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            no_compile_layers={layer.layer_name: layer},
        ),
    )
    all_reduce_inputs = _record_all_reduce(monkeypatch)

    output, residual = layer(
        positions=torch.arange(hidden_states.shape[0], device=device),
        hidden_states=hidden_states,
        residual=None,
    )

    assert len(all_reduce_inputs) == 1
    torch.testing.assert_close(all_reduce_inputs[0], attention_output + mlp_output)
    torch.testing.assert_close(
        output, hidden_states + 4 * (attention_output + mlp_output)
    )
    torch.testing.assert_close(residual, hidden_states)


class _SlowStreamRecordingMLP(torch.nn.Module):
    """Delays its stream before writing the output, so a join that does not
    wait would read an unfinished tensor."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight
        self.streams: list[torch.cuda.Stream] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.streams.append(torch.cuda.current_stream())
        torch.cuda._sleep(50_000_000)
        return torch.tanh(hidden_states @ self.weight)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA streams")
def test_fork_overlaps_on_side_stream_and_join_waits(monkeypatch) -> None:
    """Eager decode-sized batch through ``layer(...)``: the registered
    ``cohere2_mlp_fork`` / ``cohere2_mlp_join`` ops find the layer through the
    forward context, the MLP runs on the side stream, and the join makes the
    main stream wait, so the result matches the serial path."""
    hidden_states = torch.randn(8, 64, device="cuda")
    weight = torch.randn(64, 64, device="cuda")
    attention_output = torch.randn_like(hidden_states)
    mlp = _SlowStreamRecordingMLP(weight)
    moe_stream = torch.cuda.Stream()
    layer = _make_decoder(
        attention=_Branch(attention_output),
        mlp=mlp,
        dual_stream=cohere2_moe._DualStreamState(
            stream=moe_stream, token_threshold=256
        ),
    )
    monkeypatch.setattr(
        cohere2_moe,
        "get_forward_context",
        lambda: SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            no_compile_layers={layer.layer_name: layer},
        ),
    )
    monkeypatch.setattr(
        cohere2_moe, "tensor_model_parallel_all_reduce", lambda tensor: tensor
    )
    expected = hidden_states + attention_output + torch.tanh(hidden_states @ weight)

    # The first allocation on a new stream goes through cudaMalloc, which
    # synchronizes the device and would hide a join that does not wait, so
    # only the second, warm iteration exercises the race.
    for _ in range(2):
        output, residual = layer(
            positions=torch.arange(hidden_states.shape[0], device="cuda"),
            hidden_states=hidden_states,
            residual=None,
        )
        assert layer.dual_stream.join_pending is False
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(residual, hidden_states)

    assert mlp.streams == [moe_stream, moe_stream]
