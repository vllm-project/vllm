# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pin which all-reduce the ROCm DSA MoE runs for each parallel layout."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config.parallel import ParallelConfig
from vllm.model_executor.layers.fused_moe import config as fused_moe_config
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.common.ops import fused_allreduce_rms_norm as fused_norm
from vllm.models.deepseek_v32.amd import model as amd_model

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

NUM_TOKENS, HIDDEN = 3, 8


@dataclass(frozen=True)
class Layout:
    tp: int = 1
    dp: int = 1
    pcp: int = 1
    enable_ep: bool = False
    backend: str = "allgather_reducescatter"

    def __str__(self) -> str:
        parts = [f"tp{self.tp}", f"dp{self.dp}", f"pcp{self.pcp}"]
        if self.enable_ep:
            parts.append(f"ep-{self.backend}")
        return "-".join(parts)


_REDUCING_BACKENDS = (
    "mori_high_throughput",
    "mori_low_latency",
    "deepep_high_throughput",
    "deepep_low_latency",
)

CORRECT = "reduced exactly once"
ASSERTS = "runner asserts"
# DP+EP+TP>1 (SP MoE) double reduce is pre-existing; pinned to stay unchanged.
DOUBLE_REDUCED = "reduced twice (pre-existing)"

_SP = dict(tp=2, dp=2, enable_ep=True)
_PCP_TP2 = dict(tp=2, pcp=2, enable_ep=True)

# (layout, expected skip_final_all_reduce, outcome on main, outcome now)
LAYOUTS = [
    (Layout(), True, CORRECT, CORRECT),
    (Layout(tp=2), True, CORRECT, CORRECT),
    (Layout(tp=2, enable_ep=True), True, CORRECT, CORRECT),
    (Layout(dp=2), True, CORRECT, CORRECT),
    (Layout(tp=2, dp=2), True, CORRECT, CORRECT),
    # The fixed layouts: a reducing combine at TP=1.
    *[
        (Layout(dp=2, enable_ep=True, backend=b), False, ASSERTS, CORRECT)
        for b in _REDUCING_BACKENDS
    ],
    (
        Layout(pcp=2, enable_ep=True, backend="mori_high_throughput"),
        False,
        ASSERTS,
        CORRECT,
    ),
    (Layout(dp=2, enable_ep=True), True, CORRECT, CORRECT),
    (Layout(pcp=2, enable_ep=True), True, CORRECT, CORRECT),
    (Layout(**_SP), True, DOUBLE_REDUCED, DOUBLE_REDUCED),
    (Layout(**_PCP_TP2), True, CORRECT, CORRECT),
    # A reducing combine at TP>1 keeps failing loudly instead of reducing twice.
    *[(Layout(**_SP, backend=b), True, ASSERTS, ASSERTS) for b in _REDUCING_BACKENDS],
    (Layout(**_PCP_TP2, backend="mori_high_throughput"), True, ASSERTS, ASSERTS),
]

CASES = [
    pytest.param(layout, skip, on_main, now, id=str(layout))
    for layout, skip, on_main, now in LAYOUTS
]


def _use_sequence_parallel_moe(layout: Layout) -> bool:
    return ParallelConfig.use_sequence_parallel_moe.fget(
        SimpleNamespace(
            all2all_backend=layout.backend,
            enable_expert_parallel=layout.enable_ep,
            tensor_parallel_size=layout.tp,
            data_parallel_size=layout.dp,
        )
    )


def _moe_parallel_config(monkeypatch, layout: Layout) -> FusedMoEParallelConfig:
    group = SimpleNamespace(rank_in_group=0)
    monkeypatch.setattr(fused_moe_config, "get_dp_group", lambda: group)
    monkeypatch.setattr(fused_moe_config, "get_pcp_group", lambda: group)
    monkeypatch.setattr(fused_moe_config, "get_tensor_model_parallel_rank", lambda: 0)
    sp = _use_sequence_parallel_moe(layout)
    return FusedMoEParallelConfig.make(
        tp_size_=layout.tp,
        pcp_size_=layout.pcp,
        dp_size_=layout.dp,
        sp_size_=layout.tp if sp else 1,
        vllm_parallel_config=SimpleNamespace(
            enable_expert_parallel=layout.enable_ep,
            all2all_backend=layout.backend,
            enable_eplb=False,
        ),
    )


class _Norm(nn.Module):
    """Records the first residual sum it sees."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN), requires_grad=False)
        self.variance_epsilon = 1e-6
        self.seen: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return x
        out = x + residual
        if self.seen is None:
            self.seen = out.clone()
        return out, out


class _Attention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, positions, hidden_states):
        return torch.zeros_like(hidden_states)


def _build_amd_decoder_layer(monkeypatch, layout: Layout, moe_parallel_config):
    class _MoE(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            # reduce_results=True leaves skip_final_all_reduce False.
            self.experts = SimpleNamespace(
                moe_config=SimpleNamespace(
                    moe_parallel_config=moe_parallel_config,
                    skip_final_all_reduce=False,
                )
            )

        def forward(self, hidden_states):
            return hidden_states

    monkeypatch.setattr(amd_model, "DeepseekV2MoE", _MoE)
    monkeypatch.setattr(amd_model, "DeepseekV32MLAAttention", _Attention)
    monkeypatch.setattr(amd_model, "RMSNorm", _Norm)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        n_routed_experts=4,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        rms_norm_eps=1e-6,
        routed_scaling_factor=1.0,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=layout.tp,
            data_parallel_size=layout.dp,
            prefill_context_parallel_size=layout.pcp,
            enable_expert_parallel=layout.enable_ep,
            all2all_backend=layout.backend,
            use_sequence_parallel_moe=_use_sequence_parallel_moe(layout),
        ),
    )
    return amd_model.DeepseekV32DecoderLayer(
        vllm_config=vllm_config, prefix="model.layers.1", config=config
    )


def _tp_all_reduce(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=0, keepdim=True).expand_as(x).clone()


def _replicated(full: torch.Tensor, tp: int) -> torch.Tensor:
    return full.unsqueeze(0).expand(tp, *full.shape).clone()


def _tp_partial(full: torch.Tensor, tp: int) -> torch.Tensor:
    parts = torch.randn(tp, *full.shape, dtype=full.dtype)
    parts[-1] = full - parts[:-1].sum(dim=0)
    return parts


def _runner(moe_parallel_config, skip_final_all_reduce: bool, combine_reduces: bool):
    runner = object.__new__(MoERunner)
    kernel = SimpleNamespace(output_is_reduced=lambda: combine_reduces)
    attrs = {
        "moe_config": SimpleNamespace(
            tp_size=moe_parallel_config.tp_size,
            ep_size=moe_parallel_config.ep_size,
            is_sequence_parallel=moe_parallel_config.is_sequence_parallel,
            skip_final_all_reduce=skip_final_all_reduce,
            hidden_dim=HIDDEN,
            hidden_dim_unpadded=HIDDEN,
        ),
        "routed_experts": SimpleNamespace(
            quant_method=SimpleNamespace(
                moe_kernel=kernel,
                skip_forward_padding=True,
                has_unpadded_output=False,
            )
        ),
        "_shared_experts": object(),
        "routed_input_transform": None,
        "routed_output_transform": None,
        "routed_scaling_factor": 1.0,
        "router": None,
        "layer_name": "model.layers.1.mlp.experts",
    }
    for name, value in attrs.items():
        object.__setattr__(runner, name, value)
    return runner


def _layer_skip(monkeypatch, layout: Layout) -> bool:
    moe_parallel_config = _moe_parallel_config(monkeypatch, layout)
    layer = _build_amd_decoder_layer(monkeypatch, layout, moe_parallel_config)
    return layer.mlp.experts.moe_config.skip_final_all_reduce


def _outcome(monkeypatch, layout: Layout, skip_final_all_reduce: bool | None) -> str:
    """``skip_final_all_reduce=None`` uses the layer's own value."""
    torch.manual_seed(0)
    tp = layout.tp
    monkeypatch.setattr(moe_runner, "tensor_model_parallel_all_reduce", _tp_all_reduce)
    monkeypatch.setattr(fused_norm, "tensor_model_parallel_all_reduce", _tp_all_reduce)
    monkeypatch.setattr(fused_norm, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(fused_norm, "flashinfer_trtllm_fused_allreduce_norm", None)

    moe_parallel_config = _moe_parallel_config(monkeypatch, layout)
    layer = _build_amd_decoder_layer(monkeypatch, layout, moe_parallel_config)
    if skip_final_all_reduce is None:
        skip_final_all_reduce = layer.mlp.experts.moe_config.skip_final_all_reduce

    sp = moe_parallel_config.is_sequence_parallel
    combine_reduces = (
        moe_parallel_config.use_all2all_kernels and layout.backend in _REDUCING_BACKENDS
    )
    routed = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.float64)
    shared = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.float64)
    residual = torch.randn(NUM_TOKENS, HIDDEN, dtype=torch.float64)

    # Under SP each rank ends with the full output; otherwise it is TP-partial.
    routed_in = (
        _replicated(routed, tp) if sp or combine_reduces else _tp_partial(routed, tp)
    )
    shared_in = _replicated(shared, tp) if sp else _tp_partial(shared, tp)

    runner = _runner(moe_parallel_config, skip_final_all_reduce, combine_reduces)
    object.__setattr__(runner, "_forward_entry", lambda *args: (shared_in, routed_in))
    try:
        moe_out = runner.forward(
            hidden_states=torch.zeros(tp, NUM_TOKENS, HIDDEN, dtype=torch.float64),
            router_logits=torch.zeros(tp, NUM_TOKENS, 4, dtype=torch.float64),
        )
    except AssertionError as e:
        assert "skip_final_all_reduce requires an un-reduced fused output" in str(e)
        return ASSERTS

    # The next decoder layer's input_layernorm consumes the MoE output.
    layer(
        positions=torch.zeros(NUM_TOKENS),
        hidden_states=moe_out,
        residual=_replicated(residual, tp),
    )
    seen = layer.input_layernorm.seen
    moe = routed + shared
    if torch.allclose(seen, _replicated(moe + residual, tp)):
        return CORRECT
    if torch.allclose(seen, _replicated(tp * moe + residual, tp)):
        return DOUBLE_REDUCED
    return "wrong"


@pytest.mark.parametrize(("layout", "skip", "on_main", "now"), CASES)
def test_amd_dsa_skip_final_all_reduce(monkeypatch, layout, skip, on_main, now):
    assert _layer_skip(monkeypatch, layout) is skip


@pytest.mark.parametrize(("layout", "skip", "on_main", "now"), CASES)
def test_amd_dsa_moe_output_vs_main(monkeypatch, layout, skip, on_main, now):
    # main forced skip_final_all_reduce on for every layout.
    assert _outcome(monkeypatch, layout, skip_final_all_reduce=True) == on_main
    assert _outcome(monkeypatch, layout, skip_final_all_reduce=None) == now


def test_only_the_asserting_tp1_layouts_change():
    changed = [layout for layout, _, on_main, now in LAYOUTS if on_main != now]
    assert changed and all(
        layout.tp == 1 and layout.backend in _REDUCING_BACKENDS for layout in changed
    )
    for layout, _, on_main, now in LAYOUTS:
        if on_main != now:
            assert (on_main, now) == (ASSERTS, CORRECT)


def test_sequence_parallel_moe_implies_expert_parallel():
    # Rules out SP without EP, where use_all2all_kernels would be False.
    for backend in ("mori_high_throughput", "allgather_reducescatter"):
        layout = Layout(tp=2, dp=2, enable_ep=False, backend=backend)
        assert not _use_sequence_parallel_moe(layout)
