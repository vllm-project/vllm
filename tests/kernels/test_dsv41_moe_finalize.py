# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1's fused MoE top-k finalize + shared-expert add.

The fused kernel replaces the TRT-LLM finalize and the shared-expert add that
follows it, so it must reproduce them bit for bit. ``moe_unpermute`` is that
finalize: ``check_deferred_moe_finalize`` holds it to the TRTLLM-Gen experts'
own finalized output. The dispatch tests need no GPU.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe import (
    TrtLlmMxfp4ExpertsModular,
    TrtLlmMxfp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    MoEOutput,
    UnfinalizedMoEOutput,
)
from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute_unpermute_supported,
    moe_unpermute,
)
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MoE as DeepseekV4MoEBase
from vllm.models.deepseek_v41.nvidia.model import (
    DeepseekV4MoE,
    _finalize_shared_add,
)
from vllm.models.deepseek_v41.nvidia.ops.mhc import MHC_OVERLAP_MAX_TOKENS
from vllm.platforms import current_platform

MODEL_MODULE = "vllm.models.deepseek_v41.nvidia.model"
# Stands in for the experts' single-launch size, which caps every deferral.
MAX_LAUNCH_TOKENS = 1000


@pytest.mark.parametrize("num_tokens", [1, 1000])
# DeepSeek V4.1's shape, and a hidden size that ends in a partial block.
@pytest.mark.parametrize("hidden_size,top_k", [(5120, 6), (2880, 4)])
def test_finalize_shared_add_matches_finalize_then_add(
    num_tokens: int, hidden_size: int, top_k: int
):
    if not current_platform.is_cuda() or not moe_permute_unpermute_supported():
        pytest.skip("moe_unpermute requires CUDA")
    torch.manual_seed(0)
    num_routes = num_tokens * top_k
    # Rows past the routed ones stand in for the kernel's padding.
    gemm2_permuted = torch.randn(
        num_routes + 8, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    permuted_idx = torch.randperm(num_routes, device="cuda").to(torch.int32)
    permuted_idx = permuted_idx.view(num_tokens, top_k)
    expert_weights = torch.rand(num_tokens, top_k, device="cuda").softmax(dim=-1)
    shared_output = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    # One route goes to another rank's expert (-1), which the finalize skips.
    # moe_unpermute's input keeps a row for that route with zero weight
    # instead, which leaves its sum unchanged.
    unfinalized_idx = permuted_idx.clone()
    unfinalized_idx[0, -1] = -1
    reference_weights = expert_weights.clone()
    reference_weights[0, -1] = 0

    routed_output = torch.empty_like(shared_output)
    moe_unpermute(
        routed_output,
        gemm2_permuted,
        reference_weights,
        permuted_idx,
        # The kernel reads its valid-row count through this pointer either way.
        expert_first_token_offset=gemm2_permuted.new_full(
            (1,), gemm2_permuted.shape[0], dtype=torch.int64
        ),
    )
    expected = shared_output + routed_output

    actual = _finalize_shared_add(
        MoEOutput(
            routed=UnfinalizedMoEOutput(
                gemm2_permuted, expert_weights, unfinalized_idx
            ),
            shared_output=shared_output,
        )
    )
    torch.testing.assert_close(
        actual.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
    )


def _make_moe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tp_size: int = 2,
    use_native_mega_moe: bool = False,
    has_shared_experts: bool = True,
    experts_cls: type = TrtLlmMxfp4ExpertsModular,
    routed_scaling_factor: float = 1.0,
    reduce_results: bool = True,
) -> DeepseekV4MoE:
    """A DeepSeek V4.1 MoE layer holding only what its deferral reads."""
    # Skip building the gate and the experts; the fields below stand in.
    monkeypatch.setattr(
        DeepseekV4MoEBase,
        "__init__",
        lambda self, *args, **kwargs: nn.Module.__init__(self),
    )
    hf_config = SimpleNamespace(
        n_routed_experts=8, num_experts_per_tok=2, num_hidden_layers=1
    )
    moe = DeepseekV4MoE(
        SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config)),
        prefix="model.layers.0.ffn",
    )
    moe_config = FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=replace(
            FusedMoEParallelConfig.make_no_parallel(), tp_size=tp_size
        ),
        in_dtype=torch.bfloat16,
    )
    # The experts lower the deferral cap to their single-launch size.
    moe_config.limit_deferred_moe_finalize(MAX_LAUNCH_TOKENS)
    moe.use_native_mega_moe = use_native_mega_moe
    moe.shared_experts = nn.Identity() if has_shared_experts else None
    moe.reduce_results = reduce_results
    moe.experts = SimpleNamespace(
        moe_config=moe_config,
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(experts_cls=experts_cls)
        ),
        routed_scaling_factor=routed_scaling_factor,
    )
    return moe


@pytest.mark.parametrize(
    "case",
    [
        "native_mega_moe",
        "no_shared_experts",
        "monolithic_experts",
        "routed_scaling_factor",
        "tp1",
    ],
)
def test_ineligible_moe_keeps_runner_finalize(monkeypatch, case):
    moe = _make_moe(
        monkeypatch,
        tp_size=1 if case == "tp1" else 2,
        use_native_mega_moe=case == "native_mega_moe",
        has_shared_experts=case != "no_shared_experts",
        experts_cls=(
            TrtLlmMxfp4ExpertsMonolithic
            if case == "monolithic_experts"
            else TrtLlmMxfp4ExpertsModular
        ),
        routed_scaling_factor=2.5 if case == "routed_scaling_factor" else 1.0,
    )
    moe.defer_finalize(MHC_OVERLAP_MAX_TOKENS)
    assert not moe.defers_finalize(1)
    assert not moe.finalizes_in_next_layer(1)


# 0: the last local layer, a layer before an engram layer, or a model that does
# not fuse the mHC all-reduce.
@pytest.mark.parametrize("next_layer_max_tokens", [0, MHC_OVERLAP_MAX_TOKENS])
def test_deferred_calls_fold_into_next_layer_up_to_its_limit(
    monkeypatch, next_layer_max_tokens
):
    moe = _make_moe(monkeypatch)
    moe.defer_finalize(next_layer_max_tokens)
    for num_tokens in (
        0,
        1,
        MHC_OVERLAP_MAX_TOKENS,
        MHC_OVERLAP_MAX_TOKENS + 1,
        MAX_LAUNCH_TOKENS,
        MAX_LAUNCH_TOKENS + 1,
    ):
        defers = 0 < num_tokens <= MAX_LAUNCH_TOKENS
        assert moe.defers_finalize(num_tokens) == defers
        assert moe.finalizes_in_next_layer(num_tokens) == (
            defers and num_tokens <= next_layer_max_tokens
        )


@pytest.mark.parametrize("reduce_results", [False, True])
@pytest.mark.parametrize(
    "num_tokens", [MHC_OVERLAP_MAX_TOKENS + 1, MAX_LAUNCH_TOKENS + 1]
)
def test_forward_finalizes_deferred_calls_with_shared_add(
    monkeypatch, reduce_results, num_tokens
):
    moe = _make_moe(monkeypatch, reduce_results=reduce_results)
    moe.defer_finalize(MHC_OVERLAP_MAX_TOKENS)
    calls: list[str] = []

    def record(name: str):
        def call(*args):
            calls.append(name)
            return name

        return call

    monkeypatch.setattr(DeepseekV4MoEBase, "forward", record("runner"))
    monkeypatch.setattr(moe, "forward_unfinalized", record("forward_unfinalized"))
    monkeypatch.setattr(
        f"{MODEL_MODULE}._finalize_shared_add", record("finalize_shared_add")
    )
    monkeypatch.setattr(
        f"{MODEL_MODULE}.tensor_model_parallel_all_reduce", record("all_reduce")
    )
    moe(torch.empty(num_tokens, 128))
    if num_tokens <= MAX_LAUNCH_TOKENS:
        expected = ["forward_unfinalized", "finalize_shared_add"]
        expected += ["all_reduce"] if reduce_results else []
    else:
        expected = ["runner"]
    assert calls == expected
