# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


def _make_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=1,
        experts_per_token=1,
        hidden_dim=3,
        intermediate_size_per_partition=4,
        num_local_experts=1,
        num_logical_experts=1,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.float32,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=8,
    )


class _DirectOutputExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        direct_output: torch.Tensor,
        weight_and_reduce_impl: mk.TopKWeightAndReduce | None = None,
    ) -> None:
        super().__init__(
            moe_config=_make_moe_config(),
            quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        )
        self.direct_output = direct_output
        self.weight_and_reduce_impl = (
            weight_and_reduce_impl or TopKWeightAndReduceNoOP()
        )
        self.output_arg: torch.Tensor | None = None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return True

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return self.weight_and_reduce_impl

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (1,), (1,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor | None:
        self.output_arg = output
        return self.direct_output


class _TrackingNoDPEP(MoEPrepareAndFinalizeNoDPEPModular):
    def __init__(self) -> None:
        self.finalize_calls = 0

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self.finalize_calls += 1
        super().finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )


def _make_impl(
    direct_output: torch.Tensor,
    prepare_finalize: MoEPrepareAndFinalizeNoDPEPModular | None = None,
    inplace: bool = False,
) -> tuple[mk.FusedMoEKernelModularImpl, _DirectOutputExperts]:
    experts = _DirectOutputExperts(direct_output)
    impl = mk.FusedMoEKernelModularImpl(
        prepare_finalize or MoEPrepareAndFinalizeNoDPEPModular(),
        experts,
        shared_experts=None,
        inplace=inplace,
    )
    return impl, experts


def _patch_apply_inputs(
    monkeypatch,
    impl: mk.FusedMoEKernelModularImpl,
    fused_out: torch.Tensor,
    fused_out_is_direct: bool,
) -> None:
    def fake_prepare(
        hidden_states,
        topk_weights,
        topk_ids,
        global_num_experts,
        expert_map,
        apply_router_weight_on_input,
    ):
        return hidden_states, None, None, topk_ids, topk_weights

    def fake_fused_experts(**kwargs):
        return fused_out, fused_out_is_direct

    monkeypatch.setattr(impl, "_prepare", fake_prepare)
    monkeypatch.setattr(impl, "_fused_experts", fake_fused_experts)


def _apply_test_moe(
    impl: mk.FusedMoEKernelModularImpl,
    hidden_states: torch.Tensor,
    shared_experts_input: torch.Tensor | None = None,
) -> torch.Tensor:
    w1 = torch.empty(1, 4, 3)
    w2 = torch.empty(1, 3, 4)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    return impl.apply(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        shared_experts_input=shared_experts_input,
    )


def test_fused_experts_can_return_direct_output(monkeypatch):
    direct_output = torch.full((2, 3), 7.0)
    fallback_output = torch.empty_like(direct_output)
    impl, experts = _make_impl(direct_output)

    def fake_allocate_buffers(*args, **kwargs):
        return torch.empty(1), torch.empty(1), fallback_output

    monkeypatch.setattr(impl, "_allocate_buffers", fake_allocate_buffers)

    hidden_states = torch.empty(2, 3)
    w1 = torch.empty(1, 4, 3)
    w2 = torch.empty(1, 3, 4)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)

    output, output_is_direct = impl._fused_experts(
        in_dtype=hidden_states.dtype,
        a1q=hidden_states,
        a1q_scale=None,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=1,
        local_num_experts=1,
        expert_map=None,
        apply_router_weight_on_input=False,
        expert_tokens_meta=None,
    )

    assert output is direct_output
    assert output_is_direct
    assert experts.output_arg is fallback_output


def test_fused_experts_marks_returned_workspace_not_direct(monkeypatch):
    fallback_output = torch.full((2, 3), 7.0)
    workspace_alias = fallback_output.view_as(fallback_output)
    impl, experts = _make_impl(workspace_alias)

    def fake_allocate_buffers(*args, **kwargs):
        return torch.empty(1), torch.empty(1), fallback_output

    monkeypatch.setattr(impl, "_allocate_buffers", fake_allocate_buffers)

    hidden_states = torch.empty(2, 3)
    w1 = torch.empty(1, 4, 3)
    w2 = torch.empty(1, 3, 4)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)

    output, output_is_direct = impl._fused_experts(
        in_dtype=hidden_states.dtype,
        a1q=hidden_states,
        a1q_scale=None,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=1,
        local_num_experts=1,
        expert_map=None,
        apply_router_weight_on_input=False,
        expert_tokens_meta=None,
    )

    assert output is workspace_alias
    assert not output_is_direct
    assert experts.output_arg is fallback_output


def test_apply_skips_noop_copy_for_direct_output(monkeypatch):
    prepare_finalize = _TrackingNoDPEP()
    fused_out = torch.full((2, 3), 5.0)
    impl, _ = _make_impl(fused_out, prepare_finalize)
    _patch_apply_inputs(monkeypatch, impl, fused_out, fused_out_is_direct=True)

    output = _apply_test_moe(impl, torch.zeros_like(fused_out))

    assert output is fused_out
    assert prepare_finalize.finalize_calls == 0


def test_apply_copies_workspace_output_before_returning(monkeypatch):
    prepare_finalize = _TrackingNoDPEP()
    fused_out = torch.full((2, 3), 5.0)
    impl, _ = _make_impl(fused_out, prepare_finalize)
    _patch_apply_inputs(monkeypatch, impl, fused_out, fused_out_is_direct=False)

    output = _apply_test_moe(impl, torch.zeros_like(fused_out))

    assert output is not fused_out
    assert prepare_finalize.finalize_calls == 1
    torch.testing.assert_close(output, fused_out)


def test_apply_uses_generic_path_for_inplace_output(monkeypatch):
    prepare_finalize = _TrackingNoDPEP()
    fused_out = torch.full((2, 3), 5.0)
    hidden_states = torch.zeros_like(fused_out)
    impl, _ = _make_impl(fused_out, prepare_finalize, inplace=True)
    _patch_apply_inputs(monkeypatch, impl, fused_out, fused_out_is_direct=True)

    output = _apply_test_moe(impl, hidden_states)

    assert output is hidden_states
    assert prepare_finalize.finalize_calls == 1
    torch.testing.assert_close(hidden_states, fused_out)


def test_apply_uses_generic_path_with_shared_experts_input(monkeypatch):
    prepare_finalize = _TrackingNoDPEP()
    fused_out = torch.full((2, 3), 5.0)
    impl, _ = _make_impl(fused_out, prepare_finalize)
    _patch_apply_inputs(monkeypatch, impl, fused_out, fused_out_is_direct=True)

    output = _apply_test_moe(
        impl,
        torch.zeros_like(fused_out),
        shared_experts_input=torch.empty_like(fused_out),
    )

    assert output is not fused_out
    assert prepare_finalize.finalize_calls == 1
    torch.testing.assert_close(output, fused_out)
