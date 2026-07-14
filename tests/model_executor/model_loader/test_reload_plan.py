# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.model_loader.reload.plan import (
    ReloadCapability,
    ReloadCapabilityError,
    ReloadInputError,
    ReloadLifecycleError,
    ReloadPlanBuilder,
    ReloadPlanCompileError,
    ReloadStatePolicy,
    ReloadStatePolicyError,
    ReloadStorageIdentityError,
    StaleDerivedSlotError,
    refresh_reload_derived,
)


class DerivedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.runtime_weight = self.weight.detach().clone()

    def build_reload_plan(self, builder: ReloadPlanBuilder, prefix: str) -> None:
        builder.derived(
            f"{prefix}.runtime_transform",
            owner=self,
            owner_key="runtime_transform",
            outputs=("runtime_weight",),
            depends_on=(f"{prefix}.weight",),
        )

    def refresh_runtime_weight(self) -> None:
        refresh_reload_derived(
            self,
            "runtime_transform",
            {"runtime_weight": self.weight.detach() * 2},
        )


def test_direct_input_write_preserves_storage_and_updates_value():
    model = torch.nn.Linear(4, 3, bias=False)
    plan = ReloadPlanBuilder.from_model(model).seal()
    original = model.weight

    with plan.begin(expected_inputs=("model.weight",)) as transaction:
        transaction.write("model.weight", torch.full_like(model.weight, 2))
        transaction.commit_or_raise()

    assert model.weight is original
    assert torch.equal(model.weight, torch.full_like(model.weight, 2))
    assert plan.committed_epoch == 1


def test_category1_rejects_logical_slot_rebind_before_commit():
    model = torch.nn.Linear(4, 3, bias=False)
    plan = ReloadPlanBuilder.from_model(model).seal()
    transaction = plan.begin()

    model.weight = torch.nn.Parameter(model.weight.detach().clone())

    with pytest.raises(ReloadStorageIdentityError, match="model.weight"):
        transaction.commit_or_raise()

    assert plan.active_transaction is None


def test_category2_rejects_derived_slot_not_refreshed():
    model = DerivedModel()
    plan = ReloadPlanBuilder.from_model(model).seal()
    transaction = plan.begin(expected_inputs=("model.weight",))
    transaction.write("model.weight", torch.full_like(model.weight, 3))

    with pytest.raises(StaleDerivedSlotError, match="runtime_transform"):
        transaction.commit_or_raise()


def test_derived_refresh_copies_into_original_storage_and_marks_epoch():
    model = DerivedModel()
    original_runtime_weight = model.runtime_weight
    plan = ReloadPlanBuilder.from_model(model).seal()

    with plan.begin(expected_inputs=("model.weight",)) as transaction:
        transaction.write("model.weight", torch.full_like(model.weight, 3))
        model.refresh_runtime_weight()
        transaction.commit_or_raise()

    assert model.runtime_weight is original_runtime_weight
    assert torch.equal(model.runtime_weight, torch.full_like(model.weight, 6))


def test_category3_rejects_lifecycle_operations_out_of_order():
    model = torch.nn.Linear(4, 3, bias=False)
    plan = ReloadPlanBuilder.from_model(model).seal()
    transaction = plan.begin()

    with pytest.raises(ReloadLifecycleError, match="already active"):
        plan.begin()

    transaction.start_finalizing()
    with pytest.raises(ReloadLifecycleError, match="expected LOADING"):
        transaction.write("model.weight", torch.ones_like(model.weight))

    transaction.abort()


def test_missing_expected_input_fails_closed():
    model = torch.nn.Linear(4, 3, bias=False)
    plan = ReloadPlanBuilder.from_model(model).seal()
    transaction = plan.begin(expected_inputs=("model.weight",))

    with pytest.raises(ReloadInputError, match="Missing reload inputs"):
        transaction.commit_or_raise()


def test_category4_requires_state_action_at_plan_compile_time():
    builder = ReloadPlanBuilder()

    with pytest.raises(ReloadPlanCompileError, match="requires an action"):
        builder.state("prefix_cache", policy=ReloadStatePolicy.INVALIDATE)


def test_category4_rejects_preserved_state_that_fails_validation():
    builder = ReloadPlanBuilder()
    builder.state(
        "shape_only_workspace",
        policy=ReloadStatePolicy.PRESERVE,
        validate=lambda: False,
    )
    plan = builder.seal()
    transaction = plan.begin()

    with pytest.raises(ReloadStatePolicyError, match="shape_only_workspace"):
        transaction.commit_or_raise()


def test_state_invalidation_runs_before_commit():
    state: dict[str, str | None] = {"cache": "old"}
    builder = ReloadPlanBuilder()
    builder.state(
        "prefix_cache",
        policy=ReloadStatePolicy.INVALIDATE,
        action=lambda: state.update(cache=None),
    )
    plan = builder.seal()

    with plan.begin() as transaction:
        transaction.commit_or_raise()

    assert state["cache"] is None


def test_prepare_validates_without_publishing_epoch():
    model = torch.nn.Linear(4, 3, bias=False)
    plan = ReloadPlanBuilder.from_model(model).seal()
    transaction = plan.begin(expected_inputs=("model.weight",))
    transaction.write("model.weight", torch.ones_like(model.weight))

    transaction.prepare_or_raise()

    assert plan.committed_epoch == 0
    assert plan.active_transaction is transaction
    transaction.commit_prepared()
    assert plan.committed_epoch == 1


def test_eager_only_backend_rejected_for_graph_reload_before_mutation():
    builder = ReloadPlanBuilder()
    builder.eager_only("backend replaces graph-visible storage")
    plan = builder.seal()

    assert plan.capability is ReloadCapability.EAGER_ONLY
    with pytest.raises(ReloadCapabilityError, match="replaces graph-visible"):
        plan.begin(require_graph_safe=True)

    with plan.begin(require_graph_safe=False) as transaction:
        transaction.commit_or_raise()


def test_uncertified_backend_rejected_in_eager_mode_before_mutation():
    builder = ReloadPlanBuilder()
    builder.unsupported("opaque backend has no reload participant")
    plan = builder.seal()

    assert plan.capability is ReloadCapability.UNSUPPORTED
    with pytest.raises(ReloadCapabilityError, match="opaque backend"):
        plan.begin(require_graph_safe=False)


def test_unknown_dependency_is_rejected_when_plan_is_sealed():
    owner = DerivedModel()
    builder = ReloadPlanBuilder()
    builder.derived(
        "runtime_transform",
        owner=owner,
        owner_key="runtime_transform",
        outputs=("runtime_weight",),
        depends_on=("missing.weight",),
    )

    with pytest.raises(ReloadPlanCompileError, match="unknown dependencies"):
        builder.seal()


def test_flashinfer_sinks_refresh_from_source_into_stable_storage():
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    impl = object.__new__(FlashInferImpl)
    source = torch.nn.Parameter(torch.tensor([1, 2], dtype=torch.bfloat16))
    impl.sinks = source
    FlashInferImpl.process_weights_after_loading(impl, torch.bfloat16)
    original_runtime_sinks = impl.sinks

    builder = ReloadPlanBuilder()
    builder.direct_input("sinks", source, resolver=lambda: source)
    builder.derived(
        "runtime_sinks",
        owner=impl,
        owner_key="attention_sinks",
        outputs=("sinks",),
        depends_on=("sinks",),
    )
    plan = builder.seal()

    with plan.begin(expected_inputs=("sinks",)) as transaction:
        transaction.write("sinks", torch.tensor([3, 4], dtype=torch.bfloat16))
        FlashInferImpl.process_weights_after_loading(impl, torch.bfloat16)
        transaction.commit_or_raise()

    assert impl.sinks is original_runtime_sinks
    assert impl.sinks.dtype == torch.float32
    assert torch.equal(impl.sinks, torch.tensor([3, 4], dtype=torch.float32))
