# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The spec kernel registry: providers overlay the reference kernels by key, a
layer's plan kind is what the kernels it needs support, and binding and
capture run every bound kernel. CPU only, kernels are toys."""

import logging
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.capabilities import (
    SpecCapabilities,
    SpecLayerKind,
    layer_spec_limits,
)
from vllm.v1.worker.gpu.spec_decode.commit import ConvWindowAligner, KvCompactor
from vllm.v1.worker.gpu.spec_decode.kernels import (
    CommitStep,
    InTreeVerify,
    KernelContext,
    KernelRegistry,
    SpecKernel,
    SpecLayerSpec,
)
from vllm.v1.worker.gpu.spec_decode.round import PlanShape

GDN_VERIFY = ("mamba:gdn@slots", "verify")
MAMBA2_VERIFY = ("mamba:mamba2@slots", "verify")
SHORT_CONV_VERIFY = ("mamba:short_conv@slots", "verify")
STATE_COMMIT = ("layout:mamba_state@slots", "commit")
KV_COMMIT = ("layout:paged_token", "commit")


class ToyKernel(SpecKernel):
    """Supports ``kinds`` on layers whose name is not in ``declines``; records
    its binds and commits."""

    def __init__(self, kinds=("chain", "tree"), declines=()):
        self.kinds = frozenset(kinds)
        self.declines = frozenset(declines)
        self.binds: list[list[str]] = []
        self.steps: list[CommitStep] = []

    def supports(self, layer):
        return frozenset() if layer.name in self.declines else self.kinds

    def bind(self, ctx, layers):
        self.binds.append([layer.name for layer in layers])

    def verify(self, rows):
        raise AssertionError("not called on CPU")

    def commit(self, step):
        self.steps.append(step)


class Provider:
    def __init__(self, kernels, name=None):
        self.name = name or f"toy-{uuid.uuid4().hex[:8]}"
        self._kernels = kernels

    def kernels(self):
        return self._kernels


def _module():
    return SimpleNamespace(spec_verify_kernel=None)


def _mamba(name, variant="gdn", state_mode="slots"):
    kind = SpecLayerKind("mamba", variant, state_mode)
    return SpecLayerSpec(name, kind, _module(), "mamba_state")


def _attn(name, layout="paged_token"):
    return SpecLayerSpec(
        name, SpecLayerKind("attn", "full_attention"), _module(), layout
    )


SHAPES = (PlanShape(4), PlanShape(None))


def _ctx(plan_kind="tree", shapes=SHAPES):
    return KernelContext(
        device=torch.device("cpu"),
        plan_kind=plan_kind,
        max_num_reqs=3,
        max_candidates=4,
        capture_shapes=tuple(shapes),
    )


# --- keys --------------------------------------------------------------------


def test_reference_keys():
    """GDN and Mamba2 verify trees in tree; short conv has no tree kernel."""
    registry = KernelRegistry()
    assert isinstance(registry.get(GDN_VERIFY), InTreeVerify)
    assert isinstance(registry.get(MAMBA2_VERIFY), InTreeVerify)
    assert registry.get(SHORT_CONV_VERIFY) is None
    assert isinstance(registry.get(KV_COMMIT), KvCompactor)
    assert isinstance(registry.get(STATE_COMMIT), ConvWindowAligner)


def test_required_keys():
    assert KernelRegistry.required_keys(_mamba("m", "mamba2", "replay")) == [
        ("mamba:mamba2@replay", "verify"),
        ("layout:mamba_state@replay", "commit"),
    ]
    assert KernelRegistry.required_keys(_attn("a")) == [KV_COMMIT]
    assert KernelRegistry.required_keys(_attn("ring", layout="")) == []


def test_an_unknown_op_is_refused():
    with pytest.raises(ValueError, match="the op must be one of"):
        KernelRegistry([Provider({("mamba:gdn@slots", "prefill"): ToyKernel()})])


# --- providers overlay the reference -----------------------------------------


def test_a_partial_provider_overrides_one_key():
    toy = ToyKernel()
    registry = KernelRegistry([Provider({GDN_VERIFY: toy})])
    assert registry.get(GDN_VERIFY) is toy
    assert isinstance(registry.get(MAMBA2_VERIFY), InTreeVerify)
    assert isinstance(registry.get(KV_COMMIT), KvCompactor)
    assert isinstance(registry.get(STATE_COMMIT), ConvWindowAligner)


def test_a_later_provider_wins_a_key():
    first, second = ToyKernel(), ToyKernel()
    registry = KernelRegistry(
        [Provider({GDN_VERIFY: first}), Provider({GDN_VERIFY: second})]
    )
    assert registry.get(GDN_VERIFY) is second


def test_each_override_is_logged_once(caplog):
    provider = Provider({GDN_VERIFY: ToyKernel(), SHORT_CONV_VERIFY: ToyKernel()})
    with caplog.at_level(logging.INFO, logger="vllm"):
        KernelRegistry([provider])
        KernelRegistry([provider])
    lines = [r.getMessage() for r in caplog.records if provider.name in r.getMessage()]
    assert len(lines) == 2
    assert "mamba:gdn@slots/verify" in lines[0]
    assert "mamba:short_conv@slots/verify" in lines[1]


# --- plan kinds --------------------------------------------------------------


def test_reference_plan_kinds():
    registry = KernelRegistry()
    assert registry.plan_kind(_mamba("m", "gdn"))[0] == "tree"
    assert registry.plan_kind(_mamba("m", "mamba2"))[0] == "tree"
    assert registry.plan_kind(_attn("a"))[0] == "mask"
    # A cache not paged by slot keeps its own commit: nothing to need.
    assert registry.plan_kind(_attn("ring", layout=""))[0] == "mask"


@pytest.mark.parametrize(
    "layer, why",
    [
        (_mamba("m", "short_conv"), "no mamba:short_conv@slots/verify kernel"),
        (_mamba("m", "gdn", "replay"), "no mamba:gdn@replay/verify kernel"),
        (_mamba("m", "mamba2", "replay"), "no mamba:mamba2@replay/verify kernel"),
        (_attn("a", layout="paged_split_scale"), "no layout:paged_split_scale/commit"),
    ],
    ids=["short-conv", "gdn-replay", "mamba2-replay", "nvfp4-kv"],
)
def test_a_missing_key_lowers_the_layer_to_chain(layer, why):
    plan_kind, reason = KernelRegistry().plan_kind(layer)
    assert plan_kind == "chain"
    assert why in reason


def test_a_provider_for_the_missing_key_raises_the_layer():
    registry = KernelRegistry([Provider({SHORT_CONV_VERIFY: ToyKernel()})])
    assert registry.plan_kind(_mamba("m", "short_conv"))[0] == "tree"


def test_a_kernel_declining_a_layer_lowers_it_to_chain():
    toy = ToyKernel(declines={"m.1"})
    registry = KernelRegistry([Provider({GDN_VERIFY: toy})])
    assert registry.plan_kind(_mamba("m.0"))[0] == "tree"
    assert registry.plan_kind(_mamba("m.1")) == (
        "chain",
        "ToyKernel declines the layer",
    )


def test_the_layer_takes_the_lowest_kind_of_its_kernels():
    """Verify takes trees, the state commit chains only: the layer chains."""
    registry = KernelRegistry([Provider({STATE_COMMIT: ToyKernel(kinds={"chain"})})])
    assert registry.plan_kind(_mamba("m")) == ("chain", "ToyKernel verifies chain")


def test_the_registry_sets_the_model_plan_kind():
    """layer_spec_limits reads the recurrent layers' plan kind from the
    registry: a provider declining one layer lowers the whole model."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    layers = {
        f"linear.{i}": object.__new__(QwenGatedDeltaNetAttention) for i in range(2)
    }
    group: Any = SimpleNamespace(
        metadata_builders=[object.__new__(GDNAttentionMetadataBuilder)],
        kv_cache_spec=MambaSpec(
            block_size=16, shapes=((4, 8),), dtypes=(torch.float32,)
        ),
        layer_names=list(layers),
    )

    def caps(registry):
        return SpecCapabilities.from_limits(
            layer_spec_limits([[group]], layers, "slots", kernels=registry)
        )

    assert caps(KernelRegistry()).plan_kind == "tree"
    declining = KernelRegistry(
        [Provider({GDN_VERIFY: ToyKernel(declines={"linear.1"})})]
    )
    lowered = caps(declining)
    assert lowered.plan_kind == "chain"
    assert [limit.reason for limit in lowered.limits] == [
        "mamba:gdn@slots, 1 layer: ToyKernel verifies tree"
        " (a recurrent state folds in every ancestor: no masks)",
        "mamba:gdn@slots, 1 layer: ToyKernel declines the layer",
    ]


# --- bind, commit, capture ---------------------------------------------------


def _toy_registry():
    verify, state, kv = ToyKernel(), ToyKernel(), ToyKernel()
    registry = KernelRegistry(
        [Provider({GDN_VERIFY: verify, STATE_COMMIT: state, KV_COMMIT: kv})]
    )
    layers = [_mamba("m.0"), _mamba("m.1"), _attn("a.0"), _attn("ring", layout="")]
    return registry, layers, verify, state, kv


def test_bind_binds_each_kernel_once_to_its_layers():
    registry, layers, verify, state, kv = _toy_registry()
    registry.bind(_ctx("tree"), layers)
    assert verify.binds == [["m.0", "m.1"]]
    assert state.binds == [["m.0", "m.1"]]
    assert kv.binds == [["a.0"]]
    assert all(layer.module.spec_verify_kernel is verify for layer in layers[:2])


def test_a_chain_binds_the_verify_kernels_only():
    registry, layers, verify, state, kv = _toy_registry()
    registry.bind(_ctx("chain"), layers)
    assert verify.binds == [["m.0", "m.1"]] and not state.binds and not kv.binds
    assert layers[0].module.spec_verify_kernel is verify
    registry.capture(_ctx("chain"))
    assert not state.steps and not kv.steps


def test_no_spec_decode_unbinds_the_verify_kernel():
    registry, layers, verify, *_ = _toy_registry()
    registry.bind(_ctx("tree"), layers)
    registry.bind(_ctx("none"), layers)
    assert layers[0].module.spec_verify_kernel is None


def test_the_reference_verify_leaves_the_layer_on_its_own_kernels():
    layer = _mamba("m.0")
    layer.module.spec_verify_kernel = "stale"
    registry = KernelRegistry([Provider({STATE_COMMIT: ToyKernel()})])
    registry.bind(_ctx("tree"), [layer])
    assert layer.module.spec_verify_kernel is None


def test_commit_runs_every_bound_commit_kernel():
    registry, layers, _, state, kv = _toy_registry()
    registry.bind(_ctx("tree"), layers)
    step = CommitStep(result=None, num_reqs=2, count=None, seq_lens=None)
    registry.commit(step)
    assert state.steps == [step] and kv.steps == [step]


def test_capture_runs_every_bound_kernel_for_every_shape():
    shapes = (PlanShape(4), PlanShape(2), PlanShape(None))
    ctx = _ctx("tree", shapes)
    registry, layers, _, state, kv = _toy_registry()
    registry.bind(ctx, layers)
    runs = []
    registry.capture(ctx, run=lambda fn: runs.append(fn()))
    assert len(runs) == len(shapes)
    for kernel in (state, kv):
        assert [step.count.tolist() for step in kernel.steps] == [
            [4, 4, 4],
            [2, 2, 2],
            [4, 4, 4],  # varlen: up to max_candidates rows
        ]
        for step in kernel.steps:
            assert step.num_reqs == ctx.max_num_reqs
            # Identity paths: each accepted row stays where it is.
            assert step.result.path.tolist() == [[0, 1, 2, 3]] * 3
            assert torch.equal(step.result.path_len, step.count)


def test_the_runner_binds_the_kernels_with_its_resources():
    """set_spec_resources binds every kernel before capture, with the engine
    core's resources; capture_model then runs them for each capture shape."""
    from vllm.v1.worker.gpu.spec_decode.round import SpecResources
    from vllm.v1.worker.gpu.spec_decode.round_runner import SpecRound

    registry, layers, verify, state, kv = _toy_registry()
    declared = SpecResources(max_candidates=4, capture_shapes=(), max_branching=4)
    spec_round = SpecRound(
        None,  # type: ignore[arg-type]
        declared,
        max_num_reqs=3,
        max_num_tokens=12,
        num_speculative_steps=3,
        device=torch.device("cpu"),
    )
    spec_round.kernels = registry
    resources = SpecResources(
        max_candidates=4,
        capture_shapes=(PlanShape(4), PlanShape(None)),
        max_branching=2,
    )
    spec_round.set_resources(resources, SimpleNamespace(), layers)
    ctx = spec_round.kernel_context
    assert ctx is not None
    assert (ctx.plan_kind, ctx.max_candidates, ctx.capture_shapes) == (
        "tree",
        4,
        resources.capture_shapes,
    )
    assert verify.binds == state.binds == [["m.0", "m.1"]]
    assert kv.binds == [["a.0"]]
