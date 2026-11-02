# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plans with explicit positions and custom visibility masks (plan kind "mask").

The kernel tests run on a GPU, or on CPU under ``TRITON_INTERPRET=1`` (set before
anything imports Triton).
"""

import dataclasses
import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tests.v1.worker.test_round_contract import (
    DEVICE,
    _chain_positions,
    _mrope_tree_batch,
    _random_batch,
    _random_tree,
    _reference_layout,
    _runner_and_batch,
    _spec_round,
    requires_kernels,
)
from tests.v1.worker.test_spec_capabilities import _TreeConfig
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadataBuilder
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.spec_decode.capabilities import (
    SpecCapabilities,
    SpecLimit,
    layer_spec_limits,
    plan_kind_rank,
    resolve_spec_resources,
)
from vllm.v1.worker.gpu.spec_decode.kernels import KernelRegistry, SpecKernel
from vllm.v1.worker.gpu.spec_decode.plan_kernels import PlanLayout
from vllm.v1.worker.gpu.spec_decode.round import (
    PlanShape,
    SpecResources,
    VerifyPlan,
)


def _plan(parent, counts, positions=None, mask=None):
    num_reqs, max_rows = parent.shape
    as_tensor = lambda x, dtype: (  # noqa: E731
        None if x is None else torch.as_tensor(x, dtype=dtype, device=DEVICE)
    )
    return VerifyPlan(
        parent=as_tensor(parent, torch.int32),
        count=as_tensor(counts, torch.int32),
        tokens=torch.zeros(num_reqs, max_rows, dtype=torch.int32, device=DEVICE),
        positions=as_tensor(positions, torch.int32),
        mask=as_tensor(mask, torch.bool),
    )


def _mask_bits(mask, counts):
    """Expected ``tree_ancestors`` of each row of a plan with a mask: its
    lower-triangular mask row and its own bit."""
    bits = np.zeros(mask.shape[:2], dtype=np.int64)
    for b, count in enumerate(counts):
        for i in range(count):
            seen = [j for j in range(i + 1) if mask[b, i, j] or j == i]
            bits[b, i] = sum(1 << j for j in seen)
    return bits


# --- plan types ---------------------------------------------------------------


def test_mask_is_the_last_plan_kind():
    assert plan_kind_rank("mask") > plan_kind_rank("tree")


def test_resources_declare_masks_on_branching_plans_only():
    resources = SpecResources(
        max_candidates=4,
        capture_shapes=(PlanShape(4),),
        max_branching=2,
        custom_mask=True,
    )
    assert resources.plan_kind == "mask"
    with pytest.raises(ValueError, match="custom_mask needs max_branching > 1"):
        SpecResources(
            max_candidates=4,
            capture_shapes=(),
            custom_mask=True,
        )


def test_plan_positions_and_mask_are_checked_against_parent():
    parent = np.array([[-1, 0, 0]], dtype=np.int32)
    _plan(parent, [3], positions=[[0, 1, 1]], mask=np.ones((1, 3, 3)))
    with pytest.raises(ValueError, match="positions has shape"):
        _plan(parent, [3], positions=[[0, 1]])
    with pytest.raises(ValueError, match="mask has shape"):
        _plan(parent, [3], mask=np.ones((1, 3, 2)))
    with pytest.raises(ValueError, match="not a chain"):
        VerifyPlan(
            parent=torch.as_tensor(parent),
            count=None,
            is_chain=True,
            positions=torch.zeros(1, 3, dtype=torch.int32),
        )


# --- layout --------------------------------------------------------------------


def _lay_out(seed, with_positions, with_mask):
    gen = np.random.default_rng(300 + seed)
    max_rows = int(gen.integers(2, 12))
    idx_mapping, qsl, num_computed, _, counts = _random_batch(gen, max_rows)
    num_reqs, num_tokens = len(counts), int(qsl[-1])
    parent = _random_tree(gen, num_reqs, max_rows)
    offsets = gen.integers(0, 3 * max_rows, (num_reqs, max_rows)).astype(np.int32)
    mask = gen.random((num_reqs, max_rows, max_rows)) < 0.5
    plan = _plan(
        parent,
        counts,
        positions=offsets if with_positions else None,
        mask=mask if with_mask else None,
    )
    chain = _chain_positions(idx_mapping, qsl, num_computed, num_tokens)
    positions = chain.clone()
    layout = PlanLayout(num_tokens + 5, max_rows, torch.device(DEVICE))
    layout.apply(
        plan,
        plan.count,
        torch.tensor(qsl, device=DEVICE),
        positions,
        torch.zeros(num_tokens + 5, dtype=torch.int32, device=DEVICE),
        num_reqs,
        num_tokens,
        num_tokens + 3,
    )
    return SimpleNamespace(
        qsl=qsl,
        counts=counts,
        parent=parent,
        offsets=offsets,
        mask=mask,
        chain=chain,
        positions=positions,
        layout=layout,
        num_tokens=num_tokens,
    )


@requires_kernels
@pytest.mark.parametrize("with_positions", [False, True])
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_plan_layout_passes_positions_and_mask_through(seed, with_positions, with_mask):
    """A row's position is the computed prefix plus its offset (else its
    depth); the rows it sees are its mask row (else its ancestors), in the bits
    of ``tree_ancestors``; every token before a plan's rows (padding included)
    sees causally (0)."""
    run = _lay_out(seed, with_positions, with_mask)
    depth, ancestors = _reference_layout(run.parent, run.counts)
    expected_pos = run.chain.cpu().numpy().copy()
    expected_seen = np.zeros(run.num_tokens + 5, dtype=np.int64)
    seen = _mask_bits(run.mask, run.counts) if with_mask else ancestors
    for b, count in enumerate(run.counts):
        start = run.qsl[b + 1] - count
        offset = run.offsets[b, :count] if with_positions else depth[b, :count]
        expected_pos[start : start + count] += offset - np.arange(count)
        # A request without drafts is a one-row plan: its row sees itself.
        expected_seen[start : start + count] = seen[b, :count]
    layout = run.layout
    n = run.num_tokens
    assert run.positions.cpu().numpy().tolist() == expected_pos.tolist()
    assert layout.tree_ancestors[: n + 3].cpu().tolist() == (
        expected_seen[: n + 3].tolist()
    )
    chain = run.chain.cpu().numpy()
    assert layout.pos_delta[:n].cpu().numpy().tolist() == (
        (expected_pos - chain)[:n].tolist()
    )
    layout.restore_positions(run.positions, n)
    assert torch.equal(run.positions, run.chain)


@requires_kernels
def test_plan_layout_mask_example():
    """Two requests, context 10 and 20: request 0 verifies a star whose leaves
    are both at offset 1 and where row 2 does not see the root; request 1
    verifies a chain-shaped plan at offsets 0, 2, 4."""
    parent = np.array([[-1, 0, 0], [-1, 0, 1]], dtype=np.int32)
    mask = np.zeros((2, 3, 3), dtype=bool)
    mask[0, 1, 0] = True  # row 1 sees the root; row 2 sees itself only
    mask[1] = np.tril(np.ones((3, 3), dtype=bool))
    plan = _plan(parent, [3, 3], positions=[[0, 1, 1], [0, 2, 4]], mask=mask)
    qsl = torch.tensor([0, 3, 6], dtype=torch.int32, device=DEVICE)
    positions = torch.tensor([10, 11, 12, 20, 21, 22], device=DEVICE)
    layout = PlanLayout(6, 3, torch.device(DEVICE))
    layout.apply(
        plan,
        plan.count,
        qsl,
        positions,
        torch.zeros(6, dtype=torch.int32, device=DEVICE),
        2,
        6,
    )
    assert positions.tolist() == [10, 11, 11, 20, 22, 24]
    assert layout.tree_ancestors.tolist() == [0b1, 0b11, 0b100, 0b1, 0b11, 0b111]


@requires_kernels
@pytest.mark.parametrize("last_parent", range(3))
def test_mrope_rows_take_the_plan_offsets_on_every_section(last_parent):
    """With explicit positions, each M-RoPE section of a verify row is its 1-D
    plan position (prefix + offset) plus the request's M-RoPE offset; the
    prefill chunk keeps its staged positions."""
    runner, batch, req_states, state, chain, counts, parent = _mrope_tree_batch(
        last_parent
    )
    offsets = np.array([[0, 0, 0, 0], [0, 3, 3, 5], [0, 1, 4, 4]], dtype=np.int32)
    mask = np.ones((3, 4, 4), dtype=bool)
    batch.verify_plan = _plan(parent, counts, positions=offsets, mask=mask)
    runner.spec_round.layout = PlanLayout(chain.shape[0], 4, torch.device(DEVICE))
    GPUModelRunner.prepare_attn(runner, batch)
    plan_pos = batch.positions.clone().cpu()
    rope = DefaultModelState.prepare_inputs(state, batch, req_states)["positions"]
    assert rope is not None
    rope = rope.clone().cpu()
    batch.tree_pos_delta = None
    chain_inputs = DefaultModelState.prepare_inputs(state, batch, req_states)
    chain_rope = chain_inputs["positions"]
    assert chain_rope is not None
    chain_rope = chain_rope.clone().cpu()

    qsl = batch.query_start_loc.tolist()
    delta = state.rope_state.prefill_delta.gpu.tolist()
    assert torch.equal(rope[:, : qsl[1]], chain_rope[:, : qsl[1]])
    for b in (1, 2):
        start, slot = qsl[b + 1] - counts[b], int(batch.idx_mapping[b])
        rows = torch.arange(start, start + counts[b])
        assert (plan_pos[rows] - plan_pos[start]).tolist() == offsets[b].tolist()
        for j in range(3):
            assert torch.equal(rope[j, rows], plan_pos[rows] + delta[slot])
    # Request 1: 30 computed, M-RoPE offset -5; request 2: 41 computed, +7.
    for j in range(3):
        assert rope[j, 5:].tolist() == [25, 28, 28, 30, 48, 49, 52, 52]
    assert batch.tree_ancestors is not None
    assert batch.tree_ancestors[5:].tolist() == [0b1, 0b11, 0b111, 0b1111] * 2


@requires_kernels
def test_runner_hands_the_mask_and_positions_to_attention():
    """The laid-out batch carries the mask in ``tree_ancestors``, and the
    attention metadata gets the plan positions."""
    parent = np.array([[-1, 0, 0, 0]], dtype=np.int32)
    mask = np.tril(np.ones((1, 4, 4), dtype=bool))
    plan = _plan(parent, [4], positions=[[0, 1, 1, 2]], mask=mask)
    qsl = torch.tensor([0, 4], dtype=torch.int32, device=DEVICE)
    positions = torch.arange(30, 34, device=DEVICE)
    runner, batch, _ = _runner_and_batch(plan, positions, qsl, max_branching=2)
    runner.spec_round.layout = PlanLayout(4, 4, torch.device(DEVICE))
    GPUModelRunner.prepare_attn(runner, batch)
    assert batch.positions.tolist() == [30, 31, 31, 32]
    assert batch.tree_ancestors.tolist() == [0b1, 0b11, 0b111, 0b1111]

    seen = {}

    def build_metadata(common_attn_metadata, ubatch_idx=None, **kwargs):
        seen["common"] = common_attn_metadata
        return "metadata"

    builder = SimpleNamespace(supports_update_block_table=False)
    group: Any = SimpleNamespace(
        layer_names=["attn"],
        kv_cache_spec=FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
        ),
        get_metadata_builder=lambda ubatch_idx=None: builder,
        build_metadata=build_metadata,
    )
    build_attn_metadata(
        attn_groups=[[group]],
        num_reqs=1,
        num_tokens=4,
        query_start_loc_gpu=qsl,
        query_start_loc_cpu=qsl.cpu(),
        max_query_len=4,
        seq_lens=torch.tensor([34], dtype=torch.int32, device=DEVICE),
        max_seq_len=34,
        block_tables=[torch.zeros(1, 4, dtype=torch.int32, device=DEVICE)],
        slot_mappings=torch.zeros(1, 4, dtype=torch.int64, device=DEVICE),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=group.kv_cache_spec)]
        ),
        positions=batch.positions,
        tree_parents=batch.tree_parents,
        tree_ancestors=batch.tree_ancestors,
    )

    common = seen["common"]
    assert common.positions.tolist() == [30, 31, 31, 32]
    assert common.tree_ancestors.tolist() == [0b1, 0b11, 0b111, 0b1111]


# --- capabilities and lowering -------------------------------------------------


class _MaskConfig(_TreeConfig):
    """A speculator whose plans carry masks."""

    def resources(self, max_num_seqs):
        return dataclasses.replace(super().resources(max_num_seqs), custom_mask=True)


def _attn():
    return SimpleNamespace(
        kv_cache_dtype="auto",
        has_sink=False,
        sliding_window=None,
        kv_sharing_target_layer_name=None,
        impl=SimpleNamespace(logits_soft_cap=None, alibi_slopes=None),
    )


_FULL = FullAttentionSpec(
    block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
)
_MAMBA = MambaSpec(block_size=16, shapes=((4, 8),), dtypes=(torch.float32,))


def _group(builder_cls, spec, layers):
    return SimpleNamespace(
        metadata_builders=[object.__new__(builder_cls)],
        kv_cache_spec=spec,
        layer_names=list(layers),
    )


def _caps(*groups, kernels=None):
    attn_groups, layers = [], {}
    for builder_cls, spec, group_layers in groups:
        attn_groups.append([_group(builder_cls, spec, group_layers)])
        layers.update(group_layers)
    return SpecCapabilities.from_limits(
        layer_spec_limits(attn_groups, layers, "slots", kernels=kernels)
    )


def _flex_layers(n=2):
    return (
        FlexAttentionMetadataBuilder,
        _FULL,
        {f"attn.{i}": _attn() for i in range(n)},
    )


def test_flex_model_keeps_mask_plans():
    caps = _caps(_flex_layers())
    assert caps.plan_kind == "mask"
    config = _MaskConfig()
    declared = config.resources(8)
    assert resolve_spec_resources(config, declared, [caps]) is declared


@pytest.mark.parametrize(
    "layer_cls, builder_cls",
    [
        (MambaMixer2, Mamba2AttentionMetadataBuilder),
        (QwenGatedDeltaNetAttention, GDNAttentionMetadataBuilder),
    ],
    ids=["mamba2", "gdn"],
)
def test_mask_plans_on_a_hybrid_model_lower_to_trees(layer_cls, builder_cls, caplog):
    """A recurrent layer folds every ancestor into a row's state, so a mask
    plan on a hybrid model runs as a tree (the recurrent verify kernel's plan
    kind decides), with the reason in the log."""
    linear = {f"linear.{i}": object.__new__(layer_cls) for i in range(3)}
    caps = _caps((builder_cls, _MAMBA, linear), _flex_layers(1))
    assert caps.plan_kind == "tree"
    [limit] = [lim for lim in caps.limits if lim.plan_kind == "tree"]
    assert "no masks" in limit.reason
    config = _MaskConfig()
    declared = config.resources(8)
    with caplog.at_level(logging.INFO, logger="vllm"):
        effective = resolve_spec_resources(config, declared, [caps])
    assert effective.plan_kind == "tree"
    assert not effective.custom_mask
    assert effective.max_branching == declared.max_branching
    assert any("mask -> tree" in r.getMessage() for r in caplog.records)


def test_a_recurrent_kernel_that_verifies_masks_keeps_them():
    """A provider's recurrent verify kernel that supports "mask" (it applies
    the mask itself) does not lower a hybrid model's mask plans."""

    class _MaskVerify(SpecKernel):
        def supports(self, layer):
            return frozenset({"chain", "tree", "mask"})

    class _Provider:
        name = "mask-verify"

        def kernels(self):
            return {("mamba:mamba2@slots", "verify"): _MaskVerify()}

    linear = {f"linear.{i}": object.__new__(MambaMixer2) for i in range(3)}
    groups = ((Mamba2AttentionMetadataBuilder, _MAMBA, linear), _flex_layers(1))
    assert _caps(*groups).plan_kind == "tree"
    caps = _caps(*groups, kernels=KernelRegistry([_Provider()]))
    assert caps.plan_kind == "mask"


def test_mask_plans_lower_to_chains_where_attention_verifies_chains():
    config = _MaskConfig()
    caps = SpecCapabilities.from_limits(
        [SpecLimit("chain", "attn:sliding_window, 2 layers: X verifies chain")]
    )
    effective = resolve_spec_resources(config, config.resources(8), [caps])
    assert effective.plan_kind == "chain"
    assert (effective.max_branching, effective.custom_mask) == (1, False)


def test_runner_refuses_a_mask_the_effective_resources_lowered():
    """After lowering to trees the speculator must not send masks; the runner
    checks plans against the effective resources, not the declared ones."""
    spec_round = _spec_round()
    parent = np.array([[-1, 0, 0, 1]], dtype=np.int32)
    plan = _plan(parent, [4], mask=np.ones((1, 4, 4), dtype=bool))
    with pytest.raises(ValueError, match="verifies no masks"):
        spec_round._check_plan(plan, 1, chain_only=False)
    big = _plan(np.full((1, 5), -1, dtype=np.int32), [5])
    with pytest.raises(ValueError, match="max_candidates 4"):
        spec_round._check_plan(big, 1, chain_only=False)
