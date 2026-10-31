# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The speculative round contract: plan types, plan layout, commit along a path.

The kernel tests run on a GPU, or on CPU under ``TRITON_INTERPRET=1`` (set before
anything imports Triton).
"""

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.utils_tree import depths_and_ancestors, path_to, random_parents
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.platforms import current_platform
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.commit import (
    ConvWindowAligner,
    KvCompactor,
    accept_result_of,
    accepted_rows,
    kv_cache_layout,
)
from vllm.v1.worker.gpu.spec_decode.kernels import (
    CommitStep,
    KernelContext,
    KernelRegistry,
    SpecLayerSpec,
)
from vllm.v1.worker.gpu.spec_decode.plan_kernels import (
    PlanLayout,
    chain_parents,
    chain_path,
)
from vllm.v1.worker.gpu.spec_decode.round import (
    AcceptResult,
    PlanShape,
    SpecResources,
    VerifyPlan,
)
from vllm.v1.worker.gpu.spec_decode.round_runner import SpecRound, spec_kernel_layers
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

HAS_GPU = current_platform.is_cuda_alike() or current_platform.is_xpu()
DEVICE = current_platform.device_type if HAS_GPU else "cpu"
requires_kernels = pytest.mark.skipif(
    not HAS_GPU and os.environ.get("TRITON_INTERPRET") != "1",
    reason="Triton kernels need a GPU or TRITON_INTERPRET=1",
)


def test_plans_state_their_shape():
    resources = SpecResources(
        max_candidates=4, capture_shapes=(PlanShape(4), PlanShape(None))
    )
    assert resources.max_branching == 1
    parent = torch.tensor([[-1, 0, 0, 1]], dtype=torch.int32)
    plan = VerifyPlan(
        parent=parent,
        count=torch.tensor([4], dtype=torch.int32),
        tokens=torch.zeros(1, 4, dtype=torch.int32),
    )
    assert not plan.is_chain
    result = AcceptResult(
        path=torch.tensor([[0, 2]], dtype=torch.int32),
        path_len=torch.tensor([2], dtype=torch.int32),
        sampled=torch.tensor([[5, 6]]),
    )
    assert result.processed_logits is None


def test_only_a_chain_plan_may_leave_its_rows_to_the_runner():
    parent = torch.tensor([[-1, 0, 1]], dtype=torch.int32)
    # The runner's own chain plan: counts and tokens are in its buffers.
    assert VerifyPlan(parent=parent, count=None, is_chain=True).tokens is None
    with pytest.raises(ValueError, match="count and tokens"):
        VerifyPlan(parent=parent, count=None)
    with pytest.raises(ValueError, match="count and tokens"):
        VerifyPlan(parent=parent, count=torch.tensor([3], dtype=torch.int32))


def _random_batch(gen: np.random.Generator, max_rows: int):
    """A batch of verifying and non-verifying requests in shuffled request slots:
    ``(idx_mapping, query_start_loc, num_computed_tokens, query_lens, counts)``."""
    num_reqs = int(gen.integers(1, 9))
    counts = gen.integers(1, max_rows + 1, num_reqs)
    # A request that does not verify (count 1) may be a prefill chunk.
    query_lens = np.where(counts > 1, counts, gen.integers(1, 7, num_reqs))
    query_start_loc = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    num_slots = num_reqs + 3
    idx_mapping = gen.permutation(num_slots)[:num_reqs]
    num_computed = gen.integers(0, 200, num_slots).astype(np.int32)
    return idx_mapping, query_start_loc, num_computed, query_lens, counts


def _chain_positions(idx_mapping, query_start_loc, num_computed, num_tokens):
    positions = torch.full((num_tokens + 5,), -7, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.zeros(len(idx_mapping) + 2, dtype=torch.int32, device=DEVICE)
    prepare_pos_seq_lens(
        torch.tensor(idx_mapping, dtype=torch.int32, device=DEVICE),
        torch.tensor(query_start_loc, device=DEVICE),
        torch.tensor(num_computed, device=DEVICE),
        positions,
        seq_lens,
    )
    return positions


def _random_tree(gen: np.random.Generator, num_reqs: int, max_rows: int):
    return np.array(
        [random_parents(gen, max_rows) for _ in range(num_reqs)], dtype=np.int32
    )


def _reference_layout(parent, counts):
    depth = np.zeros_like(parent)
    ancestors = np.zeros(parent.shape, dtype=np.int64)
    for b, count in enumerate(counts):
        depth[b, :count], ancestors[b, :count] = depths_and_ancestors(parent[b, :count])
    return depth, ancestors


@requires_kernels
@pytest.mark.parametrize("seed", range(6))
def test_plan_layout_of_a_chain_keeps_todays_positions(seed):
    """Laid out by the plan kernel, a chain's positions are the ones the runner
    computes today, and each row's ancestors are the rows before it."""
    gen = np.random.default_rng(seed)
    max_rows = int(gen.integers(2, 9))
    idx_mapping, qsl, num_computed, _, counts = _random_batch(gen, max_rows)
    num_tokens = int(qsl[-1])
    today = _chain_positions(idx_mapping, qsl, num_computed, num_tokens)
    positions = today.clone()
    num_reqs = len(counts)
    plan = VerifyPlan(
        parent=chain_parents(num_reqs, max_rows, torch.device(DEVICE)),
        count=torch.tensor(counts, dtype=torch.int32, device=DEVICE),
        tokens=torch.zeros(num_reqs, max_rows, dtype=torch.int32, device=DEVICE),
    )
    layout = PlanLayout(num_tokens + 5, max_rows, torch.device(DEVICE))
    layout.apply(
        plan,
        plan.count,
        torch.tensor(qsl, device=DEVICE),
        positions,
        torch.zeros(num_tokens + 5, dtype=torch.int32, device=DEVICE),
        num_reqs,
        num_tokens,
    )

    assert torch.equal(positions, today)
    tree_ancestors = layout.tree_ancestors.cpu().numpy()
    for b, count in enumerate(counts):
        start = qsl[b + 1] - count
        rows = np.arange(count)
        assert (
            tree_ancestors[start : start + count].tolist()
            == ((1 << (rows + 1)) - 1).tolist()
        )


def _runner_and_batch(plan, positions, query_start_loc, max_branching=1):
    """A runner whose block tables record the positions slot mappings are computed
    from, and a batch carrying ``plan``."""
    seen = {}

    def compute_slot_mappings(idx_mapping, qsl, positions, num_tokens_padded):
        seen["positions"] = positions.clone()
        return "slot_mappings"

    num_tokens = positions.shape[0]
    max_rows = 4
    runner = SimpleNamespace(
        pcp_manager=None,
        block_tables=SimpleNamespace(
            gather_block_tables=lambda *args, **kwargs: ("block_table",),
            compute_slot_mappings=compute_slot_mappings,
        ),
        req_states=SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=None)),
        model_state=SimpleNamespace(rope_state=None),
        spec_round=_spec_round(max_branching, num_tokens, max_rows),
        input_buffers=SimpleNamespace(tree_parents=None, tree_ancestors=None),
    )
    runner.spec_round.init_tree_verify(runner.input_buffers)
    batch = SimpleNamespace(
        verify_plan=plan,
        idx_mapping=None,
        num_reqs=query_start_loc.shape[0] - 1,
        num_reqs_after_padding=query_start_loc.shape[0] - 1,
        query_start_loc=query_start_loc,
        positions=positions,
        input_ids=torch.full((num_tokens,), 7, dtype=torch.int32, device=DEVICE),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        tree_parents=None,
        tree_ancestors=None,
        tree_pos_delta=None,
    )
    return runner, batch, seen


@pytest.mark.parametrize("with_drafts", [False, True])
def test_runner_chain_plan_changes_nothing(with_drafts):
    """A speculator that never branches: the chain plan the runner builds from its
    drafts leaves positions and slot mappings as they were and launches nothing."""
    plan = (
        VerifyPlan(
            parent=chain_parents(8, 4, torch.device(DEVICE)), count=None, is_chain=True
        )
        if with_drafts
        else None
    )
    positions = torch.arange(10, 22, device=DEVICE)
    qsl = torch.tensor([0, 4, 12], dtype=torch.int32, device=DEVICE)
    runner, batch, seen = _runner_and_batch(plan, positions.clone(), qsl)

    assert GPUModelRunner.prepare_attn(runner, batch) == (
        ("block_table",),
        "slot_mappings",
    )
    assert torch.equal(seen["positions"], positions)
    assert torch.equal(batch.positions, positions)
    assert batch.tree_parents is None and batch.tree_ancestors is None


@requires_kernels
def test_runner_lays_out_a_tree_after_the_chain_slot_mappings():
    positions = torch.arange(10, 16, device=DEVICE)
    qsl = torch.tensor([0, 2, 6], dtype=torch.int32, device=DEVICE)
    parent = torch.tensor([[-1, 0, 0, 0], [-1, 0, 0, 1]], dtype=torch.int32)
    tokens = torch.tensor([[0, 0, 0, 0], [0, 31, 32, 33]], dtype=torch.int32)
    plan = VerifyPlan(
        parent=parent.to(DEVICE),
        count=torch.tensor([1, 4], dtype=torch.int32, device=DEVICE),
        tokens=tokens.to(DEVICE),
    )
    runner, batch, seen = _runner_and_batch(
        plan, positions.clone(), qsl, max_branching=2
    )

    GPUModelRunner.prepare_attn(runner, batch)
    assert torch.equal(seen["positions"], positions)
    # Request 1 verifies rows 2..5: root at 12, two children at 13, a grandchild at 14.
    assert batch.positions.tolist() == [10, 11, 12, 13, 13, 14]
    # The candidates' tokens come from the plan; the root keeps its token.
    assert batch.input_ids.tolist() == [7, 7, 7, 31, 32, 33]
    # Plan rows are the last ``count`` tokens of each query; request 0's first
    # token is in front of its plan and keeps the chain layout (causal, bits 0).
    assert batch.tree_parents.tolist() == [-1, 0, -1, 0, 0, 1]
    assert batch.tree_ancestors.tolist() == [0, 0b1, 0b1, 0b11, 0b101, 0b1011]
    runner.spec_round.restore(batch)
    assert torch.equal(batch.positions, positions)


@requires_kernels
def test_runner_lays_out_chains_too_when_the_speculator_branches():
    """With max_branching > 1 a chain (or a batch without drafts) is laid out as
    the chain of its query rows, so the captured tree graph replays current
    metadata: parents i - 1 and no ancestor bits (causal)."""
    positions = torch.arange(10, 17, device=DEVICE)
    qsl = torch.tensor([0, 2, 7], dtype=torch.int32, device=DEVICE)
    runner, batch, _ = _runner_and_batch(None, positions.clone(), qsl, max_branching=3)

    GPUModelRunner.prepare_attn(runner, batch)
    assert torch.equal(batch.positions, positions)
    assert batch.tree_parents.tolist() == [-1, 0, -1, 0, 1, 2, 3]
    assert batch.tree_ancestors.tolist() == [0] * 7


def _rope_state(num_dims, max_num_reqs, max_model_len, num_tokens, prefill, delta):
    """A RopeState with ``prefill[r]`` (``[num_dims, len]``) staged per request and
    decode offsets ``delta``. Built without ``__init__``, which pins host memory."""
    rope = object.__new__(RopeState)
    rope.num_dims, rope.max_model_len = num_dims, max_model_len
    staged = torch.zeros(
        max_num_reqs * num_dims, max_model_len, dtype=torch.int32, device=DEVICE
    )
    for r, pos in prefill.items():
        staged[r * num_dims : (r + 1) * num_dims, : pos.shape[1]] = pos
    rope.prefill_positions = SimpleNamespace(gpu=staged)
    rope.prefill_delta = SimpleNamespace(
        gpu=torch.tensor(delta, dtype=torch.int32, device=DEVICE)
    )
    rope.positions = torch.full(
        (num_dims, num_tokens + 1), -1, dtype=torch.int64, device=DEVICE
    )
    return rope


def _mrope_tree_batch(last_parent):
    """A batch of one prefill chunk and two requests verifying trees, on an M-RoPE
    model state: ``(runner, batch, req_states, model_state, chain_positions,
    counts, parent)``. ``last_parent`` is request 2's last row's parent."""
    num_dims, max_rows, max_model_len = 3, 4, 64
    counts = [1, 4, 4]
    query_lens = [5, 4, 4]  # request 0: a 5-token prefill chunk; 1, 2: decodes
    qsl = torch.tensor(np.cumsum([0, *query_lens]), dtype=torch.int32, device=DEVICE)
    num_tokens = sum(query_lens)
    idx_mapping = torch.tensor([2, 0, 1], dtype=torch.int32, device=DEVICE)
    prefill_len = torch.tensor([6, 9, 20], dtype=torch.int32, device=DEVICE)
    num_computed = torch.tensor([30, 41, 3], dtype=torch.int32, device=DEVICE)
    # Distinct sections in request slot 2's prompt (an image), so a prefill row
    # read from the wrong place or moved by a delta shows.
    prompt = torch.stack(
        [100 * (j + 1) + torch.arange(20, dtype=torch.int32) for j in range(num_dims)]
    )
    rope = _rope_state(num_dims, 3, max_model_len, num_tokens, {2: prompt}, [-5, 7, 0])
    parent = np.full((3, max_rows), -1, dtype=np.int32)
    parent[1:, 1:] = [[0, 0, 1], [0, 1, last_parent]]
    plan = VerifyPlan(
        parent=torch.tensor(parent, device=DEVICE),
        count=torch.tensor(counts, dtype=torch.int32, device=DEVICE),
        tokens=torch.zeros(3, max_rows, dtype=torch.int32, device=DEVICE),
    )
    chain = _chain_positions(
        idx_mapping.cpu().numpy(),
        qsl.cpu().numpy(),
        num_computed.cpu().numpy(),
        num_tokens,
    )[:num_tokens]
    runner, batch, _ = _runner_and_batch(plan, chain.clone(), qsl, max_branching=2)
    batch.idx_mapping = idx_mapping
    req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=prefill_len),
        num_computed_tokens=SimpleNamespace(gpu=num_computed),
    )
    model_state = SimpleNamespace(rope_state=rope)
    return runner, batch, req_states, model_state, chain, counts, parent


def _paths(parent_row, count):
    """Every root-to-leaf path of a request's plan, as plan rows."""
    children = {
        i: [c for c in range(1, count) if parent_row[c] == i] for i in range(count)
    }
    paths, stack = [], [[0]]
    while stack:
        path = stack.pop()
        kids = children[path[-1]]
        if not kids:
            paths.append(path)
        stack.extend(path + [c] for c in kids)
    return paths


@requires_kernels
@pytest.mark.parametrize("last_parent", range(3))
def test_mrope_tree_positions_are_the_chain_positions_of_each_path(last_parent):
    """M-RoPE positions of a laid-out plan: every section of a verify row is its
    1-D plan position plus the request's M-RoPE offset, and along each
    root-to-leaf path they are the positions that path verified as a chain gets.
    Prefill rows keep their staged positions."""
    runner, batch, req_states, state, chain, counts, parent = _mrope_tree_batch(
        last_parent
    )
    GPUModelRunner.prepare_attn(runner, batch)
    plan_pos = batch.positions.clone()
    rope = DefaultModelState.prepare_inputs(state, batch, req_states)["positions"]
    assert rope is not None
    rope = rope.clone().cpu()
    # The same request rows verified as a chain: no plan delta.
    batch.tree_pos_delta = None
    chain_inputs = DefaultModelState.prepare_inputs(state, batch, req_states)
    chain_rope = chain_inputs["positions"]
    assert chain_rope is not None
    chain_rope = chain_rope.clone().cpu()

    qsl = batch.query_start_loc.tolist()
    delta = state.rope_state.prefill_delta.gpu.tolist()
    for b, count in enumerate(counts):
        start = qsl[b + 1] - count
        slot = int(batch.idx_mapping[b])
        if b == 0:  # prefill chunk: staged positions, untouched by the plan
            assert torch.equal(rope[:, qsl[0] : qsl[1]], chain_rope[:, : qsl[1]])
            assert rope[:, 0].tolist() == [103, 203, 303]
            continue
        rows = torch.arange(start, start + count)
        for j in range(rope.shape[0]):
            assert torch.equal(rope[j, rows], plan_pos[rows].cpu() + delta[slot])
        for path in _paths(parent[b], count):
            along = torch.tensor([start + r for r in path])
            assert torch.equal(
                rope[:, along], chain_rope[:, start : start + len(path)]
            ), (b, path)


@requires_kernels
def test_mrope_tree_positions_shift_every_section_by_the_plan_depth():
    """Each section of a row's M-RoPE position moves by plan position - chain
    position, as the 1-D positions do. Example: request 1 (30 computed, offset
    -5) verifies parents [-1, 0, 0, 1]; request 2 (41 computed, offset 7)
    verifies [-1, 0, 1, 1]."""
    runner, batch, req_states, state, chain, _, _ = _mrope_tree_batch(1)
    GPUModelRunner.prepare_attn(runner, batch)
    rope = DefaultModelState.prepare_inputs(state, batch, req_states)["positions"]
    assert rope is not None
    rope = rope.clone().cpu()
    batch.tree_pos_delta = None
    shifted = DefaultModelState.prepare_inputs(state, batch, req_states)["positions"]
    assert shifted is not None
    shifted = shifted.clone().cpu()
    shifted += (batch.positions - chain).cpu()

    assert torch.equal(rope, shifted)
    for j in range(3):
        assert rope[j, 5:].tolist() == [25, 26, 26, 27, 48, 49, 50, 50]


def _spec_round(max_branching=2, max_num_tokens=16, max_rows=4, max_num_reqs=8):
    resources = SpecResources(
        max_candidates=max_rows, capture_shapes=(), max_branching=max_branching
    )
    return SpecRound(
        None,  # type: ignore[arg-type]
        resources,
        max_num_reqs,
        max_num_tokens,
        max_rows - 1,
        torch.device(DEVICE),
    )


def test_runner_tree_buffers_back_every_batch_once_the_speculator_branches():
    spec_round = _spec_round()
    buffers = SimpleNamespace(tree_parents=None, tree_ancestors=None)
    spec_round.init_tree_verify(buffers)
    assert spec_round.layout is not None
    assert buffers.tree_parents is spec_round.layout.tree_parents
    assert buffers.tree_ancestors is spec_round.layout.tree_ancestors
    chain = _spec_round(max_branching=1)
    chain_buffers = SimpleNamespace(tree_parents=None, tree_ancestors=None)
    chain.init_tree_verify(chain_buffers)
    assert chain.layout is None and chain_buffers.tree_parents is None


def _random_paths(
    gen: np.random.Generator, parent: np.ndarray, counts: np.ndarray, to_last=False
):
    """A root-to-node path in each request's tree: ``(path [R, C], path_len [R])``,
    path_len 0 for some requests (a prefill chunk that sampled nothing). With
    ``to_last`` the path ends at the request's last row."""
    num_reqs, max_rows = parent.shape
    path = np.zeros((num_reqs, max_rows), dtype=np.int32)
    path_len = np.zeros(num_reqs, dtype=np.int32)
    for b in range(num_reqs):
        node = counts[b] - 1 if to_last else int(gen.integers(0, counts[b]))
        chain = path_to(parent[b], node)
        path[b, : len(chain)] = chain
        path_len[b] = 0 if gen.random() < 0.15 else len(chain)
    return path, path_len


def _mamba_state(max_num_reqs: int) -> MambaHybridModelState:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (max_num_reqs,), 99, dtype=torch.int32, device=DEVICE
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    return state


@requires_kernels
@pytest.mark.parametrize("seed", range(4))
def test_mamba_resumes_from_the_last_row_of_the_accepted_path(seed):
    """The GDN kernels read state row num_accepted_tokens - 1, which must be the
    last accepted row; for a chain that is today's max(num_sampled, 1) - 1."""
    gen = np.random.default_rng(200 + seed)
    num_reqs, max_rows, num_slots = int(gen.integers(1, 8)), int(gen.integers(2, 9)), 12
    counts = gen.integers(1, max_rows + 1, num_reqs)
    parent = _random_tree(gen, num_reqs, max_rows)
    path, path_len = _random_paths(gen, parent, counts)
    idx = gen.permutation(num_slots)[:num_reqs].astype(np.int32)
    idx[gen.random(num_reqs) < 0.2] = -1  # filtered rows under PP
    idx_mapping = torch.tensor(idx, device=DEVICE)
    result = AcceptResult(
        path=torch.tensor(path, device=DEVICE),
        path_len=torch.tensor(path_len, device=DEVICE),
        sampled=torch.zeros(num_reqs, max_rows, dtype=torch.int64, device=DEVICE),
    )
    state = _mamba_state(num_slots)
    state.postprocess_state(
        idx_mapping, result.path_len, accepted_row=accepted_rows(result, num_reqs)
    )
    got = state.num_accepted_tokens_gpu.cpu().numpy()
    for b in range(num_reqs):
        if idx[b] >= 0:
            assert got[idx[b]] - 1 == path[b, max(path_len[b], 1) - 1]
    assert (np.delete(got, idx[idx >= 0]) == 99).all()

    # A chain path gives exactly what today's num_sampled path stores.
    chain = chain_path(num_reqs, max_rows, torch.device(DEVICE))
    chain_result = AcceptResult(chain, result.path_len, result.sampled)
    via_path, today = _mamba_state(num_slots), _mamba_state(num_slots)
    via_path.postprocess_state(
        idx_mapping, result.path_len, accepted_row=accepted_rows(chain_result, num_reqs)
    )
    today.postprocess_state(idx_mapping, result.path_len)
    assert torch.equal(via_path.num_accepted_tokens_gpu, today.num_accepted_tokens_gpu)


def test_mamba_rejects_a_tree_commit_it_cannot_follow():
    state = _mamba_state(4)
    state._align_mode = True
    with pytest.raises(NotImplementedError, match="align"):
        state.postprocess_state(
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            accepted_row=torch.zeros(1, dtype=torch.int32),
        )


def test_chain_accept_result_is_a_view_of_the_constant_path():
    chain = chain_path(8, 4, torch.device(DEVICE))
    sampled = torch.arange(12, device=DEVICE).view(3, 4)
    num_sampled = torch.tensor([4, 1, 0], dtype=torch.int32, device=DEVICE)
    output = SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=4 - num_sampled,
    )
    result = accept_result_of(output, chain)
    assert result.path.data_ptr() == chain.data_ptr()
    assert result.path.tolist() == [[0, 1, 2, 3]] * 3
    assert result.path_len is num_sampled and result.sampled is sampled


def _commit_runner(plan, model_state):
    runner = SimpleNamespace(
        spec_round=_spec_round(max_branching=1, max_rows=3, max_num_reqs=2),
        model_state=model_state,
        is_last_pp_rank=False,
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(gpu="num_computed"),
            last_sampled_tokens=None,
            all_token_ids=SimpleNamespace(gpu=None),
            total_len=SimpleNamespace(gpu=None),
            draft_tokens=None,
        ),
    )
    batch = SimpleNamespace(
        verify_plan=plan, num_reqs=2, seq_lens=torch.tensor([7, 9], dtype=torch.int32)
    )
    return runner, batch


def test_chain_commit_moves_nothing(monkeypatch):
    plan = VerifyPlan(
        parent=chain_parents(2, 3, torch.device("cpu")), count=None, is_chain=True
    )
    model_state = SimpleNamespace(postprocess_state=Mock())
    runner, batch = _commit_runner(plan, model_state)
    runner.spec_round.kernels = SimpleNamespace(
        commit=Mock(side_effect=AssertionError("no compaction"))
    )
    result = AcceptResult(
        chain_path(2, 3, torch.device("cpu")),
        torch.tensor([3, 1], dtype=torch.int32),
        torch.zeros(2, 3, dtype=torch.int64),
    )
    assert runner.spec_round.commit(batch, result) is None
    assert runner.spec_round.commit(batch, None) is None

    monkeypatch.setattr(model_runner_module, "post_update", Mock())
    GPUModelRunner.postprocess_sampled(runner, "idx", "sampled", "n", "r", None)
    # A chain passes no accepted_row: model states without recurrent state
    # keep main's postprocess_state signature.
    model_state.postprocess_state.assert_called_once_with("idx", "n", "num_computed")


def test_tree_commit_compacts_and_resumes_from_the_path(monkeypatch):
    plan = VerifyPlan(
        parent=torch.tensor([[-1, 0, 0], [-1, 0, 1]], dtype=torch.int32),
        count=torch.tensor([3, 3], dtype=torch.int32),
        tokens=torch.zeros(2, 3, dtype=torch.int32),
    )
    model_state = SimpleNamespace(postprocess_state=Mock())
    runner, batch = _commit_runner(plan, model_state)
    # The reference commit kernels as bound for a model without conv states.
    registry = KernelRegistry()
    kv_commit = registry.get(("layout:paged_token", "commit"))
    state_commit = registry.get(("layout:mamba_state@slots", "commit"))
    assert isinstance(kv_commit, KvCompactor) and state_commit is not None
    compact = Mock()
    monkeypatch.setattr(kv_commit, "commit", compact)
    registry._commits = [kv_commit, state_commit]
    runner.spec_round.kernels = registry
    runner.spec_round.kernel_context = object()
    result = AcceptResult(
        torch.tensor([[0, 2, 0], [0, 1, 2]], dtype=torch.int32),
        torch.tensor([2, 3], dtype=torch.int32),
        torch.zeros(2, 3, dtype=torch.int64),
    )
    row = runner.spec_round.commit(batch, result)
    assert row is not None
    assert row.tolist() == [2, 2]
    compact.assert_called_once_with(
        CommitStep(result, 2, count=plan.count, seq_lens=batch.seq_lens)
    )

    monkeypatch.setattr(model_runner_module, "post_update", Mock())
    GPUModelRunner.postprocess_sampled(runner, "idx", "s", "n", "r", accepted_row=row)
    model_state.postprocess_state.assert_called_once_with(
        "idx", "n", "num_computed", accepted_row=row
    )


def _caches(gen, kind, num_blocks, block_size, heads, dim, dtype):
    if kind == "packed":  # K and V in the content: [blocks, heads, block, 2 * dim]
        shape = (num_blocks, heads, block_size, 2 * dim)
    elif kind == "packed_nhd":  # an NHD physical order behind the packed view
        cache = torch.empty(num_blocks, block_size, heads, 2 * dim, dtype=dtype)
        cache = cache.permute(0, 2, 1, 3)
        cache.copy_(torch.randn(cache.shape).to(dtype))
        return cache.to(DEVICE)
    elif kind == "two_first":  # FlashAttention: [2, blocks, block, heads, dim]
        shape = (2, num_blocks, block_size, heads, dim)
    elif kind == "blocks_first":  # FlashInfer: [blocks, 2, block, heads, dim]
        shape = (num_blocks, 2, block_size, heads, dim)
    elif kind == "heads_major":  # an HND physical order behind the NHD view
        cache = torch.empty(num_blocks, 2, heads, block_size, dim, dtype=dtype)
        cache = cache.permute(0, 1, 3, 2, 4)
        cache.copy_(torch.randn(cache.shape).to(dtype))
        return cache.to(DEVICE)
    else:  # MLA: [blocks, block, dim]
        shape = (num_blocks, block_size, dim)
    return torch.randn(shape).to(dtype).to(DEVICE)


def _token_view(cache, kind, kernel_block_size):
    """``[planes, tokens, heads, dim]`` view of a cache by kernel-block token id."""
    if kind in ("packed", "packed_nhd"):
        return cache.transpose(1, 2).flatten(0, 1)[None]
    if kind == "two_first":
        return cache.flatten(1, 2)
    if kind in ("blocks_first", "heads_major"):
        return cache.transpose(0, 1).flatten(1, 2)
    return cache.flatten(0, 1)[None, :, None, :]


@requires_kernels
@pytest.mark.parametrize(
    "kind", ["packed", "packed_nhd", "two_first", "blocks_first", "heads_major", "mla"]
)
@pytest.mark.parametrize("seed", range(3))
def test_kv_compaction_moves_the_path_to_the_chain_slots(kind, seed):
    gen = np.random.default_rng(300 + seed)
    table_block_size, kernel_block_size = 8, 4
    num_table_blocks, max_rows = 24, int(gen.integers(4, 10))
    num_kernel_blocks = num_table_blocks * (table_block_size // kernel_block_size)
    heads, dim = (1, 576) if kind == "mla" else (2, 24)
    dtype = torch.bfloat16 if seed % 2 else torch.float16
    num_reqs = 5
    # Mostly full trees with paths to their last row, so most requests move KV.
    counts = np.where(
        gen.random(num_reqs) < 0.8, max_rows, gen.integers(1, max_rows + 1, num_reqs)
    ).astype(np.int32)
    parent = _random_tree(gen, num_reqs, max_rows)
    path, path_len = _random_paths(gen, parent, counts, to_last=True)
    moves = [
        (b, j) for b in range(num_reqs) for j in range(path_len[b]) if path[b, j] != j
    ]
    assert moves, "the seed must exercise a copy"
    seq_lens = gen.integers(counts, 4 * table_block_size).astype(np.int32)
    # Requests get disjoint random blocks; the tables are padded past num_reqs.
    blocks = gen.permutation(num_table_blocks).astype(np.int32)
    tables = np.full((num_reqs + 2, 5), -1, dtype=np.int32)
    tables[:num_reqs, :4] = blocks[: 4 * num_reqs].reshape(num_reqs, 4)
    layers = [
        _caches(gen, kind, num_kernel_blocks, kernel_block_size, heads, dim, dtype)
        for _ in range(2)
    ]
    before = [layer.clone() for layer in layers]
    table = torch.tensor(tables, device=DEVICE)
    compactor = _bound_compactor(layers, num_kernel_blocks, table, table_block_size)
    result = AcceptResult(
        path=torch.tensor(path, device=DEVICE),
        path_len=torch.tensor(path_len, device=DEVICE),
        sampled=torch.zeros(num_reqs, max_rows, dtype=torch.int64, device=DEVICE),
    )
    compactor.commit(
        CommitStep(
            result,
            num_reqs,
            count=torch.tensor(counts, device=DEVICE),
            seq_lens=torch.tensor(seq_lens, device=DEVICE),
        )
    )

    def token(b, position):
        block = tables[b, position // table_block_size]
        return block * table_block_size + position % table_block_size

    for layer, old in zip(layers, before):
        expected = _token_view(old.clone(), kind, kernel_block_size)
        source = _token_view(old, kind, kernel_block_size)
        for b in range(num_reqs):
            context = seq_lens[b] - counts[b]
            for j in range(1, path_len[b]):
                expected[:, token(b, context + j)] = source[
                    :, token(b, context + path[b, j])
                ]
        got = _token_view(layer, kind, kernel_block_size)
        assert torch.equal(got.cpu(), expected.cpu())


@requires_kernels
def test_kv_compaction_of_a_chain_path_changes_nothing():
    layers = [torch.randn(2, 8, 4, 2, 16).to(DEVICE)]
    before = layers[0].clone()
    table = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32, device=DEVICE)
    compactor = _bound_compactor(layers, 8, table, 4)
    result = AcceptResult(
        path=chain_path(2, 4, torch.device(DEVICE)),
        path_len=torch.tensor([4, 2], dtype=torch.int32, device=DEVICE),
        sampled=torch.zeros(2, 4, dtype=torch.int64, device=DEVICE),
    )
    compactor.commit(
        CommitStep(
            result,
            2,
            count=torch.tensor([4, 4], dtype=torch.int32, device=DEVICE),
            seq_lens=torch.tensor([7, 5], dtype=torch.int32, device=DEVICE),
        )
    )
    assert torch.equal(layers[0], before)


def test_kv_compactor_takes_block_sizes_from_the_kv_cache_group():
    """A hybrid model's attention group blocks are larger than the layer's spec
    says; the cache is allocated, and the block table kept, in the group's."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    def attention(block_size):
        return FullAttentionSpec(
            block_size=block_size, num_kv_heads=4, head_size=256, dtype=torch.bfloat16
        )

    num_blocks, group_block, kernel_block = 3, 1280, 128
    kernel_blocks = num_blocks * group_block // kernel_block
    cache = torch.zeros(kernel_blocks, 4, kernel_block, 512, dtype=torch.bfloat16)
    table = torch.zeros(2, 4, dtype=torch.int32)
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.compilation_config = SimpleNamespace(
        static_forward_context={"attn": SimpleNamespace(kv_cache=cache)}
    )
    runner.kv_cache_config = SimpleNamespace(num_blocks=num_blocks)
    group: Any = SimpleNamespace(
        layer_names=["attn"], kv_cache_spec=attention(group_block), kv_cache_group_id=0
    )
    runner.attn_groups = [[group]]
    runner.kernel_block_sizes = [kernel_block]
    runner.block_tables = SimpleNamespace(input_block_tables=[table])
    runner.cache_config = CacheConfig()

    compactor = KvCompactor()
    compactor.bind(_kernel_context(runner.device), spec_kernel_layers(runner))

    [(layout, *_, table_sizes)] = compactor.launches
    assert layout.block_size == kernel_block
    assert table_sizes.tolist() == [group_block]


def test_kv_cache_layout_refuses_what_it_cannot_read():
    with pytest.raises(NotImplementedError, match="layout"):
        kv_cache_layout(torch.zeros(3, 4, 5, 6), num_blocks=8)
    with pytest.raises(NotImplementedError, match="copy"):
        kv_cache_layout(torch.zeros(2, 8, 4, 2, 16, dtype=torch.float64), 8)


def _kernel_context(device) -> KernelContext:
    return KernelContext(
        device=device,
        plan_kind="tree",
        max_num_reqs=4,
        max_candidates=4,
        capture_shapes=(PlanShape(4),),
    )


def _bound_compactor(layers, blocks, table, block_size):
    compactor = KvCompactor()
    compactor.bind(
        _kernel_context(torch.device(DEVICE)),
        [
            _layer(f"l{i}", SimpleNamespace(kv_cache=cache), table, block_size, blocks)
            for i, cache in enumerate(layers)
        ],
    )
    return compactor


def _layer(name, module, block_table, block_size=0, num_blocks=0):
    return SpecLayerSpec(
        name,
        kind=None,
        module=module,
        block_table=block_table,
        block_size=block_size,
        num_blocks=num_blocks,
    )


def _chain_order_window_commit(window, slot, path, length, width):
    """A commit that moves the path to chain order: path row ``j`` moves from
    column ``(k-2) + path[j]`` to ``(k-2) + j``; the next step then reads the
    ``k - 1`` columns ending at ``(k-2) + length - 1``."""
    window = window.clone()
    for j in range(1, length):
        window[slot, :, width - 2 + j] = window[slot, :, width - 2 + path[j]]
    end = width - 2 + length - 1
    return window[slot, :, end - (width - 2) : end + 1]


@requires_kernels
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("seed", range(3))
def test_conv_window_alignment_reads_what_a_chain_order_commit_reads(width, seed):
    """The engine keeps the state where the last accepted row wrote it and lines
    the conv window up to end at that row; the next step reads the same k - 1
    conv inputs as after moving the path to chain order."""
    gen = np.random.default_rng(900 + seed)
    num_reqs, num_rows, dim, num_slots = 5, 6, 40, 9
    window = torch.randn(num_slots, dim, width - 2 + num_rows + 1).to(DEVICE)
    # Column 0 of each request's table row is its state slot; request 4 is idle.
    slots = [3, 1, 7, 5, 0]
    table = torch.zeros(num_reqs + 2, 4, dtype=torch.int32)
    table[:num_reqs, 0] = torch.tensor(slots)
    paths, lengths = [], []
    for _ in range(num_reqs):
        length = int(gen.integers(1, num_rows + 1))
        rows = sorted(gen.choice(np.arange(1, num_rows), length - 1, replace=False))
        paths.append([0, *map(int, rows)] + [0] * (num_rows - length))
        lengths.append(length)
    result = AcceptResult(
        torch.tensor(paths, dtype=torch.int32, device=DEVICE),
        torch.tensor(lengths, dtype=torch.int32, device=DEVICE),
        torch.zeros(num_reqs, num_rows, dtype=torch.int64, device=DEVICE),
    )
    expected = [
        _chain_order_window_commit(window, slots[b], paths[b], lengths[b], width)
        for b in range(num_reqs - 1)
    ]
    before = window.clone()

    layer = SimpleNamespace(
        kv_cache=[window if is_conv_state_dim_first() else window.mT],
        conv1d=SimpleNamespace(weight=torch.empty(dim, 1, width)),
    )
    aligner = ConvWindowAligner()
    aligner.bind(
        _kernel_context(window.device),
        [_layer("conv", layer, table.to(DEVICE))],
    )
    aligner.commit(CommitStep(result, num_reqs, count=None, seq_lens=None))

    for b in range(num_reqs - 1):
        last = paths[b][lengths[b] - 1]
        end = width - 2 + last
        assert torch.equal(
            window[slots[b], :, end - (width - 2) : end + 1], expected[b]
        )
    # The null slot (an idle request) and other slots are untouched.
    untouched = [s for s in range(num_slots) if s not in slots[:-1]]
    assert torch.equal(window[untouched], before[untouched])


# --- SpecResources and the candidate budget on the speculator config --------


class _StaticSpec:
    """The SpeculativeConfig fields the default resources and budget read."""

    num_speculative_tokens = 3
    enable_adaptive_verification = False
    num_speculative_tokens_per_batch_size = None

    def uses_dynamic_speculative_decoding(self) -> bool:
        return False


_BUILTIN_CONFIGS = [
    "ngram.config.NgramGPUSpeculatorConfig",
    "eagle.config.EagleSpeculatorConfig",
    "eagle.config.Eagle3SpeculatorConfig",
    "mtp.config.MTPSpeculatorConfig",
    "dflash.config.DFlashSpeculatorConfig",
    "dspark.config.DSparkSpeculatorConfig",
    "standalone_ar.config.StandaloneARSpeculatorConfig",
]


@pytest.mark.parametrize("qualname", _BUILTIN_CONFIGS)
def test_builtin_configs_keep_todays_resources_and_budget(qualname):
    """Built-in methods draft K per step through the default declaration:
    K + 1 candidates, the uniform K + 1 decode graph, K state rows, chains,
    and padded new decodes, at every batch size."""
    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.worker.gpu.spec_decode.round import BudgetView

    config_cls = resolve_obj_by_qualname(f"vllm.v1.worker.gpu.spec_decode.{qualname}")
    config = config_cls(_StaticSpec())
    assert config.resources(16) == SpecResources(
        max_candidates=4, capture_shapes=(PlanShape(4),)
    )
    assert config.pads_new_decode_rows
    for num_reqs in (1, 7, 16):
        assert (
            config.candidate_budget(BudgetView(num_reqs, num_reqs, 0, 0, 16, 16)) == 4
        )


def test_speculator_config_with_its_budget_state_pickles():
    """The scheduler process holds the config: it must cross process
    boundaries with the VllmConfig."""
    import pickle

    config = SpeculatorConfig(_StaticSpec())
    clone = pickle.loads(pickle.dumps(config))
    assert clone.resources(4) == config.resources(4)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(max_candidates=0, capture_shapes=()), "max_candidates"),
        (dict(max_candidates=2, capture_shapes=(PlanShape(3),)), "outside"),
        (dict(max_candidates=2, capture_shapes=(PlanShape(0),)), "outside"),
        (
            dict(max_candidates=2, capture_shapes=(), max_branching=0),
            "max_branching",
        ),
    ],
)
def test_spec_resources_reject_shapes_they_cannot_hold(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SpecResources(**kwargs)


def test_spec_resources_list_uniform_counts_and_varlen():
    resources = SpecResources(
        max_candidates=4,
        capture_shapes=(PlanShape(4), PlanShape(None), PlanShape(1)),
    )
    assert resources.uniform_counts == [1, 4]
    assert resources.varlen
