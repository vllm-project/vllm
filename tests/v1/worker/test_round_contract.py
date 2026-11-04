# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The speculative round contract: plan types, plan layout, tree accept, and the
commit along the accepted path.

A request's tree is ``parent``: ``parent[i]`` is the row that row ``i`` extends,
``-1`` for the root (row 0), and a parent always precedes its child.

The kernel tests run on a GPU, or on CPU under ``TRITON_INTERPRET=1`` (set before
anything imports Triton).
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CacheConfig
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
)
from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.spec_decode.commit import (
    KvCompactor,
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
    VerifyOutput,
    VerifyPlan,
)
from vllm.v1.worker.gpu.spec_decode.round_runner import SpecRound, spec_kernel_layers
from vllm.v1.worker.gpu.spec_decode.tree_accept import accept_greedy_tree, walk_tree

HAS_GPU = current_platform.is_cuda_alike() or current_platform.is_xpu()
DEVICE = current_platform.device_type if HAS_GPU else "cpu"
requires_kernels = pytest.mark.skipif(
    not HAS_GPU and os.environ.get("TRITON_INTERPRET") != "1",
    reason="Triton kernels need a GPU or TRITON_INTERPRET=1",
)


# --- Reference trees --------------------------------------------------------


def _random_parents(gen: np.random.Generator, num_rows: int) -> list[int]:
    """A random tree: each row extends a uniformly drawn earlier row."""
    return [-1] + [int(gen.integers(0, row)) for row in range(1, num_rows)]


def _random_tree(gen: np.random.Generator, num_reqs: int, max_rows: int):
    return np.array(
        [_random_parents(gen, max_rows) for _ in range(num_reqs)], dtype=np.int32
    )


def _reference_layout(parent, counts):
    """Each row's depth and its ancestor bitmask (bit ``j``: row ``j`` is on the
    row's path from the root, the row itself included)."""
    depth = np.zeros_like(parent)
    ancestors = np.zeros(parent.shape, dtype=np.int64)
    for b, count in enumerate(counts):
        for row in range(count):
            up = int(parent[b, row])
            depth[b, row] = 0 if row == 0 else depth[b, up] + 1
            ancestors[b, row] = (1 << row) | (0 if row == 0 else ancestors[b, up])
    return depth, ancestors


def _path_to(parent, row: int) -> list[int]:
    """The rows from the root down to ``row``."""
    path = [row]
    while int(parent[path[-1]]) >= 0:
        path.append(int(parent[path[-1]]))
    return path[::-1]


# --- Plan and resource types ------------------------------------------------


def test_plan_validation():
    parent = torch.tensor([[-1, 0, 0, 1]], dtype=torch.int32)
    plan = VerifyPlan(
        parent=parent,
        count=torch.tensor([4], dtype=torch.int32),
        tokens=torch.zeros(1, 4, dtype=torch.int32),
    )
    assert not plan.is_chain
    # Only the runner's own chain plan leaves its counts and tokens to the runner.
    assert VerifyPlan(parent=parent, count=None, is_chain=True).tokens is None
    with pytest.raises(ValueError, match="count and tokens"):
        VerifyPlan(parent=parent, count=None)
    with pytest.raises(ValueError, match="count and tokens"):
        VerifyPlan(parent=parent, count=torch.tensor([4], dtype=torch.int32))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(max_candidates=0, capture_shapes=()), "max_candidates"),
        (dict(max_candidates=2, capture_shapes=(PlanShape(3),)), "outside"),
        (dict(max_candidates=2, capture_shapes=(PlanShape(0),)), "outside"),
        (dict(max_candidates=2, capture_shapes=(), max_branching=0), "max_branching"),
        (
            dict(
                max_candidates=4,
                capture_shapes=(PlanShape(4), PlanShape(None), PlanShape(1)),
            ),
            None,
        ),
    ],
)
def test_spec_resources(kwargs, match):
    if match is not None:
        with pytest.raises(ValueError, match=match):
            SpecResources(**kwargs)
        return
    resources = SpecResources(**kwargs)
    assert resources.max_branching == 1
    assert resources.uniform_counts == [1, 4]
    assert resources.varlen


# --- Plan layout ------------------------------------------------------------


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
    """The positions the runner computes today."""
    positions = torch.full((num_tokens,), -7, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.zeros(len(idx_mapping) + 2, dtype=torch.int32, device=DEVICE)
    prepare_pos_seq_lens(
        torch.tensor(idx_mapping, dtype=torch.int32, device=DEVICE),
        torch.tensor(query_start_loc, device=DEVICE),
        torch.tensor(num_computed, device=DEVICE),
        positions,
        seq_lens,
    )
    return positions


def _runner_and_batch(plan, positions, query_start_loc, max_branching=1, max_rows=4):
    """The runner members ``prepare_attn`` reads, with block tables that record
    the positions slot mappings are computed from, and a batch carrying
    ``plan``."""
    seen = {}

    def compute_slot_mappings(idx_mapping, qsl, positions, num_tokens_padded):
        seen["positions"] = positions.clone()
        return "slot_mappings"

    num_tokens = positions.shape[0]
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


@requires_kernels
@pytest.mark.parametrize("seed", range(4))
def test_chain_layout_is_todays(seed):
    """A chain keeps today's positions. Laid out by the plan kernel, each row's
    ancestors are the rows before it; through the runner, a speculator that
    never branches launches nothing, with or without drafts."""
    gen = np.random.default_rng(seed)
    max_rows = int(gen.integers(2, 9))
    idx_mapping, qsl, num_computed, _, counts = _random_batch(gen, max_rows)
    num_tokens, num_reqs = int(qsl[-1]), len(counts)
    today = _chain_positions(idx_mapping, qsl, num_computed, num_tokens)
    positions = today.clone()
    plan = VerifyPlan(
        parent=chain_parents(num_reqs, max_rows, torch.device(DEVICE)),
        count=torch.tensor(counts, dtype=torch.int32, device=DEVICE),
        tokens=torch.zeros(num_reqs, max_rows, dtype=torch.int32, device=DEVICE),
    )
    layout = PlanLayout(num_tokens, max_rows, torch.device(DEVICE))
    layout.apply(
        plan,
        plan.count,
        torch.tensor(qsl, device=DEVICE),
        positions,
        torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE),
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

    runner_chain = VerifyPlan(
        parent=chain_parents(num_reqs, max_rows, torch.device(DEVICE)),
        count=None,
        is_chain=True,
    )
    qsl_gpu = torch.tensor(qsl, device=DEVICE)
    for runner_plan in (None, runner_chain):
        runner, batch, seen = _runner_and_batch(
            runner_plan, today.clone(), qsl_gpu, max_rows=max_rows
        )
        assert GPUModelRunner.prepare_attn(runner, batch) == (
            ("block_table",),
            "slot_mappings",
        )
        assert torch.equal(seen["positions"], today)
        assert torch.equal(batch.positions, today)
        assert batch.tree_parents is None and batch.tree_ancestors is None


@requires_kernels
@pytest.mark.parametrize("seed", range(4))
def test_tree_layout_matches_reference(seed):
    """The runner lays out random trees after the chain slot mappings. Plan rows
    are the last ``count`` tokens of each query: each candidate sits at the
    root's position plus its depth, with the plan's token, a query-local parent
    and plan-relative ancestor bits. Tokens in front of a plan keep the chain
    layout (causal, no ancestor bits). ``restore`` brings the chain back."""
    gen = np.random.default_rng(100 + seed)
    max_rows = int(gen.integers(2, 9))
    idx_mapping, qsl, num_computed, query_lens, counts = _random_batch(gen, max_rows)
    num_tokens, num_reqs = int(qsl[-1]), len(counts)
    chain = _chain_positions(idx_mapping, qsl, num_computed, num_tokens)
    parent = _random_tree(gen, num_reqs, max_rows)
    tokens = gen.integers(100, 1000, (num_reqs, max_rows)).astype(np.int32)
    plan = VerifyPlan(
        parent=torch.tensor(parent, device=DEVICE),
        count=torch.tensor(counts, dtype=torch.int32, device=DEVICE),
        tokens=torch.tensor(tokens, device=DEVICE),
    )
    runner, batch, seen = _runner_and_batch(
        plan,
        chain.clone(),
        torch.tensor(qsl, device=DEVICE),
        max_branching=2,
        max_rows=max_rows,
    )

    GPUModelRunner.prepare_attn(runner, batch)

    depth, ancestors = _reference_layout(parent, counts)
    want_pos, want_ids = chain.tolist(), [7] * num_tokens
    want_parents, want_ancestors = [], []
    for b, count in enumerate(counts):
        offset = int(query_lens[b] - count)
        root = int(qsl[b]) + offset
        for t in range(int(query_lens[b])):
            row = t - offset
            want_parents.append(t - 1 if row <= 0 else offset + parent[b, row])
            want_ancestors.append(0 if row < 0 else int(ancestors[b, row]))
            if row > 0:
                want_pos[root + row] = want_pos[root] + int(depth[b, row])
                want_ids[root + row] = int(tokens[b, row])
    assert torch.equal(seen["positions"], chain)
    assert batch.positions.tolist() == want_pos
    assert batch.input_ids.tolist() == want_ids
    assert batch.tree_parents.tolist() == want_parents
    assert batch.tree_ancestors.tolist() == want_ancestors
    runner.spec_round.restore(batch)
    assert torch.equal(batch.positions, chain)


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
    )
    runner, batch, _ = _runner_and_batch(plan, chain.clone(), qsl, max_branching=2)
    batch.idx_mapping = idx_mapping
    req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=prefill_len),
        num_computed_tokens=SimpleNamespace(gpu=num_computed),
    )
    model_state = SimpleNamespace(rope_state=rope)
    return runner, batch, req_states, model_state, chain, counts, parent


def _leaf_paths(parent_row, count):
    """Every root-to-leaf path of a request's plan, as plan rows."""
    leaves = set(range(count)) - {int(p) for p in parent_row[1:count]}
    return [_path_to(parent_row, leaf) for leaf in sorted(leaves)]


@requires_kernels
@pytest.mark.parametrize("last_parent", range(3))
def test_mrope_tree_positions(last_parent):
    """Every M-RoPE section of a verify row moves by plan position - chain
    position, as the 1-D position does, so along each root-to-leaf path the
    positions are the ones that path verified as a chain gets. Prefill rows keep
    their staged positions. Request 1 (30 computed, offset -5) verifies parents
    [-1, 0, 0, 1]; request 2 (41 computed, offset 7) [-1, 0, 1, last_parent]."""
    runner, batch, req_states, state, chain, counts, parent = _mrope_tree_batch(
        last_parent
    )
    GPUModelRunner.prepare_attn(runner, batch)
    rope = DefaultModelState.prepare_inputs(state, batch, req_states)["positions"]
    assert rope is not None
    rope = rope.clone().cpu()
    # The same request rows verified as a chain: no plan delta.
    batch.tree_pos_delta = None
    chain_rope = DefaultModelState.prepare_inputs(state, batch, req_states)["positions"]
    assert chain_rope is not None
    chain_rope = chain_rope.clone().cpu()

    assert torch.equal(rope, chain_rope + (batch.positions - chain).cpu())
    assert rope[:, 0].tolist() == [103, 203, 303]
    qsl = batch.query_start_loc.tolist()
    for b in (1, 2):
        start = qsl[b + 1] - counts[b]
        for path in _leaf_paths(parent[b], counts[b]):
            along = torch.tensor([start + r for r in path])
            assert torch.equal(
                rope[:, along], chain_rope[:, start : start + len(path)]
            ), (b, path)
    if last_parent == 1:
        for j in range(3):
            assert rope[j, 5:].tolist() == [25, 26, 26, 27, 48, 49, 50, 50]


# --- Tree accept ------------------------------------------------------------


def _reference_walk(parent, count, drafted, target):
    """Root first: the child whose drafted token equals the current row's target
    token, the lowest row on ties, until there is none."""
    path, sampled = [0], [target[0]]
    node = 0
    while True:
        children = [
            r
            for r in range(1, count)
            if parent[r] == node and drafted[r] != -1 and drafted[r] == target[node]
        ]
        if not children:
            return path, sampled
        node = children[0]
        path.append(node)
        sampled.append(target[node])


def _verify_output(drafted, row_start, logits=None, num_reqs=0):
    return VerifyOutput(
        logits=logits,
        row_start=row_start,
        idx_mapping=torch.arange(num_reqs, device=DEVICE),
        idx_mapping_np=np.arange(num_reqs),
        expanded_idx_mapping=None,
        expanded_local_pos=None,
        positions=None,
        drafted=drafted,
        draft_tokens=None,
        seq_lens_upper_bound_np=None,
    )


def _tree_plan(parent, counts):
    parent = torch.tensor(parent, dtype=torch.int32, device=DEVICE)
    return VerifyPlan(
        parent=parent,
        count=torch.tensor(counts, dtype=torch.int32, device=DEVICE),
        tokens=torch.zeros_like(parent),
    )


@requires_kernels
@pytest.mark.parametrize("seed", range(4))
def test_walk_tree_matches_reference(seed):
    """The walk follows the child holding the target token. Rows past a
    request's count and masked drafts (-1) are never accepted."""
    gen = np.random.default_rng(seed)
    num_reqs, num_cols, vocab = 7, 6, 3
    parent = _random_tree(gen, num_reqs, num_cols)
    counts = gen.integers(1, num_cols + 1, num_reqs)
    row_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    drafted = [
        -1 if r == 0 else int(gen.integers(0, vocab)) for c in counts for r in range(c)
    ]
    # Half the candidates match their parent's target, so paths grow.
    target = [int(gen.integers(0, vocab)) for _ in drafted]
    for b, count in enumerate(counts):
        for r in range(1, count):
            if gen.random() < 0.5:
                target[row_start[b] + parent[b, r]] = drafted[row_start[b] + r]
    output = _verify_output(
        torch.tensor(drafted, dtype=torch.int64, device=DEVICE),
        torch.tensor(row_start, device=DEVICE),
        num_reqs=num_reqs,
    )

    result = walk_tree(
        _tree_plan(parent, counts),
        output,
        torch.tensor(target, dtype=torch.int64, device=DEVICE),
        8,
    )

    assert result.sampled.shape == (num_reqs, 8)
    for b, count in enumerate(counts):
        rows = slice(row_start[b], row_start[b + 1])
        path, sampled = _reference_walk(parent[b], count, drafted[rows], target[rows])
        n = int(result.path_len[b])
        assert result.path[b, :n].tolist() == path
        assert result.sampled[b, :n].tolist() == sampled

    # Row 1 matches but is masked (-1); row 2 matches but is past count.
    output = _verify_output(
        torch.tensor([-1, -1], dtype=torch.int64, device=DEVICE),
        torch.tensor([0, 2], dtype=torch.int32, device=DEVICE),
        num_reqs=1,
    )
    target = torch.tensor([5, 9], dtype=torch.int64, device=DEVICE)
    result = walk_tree(_tree_plan([[-1, 0, 0]], [2]), output, target)
    assert result.path_len.tolist() == [1]
    assert result.sampled[0, 0].item() == 5


def _greedy_sampler(temperature, processed=None):
    """The two sampler members the default tree accept reads."""

    def apply_sampling_params(logits, *args):
        return logits if processed is None else processed

    return SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(np=np.asarray(temperature, dtype=np.float32))
        ),
        apply_sampling_params=apply_sampling_params,
    )


@requires_kernels
def test_default_tree_accept_walks_the_argmax():
    """Greedy requests walk the argmax of the processed logits, the ones the
    runner takes logprobs from."""
    logits = torch.zeros(4, 8, device=DEVICE)
    processed = torch.zeros(4, 8, device=DEVICE)
    processed[0, 3] = processed[2, 6] = processed[3, 1] = processed[1, 0] = 1.0
    output = _verify_output(
        torch.tensor([-1, 2, 3, 6], dtype=torch.int64, device=DEVICE),
        torch.tensor([0, 4], dtype=torch.int32, device=DEVICE),
        logits=logits,
        num_reqs=1,
    )
    plan = _tree_plan([[-1, 0, 0, 2]], [4])
    result = accept_greedy_tree(plan, output, _greedy_sampler([0.0], processed), 4)
    # Root's argmax 3 -> row 2 (token 3); its argmax 6 -> row 3; then token 1.
    assert result.path[0, :3].tolist() == [0, 2, 3]
    assert result.sampled[0, :3].tolist() == [3, 6, 1]
    assert result.processed_logits is processed


def test_default_tree_accept_refuses_sampled():
    output = _verify_output(None, None, logits=torch.zeros(2, 4), num_reqs=1)
    with pytest.raises(ValueError, match="greedy"):
        accept_greedy_tree(
            _tree_plan([[-1, 0]], [2]), output, _greedy_sampler([0.7]), 2
        )


# --- Commit: recurrent state ------------------------------------------------


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
        chain = _path_to(parent[b], node)
        path[b, : len(chain)] = chain
        path_len[b] = 0 if gen.random() < 0.15 else len(chain)
    return path, path_len


def _mamba_state(max_num_reqs: int) -> MambaHybridModelState:
    """The members ``postprocess_state`` reads; ``__init__`` needs a model."""
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
def test_mamba_resumes_from_the_last_accepted_row(seed):
    """The GDN kernels read state row num_accepted_tokens - 1, which must be the
    last accepted row; for a chain that is today's max(num_sampled, 1) - 1. A
    state that cannot follow a tree path refuses it."""
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

    aligned = _mamba_state(4)
    aligned._align_mode = True
    with pytest.raises(NotImplementedError, match="align"):
        aligned.postprocess_state(
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            accepted_row=torch.zeros(1, dtype=torch.int32),
        )


# --- Commit: KV compaction --------------------------------------------------


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


def _token_view(cache, kind):
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
@pytest.mark.parametrize("path_kind", ["tree", "chain"])
@pytest.mark.parametrize("seed", range(2))
def test_kv_compaction_moves_the_path(kind, path_kind, seed):
    """The KV of each accepted path row lands in the chain slot of its path
    index, in every cache layout; a chain path moves nothing."""
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
    if path_kind == "chain":
        path = np.tile(np.arange(max_rows, dtype=np.int32), (num_reqs, 1))
    else:
        moves = [
            (b, j)
            for b in range(num_reqs)
            for j in range(path_len[b])
            if path[b, j] != j
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
        expected = _token_view(old.clone(), kind)
        source = _token_view(old, kind)
        for b in range(num_reqs):
            context = seq_lens[b] - counts[b]
            for j in range(1, path_len[b]):
                expected[:, token(b, context + j)] = source[
                    :, token(b, context + path[b, j])
                ]
        assert torch.equal(_token_view(layer, kind).cpu(), expected.cpu())
    if path_kind == "chain":
        for layer, old in zip(layers, before):
            assert torch.equal(layer, old)


def _attention_runner(specs, modules, kernel_block_size, num_blocks):
    """The runner members ``spec_kernel_layers`` reads: one KV cache group per
    layer of ``specs``; ``modules`` are the layers in the forward context."""
    return SimpleNamespace(
        device=torch.device("cpu"),
        compilation_config=SimpleNamespace(static_forward_context=modules),
        kv_cache_config=SimpleNamespace(num_blocks=num_blocks),
        attn_groups=[
            [
                SimpleNamespace(
                    layer_names=[name], kv_cache_spec=spec, kv_cache_group_id=g
                )
            ]
            for g, (name, spec) in enumerate(specs.items())
        ],
        kernel_block_sizes=[kernel_block_size] * len(specs),
        block_tables=SimpleNamespace(
            input_block_tables=[torch.zeros(2, 4, dtype=torch.int32)] * len(specs)
        ),
        cache_config=CacheConfig(),
    )


def _attention_spec(cls, block_size=16, num_kv_heads=1, head_size=64):
    return cls(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("case", ["hybrid_group_blocks", "ring_shared_hidden"])
def test_kv_compactor_binds_the_layers_it_moves(case):
    """A hybrid model's attention group blocks are larger than the layer's spec
    says: the cache is allocated, and the block table kept, in the group's. The
    DSv4 ring (not paged by slot) keeps its own chain commit, a layer sharing
    another's KV cache moves with it, and a hidden-state cache is not a decoder
    layer: none of these is compacted on its own."""
    if case == "hybrid_group_blocks":
        num_blocks, group_block, kernel_block = 3, 1280, 128
        kernel_blocks = num_blocks * group_block // kernel_block
        cache = torch.zeros(kernel_blocks, 4, kernel_block, 512, dtype=torch.bfloat16)
        specs = {
            "attn": _attention_spec(
                FullAttentionSpec, group_block, num_kv_heads=4, head_size=256
            )
        }
        modules = {"attn": SimpleNamespace(kv_cache=cache)}
        runner = _attention_runner(specs, modules, kernel_block, num_blocks)
    else:
        cache = torch.zeros(4, 2, 16, 1, 64, dtype=torch.bfloat16)
        specs = {
            "attn": _attention_spec(FullAttentionSpec),
            "ring": _attention_spec(CircularBufferSpec),
            "hidden": _attention_spec(HiddenStateCacheSpec),
            "shared": _attention_spec(FullAttentionSpec),
        }
        # Only the paged layers have a cache the compactor can read.
        modules = {
            "attn": SimpleNamespace(kv_cache=cache),
            "shared": SimpleNamespace(
                kv_cache=cache, kv_sharing_target_layer_name="attn"
            ),
        }
        runner = _attention_runner(specs, modules, 16, 4)
    layers = spec_kernel_layers(runner)  # type: ignore[arg-type]
    registry = KernelRegistry()
    registry.bind(_kernel_context(runner.device), layers)
    compactor = registry.get(("layout:paged_token", "commit"))
    assert isinstance(compactor, KvCompactor)

    [(layout, *_, table_sizes)] = compactor.launches
    if case == "hybrid_group_blocks":
        assert layout.block_size == kernel_block
        assert table_sizes.tolist() == [group_block]
    else:
        assert [(layer.name, layer.layout) for layer in layers] == [
            ("attn", "paged_token"),
            ("ring", ""),
        ]


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


def _bound_compactor(layers, num_blocks, block_table, block_size):
    compactor = KvCompactor()
    compactor.bind(
        _kernel_context(torch.device(DEVICE)),
        [
            SpecLayerSpec(
                f"l{i}",
                kind=None,  # type: ignore[arg-type]
                module=SimpleNamespace(kv_cache=cache),
                block_table=block_table,
                block_size=block_size,
                num_blocks=num_blocks,
            )
            for i, cache in enumerate(layers)
        ],
    )
    return compactor


# --- Built-in speculators keep today's resources and budget --------------


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
