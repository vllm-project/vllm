# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The speculative round contract: plan types, plan layout, tree accept, and the
commit along the accepted path.

A request's tree is ``parent``: ``parent[i]`` is the row that row ``i`` extends,
``-1`` for the root (row 0), and a parent always precedes its child.

The kernel tests run on a GPU, or on CPU under ``TRITON_INTERPRET=1``.
"""

import os

import numpy as np
import pytest
import torch

from vllm.config import CacheConfig, CompilationConfig
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates
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
    BudgetView,
    PlanShape,
    SpecResources,
    VerifyOutput,
    VerifyPlan,
)
from vllm.v1.worker.gpu.spec_decode.round_runner import SpecRound, spec_kernel_layers
from vllm.v1.worker.gpu.spec_decode.tree_accept import accept_greedy_tree, walk_tree
from vllm.v1.worker.utils import AttentionGroup

HAS_GPU = current_platform.is_cuda_alike() or current_platform.is_xpu()
DEVICE = torch.device(current_platform.device_type if HAS_GPU else "cpu")
requires_kernels = pytest.mark.skipif(
    not HAS_GPU and os.environ.get("TRITON_INTERPRET") != "1",
    reason="Triton kernels need a GPU or TRITON_INTERPRET=1",
)


def _t(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype, device=DEVICE)


def _random_tree(gen, num_reqs, max_rows):
    """Each row extends a uniformly drawn earlier row."""
    rows = [
        [-1] + [int(gen.integers(0, r)) for r in range(1, max_rows)]
        for _ in range(num_reqs)
    ]
    return np.array(rows, dtype=np.int32)


def _path_to(parent, row):
    """The rows from the root down to ``row``."""
    path = [row]
    while int(parent[path[-1]]) >= 0:
        path.append(int(parent[path[-1]]))
    return path[::-1]


def _plan(parent, counts, tokens=None):
    parent = _t(parent)
    return VerifyPlan(
        parent=parent,
        count=_t(counts),
        tokens=torch.zeros_like(parent) if tokens is None else _t(tokens),
    )


# --- Plan and resource types ------------------------------------------------

_PARENT = [[-1, 0, 0, 1]]


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: VerifyPlan(parent=_t(_PARENT), count=None), "count and tokens"),
        (lambda: VerifyPlan(parent=_t(_PARENT), count=_t([4])), "count and tokens"),
        (lambda: SpecResources(max_candidates=0, capture_shapes=()), "max_candidates"),
        (
            lambda: SpecResources(max_candidates=2, capture_shapes=(PlanShape(3),)),
            "outside",
        ),
        (
            lambda: SpecResources(max_candidates=2, capture_shapes=(), max_branching=0),
            "max_branching",
        ),
    ],
    ids=["plan-no-count", "plan-no-tokens", "candidates", "shape", "branching"],
)
def test_plan_and_resources_are_validated(make, match):
    with pytest.raises(ValueError, match=match):
        make()


@pytest.mark.parametrize("is_chain", [False, True])
@pytest.mark.parametrize(
    "max_candidates, rows, match",
    [
        (3, [4], "max_candidates 3"),
        (4, [4, 4], "every request"),
        (8, [5], "verifies up to 5"),
    ],
)
def test_spec_round_checks_a_plan_with_counts(is_chain, max_candidates, rows, match):
    """A plan with counts is laid out from its parents, so its shape is
    checked even when it is a chain: it fits the resources and has a column
    for every row it verifies."""
    spec_round = object.__new__(SpecRound)
    spec_round.resources = SpecResources(
        max_candidates=max_candidates, capture_shapes=()
    )
    plan = _plan(_PARENT, [4])
    plan.is_chain = is_chain
    with pytest.raises(ValueError, match=match):
        spec_round._check_plan(plan, np.array(rows), chain_only=False)


def test_valid_plan_and_resources():
    # Only the runner's own chain plan leaves its counts and tokens to it.
    assert VerifyPlan(parent=_t(_PARENT), count=None, is_chain=True).tokens is None
    assert not _plan(_PARENT, [4]).is_chain
    resources = SpecResources(
        max_candidates=4, capture_shapes=(PlanShape(4), PlanShape(None), PlanShape(1))
    )
    assert resources.max_branching == 1
    assert resources.uniform_counts == [1, 4]
    assert resources.varlen


# --- Plan layout ------------------------------------------------------------


def _random_batch(gen, max_rows):
    """Verifying and non-verifying requests (count 1, maybe a prefill chunk):
    ``(query_start_loc, query_lens, counts)``."""
    num_reqs = int(gen.integers(1, 9))
    counts = gen.integers(1, max_rows + 1, num_reqs)
    query_lens = np.where(counts > 1, counts, gen.integers(1, 7, num_reqs))
    qsl = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    return qsl, query_lens, counts


@requires_kernels
@pytest.mark.parametrize("kind", ["tree", "bad-tree", "chain-as-tree", "runner-chain"])
@pytest.mark.parametrize("seed", range(3))
def test_layout_matches_reference(kind, seed):
    """Plan rows are the last ``count`` tokens of each query: each sits at the
    root's position plus its depth, with the plan's token, a query-local parent
    and plan-relative ancestor bits. Tokens in front of a plan, and every token
    of the runner's chain, attend causally (no ancestor bits). A parent that is
    not an earlier row is a chain step. ``restore`` brings the chain positions
    back."""
    gen = np.random.default_rng(seed)
    max_rows = int(gen.integers(2, 9))
    qsl, query_lens, counts = _random_batch(gen, max_rows)
    num_reqs, num_tokens = len(counts), int(qsl[-1])
    if kind in ("tree", "bad-tree"):
        parent = _random_tree(gen, num_reqs, max_rows)
    else:
        parent = chain_parents(num_reqs, max_rows, DEVICE).cpu().numpy()
    tokens = gen.integers(100, 1000, (num_reqs, max_rows)).astype(np.int32)
    if kind == "runner-chain":
        plan = VerifyPlan(parent=_t(parent), count=None, is_chain=True)
    elif kind == "bad-tree":
        # Parents at their own row or later are laid out as chain steps.
        rows = np.arange(max_rows)
        bad = gen.random(parent.shape) < 0.5
        late = rows + gen.integers(0, 3, parent.shape)
        plan = _plan(np.where(bad, late, parent), counts, tokens)
        parent = np.where(bad, rows - 1, parent)
    else:
        plan = _plan(parent, counts, tokens)
    # Each query continues its request's sequence.
    starts = gen.integers(0, 200, num_reqs)
    chain = torch.tensor(
        [int(starts[b]) + t for b in range(num_reqs) for t in range(query_lens[b])]
    ).to(DEVICE)
    positions = chain.clone()
    input_ids = torch.full((num_tokens,), 7, dtype=torch.int32, device=DEVICE)
    layout = PlanLayout(num_tokens, max_rows, DEVICE)

    layout.apply(plan, _t(counts), _t(qsl), positions, input_ids, num_reqs, num_tokens)

    want_pos, want_ids = chain.tolist(), [7] * num_tokens
    want_parents, want_ancestors = [], []
    for b, count in enumerate(counts):
        offset = int(query_lens[b] - count)
        root = int(qsl[b]) + offset
        for t in range(int(query_lens[b])):
            row = t - offset
            want_parents.append(t - 1 if row <= 0 else offset + int(parent[b, row]))
            path = _path_to(parent[b], row) if row >= 0 else []
            bits = sum(1 << r for r in path)
            want_ancestors.append(0 if kind == "runner-chain" else bits)
            if row > 0:
                want_pos[root + row] = want_pos[root] + len(path) - 1
                if kind != "runner-chain":
                    want_ids[root + row] = int(tokens[b, row])
    assert positions.tolist() == want_pos
    assert input_ids.tolist() == want_ids
    assert layout.tree_parents[:num_tokens].tolist() == want_parents
    assert layout.tree_ancestors[:num_tokens].tolist() == want_ancestors
    assert layout.pos_delta[:num_tokens].tolist() == (positions - chain).tolist()
    layout.restore_positions(positions, num_tokens)
    assert torch.equal(positions, chain)


@requires_kernels
@pytest.mark.parametrize("max_branching", [1, 2])
def test_spec_round_lays_out_only_a_branching_speculator(max_branching):
    """A speculator that never branches launches nothing; one that branches
    lays out every batch, a batch without drafts as the chain of its rows."""
    resources = SpecResources(
        max_candidates=4, capture_shapes=(), max_branching=max_branching
    )
    spec_round = SpecRound(
        None,  # type: ignore[arg-type]
        resources,
        4,
        16,
        3,
        DEVICE,
        plans=max_branching > 1,
    )
    buffers = InputBuffers(4, 16, DEVICE)
    # The memory profile runs before the spec kernels are bound, so the layers
    # run their own chain kernels: set_resources lays out the trees.
    spec_round.kernels = KernelRegistry()
    spec_round.set_resources(resources, buffers, [])
    batch = InputBatch.make_dummy(2, 6, buffers)
    batch.positions.copy_(torch.arange(6))

    spec_round.lay_out(batch)

    assert batch.positions.tolist() == list(range(6))
    if max_branching == 1:
        assert batch.tree_parents is None and batch.tree_ancestors is None
    else:
        assert batch.tree_parents is not None and batch.tree_ancestors is not None
        assert batch.tree_parents.tolist() == [-1, 0, 1, -1, 0, 1]
        assert batch.tree_ancestors.tolist() == [0] * 6  # a chain: causal

    # A speculator without ``plan`` verifies the scheduled chains: no result.
    batch.verify_plan = spec_round.chain_plan
    zeros = torch.zeros(2, 4, dtype=torch.int32, device=DEVICE)
    sampled = SamplerOutput(zeros, None, None, zeros[:, 0], zeros[:, 0])
    result = spec_round.accept_result(sampled, batch)
    assert (result is None) == (max_branching == 1)


class _Staged:
    """A UVA-backed buffer's device view; ``RopeState.__init__`` pins host
    memory, which needs an accelerator."""

    def __init__(self, gpu):
        self.gpu = gpu


@requires_kernels
@pytest.mark.parametrize("last_parent", range(3))
def test_mrope_tree_positions(last_parent):
    """Every M-RoPE section of a verify row moves by plan - chain position, so
    along each root-to-leaf path the positions are the ones that path verified
    as a chain gets. Prefill rows keep their staged positions."""
    num_dims, max_rows, max_model_len = 3, 4, 64
    counts, query_lens = [1, 4, 4], [5, 4, 4]  # request 0: a prefill chunk
    qsl = np.cumsum([0, *query_lens]).astype(np.int32)
    num_tokens = int(qsl[-1])
    idx_mapping = _t([2, 0, 1])
    num_computed = _t([30, 41, 3])
    rope = object.__new__(RopeState)
    rope.num_dims, rope.max_model_len = num_dims, max_model_len
    staged = torch.zeros(3 * num_dims, max_model_len, dtype=torch.int32, device=DEVICE)
    # Distinct sections in slot 2's prompt, so a misread prefill row shows.
    for j in range(num_dims):
        staged[2 * num_dims + j, :20] = 100 * (j + 1) + torch.arange(20)
    rope.prefill_positions = _Staged(staged)
    rope.prefill_delta = _Staged(_t([-5, 7, 0]))
    rope.positions = torch.zeros(num_dims, num_tokens + 1, dtype=torch.int64).to(DEVICE)
    parent = np.full((3, max_rows), -1, dtype=np.int32)
    parent[1:, 1:] = [[0, 0, 1], [0, 1, last_parent]]
    chain_pos = torch.zeros(num_tokens, dtype=torch.int64, device=DEVICE)
    input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    layout = PlanLayout(num_tokens, max_rows, DEVICE)
    plan = _plan(parent, counts)
    layout.apply(plan, plan.count, _t(qsl), chain_pos, input_ids, 3, num_tokens)

    def rope_positions(pos_delta):
        args = (idx_mapping, _t(qsl), _t([6, 9, 20]), num_computed)
        rope.prepare_positions(*args, pos_delta=pos_delta)
        return rope.get_positions(num_tokens).clone().cpu()

    tree, chain = rope_positions(layout.pos_delta), rope_positions(None)

    assert torch.equal(tree, chain + layout.pos_delta[:num_tokens].cpu())
    assert tree[:, 0].tolist() == [103, 203, 303]
    for b in (1, 2):
        start = int(qsl[b + 1]) - counts[b]
        for leaf in set(range(4)) - set(parent[b, 1:].tolist()):
            path = _path_to(parent[b], leaf)
            along = torch.tensor([start + r for r in path])
            assert torch.equal(tree[:, along], chain[:, start : start + len(path)])


# --- Tree accept ------------------------------------------------------------


def _verify_output(drafted, row_start, logits=None, num_reqs=1):
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


def _reference_walk(parent, count, drafted, target):
    """From the root: the lowest child whose drafted token is the current row's
    target, until there is none."""
    path = [0]
    while True:
        node = path[-1]
        children = [
            r
            for r in range(1, count)
            if parent[r] == node and drafted[r] != -1 and drafted[r] == target[node]
        ]
        if not children:
            return path, [target[r] for r in path]
        path.append(children[0])


@requires_kernels
@pytest.mark.parametrize("seed", range(3))
def test_walk_tree_matches_reference(seed):
    """The walk follows the child holding the target token; rows past a
    request's count and masked drafts (-1) are never accepted."""
    gen = np.random.default_rng(seed)
    num_reqs, num_cols, vocab = 7, 6, 3
    parent = _random_tree(gen, num_reqs, num_cols)
    counts = gen.integers(1, num_cols + 1, num_reqs)
    row_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    drafted = [
        -1 if r == 0 or gen.random() < 0.1 else int(gen.integers(0, vocab))
        for c in counts
        for r in range(c)
    ]
    target = [int(gen.integers(0, vocab)) for _ in drafted]
    for b, count in enumerate(counts):  # half the candidates match: paths grow
        for r in range(1, count):
            if drafted[row_start[b] + r] != -1 and gen.random() < 0.5:
                target[row_start[b] + parent[b, r]] = drafted[row_start[b] + r]
    output = _verify_output(_t(drafted, torch.int64), _t(row_start), num_reqs=num_reqs)

    result = walk_tree(_plan(parent, counts), output, _t(target, torch.int64), 8)

    assert result.sampled.shape == (num_reqs, 8)
    for b, count in enumerate(counts):
        rows = slice(row_start[b], row_start[b + 1])
        path, sampled = _reference_walk(parent[b], count, drafted[rows], target[rows])
        n = int(result.path_len[b])
        assert result.path[b, :n].tolist() == path
        assert result.sampled[b, :n].tolist() == sampled


def _greedy_sampler(temperature, processed):
    """A sampler whose sampling params turn logits into ``processed``, built
    without ``__init__`` (it needs a vLLM config and pinned host memory)."""
    sampler = object.__new__(Sampler)
    sampler.sampling_states = object.__new__(SamplingStates)
    sampler.sampling_states.temperature = object.__new__(UvaBackedTensor)
    sampler.sampling_states.temperature.np = np.array([temperature], np.float32)
    sampler.apply_sampling_params = lambda *args: processed  # type: ignore[method-assign]
    return sampler


@requires_kernels
def test_default_tree_accept_walks_the_greedy_argmax():
    """Greedy requests walk the argmax of the processed logits (the ones the
    runner takes logprobs from); sampled requests are refused."""
    processed = torch.zeros(4, 8, device=DEVICE)
    processed[0, 3] = processed[2, 6] = processed[3, 1] = processed[1, 0] = 1.0
    output = _verify_output(
        _t([-1, 2, 3, 6], torch.int64), _t([0, 4]), logits=torch.zeros_like(processed)
    )
    plan = _plan([[-1, 0, 0, 2]], [4])

    result = accept_greedy_tree(plan, output, _greedy_sampler(0.0, processed), 4)

    # Root's argmax 3 -> row 2 (token 3); its argmax 6 -> row 3; then token 1.
    assert result.path[0, :3].tolist() == [0, 2, 3]
    assert result.sampled[0, :3].tolist() == [3, 6, 1]
    assert result.processed_logits is processed
    with pytest.raises(ValueError, match="greedy"):
        accept_greedy_tree(plan, output, _greedy_sampler(0.7, processed), 4)


# --- Commit -----------------------------------------------------------------


def _random_paths(gen, parent, counts, to_last=False):
    """A root-to-node path per request: ``(path [R, C], path_len [R])``, length 0
    for some (a prefill chunk that sampled nothing)."""
    num_reqs, max_rows = parent.shape
    path = np.zeros((num_reqs, max_rows), dtype=np.int32)
    path_len = np.zeros(num_reqs, dtype=np.int32)
    for b in range(num_reqs):
        node = counts[b] - 1 if to_last else int(gen.integers(0, counts[b]))
        rows = _path_to(parent[b], node)
        path[b, : len(rows)] = rows
        path_len[b] = 0 if gen.random() < 0.15 else len(rows)
    return path, path_len


def _mamba_state(max_num_reqs, align=False):
    """The members ``postprocess_state`` reads; ``__init__`` needs a model."""
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full((max_num_reqs,), 99).to(
        DEVICE, torch.int32
    )
    state._align_mode = align
    state.recoverssm = None
    state._mamba_ctx = None
    return state


@requires_kernels
@pytest.mark.parametrize("seed", range(3))
def test_mamba_resumes_from_the_last_accepted_row(seed):
    """The GDN kernels read state row num_accepted_tokens - 1: the last accepted
    row, for a chain today's max(num_sampled, 1) - 1. Rows filtered under PP
    (-1) are left alone; the aligned state mode refuses tree paths."""
    gen = np.random.default_rng(200 + seed)
    num_reqs, max_rows, num_slots = int(gen.integers(1, 8)), int(gen.integers(2, 9)), 12
    counts = gen.integers(1, max_rows + 1, num_reqs)
    path, path_len = _random_paths(gen, _random_tree(gen, num_reqs, max_rows), counts)
    idx = gen.permutation(num_slots)[:num_reqs].astype(np.int32)
    idx[gen.random(num_reqs) < 0.2] = -1
    sampled = torch.zeros(num_reqs, max_rows, dtype=torch.int64, device=DEVICE)
    tree = AcceptResult(_t(path), _t(path_len), sampled)
    chain = AcceptResult(chain_path(num_reqs, max_rows, DEVICE), _t(path_len), sampled)
    states = {}
    for name, result in [("tree", tree), ("chain", chain), ("today", None)]:
        states[name] = _mamba_state(num_slots)
        rows = None if result is None else accepted_rows(result, num_reqs)
        states[name].postprocess_state(_t(idx), _t(path_len), accepted_row=rows)

    got = states["tree"].num_accepted_tokens_gpu.cpu().numpy()
    for b in np.flatnonzero(idx >= 0):
        assert got[idx[b]] - 1 == path[b, max(path_len[b], 1) - 1]
    assert (np.delete(got, idx[idx >= 0]) == 99).all()
    assert torch.equal(
        states["chain"].num_accepted_tokens_gpu,
        states["today"].num_accepted_tokens_gpu,
    )
    with pytest.raises(NotImplementedError, match="align"):
        _mamba_state(4, align=True).postprocess_state(
            _t([0]), _t([1]), accepted_row=_t([0])
        )


# Cache layout -> (shape, physical permutation or None): K and V packed in the
# content, FlashAttention, FlashInfer, the NHD/HND physical orders behind each
# view, and MLA.
_CACHES = {
    "packed": ((None, "H", "B", "2D"), None),
    "packed_nhd": ((None, "B", "H", "2D"), (0, 2, 1, 3)),
    "mla": ((None, "B", "D"), None),
}


def _cache(kind, num_blocks, block, heads, dim, dtype):
    sizes = {None: num_blocks, "B": block, "H": heads, "D": dim, "2D": 2 * dim, 2: 2}
    shape, perm = _CACHES[kind]
    cache = torch.randn([sizes[s] for s in shape]).to(dtype)
    return (cache if perm is None else cache.permute(perm)).to(DEVICE)


def _token_view(cache, kind):
    """``[planes, tokens, heads, dim]`` view of a cache by kernel-block token id."""
    if kind.startswith("packed"):
        return cache.transpose(1, 2).flatten(0, 1)[None]
    return cache.flatten(0, 1)[None, :, None, :]


def _module(**attrs):
    module = torch.nn.Module()
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


def _kernel_context():
    return KernelContext(
        device=DEVICE,
        plan_kind="tree",
        max_num_reqs=4,
        max_candidates=4,
        capture_shapes=(PlanShape(4),),
    )


@requires_kernels
@pytest.mark.parametrize("kind", list(_CACHES))
@pytest.mark.parametrize("path_kind", ["tree", "chain"])
def test_kv_compaction_moves_the_path(kind, path_kind):
    """The KV of each accepted path row lands in the chain slot of its path
    index, in every cache layout; a chain path moves nothing."""
    gen = np.random.default_rng(300)
    table_block, kernel_block, num_table_blocks, max_rows = 8, 4, 24, 7
    num_kernel_blocks = num_table_blocks * table_block // kernel_block
    heads, dim = (1, 576) if kind == "mla" else (2, 24)
    num_reqs = 5
    counts = np.full(num_reqs, max_rows, dtype=np.int32)
    counts[-1] = 3
    path, path_len = _random_paths(
        gen, _random_tree(gen, num_reqs, max_rows), counts, to_last=True
    )
    if path_kind == "chain":
        path = np.tile(np.arange(max_rows, dtype=np.int32), (num_reqs, 1))
    else:
        assert (path[:, 1:] != np.arange(1, max_rows)).any(), "seed must move KV"
    seq_lens = gen.integers(counts, 4 * table_block).astype(np.int32)
    # Requests get disjoint random blocks; the tables are padded past num_reqs.
    tables = np.full((num_reqs + 2, 5), -1, dtype=np.int32)
    blocks = gen.permutation(num_table_blocks)[: 4 * num_reqs]
    tables[:num_reqs, :4] = blocks.reshape(num_reqs, 4)
    caches = [
        _cache(kind, num_kernel_blocks, kernel_block, heads, dim, torch.bfloat16)
        for _ in range(2)
    ]
    table = _t(tables)  # the compactor keeps its pointer
    before = [cache.clone() for cache in caches]
    compactor = KvCompactor()
    compactor.bind(
        _kernel_context(),
        [
            SpecLayerSpec(
                f"l{i}",
                kind=None,  # type: ignore[arg-type]
                module=_module(kv_cache=cache),
                block_table=table,
                block_size=table_block,
                num_blocks=num_kernel_blocks,
            )
            for i, cache in enumerate(caches)
        ],
    )
    sampled = torch.zeros(num_reqs, max_rows, dtype=torch.int64, device=DEVICE)
    compactor.commit(
        CommitStep(
            AcceptResult(_t(path), _t(path_len), sampled),
            num_reqs,
            count=_t(counts),
            seq_lens=_t(seq_lens),
        )
    )

    def token(b, position):
        return tables[b, position // table_block] * table_block + position % table_block

    for cache, old in zip(caches, before):
        expected, source = _token_view(old.clone(), kind), _token_view(old, kind)
        for b in range(num_reqs):
            context = seq_lens[b] - counts[b]
            for j in range(1, path_len[b]):
                expected[:, token(b, context + j)] = source[
                    :, token(b, context + path[b, j])
                ]
        assert torch.equal(_token_view(cache, kind).cpu(), expected.cpu())


def _spec(cls, block_size=16, num_kv_heads=1, head_size=64):
    return cls(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def _runner(specs, modules, kernel_block_size, num_blocks):
    """A runner with the members ``spec_kernel_layers`` reads: one KV cache
    group per layer of ``specs``; ``modules`` are the forward context."""
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.compilation_config = CompilationConfig()
    runner.compilation_config.static_forward_context = modules
    runner.kv_cache_config = KVCacheConfig(num_blocks, [], [])
    runner.attn_groups = [
        [AttentionGroup(FlashAttentionBackend, [name], spec, g)]
        for g, (name, spec) in enumerate(specs.items())
    ]
    runner.kernel_block_sizes = [kernel_block_size] * len(specs)
    runner.block_tables = object.__new__(BlockTables)
    runner.block_tables.input_block_tables = [
        torch.zeros(2, 4, dtype=torch.int32)
    ] * len(specs)
    runner.cache_config = CacheConfig()
    return runner


@pytest.mark.parametrize("case", ["hybrid_group_blocks", "ring_shared_hidden"])
def test_kv_compactor_binds_the_layers_it_moves(case):
    """A hybrid model's attention group blocks are larger than the layer's spec
    says: the cache is allocated, and the block table kept, in the group's. The
    DSv4 ring (not paged by slot) keeps its own commit, a layer sharing another's
    KV cache moves with it, and a hidden-state cache is not a decoder layer."""
    if case == "hybrid_group_blocks":
        num_blocks, group_block, kernel_block = 3, 1280, 128
        cache = torch.zeros(
            num_blocks * group_block // kernel_block, 4, kernel_block, 512
        ).to(torch.bfloat16)
        specs = {"attn": _spec(FullAttentionSpec, group_block, 4, 256)}
        modules = {"attn": _module(kv_cache=cache)}
        runner = _runner(specs, modules, kernel_block, num_blocks)
    else:
        cache = torch.zeros(4, 1, 16, 128, dtype=torch.bfloat16)
        specs = {
            "attn": _spec(FullAttentionSpec),
            "ring": _spec(CircularBufferSpec),
            "hidden": _spec(HiddenStateCacheSpec),
            "shared": _spec(FullAttentionSpec),
        }
        modules = {
            "attn": _module(kv_cache=cache),
            "shared": _module(kv_cache=cache, kv_sharing_target_layer_name="attn"),
        }
        runner = _runner(specs, modules, 16, 4)

    layers = spec_kernel_layers(runner)
    registry = KernelRegistry()
    registry.bind(_kernel_context(), layers)
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


@pytest.mark.parametrize(
    "cache, match",
    [
        (torch.zeros(3, 4, 5, 6), "layout"),
        (torch.zeros(8, 2, 4, 16, dtype=torch.float64), "copy"),
        # K/V-split views: [2, blocks, ...] and [blocks, 2, ...] read the same
        # when there are 2 blocks, and vLLM does not give layers either.
        (torch.zeros(2, 8, 4, 2, 16), "layout"),
        (torch.zeros(8, 2, 4, 2, 16), "layout"),
    ],
)
def test_kv_cache_layout_refuses_what_it_cannot_read(cache, match):
    with pytest.raises(NotImplementedError, match=match):
        kv_cache_layout(cache, num_blocks=8)


# --- Built-in speculators keep today's resources and budget --------------


class _StaticSpec:
    """The SpeculativeConfig fields the default resources and budget read."""

    num_speculative_tokens = 3
    enable_adaptive_verification = False
    num_speculative_tokens_per_batch_size = None

    def uses_dynamic_speculative_decoding(self) -> bool:
        return False


@pytest.mark.parametrize(
    "qualname",
    [
        "ngram.config.NgramGPUSpeculatorConfig",
        "eagle.config.EagleSpeculatorConfig",
        "eagle.config.Eagle3SpeculatorConfig",
        "mtp.config.MTPSpeculatorConfig",
        "dflash.config.DFlashSpeculatorConfig",
        "dspark.config.DSparkSpeculatorConfig",
        "standalone_ar.config.StandaloneARSpeculatorConfig",
    ],
)
def test_builtin_configs_keep_todays_resources_and_budget(qualname):
    """Built-in methods draft K per step through the default declaration: K + 1
    candidates, the uniform K + 1 decode graph and chains, at every batch
    size."""
    config_cls = resolve_obj_by_qualname(f"vllm.v1.worker.gpu.spec_decode.{qualname}")
    config = config_cls(_StaticSpec())
    assert config.resources(16) == SpecResources(
        max_candidates=4, capture_shapes=(PlanShape(4),)
    )
    for num_reqs in (1, 7, 16):
        view = BudgetView(num_reqs, num_reqs, 0, 0, 16, 16)
        assert config.candidate_budget(view) == 4
