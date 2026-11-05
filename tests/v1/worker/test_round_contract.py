# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan types, plan layout, tree accept and commit; kernels run on a GPU or on CPU
under ``TRITON_INTERPRET=1``. ``parent[i]`` is the row that row ``i`` extends."""

import os
import types

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
from vllm.v1.spec_decode.resources import BudgetView, SpecResources
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.capabilities import SpecLayerKind
from vllm.v1.worker.gpu.spec_decode.commit import accepted_rows
from vllm.v1.worker.gpu.spec_decode.kernels import (
    CommitStep,
    KernelContext,
    KernelRegistry,
    SpecLayerSpec,
)
from vllm.v1.worker.gpu.spec_decode.plan_kernels import (
    PlanLayout,
    arange_table,
)
from vllm.v1.worker.gpu.spec_decode.round import (
    AcceptResult,
    TargetTokens,
    VerifyPlan,
    check_accept_result,
)
from vllm.v1.worker.gpu.spec_decode.round_runner import SpecRound, spec_kernel_layers
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
        tokens=torch.zeros_like(parent) if tokens is None else _t(tokens),
        rows=np.asarray(counts, dtype=np.int32),
    )


_PARENT = [[-1, 0, 0, 1]]


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: VerifyPlan(parent=_t(_PARENT)), "tokens"),
        # A chain verifies the drafted tokens.
        (
            lambda: VerifyPlan(
                parent=_t([[-1, 0, 1, 2]]), tokens=_t([[0, 7, 8, 9]]), is_chain=True
            ),
            "chain",
        ),
        (lambda: SpecResources(capture_rows=()), "capture_rows"),
        (lambda: SpecResources(capture_rows=(0,)), "capture_rows"),
        (lambda: SpecResources(capture_rows=(None,)), "capture_rows"),
        (lambda: SpecResources(capture_rows=(2,), max_branching=0), "max_branching"),
    ],
    ids=[
        "plan-no-tokens",
        "chain-tokens",
        "no-rows",
        "zero-rows",
        "none-rows",
        "branching",
    ],
)
def test_plan_and_resources_are_validated(make, match):
    with pytest.raises(ValueError, match=match):
        make()
    # Only a chain may leave its tokens to the runner's draft buffer.
    assert VerifyPlan(parent=_t(_PARENT), is_chain=True).tokens is None
    resources = SpecResources(capture_rows=(4, 1, 4))
    assert (resources.max_branching, resources.max_rows) == (1, 4)


@pytest.mark.parametrize(
    "max_rows, rows, match",
    [
        (3, [4], "num_speculative_tokens \\+ 1 = 3"),
        (4, [4, 4], "every request"),
        (8, [5], "verifies up to 5"),
    ],
)
def test_spec_round_checks_a_plans_shape(max_rows, rows, match):
    """A plan fits the runner's buffers and has a column for every row it
    verifies."""
    spec_round = object.__new__(SpecRound)
    spec_round.max_rows = max_rows
    plan = _plan(_PARENT, [4])
    with pytest.raises(AssertionError, match=match):
        spec_round._check_plan(plan, np.array(rows))


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
    root's position plus its depth, with the plan's token and a query-local
    parent. Tokens in front of a plan chain on the token before them. A parent
    that is not an earlier row is a chain step. ``restore`` brings the chain
    positions back."""
    gen = np.random.default_rng(seed)
    max_rows = int(gen.integers(2, 9))
    qsl, query_lens, counts = _random_batch(gen, max_rows)
    num_reqs, num_tokens = len(counts), int(qsl[-1])
    if kind in ("tree", "bad-tree"):
        parent = _random_tree(gen, num_reqs, max_rows)
    else:
        parent = arange_table(num_reqs, max_rows, DEVICE, start=-1).cpu().numpy()
    tokens = gen.integers(100, 1000, (num_reqs, max_rows)).astype(np.int32)
    if kind == "runner-chain":
        plan = VerifyPlan(parent=_t(parent), is_chain=True)
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
    want_parents = []
    for b, count in enumerate(counts):
        offset = int(query_lens[b] - count)
        root = int(qsl[b]) + offset
        for t in range(int(query_lens[b])):
            row = t - offset
            want_parents.append(t - 1 if row <= 0 else offset + int(parent[b, row]))
            if row > 0:
                path = _path_to(parent[b], row)
                want_pos[root + row] = want_pos[root] + len(path) - 1
                if kind != "runner-chain":
                    want_ids[root + row] = int(tokens[b, row])
    assert positions.tolist() == want_pos
    assert input_ids.tolist() == want_ids
    assert layout.tree_parents[:num_tokens].tolist() == want_parents
    assert layout.pos_delta[:num_tokens].tolist() == (positions - chain).tolist()
    layout.restore_positions(positions, num_tokens)
    assert torch.equal(positions, chain)


def _spec_round(max_branching):
    """A SpecRound over 4 requests of up to 4 rows; it plans when it branches.
    The memory profile runs before the spec kernels are bound, so the layers
    run their own chain kernels: set_resources lays out the trees."""
    resources = SpecResources(capture_rows=(4,), max_branching=max_branching)
    plans = max_branching > 1
    spec_round = SpecRound(None, resources, 4, 16, 3, DEVICE, plans=plans)  # type: ignore[arg-type]
    buffers = InputBuffers(4, 16, DEVICE)
    spec_round.kernels = KernelRegistry()
    spec_round.set_resources(resources, buffers, [])
    return spec_round, buffers


@requires_kernels
@pytest.mark.parametrize("max_branching", [1, 2])
def test_spec_round_lays_out_only_a_branching_speculator(max_branching):
    """A speculator that never branches launches nothing; one that branches
    lays out every batch, a batch without drafts as the chain of its rows."""
    spec_round, buffers = _spec_round(max_branching)
    batch = InputBatch.make_dummy(2, 6, buffers)
    batch.positions.copy_(torch.arange(6))

    spec_round.lay_out(batch)

    assert batch.positions.tolist() == list(range(6))
    if max_branching == 1:
        assert batch.tree_parents is None
    else:
        assert batch.tree_parents is not None
        assert batch.tree_parents.tolist() == [-1, 0, 1, -1, 0, 1]

    # A speculator without ``plan`` verifies the scheduled chains: no result.
    batch.verify_plan = spec_round.chain_plan
    zeros = torch.zeros(2, 4, dtype=torch.int32, device=DEVICE)
    sampled = SamplerOutput(zeros, None, None, zeros[:, 0], zeros[:, 0])
    result = spec_round.accept_result(sampled, batch)
    assert (result is None) == (max_branching == 1)


def test_reference_tree_attention_follows_parents():
    """The test plugin's tree attention, masked from vLLM's parents: each plan
    row attends exactly as the chain of its path would."""
    reference = pytest.importorskip("vllm_add_test_speculators.reference")
    tree_attn = pytest.importorskip("vllm_add_test_speculators.tree_attn")
    torch.manual_seed(0)
    block_size, heads, dim, context = 4, 2, 8, 5
    parents = [-1, 0, 0, 1]  # query rows: the root, two children, a grandchild
    num_rows, seq_len = len(parents), context + len(parents)
    keys = torch.randn(3, heads, block_size, dim)
    values = torch.randn(3, heads, block_size, dim)
    query = torch.randn(num_rows, heads, dim)
    qsl = torch.tensor([0, num_rows])
    metadata = types.SimpleNamespace(
        query_start_loc=qsl,
        seq_lens=torch.tensor([seq_len]),
        block_table=torch.tensor([[2, 0, 1]]),
        tree_ancestors=reference.tree_ancestors(torch.tensor(parents), qsl),
    )
    output = torch.zeros(num_rows, heads * dim)

    tree_attn.tree_attention(query, keys, values, metadata, 0.5, output)

    flat_k = keys[[2, 0, 1]].transpose(1, 2).flatten(0, 1)  # [token, head, dim]
    flat_v = values[[2, 0, 1]].transpose(1, 2).flatten(0, 1)
    for row, path in enumerate([[0], [0, 1], [0, 2], [0, 1, 3]]):
        seen = list(range(context)) + [context + r for r in path]
        scores = torch.einsum("hd,khd->hk", query[row], flat_k[seen]) * 0.5
        want = torch.einsum("hk,khd->hd", scores.softmax(-1), flat_v[seen])
        torch.testing.assert_close(output[row], want.flatten())


def test_reference_tree_ancestors():
    """The test plugin's mask from vLLM's parents: a row sees itself and its
    ancestors; a chain request (prompt rows included) attends causally."""
    reference = pytest.importorskip("vllm_add_test_speculators.reference")
    # A chain of 3, then a prompt row in front of the plan [-1, 0, 0, 1].
    parents = _t([-1, 0, 1, -1, 0, 1, 1, 2])
    masks = reference.tree_ancestors(parents, _t([0, 3, 8]))
    assert masks[0] is None
    assert masks[1].int().tolist() == [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 1, 1, 0, 1],
    ]


@requires_kernels
def test_spec_round_verifies_the_rows_of_each_query():
    """The runner takes each request's verify rows from its query length (the
    plan's rows, trimmed by plan_batch), for the layout and the commit: a plan
    carries no device count that could disagree with the batch."""
    spec_round, buffers = _spec_round(2)
    batch = InputBatch.make_dummy(2, 5, buffers)
    batch.query_start_loc[:3].copy_(_t([0, 2, 5]))
    batch.positions.copy_(torch.arange(5))
    parent = [[-1, 0, 0, -1], [-1, 0, 0, 1]]
    batch.verify_plan = VerifyPlan(
        parent=_t(parent),
        tokens=torch.zeros(2, 4, dtype=torch.int32, device=DEVICE),
        rows=np.array([2, 3], dtype=np.int32),
    )

    spec_round.lay_out(batch)

    assert spec_round.rows is not None
    assert spec_round.rows[:2].tolist() == [2, 3]
    assert batch.tree_parents is not None
    assert batch.tree_parents.tolist() == [-1, 0, -1, 0, 0]


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
    layout.apply(plan, _t(counts), _t(qsl), chain_pos, input_ids, 3, num_tokens)

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


@pytest.mark.parametrize("seed", range(3))
def test_walk_tree_matches_reference(seed):
    """The test plugin's walk follows the child holding the target token; rows past a
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
    output = TargetTokens(
        tokens=_t(target, torch.int64),
        drafted=_t(drafted, torch.int64),
        row_start=_t(row_start),
    )
    reference = pytest.importorskip("vllm_add_test_speculators.reference")

    plan = _plan(parent, counts)
    result = reference.walk_tree(plan, output, 8)
    check_accept_result(plan, output, result)

    assert result.sampled.shape == (num_reqs, 8)
    for b, count in enumerate(counts):
        rows = slice(row_start[b], row_start[b + 1])
        path, sampled = _reference_walk(parent[b], count, drafted[rows], target[rows])
        n = int(result.path_len[b])
        assert result.path[b, :n].tolist() == path
        assert result.sampled[b, :n].tolist() == sampled


def _check(parent, counts, path, path_len):
    """``check_accept_result`` on CPU: a failed device assert would end a GPU
    process."""
    cpu = torch.device("cpu")
    row_start = torch.tensor(np.concatenate([[0], np.cumsum(counts)]), device=cpu)
    tokens = torch.zeros(int(row_start[-1]), dtype=torch.int64)
    parent = torch.tensor(parent, dtype=torch.int32)
    check_accept_result(
        VerifyPlan(parent=parent, tokens=torch.zeros_like(parent)),
        TargetTokens(tokens=tokens, drafted=tokens, row_start=row_start.int()),
        AcceptResult(
            torch.tensor(path, dtype=torch.int32),
            torch.tensor(path_len, dtype=torch.int32),
            torch.zeros(len(counts), 4, dtype=torch.int64),
        ),
    )


_TREE = [[-1, 0, 0, 1], [-1, 0, 1, 2]]


def test_check_accept_result_takes_plan_paths():
    """Every root-to-row path of the plan passes, padding past path_len ignored."""
    _check(_TREE, [4, 3], [[0, 1, 3, 9], [0, 1, 2, 9]], [3, 3])
    _check(_TREE, [4, 3], [[0, 2, 0, 0], [0, 0, 0, 0]], [2, 1])


@pytest.mark.parametrize(
    "path, path_len",
    [
        pytest.param([[1, 3, 0, 0], [0, 1, 0, 0]], [2, 2], id="not-at-root"),
        pytest.param([[0, 2, 3, 0], [0, 1, 0, 0]], [3, 2], id="not-a-child"),
        pytest.param([[0, 1, 0, 0], [0, 1, 2, 3]], [2, 4], id="past-rows"),
        pytest.param([[0, 1, 0, 0], [0, 1, 0, 0]], [0, 2], id="empty"),
        pytest.param([[0, 1, 1, 0], [0, 1, 0, 0]], [3, 2], id="repeat"),
    ],
)
def test_check_accept_result_rejects(path, path_len):
    """A speculator's path that is not in the plan fails vLLM's check."""
    with pytest.raises(RuntimeError, match="not in the plan"):
        _check(_TREE, [4, 3], path, path_len)


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
    row. Rows filtered under PP (-1) are left alone; the aligned state mode
    refuses tree paths."""
    gen = np.random.default_rng(200 + seed)
    num_reqs, max_rows, num_slots = int(gen.integers(1, 8)), int(gen.integers(2, 9)), 12
    counts = gen.integers(1, max_rows + 1, num_reqs)
    path, path_len = _random_paths(gen, _random_tree(gen, num_reqs, max_rows), counts)
    idx = gen.permutation(num_slots)[:num_reqs].astype(np.int32)
    idx[gen.random(num_reqs) < 0.2] = -1
    sampled = torch.zeros(num_reqs, max_rows, dtype=torch.int64, device=DEVICE)
    result = AcceptResult(_t(path), _t(path_len), sampled)
    state = _mamba_state(num_slots)
    rows = accepted_rows(result, num_reqs)
    state.postprocess_state(_t(idx), _t(path_len), accepted_row=rows)

    got = state.num_accepted_tokens_gpu.cpu().numpy()
    for b in np.flatnonzero(idx >= 0):
        assert got[idx[b]] - 1 == path[b, max(path_len[b], 1) - 1]
    assert (np.delete(got, idx[idx >= 0]) == 99).all()
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
        max_rows=4,
        capture_rows=(4,),
    )


def _reference_kernels():
    """The test plugin's reference kernels: vLLM keeps no KV compaction."""
    reference = pytest.importorskip("vllm_add_test_speculators.reference")
    return KernelRegistry([reference.ReferenceKernels()])


@pytest.mark.parametrize("kind", list(_CACHES))
def test_kv_compaction_moves_the_path(kind):
    """The KV of each accepted path row lands in the chain slot of its path
    index, in every cache layout, through the layouts vLLM describes."""
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
    table = _t(tables)
    before = [cache.clone() for cache in caches]
    registry = _reference_kernels()
    registry.bind(
        _kernel_context(),
        [
            SpecLayerSpec(
                f"l{i}",
                kind=SpecLayerKind("attn", "full_attention"),
                module=_module(kv_cache=cache),
                layout="paged_token",
                block_table=table,
                block_size=table_block,
                num_blocks=num_kernel_blocks,
            )
            for i, cache in enumerate(caches)
        ],
    )
    sampled = torch.zeros(num_reqs, max_rows, dtype=torch.int64, device=DEVICE)
    registry.commit(
        CommitStep(
            AcceptResult(_t(path), _t(path_len), sampled),
            num_reqs,
            rows=_t(counts),
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
def test_kv_commit_binds_the_layers_it_moves(case):
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
    registry = _reference_kernels()
    registry.bind(_kernel_context(), layers)
    compactor = registry.get(("layout:paged_token", "commit"))

    [layout] = compactor.layouts
    if case == "hybrid_group_blocks":
        assert layout.block_size == kernel_block
        assert layout.table_block_size == group_block
    else:
        assert [(layer.name, layer.layout) for layer in layers] == [
            ("attn", "paged_token"),
            ("ring", ""),
        ]
        assert layout.caches == (cache,)


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
        # No built-in config overrides the defaults.
        "ngram.config.NgramGPUSpeculatorConfig",
        "dspark.config.DSparkSpeculatorConfig",
    ],
)
def test_builtin_configs_keep_todays_resources_and_budget(qualname):
    """Built-in methods draft K per step through the default declaration: K + 1
    candidates, the uniform K + 1 decode graph and chains, at every batch
    size."""
    config_cls = resolve_obj_by_qualname(f"vllm.v1.worker.gpu.spec_decode.{qualname}")
    config = config_cls(_StaticSpec())
    assert config.resources(16) == SpecResources(capture_rows=(4,))
    for num_reqs in (1, 7, 16):
        view = BudgetView(num_reqs, num_reqs, 0, 0, 16, 16)
        assert config.candidate_budget(view) == 4
