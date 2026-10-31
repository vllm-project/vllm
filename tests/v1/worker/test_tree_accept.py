# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine's walk down a verify tree (``spec_decode/tree_accept.py``).

The Triton kernel runs under ``TRITON_INTERPRET=1`` on CPU.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.utils_tree import random_parents
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.round import VerifyOutput, VerifyPlan
from vllm.v1.worker.gpu.spec_decode.tree_accept import accept_greedy_tree, walk_tree

HAS_GPU = current_platform.is_cuda_alike()
DEVICE = current_platform.device_type if HAS_GPU else "cpu"
pytestmark = pytest.mark.skipif(
    not HAS_GPU and os.environ.get("TRITON_INTERPRET") != "1",
    reason="Triton kernels need a GPU or TRITON_INTERPRET=1",
)


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


def _random_tree(gen, num_cols, vocab):
    count = int(gen.integers(1, num_cols + 1))
    parent = random_parents(gen, num_cols)
    drafted = [-1] + [int(gen.integers(0, vocab)) for _ in range(1, num_cols)]
    return parent, count, drafted


def _output(drafted, row_start, logits=None, num_reqs=0):
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


def _batch(gen, num_reqs, num_cols, vocab):
    trees = [_random_tree(gen, num_cols, vocab) for _ in range(num_reqs)]
    counts = [count for _, count, _ in trees]
    row_start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    drafted = [d for (_, c, dr) in trees for d in dr[:c]]
    # A third of the rows match their parent's target, so paths grow.
    target = [int(gen.integers(0, vocab)) for _ in drafted]
    for b, (parent, count, dr) in enumerate(trees):
        for r in range(1, count):
            if gen.random() < 0.5:
                target[row_start[b] + parent[r]] = dr[r]
    plan = VerifyPlan(
        parent=torch.tensor([p for p, _, _ in trees], dtype=torch.int32).to(DEVICE),
        count=torch.tensor(counts, dtype=torch.int32, device=DEVICE),
        tokens=torch.zeros(num_reqs, num_cols, dtype=torch.int32, device=DEVICE),
    )
    output = _output(
        torch.tensor(drafted, dtype=torch.int64, device=DEVICE),
        torch.tensor(row_start, device=DEVICE),
        num_reqs=num_reqs,
    )
    return trees, row_start, target, plan, output


@pytest.mark.parametrize("seed", range(4))
def test_walk_follows_the_child_with_the_target_token(seed):
    gen = np.random.default_rng(seed)
    num_reqs, num_cols, vocab = 7, 6, 3
    trees, row_start, target, plan, output = _batch(gen, num_reqs, num_cols, vocab)

    result = walk_tree(
        plan, output, torch.tensor(target, dtype=torch.int64, device=DEVICE), 8
    )

    assert result.sampled.shape == (num_reqs, 8)
    for b, (parent, count, drafted) in enumerate(trees):
        rows = target[row_start[b] : row_start[b + 1]]
        path, sampled = _reference_walk(parent, count, drafted[:count], rows)
        n = int(result.path_len[b])
        assert n == len(path)
        assert result.path[b, :n].tolist() == path
        assert result.sampled[b, :n].tolist() == sampled


def test_rows_past_count_and_masked_drafts_are_never_accepted():
    # Row 1 matches but is masked (-1); row 2 matches but is past count.
    plan = VerifyPlan(
        parent=torch.tensor([[-1, 0, 0]], dtype=torch.int32, device=DEVICE),
        count=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        tokens=torch.zeros(1, 3, dtype=torch.int32, device=DEVICE),
    )
    output = _output(
        torch.tensor([-1, -1], dtype=torch.int64, device=DEVICE),
        torch.tensor([0, 2], dtype=torch.int32, device=DEVICE),
        num_reqs=1,
    )
    target = torch.tensor([5, 9], dtype=torch.int64, device=DEVICE)
    result = walk_tree(plan, output, target)
    assert result.path_len.tolist() == [1]
    assert result.sampled[0, 0].item() == 5


def _greedy_sampler(temperature, processed=None):
    def apply_sampling_params(logits, *args):
        return logits if processed is None else processed

    return SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(np=np.asarray(temperature, dtype=np.float32))
        ),
        apply_sampling_params=apply_sampling_params,
    )


def test_default_tree_accept_walks_the_argmax_of_the_processed_logits():
    plan = VerifyPlan(
        parent=torch.tensor([[-1, 0, 0, 2]], dtype=torch.int32, device=DEVICE),
        count=torch.tensor([4], dtype=torch.int32, device=DEVICE),
        tokens=torch.zeros(1, 4, dtype=torch.int32, device=DEVICE),
    )
    logits = torch.zeros(4, 8, device=DEVICE)
    processed = torch.zeros(4, 8, device=DEVICE)
    processed[0, 3] = processed[2, 6] = processed[3, 1] = processed[1, 0] = 1.0
    output = _output(
        torch.tensor([-1, 2, 3, 6], dtype=torch.int64, device=DEVICE),
        torch.tensor([0, 4], dtype=torch.int32, device=DEVICE),
        logits=logits,
        num_reqs=1,
    )
    result = accept_greedy_tree(plan, output, _greedy_sampler([0.0], processed), 4)
    # Root's argmax 3 -> row 2 (token 3); its argmax 6 -> row 3; then token 1.
    assert result.path[0, :3].tolist() == [0, 2, 3]
    assert result.sampled[0, :3].tolist() == [3, 6, 1]
    assert result.processed_logits is processed


def test_default_tree_accept_refuses_sampled_requests():
    plan = VerifyPlan(
        parent=torch.tensor([[-1, 0]], dtype=torch.int32, device=DEVICE),
        count=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        tokens=torch.zeros(1, 2, dtype=torch.int32, device=DEVICE),
    )
    output = _output(None, None, logits=torch.zeros(2, 4), num_reqs=1)
    with pytest.raises(ValueError, match="greedy"):
        accept_greedy_tree(plan, output, _greedy_sampler([0.7]), 2)
