# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.two_tensor import TwoTensor

from vllm.config import ParallelConfig
from vllm.model_executor import parameter
from vllm.model_executor.model_loader.weight_checksum import (
    compute_tensor_digests,
    select_reset_weights,
    zero_selected_weights,
    zero_weights,
)
from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker

pytestmark = pytest.mark.cpu_test


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        nn.init.ones_(self.linear.weight)
        nn.init.ones_(self.linear.bias)
        # A strided view, which cannot be reinterpreted as bytes in place.
        self.strided = nn.Parameter(torch.arange(1.0, 7.0)[::2])
        self.scalar = nn.Parameter(torch.tensor(2.0))
        # Loading never restores buffers, so reset must leave them alone.
        self.register_buffer("k_scale", torch.tensor(2.0))


def test_digests_cover_every_parameter():
    assert set(compute_tensor_digests(_Model())) == {
        "linear.weight",
        "linear.bias",
        "strided",
        "scalar",
    }


def test_zero_weights_changes_every_parameter_and_no_buffer():
    model = _Model()
    before = compute_tensor_digests(model)
    zero_weights(model)
    after = compute_tensor_digests(model)
    assert all(before[name] != after[name] for name in before)
    assert model.k_scale.item() == 2.0


def test_shared_weight_partitions_are_hashed_and_reset(monkeypatch):
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    model = nn.Module()
    model.transform = parameter.SharedWeightParameter(weight_loader=None)
    model.transform.add_partition(0, object(), 2, 2)
    model.transform.partitions[0].data.fill_(3.0)

    before = compute_tensor_digests(model)
    zero_weights(model)
    assert set(before) == {"transform.0"}
    assert compute_tensor_digests(model) != before


def test_tensor_subclass_inner_tensors_are_hashed_and_reset():
    model = nn.Module()
    model.weight = nn.Parameter(
        TwoTensor(torch.ones(2, 2), torch.full((2, 2), 2.0)), requires_grad=False
    )

    before = compute_tensor_digests(model)
    zero_weights(model)
    assert set(before) == {"weight.a", "weight.b"}
    assert all(
        before[name] != digest for name, digest in compute_tensor_digests(model).items()
    )


@pytest.mark.parametrize("model", ["target", "draft", "both"])
def test_reset_covers_selected_model_weights(model):
    target = nn.Module()
    target.embed_tokens = nn.Embedding(4, 2)
    target.lm_head = nn.Linear(2, 4, bias=False)
    target.private = nn.Parameter(torch.full((2,), 3.0))
    nn.init.ones_(target.embed_tokens.weight)
    nn.init.ones_(target.lm_head.weight)

    draft = nn.Module()
    draft.embed_tokens = target.embed_tokens
    draft.lm_head = nn.Linear(2, 4, bias=False)
    draft.lm_head.weight = nn.Parameter(target.lm_head.weight.detach())
    # Equal values in separate storage do not make this a shared weight.
    draft.private = nn.Parameter(target.private.detach().clone())
    draft.already_zero = nn.Parameter(torch.zeros(1))
    before_target = compute_tensor_digests(target)
    before_draft = compute_tensor_digests(draft)
    addresses = [
        p.data_ptr() for module in (target, draft) for p in module.parameters()
    ]
    shared = {"embed_tokens.weight", "lm_head.weight"}
    assert all(before_target[name] == before_draft[name] for name in shared)

    plan = select_reset_weights(target, draft, model)
    assert not isinstance(plan, str)
    zero_selected_weights(plan)

    assert addresses == [
        p.data_ptr() for module in (target, draft) for p in module.parameters()
    ]
    assert draft.embed_tokens is target.embed_tokens
    assert draft.lm_head.weight.data_ptr() == target.lm_head.weight.data_ptr()
    after_target = compute_tensor_digests(target)
    after_draft = compute_tensor_digests(draft)
    assert {key for key in after_target if after_target[key] != before_target[key]} == (
        set(before_target) if model != "draft" else shared
    )
    assert {key for key in after_draft if after_draft[key] != before_draft[key]} == (
        shared | ({"private"} if model != "target" else set())
    )
    for weight in plan:
        torch.testing.assert_close(weight, torch.zeros_like(weight))


def test_target_reset_without_draft():
    target = _Model()
    plan = select_reset_weights(target, None, "target")
    assert not isinstance(plan, str)
    zero_selected_weights(plan)
    assert all(torch.count_nonzero(weight) == 0 for weight in target.parameters())
    assert target.k_scale.item() == 2.0


@pytest.mark.parametrize("model", ["draft", "both"])
def test_reset_accepts_a_draft_with_only_shared_weights(model):
    target = nn.Linear(2, 2, bias=False)
    nn.init.ones_(target.weight)
    draft = nn.Sequential(target)
    address = target.weight.data_ptr()
    plan = select_reset_weights(target, draft, model)
    assert not isinstance(plan, str)
    zero_selected_weights(plan)
    zero_selected_weights(plan)
    assert draft[0].weight is target.weight
    assert target.weight.data_ptr() == address
    torch.testing.assert_close(target.weight, torch.zeros_like(target.weight))


def test_draft_only_reset_requires_nonempty_weights():
    target = nn.Linear(2, 2, bias=False)
    draft = nn.Module()
    draft.weight = nn.Parameter(torch.empty(0))
    assert select_reset_weights(target, draft, "draft") == (
        "Weight Checker found no resettable draft weights."
    )


@pytest.mark.parametrize("writer", ["target", "draft"])
def test_reset_checks_shared_writes_against_the_other_models_buffers(writer):
    target = nn.Linear(2, 2, bias=False)
    draft = nn.Module()
    draft.weight = target.weight
    other = target if writer == "draft" else draft
    other.register_buffer("alias", target.weight.detach())
    before = other.alias.clone()

    for model in (writer, "both"):
        error = select_reset_weights(target, draft, model)
        assert isinstance(error, str)
        assert f"{writer} parameter 'weight' overlaps buffer 'alias'" in error
    torch.testing.assert_close(other.alias, before)


@pytest.mark.parametrize("model", ["target", "draft", "both"])
def test_reset_rejects_partial_overlap_before_mutation(model):
    storage = torch.ones(8)
    target, draft = nn.Module(), nn.Module()
    target.weight = nn.Parameter(storage[:5])
    draft.weight = nn.Parameter(storage[4:])
    error = select_reset_weights(target, draft, model)
    assert isinstance(error, str) and "ambiguous overlapping storage" in error
    torch.testing.assert_close(storage, torch.ones(8))


def test_reset_allows_views_with_disjoint_address_ranges():
    storage = torch.ones(2, 4)
    left, right = storage[0], storage[1]
    target, draft = nn.Module(), nn.Module()
    target.weight = nn.Parameter(left)
    draft.weight = nn.Parameter(right)
    plan = select_reset_weights(target, draft, "draft")
    assert not isinstance(plan, str)
    zero_selected_weights(plan)
    torch.testing.assert_close(target.weight, torch.ones_like(left))
    torch.testing.assert_close(draft.weight, torch.zeros_like(right))


@pytest.mark.parametrize("kind", ["interleaved", "columns"])
def test_reset_conservatively_rejects_strided_views_with_overlapping_ranges(kind):
    storage = torch.ones(2, 4)
    if kind == "interleaved":
        left, right = storage[:, ::2], storage[:, 1::2]
    else:
        left, right = storage[:, :2], storage[:, 2:]
    target, draft = nn.Module(), nn.Module()
    target.weight = nn.Parameter(left)
    draft.weight = nn.Parameter(right)
    error = select_reset_weights(target, draft, "draft")
    assert isinstance(error, str) and "ambiguous overlapping storage" in error
    torch.testing.assert_close(storage, torch.ones_like(storage))


@pytest.mark.parametrize("owner", ["target", "draft"])
def test_reset_rejects_a_write_through_the_other_models_buffer(owner):
    target, draft = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
    selected, other = (target, draft) if owner == "target" else (draft, target)
    other.register_buffer("alias", selected.weight.detach())
    before = other.alias.clone()
    for model in (owner, "both"):
        error = select_reset_weights(target, draft, model)
        assert isinstance(error, str) and "overlaps buffer 'alias'" in error
    other_plan = select_reset_weights(
        target, draft, "draft" if owner == "target" else "target"
    )
    assert not isinstance(other_plan, str)
    zero_selected_weights(other_plan)
    torch.testing.assert_close(other.alias, before)


def test_dense_dp_replicas_get_distinct_key_prefixes(monkeypatch):
    for name in ("get_tp_group", "get_pp_group", "get_pcp_group"):
        monkeypatch.setattr(gpu_worker, name, lambda: SimpleNamespace(rank_in_group=0))
    prefixes = []
    for dp_rank in (0, 1):
        worker = object.__new__(Worker)
        worker.parallel_config = ParallelConfig(
            data_parallel_size=2, data_parallel_rank=dp_rank
        )
        # Dense DP resets data_parallel_rank to 0 in each engine.
        worker.parallel_config.reconfigure_for_independent_dp_rank()
        prefixes.append(worker._weight_checksum_prefix())
    assert prefixes == ["dp0:pp0:pcp0:tp0:", "dp1:pp0:pcp0:tp0:"]
