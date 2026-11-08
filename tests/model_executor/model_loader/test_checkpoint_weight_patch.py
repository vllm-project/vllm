# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    _validate_sparse_patches,
    load_checkpoint_weight_patches,
)

pytestmark = pytest.mark.cpu_test


class _PackedTPModel(torch.nn.Module):
    """Model that packs the TP rank 1 shards of Q and K into one parameter."""

    def __init__(self):
        super().__init__()
        self.packed_weight = torch.nn.Parameter(torch.full((4,), -1.0))
        self.load_calls: list[list[str]] = []

    def load_weights(self, weights):
        weights = list(weights)
        self.load_calls.append([name for name, _ in weights])
        loaded_names = set()
        for name, loaded_weight in weights:
            if name == "q_proj.weight":
                destination = self.packed_weight.data[:2]
            elif name == "k_proj.weight":
                destination = self.packed_weight.data[2:]
            else:
                raise AssertionError(name)
            # Slice each full checkpoint tensor and load its second half for TP rank 1.
            destination.copy_(loaded_weight.narrow(0, 2, 2))
            loaded_names.add(name)
        return loaded_names


def _make_patch(
    name: str,
    *,
    values: list[float],
    indices: list[int] | None = None,
    shape: tuple[int, ...] = (4,),
    dtype: torch.dtype = torch.float32,
) -> CheckpointWeightPatch:
    return CheckpointWeightPatch(
        name=name,
        shape=shape,
        dtype=dtype,
        values=torch.tensor(values, dtype=dtype),
        indices=None if indices is None else torch.tensor(indices, dtype=torch.int32),
    )


def _q(indices: list[int], values: list[float]) -> CheckpointWeightPatch:
    return _make_patch("q_proj.weight", indices=indices, values=values)


def _k(
    indices: list[int], values: list[float], shape: tuple[int, ...] = (4,)
) -> CheckpointWeightPatch:
    return _make_patch("k_proj.weight", indices=indices, values=values, shape=shape)


@pytest.mark.parametrize("max_chunk_bytes", [1, 1 << 20])
@pytest.mark.parametrize(
    ("patches", "validate_unique", "expected"),
    [
        # Valid: indices may restart or repeat across patches, or come out of order.
        (
            [_k([5], [2.0], shape=(8,)), _q([3], [1.0])],
            True,
            [-1.0, 1.0, -1.0, -1.0],
        ),
        ([_q([2], [1.0]), _q([3, 2], [2.0, 3.0])], True, [3.0, 2.0, -1.0, -1.0]),
        ([_q([], []), _k([3], [4.0]), _k([], [])], True, [-1.0, -1.0, -1.0, 4.0]),
        ([_q([2, 2], [7.0, 7.0])], False, [7.0, -1.0, -1.0, -1.0]),
        # Invalid values and indices, in the first or a later patch.
        ([_q([2], [1.0]), _k([2], [float("nan")])], True, None),
        ([_q([-1, 2], [1.0, 2.0]), _k([1], [3.0])], True, None),
        ([_q([2, 4], [1.0, 2.0])], True, None),
        # Out of order, so the ends of a patch say nothing about its range.
        ([_q([3, -1], [1.0, 2.0])], False, None),
        ([_q([4, 3], [1.0, 2.0])], False, None),
        # Each patch has its own limit.
        ([_q([1, 0], [1.0, 2.0]), _k([3], [1.0], shape=(2,))], True, None),
        # Dtypes that cannot be concatenated are still checked.
        (
            [
                _make_patch(
                    "q_proj.weight",
                    indices=[4],
                    values=[1.0],
                    dtype=torch.float8_e4m3fn,
                ),
                _k([2], [1.0]),
            ],
            True,
            None,
        ),
        # Duplicates: unsorted, at the end of a patch, at the start of a later one.
        ([_q([3, 2, 3], [1.0, 2.0, 3.0])], True, None),
        ([_q([1, 2, 2], [1.0, 2.0, 3.0]), _k([3], [1.0])], True, None),
        ([_q([1], [1.0]), _k([2, 2], [1.0, 2.0])], True, None),
    ],
)
def test_sparse_patches_are_validated_before_loading(
    patches, validate_unique, expected, max_chunk_bytes
):
    model = _PackedTPModel()
    kwargs = {
        "max_chunk_bytes": max_chunk_bytes,
        "validate_unique_indices": validate_unique,
    }

    if expected is None:
        with pytest.raises(ValueError, match="Sparse checkpoint patches contain"):
            load_checkpoint_weight_patches(model, patches, **kwargs)
        assert model.load_calls == []
    else:
        load_checkpoint_weight_patches(model, patches, **kwargs)
        assert model.packed_weight.tolist() == expected


@pytest.mark.parametrize("indices", [[0, 2**31 - 1], [2**31 - 1, 0]])
def test_int32_indices_are_valid_in_tensors_past_the_int32_range(indices):
    # Through the helper, since loading would allocate the whole tensor.
    patch = _make_patch(
        "q_proj.weight", indices=indices, values=[1.0, 2.0], shape=(1 << 31,)
    )
    _validate_sparse_patches([patch], 1 << 20, validate_unique_indices=True)


def test_dense_and_sparse_patches_follow_packed_tp_loader():
    model = _PackedTPModel()

    # Seed the TP-local packed weights through the dense path.
    dense_loaded = load_checkpoint_weight_patches(
        model,
        [
            _make_patch("q_proj.weight", values=[0.0, 1.0, 2.0, 3.0]),
            _make_patch("k_proj.weight", values=[10.0, 11.0, 12.0, 13.0]),
        ],
    )
    assert dense_loaded == {"q_proj.weight", "k_proj.weight"}
    assert torch.equal(model.packed_weight, torch.tensor([2.0, 3.0, 12.0, 13.0]))

    # Sparse indices address the full checkpoint tensor; index 0 is outside this shard.
    model.load_calls.clear()
    original_copy = torch.Tensor.copy_
    sparse_loaded = load_checkpoint_weight_patches(
        model,
        [
            _make_patch(
                "q_proj.weight",
                indices=[0, 3],
                values=[100.0, 30.0],
            ),
            _make_patch("k_proj.weight", indices=[2], values=[20.0]),
            _make_patch("q_proj.weight", indices=[2], values=[22.0]),
        ],
    )

    assert sparse_loaded == {"q_proj.weight", "k_proj.weight"}
    assert model.load_calls == [
        ["q_proj.weight", "k_proj.weight"],
        ["q_proj.weight"],
    ]
    assert torch.equal(model.packed_weight, torch.tensor([22.0, 30.0, 20.0, 13.0]))
    assert torch.Tensor.copy_ is original_copy

    # The empty result tells the helper that the loader intentionally made no write.
    class NoLocalWeightModel(torch.nn.Module):
        """Model that deliberately owns none of the supplied weights."""

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def load_weights(self, _weights):
            return set()

    no_local_weight_model = NoLocalWeightModel()
    no_local_weight_loaded = load_checkpoint_weight_patches(
        no_local_weight_model,
        [_make_patch("q_proj.weight", indices=[3], values=[30.0])],
    )
    assert no_local_weight_loaded == set()
    assert torch.equal(no_local_weight_model.weight, torch.tensor([1.0]))
