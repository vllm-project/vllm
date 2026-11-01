# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
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
) -> CheckpointWeightPatch:
    return CheckpointWeightPatch(
        name=name,
        shape=(4,),
        dtype=torch.float32,
        values=torch.tensor(values),
        indices=None if indices is None else torch.tensor(indices, dtype=torch.int32),
    )


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


def _tied_embedding_model(vocab: int, hidden: int, *, dtype: torch.dtype):
    from types import SimpleNamespace

    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from vllm.model_executor.models.utils import AutoWeightsLoader

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = VocabParallelEmbedding(
                vocab, hidden, params_dtype=dtype
            )

    class TiedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = ParallelLMHead(vocab, hidden, params_dtype=dtype)
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        def load_weights(self, weights):
            return AutoWeightsLoader(self).load_weights(weights)

    return TiedModel, SimpleNamespace(rank_in_group=0, world_size=1), parallel_state


def test_tied_embedding_patches_survive_default_chunking(monkeypatch):
    """Regression for #60151: tied alias + canonical must not fail when
    chunking splits them across load_weights calls."""
    vocab, hidden = 2048, 1024
    TiedModel, tp, parallel_state = _tied_embedding_model(
        vocab, hidden, dtype=torch.bfloat16
    )
    monkeypatch.setattr(parallel_state, "_TP", tp)

    def make_patches():
        numel = vocab * hidden
        indices = torch.arange(0, numel, 17)
        values = torch.ones(indices.numel(), dtype=torch.bfloat16)
        return [
            CheckpointWeightPatch(
                name,
                (vocab, hidden),
                torch.bfloat16,
                values.clone(),
                indices.clone(),
            )
            for name in ("lm_head.weight", "model.embed_tokens.weight")
        ]

    model = TiedModel().requires_grad_(False)
    load_checkpoint_weight_patches(model, make_patches(), max_chunk_bytes=4 << 30)

    model = TiedModel().requires_grad_(False)
    loaded = load_checkpoint_weight_patches(
        model, make_patches(), max_chunk_bytes=1 << 20
    )
    assert "model.embed_tokens.weight" in loaded
    assert torch.count_nonzero(model.model.embed_tokens.weight) > 0


def test_tied_alias_only_patch_loads_under_canonical_name(monkeypatch):
    vocab, hidden = 128, 32
    TiedModel, tp, parallel_state = _tied_embedding_model(
        vocab, hidden, dtype=torch.float32
    )
    monkeypatch.setattr(parallel_state, "_TP", tp)

    indices = torch.tensor([0, 1, 2], dtype=torch.int64)
    values = torch.tensor([3.0, 4.0, 5.0])
    model = TiedModel().requires_grad_(False)
    loaded = load_checkpoint_weight_patches(
        model,
        [
            CheckpointWeightPatch(
                "lm_head.weight",
                (vocab, hidden),
                torch.float32,
                values,
                indices,
            )
        ],
    )
    assert loaded == {"model.embed_tokens.weight"}
    flat = model.model.embed_tokens.weight.detach().flatten()
    assert flat[0].item() == 3.0
    assert flat[1].item() == 4.0
    assert flat[2].item() == 5.0
