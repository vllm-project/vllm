# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import regex as re
import torch

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


class ModuleWithBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)

    def forward(self, x):
        return self.bn(x)


class ModuleWithNestedBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nested_mod = ModuleWithBatchNorm()

    def forward(self, x):
        return self.nested_mod(x)


@pytest.mark.cpu_test
def test_module_with_batchnorm_can_load():
    """Ensure the auto weight loader can load batchnorm stats."""
    mod = ModuleWithBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        yield from mod.state_dict().items()

    new_mod = ModuleWithBatchNorm()

    assert not torch.all(new_mod.bn.running_mean == mod.bn.running_mean)
    assert not torch.all(new_mod.bn.running_var == mod.bn.running_var)
    assert new_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod)
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(new_mod.bn.running_mean == mod.bn.running_mean)
    assert torch.all(new_mod.bn.running_var == mod.bn.running_var)
    assert new_mod.bn.num_batches_tracked.item() == 1


@pytest.mark.cpu_test
def test_module_with_child_containing_batchnorm_can_autoload():
    """Ensure the auto weight loader can load nested modules batchnorm stats."""
    mod = ModuleWithNestedBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        yield from mod.state_dict().items()

    new_mod = ModuleWithNestedBatchNorm()

    assert not torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert not torch.all(
        new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var
    )
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod)
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert torch.all(new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var)
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 1


VOCAB_SIZE = 16
HIDDEN_SIZE = 2


class ModuleWithTiedWeights(torch.nn.Module):
    """Mimics how models tie `lm_head` to the input embeddings."""

    def __init__(self, tie: bool):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.lm_head = ParallelLMHead(VOCAB_SIZE, HIDDEN_SIZE)
        if tie:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)


def make_embedding_weights(value: float) -> torch.Tensor:
    return torch.full((VOCAB_SIZE, HIDDEN_SIZE), value)


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
@pytest.mark.parametrize("tie", [True, False])
def test_module_skip_tied_weights(tie: bool):
    """Tied weights must be loaded once, under the first of their names."""
    mod = ModuleWithTiedWeights(tie)

    weights = [
        ("model.embed_tokens.weight", make_embedding_weights(1.0)),
        ("lm_head.weight", make_embedding_weights(2.0)),
    ]
    loaded = AutoWeightsLoader(mod).load_weights(iter(weights))

    if tie:
        assert loaded == {"model.embed_tokens.weight"}
        assert torch.all(mod.lm_head.weight[:VOCAB_SIZE] == 1.0)
    else:
        assert loaded == {"model.embed_tokens.weight", "lm_head.weight"}
        assert torch.all(mod.lm_head.weight[:VOCAB_SIZE] == 2.0)


@pytest.mark.cpu_test
@pytest.mark.usefixtures("dist_init")
def test_module_skip_tied_weights_without_canonical():
    """Skipping a tied weight must not leave the shared weight uninitialized."""
    mod = ModuleWithTiedWeights(tie=True)

    weights = [("lm_head.weight", make_embedding_weights(2.0))]
    with pytest.raises(ValueError, match="model.embed_tokens.weight"):
        AutoWeightsLoader(mod).load_weights(iter(weights))


class ModuleWithSharedParam(torch.nn.Module):
    """Mimics an MoE router shared between the MLP and its fused experts."""

    def __init__(self):
        super().__init__()
        self.experts = torch.nn.Module()
        self.experts.gate = torch.nn.Linear(2, 2, bias=False)
        self.gate = self.experts.gate


@pytest.mark.cpu_test
def test_module_load_shared_params_that_are_not_tied_embeddings():
    """Only tied embeddings are skipped; other shared params must still load."""
    mod = ModuleWithSharedParam()

    weights = [("gate.weight", torch.Tensor([[1, 2], [3, 4]]))]
    loaded = AutoWeightsLoader(mod).load_weights(iter(weights))

    assert loaded == {"gate.weight"}
    assert torch.all(mod.gate.weight == torch.Tensor([[1, 2], [3, 4]]))


class raise_if_cuda_sync:
    def __enter__(self):
        self.previous_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")

    def __exit__(self, exception_type, exception_value, traceback):
        torch.cuda.set_sync_debug_mode(self.previous_debug_mode)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_merge_multimodal_embeddings_no_sync():
    inputs_embeds = torch.zeros(
        [5, 10], dtype=torch.bfloat16, device=f"{DEVICE_TYPE}:0"
    )
    multimodal_embeddings = [
        torch.ones([3, 10], dtype=torch.bfloat16, device=f"{DEVICE_TYPE}:0")
    ]
    is_multimodal = torch.tensor([True, False, True, True, False], device="cpu")
    with raise_if_cuda_sync():
        _merge_multimodal_embeddings(
            inputs_embeds, multimodal_embeddings, is_multimodal
        )


@pytest.mark.cpu_test
def test_get_rename_mapper_keeps_only_renames():
    """`None` means "do not load", which is meaningless to the consumers of
    this mapper (LoRA name parsing, quantization config layer lists), and
    applying it would silently shrink their lists."""
    mapper = WeightsMapper(
        orig_to_new_regex={re.compile(r"^drop_regex\."): None},
        orig_to_new_substr={"drop_substr": None, "keep_substr": "kept"},
        orig_to_new_stacked={".q_proj": (".qkv_proj", "q")},
        orig_to_new_prefix={"drop_prefix.": None, "keep_prefix.": "kept."},
        orig_to_new_suffix={".drop_suffix": None},
    )
    renames = mapper.get_rename_mapper()

    assert renames.orig_to_new_regex == {}
    assert renames.orig_to_new_substr == {"keep_substr": "kept"}
    assert renames.orig_to_new_stacked == {}
    assert renames.orig_to_new_prefix == {"keep_prefix.": "kept."}
    assert renames.orig_to_new_suffix == {}

    # Names the full mapper drops now survive unchanged.
    for name in ("drop_regex.w", "drop_substr.w", "drop_prefix.w", "w.drop_suffix"):
        assert mapper._map_name(name) is None
        assert renames._map_name(name) == name
