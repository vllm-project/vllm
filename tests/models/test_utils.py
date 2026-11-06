# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import regex as re
import torch

from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models import ernie_mtp, mimo_mtp
from vllm.model_executor.models.bloom import BloomForCausalLM
from vllm.model_executor.models.falcon import deinterleave_qkv
from vllm.model_executor.models.openvla import OpenVLAForActionPrediction
from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM
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


def test_weights_mapper_stacks_one_weight_into_several_shards():
    """One checkpoint tensor can feed more than one shard of a stacked
    parameter, and a dropped name stays dropped rather than reappearing as a
    shard of the stacked one."""
    mapper = WeightsMapper(
        orig_to_new_substr={"layers.1.k_proj.": None},
        orig_to_new_stacked={".k_proj.": (".qkv_proj.", ["k", "v"])},
    )
    weight = torch.ones(2)
    weights = [(f"layers.{i}.k_proj.weight", weight) for i in (0, 1)]

    mapped = list(mapper.apply(weights))

    assert [(name, w.shard_id) for name, w in mapped] == [
        ("layers.0.qkv_proj.weight", "k"),
        ("layers.0.qkv_proj.weight", "v"),
    ]
    # Each shard needs its own tensor object, but they alias one allocation.
    assert mapped[0][1] is not mapped[1][1]
    assert mapped[0][1].data_ptr() == mapped[1][1].data_ptr() == weight.data_ptr()


def _double(weights):
    """Stateless recode: double every tensor."""
    for name, tensor in weights:
        yield name, tensor * 2


def _pair_halves(weights):
    """Stateful merge: cat a name's ``.first``/``.second`` halves on arrival."""
    pending: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in weights:
        for half in (".first", ".second"):
            if name.endswith(half):
                base = name[: -len(half)]
                parts = pending.setdefault(base, {})
                parts[half] = tensor
                if len(parts) == 2:
                    del pending[base]
                    yield base, torch.cat([parts[".first"], parts[".second"]])
                break
        else:
            yield name, tensor
    if pending:
        raise ValueError(f"unpaired: {sorted(pending)}")


def test_weights_mapper_transform_runs_before_name_mapping():
    """Transform-emitted names flow through the name mappings, and an N->1
    merge completes however its inputs are ordered."""
    mapper = WeightsMapper(
        orig_to_new_suffix={".w": ".w_renamed"},
        weights_transforms=[_pair_halves],
    )
    first, second = torch.ones(2), torch.zeros(2)
    keep = torch.ones(1)
    weights = [("keep.w", keep), ("e.w.second", second), ("e.w.first", first)]

    mapped = list(mapper.apply(weights))

    assert [name for name, _ in mapped] == ["keep.w_renamed", "e.w_renamed"]
    assert mapped[0][1] is keep
    assert torch.equal(mapped[1][1], torch.cat([first, second]))


def test_weights_mapper_transform_can_recode_values():
    mapper = WeightsMapper(weights_transforms=[_double])

    mapped = list(mapper.apply([("w", torch.ones(2))]))

    assert torch.equal(mapped[0][1], torch.full((2,), 2.0))


def test_weights_mapper_transform_raises_at_iteration_not_at_apply():
    """Incomplete input raises when the stream is exhausted; merely calling
    apply() (or the transform) does nothing yet."""
    mapper = WeightsMapper(weights_transforms=[_pair_halves])
    stream = mapper.apply([("e.w.first", torch.ones(2))])

    with pytest.raises(ValueError, match="unpaired"):
        list(stream)


def test_weights_mapper_transform_state_is_fresh_per_apply():
    """The mapper is shared and long-lived; each apply() must start clean."""
    mapper = WeightsMapper(weights_transforms=[_pair_halves])
    with pytest.raises(ValueError, match="unpaired"):
        list(mapper.apply([("e.w.first", torch.ones(2))]))
    # A leaked pending entry would pair with this stream instead of raising.
    with pytest.raises(ValueError, match="unpaired"):
        list(mapper.apply([("e.w.second", torch.zeros(2))]))

    first, second = torch.ones(2), torch.zeros(2)
    mapped = list(mapper.apply([("e.w.second", second), ("e.w.first", first)]))
    assert torch.equal(mapped[0][1], torch.cat([first, second]))


def test_weights_mapper_or_concatenates_transforms_in_order():
    """Self's transforms run before other's; a mapper without any adds none."""
    seen = []

    def tag(label):
        def rename(weights):
            for name, tensor in weights:
                seen.append((label, name))
                yield f"{name}.{label}", tensor

        return rename

    mapper = WeightsMapper(weights_transforms=[tag("x"), tag("y")]) | WeightsMapper(
        weights_transforms=[tag("z")]
    )
    (mapped,) = list(mapper.apply([("w", torch.ones(1))]))

    assert mapped[0] == "w.x.y.z"
    assert seen == [("x", "w"), ("y", "w.x"), ("z", "w.x.y")]
    assert (WeightsMapper() | WeightsMapper()).weights_transforms == []
    one_side = WeightsMapper() | WeightsMapper(weights_transforms=[_pair_halves])
    assert one_side.weights_transforms == [_pair_halves]


def _double_w(weights):
    """Stateless dispatch by name, the idiomatic inline ``endswith`` form."""
    for name, tensor in weights:
        if name.endswith(".w"):
            tensor = tensor * 2
        yield name, tensor


def test_weights_mapper_transforms_see_earlier_emitted_names():
    mapper = WeightsMapper(weights_transforms=[_pair_halves, _double_w])

    (mapped,) = list(
        mapper.apply([("e.w.second", torch.zeros(1)), ("e.w.first", torch.ones(1))])
    )

    # The halves merge into "e.w", which the suffix mapping then doubles.
    assert torch.equal(mapped[1], torch.tensor([2.0, 0.0]))


def _qkv_module(output_dim: int | None, shape: tuple[int, ...]):
    """A module with a `query_key_value.weight` tagged with `output_dim`."""
    module = torch.nn.Module()
    module.query_key_value = torch.nn.Module()
    weight = torch.nn.Parameter(torch.zeros(shape))
    if output_dim is not None:
        weight.output_dim = output_dim
    module.query_key_value.weight = weight
    return module


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads"),
    [(2, 2), (4, 2), (4, 1)],
    ids=["bloom", "new_decoder", "multi_query"],
)
@pytest.mark.parametrize("output_dim", [0, 1], ids=["dense", "gptq"])
def test_deinterleave_qkv_restores_block_layout(num_heads, num_kv_heads, output_dim):
    q_per_kv = num_heads // num_kv_heads
    q = torch.arange(num_heads * 3.0).reshape(num_heads, 3)
    k = 100 + torch.arange(num_kv_heads * 3.0).reshape(num_kv_heads, 3)
    v = 200 + torch.arange(num_kv_heads * 3.0).reshape(num_kv_heads, 3)
    expected = torch.cat([q, k, v])
    groups = [
        [q[g * q_per_kv : (g + 1) * q_per_kv], k[g:][:1], v[g:][:1]]
        for g in range(num_kv_heads)
    ]
    interleaved = torch.cat([torch.cat(group) for group in groups])
    if output_dim == 1:
        interleaved, expected = interleaved.unsqueeze(0), expected.unsqueeze(0)
    module = _qkv_module(output_dim, interleaved.shape)

    weights = [("query_key_value.weight", interleaved)]
    ((_, loaded),) = deinterleave_qkv(weights, module, num_heads, num_kv_heads)

    assert torch.equal(loaded, expected)


def test_deinterleave_qkv_skips_weights_without_a_local_output_dim():
    """PP-missing layers and unpartitioned parameters must pass through as is."""
    weight = torch.arange(12.0).reshape(6, 2)
    weights = [("query_key_value.weight", weight)]

    ((_, missing),) = deinterleave_qkv(weights, torch.nn.Module(), 2, 2)
    no_dim = _qkv_module(None, weight.shape)
    ((_, untagged),) = deinterleave_qkv(weights, no_dim, 2, 2)

    assert missing is weight
    assert untagged is weight


def test_bloom_mapper_adds_the_prefix_without_renaming_quant_config_names():
    mapper = BloomForCausalLM.hf_to_vllm_mapper

    mapped = [name for name, _ in mapper.apply([("h.0.w", torch.ones(1))])]

    assert mapped == ["transformer.h.0.w"]
    assert mapper.get_rename_mapper().map_name("mlp.dense") == "mlp.dense"


@pytest.mark.parametrize(
    ("module", "name", "kept"),
    [
        (mimo_mtp, "model.mtp_layers.0.self_attn.q_proj.weight", True),
        (mimo_mtp, "model.embed_tokens.weight", True),
        (mimo_mtp, "lm_head.weight", True),
        (mimo_mtp, "model.layers.0.self_attn.q_proj.weight", False),
        (ernie_mtp, "mtp_block.0.mlp.gate_proj.weight", True),
        (ernie_mtp, "model.embed_tokens.weight", True),
        (ernie_mtp, "model.layers.0.mlp.gate_proj.weight", False),
    ],
)
def test_mtp_mappers_keep_only_mtp_weights(module, name, kept):
    mapper = WeightsMapper(orig_to_new_regex={module._NON_MTP_WEIGHT: None})

    assert (mapper.map_name(name) == name) is kept


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("vision_backbone.featurizer.x", "vision_backbone.dinov2_featurizer.x"),
        ("vision_backbone.fused_featurizer.x", "vision_backbone.siglip_featurizer.x"),
        ("a.ls1.scale_factor", "a.ls1.gamma"),
        ("a.ls2.scale_factor", "a.ls2.gamma"),
    ],
)
def test_openvla_mapper_renames_vision_weights(name, expected):
    assert OpenVLAForActionPrediction.hf_to_vllm_mapper.map_name(name) == expected


def test_qwen2_moe_mapper_unsqueezes_only_1d_shared_expert_gates():
    name = "model.layers.0.mlp.shared_expert_gate.weight"
    weights = [(name, torch.ones(4)), (name, torch.ones(1, 4))]

    mapped = Qwen2MoeForCausalLM.hf_to_vllm_mapper.apply(weights)

    assert [tensor.dim() for _, tensor in mapped] == [2, 2]
