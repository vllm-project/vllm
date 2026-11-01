# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import regex as re
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    RankLocalTransform,
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


def _pp_rank_1_module() -> torch.nn.Module:
    module = torch.nn.Module()
    module.layers = torch.nn.ModuleList(
        [PPMissingLayer(), torch.nn.Linear(2, 2, bias=False)]
    )
    return module


def test_auto_weights_loader_keeps_other_stages_for_plain_transforms():
    """A plain transform may move another stage's weight into this stage."""
    module = _pp_rank_1_module()

    def move_to_layer_1(weights):
        for name, tensor in weights:
            yield name.replace("layers.0.", "layers.1."), tensor

    loaded = AutoWeightsLoader(module).load_weights(
        [("layers.0.weight", torch.full((2, 2), 3.0))],
        mapper=WeightsMapper(weights_transforms=[move_to_layer_1]),
    )

    assert loaded == {"layers.1.weight"}
    assert torch.equal(module.layers[1].weight, torch.full((2, 2), 3.0))


def test_auto_weights_loader_drops_other_stages_before_rank_local_transforms():
    """A rank-local merge never sees the parts of another stage's weights."""
    module = _pp_rank_1_module()
    seen: list[str] = []

    def record(weights):
        for name, tensor in weights:
            seen.append(name)
            yield name, tensor

    weights = [
        (f"layers.{i}.weight{half}", torch.full((1, 2), float(2 * i + j)))
        for i in range(2)
        for j, half in enumerate((".first", ".second"))
    ]
    transform = RankLocalTransform(lambda w: _pair_halves(record(w)))

    loaded = AutoWeightsLoader(module).load_weights(
        weights, mapper=WeightsMapper(weights_transforms=[transform])
    )

    assert loaded == {"layers.1.weight"}
    assert seen == ["layers.1.weight.first", "layers.1.weight.second"]
    assert torch.equal(module.layers[1].weight, torch.tensor([[2.0] * 2, [3.0] * 2]))


def _ep_rank_1_experts(
    num_experts: int, *, enable_eplb: bool = False, use_global_sf: bool = False
) -> MoERunner:
    """A MoE layer on expert parallel rank 1 of 2, holding the upper half."""
    parallel_config = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        use_ep=True,
        ep_size=2,
        ep_rank=1,
        enable_eplb=enable_eplb,
    )
    experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(experts)
    experts.moe_config = SimpleNamespace(
        moe_parallel_config=parallel_config, num_logical_experts=num_experts
    )
    experts.quant_config = None
    experts.quant_method = SimpleNamespace(use_global_sf=use_global_sf)
    experts.expert_map_manager = ExpertMapManager(
        max_num_batched_tokens=16,
        top_k=2,
        global_num_experts=num_experts,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel_config,
        placement_strategy="linear",
        enable_eplb=False,
    )
    runner = MoERunner.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = experts
    return runner


def test_auto_weights_loader_drops_other_ranks_experts_before_rank_local_transforms():
    """A rank-local transform only sees the expert weights this rank loads."""
    module = torch.nn.Module()
    module.layers = torch.nn.ModuleList(
        [PPMissingLayer(), *(torch.nn.Module() for _ in range(3))]
    )
    module.layers[1].experts = _ep_rank_1_experts(4)
    module.layers[2].experts = _ep_rank_1_experts(8, use_global_sf=True)
    module.layers[3].experts = _ep_rank_1_experts(4, enable_eplb=True)
    kept = [
        "layers.1.experts.2.w1.weight",
        "layers.1.experts.4.w1.weight",  # past the routed experts
        "layers.1.experts.w13_weight",  # all experts in one tensor
        "layers.1.shared_experts.0.w1.weight",  # not a MoE layer
        "layers.2.experts.0.w1.input_scale",  # reduced to a global scale
        "layers.2.experts.4.w1.weight_scale",
        "layers.3.experts.0.w1.weight",  # EPLB loads by physical expert ID
    ]
    dropped = [
        "layers.0.experts.2.w1.weight",  # on another pipeline stage
        "layers.1.experts.1.w1.weight",
        "layers.1.experts.1.w1.input_scale",
        "layers.2.experts.3.w1.weight_scale",
    ]
    seen: list[str] = []

    def record(weights):
        seen.extend(name for name, _ in weights)
        yield from ()

    AutoWeightsLoader(module).load_weights(
        [(name, torch.zeros(1)) for name in kept + dropped],
        mapper=WeightsMapper(weights_transforms=[RankLocalTransform(record)]),
    )

    assert seen == kept
