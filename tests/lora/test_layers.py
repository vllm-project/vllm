import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from vllm.config import LoRAConfig
from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLora, RowParallelLinearWithShardedLoRA)
# yapf conflicts with isort for this block
# yapf: disable
from vllm.lora.layers import (BaseLayerWithLoRA, ColumnParallelLinearWithLoRA,
                              LogitsProcessorWithLoRA, LoRAMapping,
                              MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLora,
                              QKVParallelLinearWithLora,
                              RowParallelLinearWithLoRA,
                              VocabParallelEmbeddingWithLoRA)
# yapf: enable
from vllm.lora.models import (LoRALayerWeights, PackedLoRALayerWeights,
                              convert_mapping)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.utils import set_random_seed

from .utils import DummyLoRAManager

TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.float32: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 2e-2),
}
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


def get_random_id_to_index(num_loras: int,
                           num_slots: int,
                           log: bool = True) -> List[Optional[int]]:
    """Creates a random lora_id_to_index mapping.

    Args:
        num_loras: The number of active loras in the mapping.
        num_slots: The number of slots in the mapping. Must be larger
            than num_loras.
        log: Whether to log the output.
    """

    if num_loras > num_slots:
        raise ValueError(
            f"num_loras is higher than num_slots: {num_loras} > {num_slots}. "
            "num_loras must be less than or equal to num_slots.")

    slots: List[Optional[int]] = [None] * num_slots
    random_slot_selections = (torch.randperm(num_slots)[:num_loras]).tolist()
    for lora_id, slot_idx in enumerate(random_slot_selections, start=1):
        slots[slot_idx] = lora_id

    if log:
        print(f"Created lora_id_to_index mapping: {slots}.")

    return slots


def populate_loras(
    id_to_index: List[Optional[int]],
    layer: BaseLayerWithLoRA,
    layer_weights: torch.Tensor,
    generate_embeddings_tensor: int = 0,
    repeats: int = 1,
) -> Tuple[Dict[int, LoRALayerWeights], Dict[int, List[LoRALayerWeights]]]:
    """This method populates the lora layers with lora weights.

    Args:
        id_to_index: a list of lora ids. The index of the lora id
            represents which memory slot the lora matrices are
            stored in. A None value indicates a free slot.
        layer: the LoRAlayer to populate.
        layer_weights: the PyTorch tensor containing the layer's
            weights.
        generate_embeddings_tensor: whether to generate an
            embeddings tensor for each LoRA.
        repeats: must only be set for column parallel packed
            layers. Indicates the number of loras to compose
            together to create a single lora layer.
    """

    # Dictionary that maps the lora ID to the
    # corresponding lora weights.
    lora_dict: Dict[int, LoRALayerWeights] = dict()

    # Dictionary that maps the lora ID to the
    # corresponding subloras.
    sublora_dict: Dict[int, List[LoRALayerWeights]] = dict()

    for slot_idx, lora_id in enumerate(id_to_index):
        if lora_id is not None:
            subloras = []
            sublora_len = layer_weights.shape[0] // repeats
            for i in range(repeats):
                sublora = DummyLoRAManager().init_random_lora(
                    module_name=f"fake_{i}",
                    weight=layer_weights,
                    generate_embeddings_tensor=generate_embeddings_tensor,
                )
                sublora.lora_b = sublora.lora_b[:, (sublora_len *
                                                    i):(sublora_len * (i + 1))]
                sublora.optimize()
                subloras.append(sublora)

            lora = PackedLoRALayerWeights.pack(
                subloras) if repeats > 1 else subloras[0]

            layer.set_lora(
                slot_idx,
                lora_a=lora.lora_a,
                lora_b=lora.lora_b,
                embeddings_tensor=lora.embeddings_tensor,
            )

            lora_dict[lora_id] = lora
            sublora_dict[lora_id] = subloras

    return lora_dict, sublora_dict


def create_random_inputs(
    active_lora_ids: List[int],
    num_inputs: int,
    input_size: Tuple[int, ...],
    input_range: Tuple[float, float],
    input_type: torch.dtype = torch.int,
) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """Creates random inputs.

    Args:
        active_lora_ids: lora IDs of active lora weights.
        num_inputs: the number of inputs to create.
        input_size: the size of each individual input.
        input_range: the range of values to include in the input.
            input_range[0] <= possible input values < input_range[1]
        input_type: the type of values in the input.
    """

    low, high = input_range

    inputs, index_mapping, prompt_mapping = [], [], []
    for _ in range(num_inputs):
        if input_type == torch.int:
            inputs.append(
                torch.randint(low=int(low), high=int(high), size=input_size))
        else:
            inputs.append(
                torch.rand(size=input_size, dtype=input_type) * high + low)

        lora_id = random.choice(active_lora_ids)
        index_mapping += [lora_id] * input_size[0]
        prompt_mapping += [lora_id]

    return inputs, index_mapping, prompt_mapping


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4, 8])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("vocab_size", [512, 32000, 64000, 128000])
def test_embeddings(dist_init, num_loras, device, vocab_size) -> None:

    torch.set_default_device(device)
    max_loras = 8
    lora_config = LoRAConfig(max_loras=max_loras,
                             max_lora_rank=8,
                             lora_dtype=torch.float16)

    def create_random_embedding_layer():
        embedding = VocabParallelEmbedding(vocab_size, 256)
        embedding.weight.data = torch.rand_like(embedding.weight.data)
        embedding.weight.data[vocab_size:, :] = 0
        lora_embedding = VocabParallelEmbeddingWithLoRA(embedding)
        lora_embedding.create_lora_weights(max_loras, lora_config)

        return embedding, lora_embedding

    for i in range(10):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)
        embedding, lora_embedding = create_random_embedding_layer()

        lora_dict, _ = populate_loras(
            id_to_index,
            layer=lora_embedding,
            layer_weights=embedding.weight.T,
        )

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=num_loras * 3,
            input_size=(200, ),
            input_range=(1, vocab_size),
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)

        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras,
                                       vocab_size,
                                       lora_config.lora_extra_vocab_size)
        lora_embedding.set_mapping(*mapping_info)

        lora_result = lora_embedding(torch.cat(inputs))

        expected_results = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = embedding(input_)
            after_a = F.embedding(
                input_,
                lora.lora_a,
            )
            result += (after_a @ lora.lora_b)
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)

        # Check that resetting the lora weights succeeds

        for slot_idx in range(max_loras):
            lora_embedding.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=num_loras * 3,
            input_size=(200, ),
            input_range=(1, vocab_size),
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)

        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras,
                                       vocab_size,
                                       lora_config.lora_extra_vocab_size)
        lora_embedding.set_mapping(*mapping_info, )

        lora_result = lora_embedding(torch.cat(inputs))
        expected_result = embedding(torch.cat(inputs))

        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)


@torch.inference_mode()
# @pytest.mark.skip(
#     reason="Fails when loras are in any slot other than the first.")
@pytest.mark.parametrize("num_loras", [1, 2, 4, 8])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("vocab_size", [512, 32000, 64000, 128000])
def test_embeddings_with_new_embeddings(dist_init, num_loras, device,
                                        vocab_size) -> None:

    torch.set_default_device(device)
    max_loras = 8
    lora_config = LoRAConfig(max_loras=max_loras,
                             max_lora_rank=8,
                             lora_dtype=torch.float16)

    def create_random_embedding_layer():
        embedding = VocabParallelEmbedding(vocab_size, 256)
        embedding_data = torch.rand_like(embedding.weight.data)
        embedding.weight.data = embedding_data
        embedding.weight.data[vocab_size:, :] = 0
        expanded_embedding = VocabParallelEmbedding(
            vocab_size + lora_config.lora_extra_vocab_size * max_loras,
            256,
            org_num_embeddings=vocab_size)
        expanded_embedding.weight.data[:vocab_size, :] = embedding_data
        # We need to deepcopy the embedding as it will be modified
        # in place
        lora_embedding = VocabParallelEmbeddingWithLoRA(
            deepcopy(expanded_embedding))
        lora_embedding.create_lora_weights(max_loras, lora_config)

        return expanded_embedding, lora_embedding

    for i in range(10):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)
        expanded_embedding, lora_embedding = create_random_embedding_layer()
        lora_dict, _ = populate_loras(
            id_to_index,
            layer=lora_embedding,
            layer_weights=torch.zeros(
                (256, vocab_size + lora_config.lora_extra_vocab_size)),
            generate_embeddings_tensor=256,
        )

        # All embeddings tensors have the same shape.
        embeddings_tensors = [
            lora_dict[id].embeddings_tensor for id in sorted(lora_dict.keys())
        ]
        embeddings_tensor_len = embeddings_tensors[0].shape[0]

        # Add empty embeddings_tensors for unoccupied lora slots.
        for _ in range(max_loras - len(embeddings_tensors)):
            embeddings_tensors.append(torch.zeros(embeddings_tensors[0].shape))

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=num_loras * 3,
            input_size=(200, ),
            input_range=(1, vocab_size),
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)

        original_inputs = deepcopy(inputs)

        # Force some of the inputs to be in the extended embeddings range
        # to guarantee that their behavior is tested.
        for input_, original_input_, lora_id in zip(inputs, original_inputs,
                                                    prompt_mapping):
            embedding_id = lora_id - 1
            input_[-1] = vocab_size + (embedding_id * embeddings_tensor_len)
            original_input_[-1] = vocab_size
            input_[-2] = vocab_size + (
                (embedding_id + 1) * embeddings_tensor_len - 1)
            original_input_[-2] = vocab_size + embeddings_tensor_len - 1

        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras,
                                       vocab_size,
                                       lora_config.lora_extra_vocab_size)
        lora_embedding.set_mapping(*mapping_info, )

        expanded_embedding.weight[vocab_size:vocab_size +
                                  (embeddings_tensor_len *
                                   max_loras)] = torch.cat(embeddings_tensors)

        lora_result = lora_embedding(torch.cat(original_inputs))

        expected_results = []
        for input_, original_input_, lora_id in zip(inputs, original_inputs,
                                                    prompt_mapping):
            lora = lora_dict[lora_id]
            result = expanded_embedding(input_)
            after_a = F.embedding(
                original_input_,
                lora.lora_a,
            )
            result += (after_a @ lora.lora_b)
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)

        # Check that resetting the lora weights succeeds

        for slot_idx in range(max_loras):
            lora_embedding.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=num_loras * 3,
            input_size=(200, ),
            input_range=(1, vocab_size),
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)

        original_inputs = deepcopy(inputs)

        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras,
                                       vocab_size,
                                       lora_config.lora_extra_vocab_size)
        lora_embedding.set_mapping(*mapping_info, )

        lora_result = lora_embedding(torch.cat(original_inputs))
        expected_result = expanded_embedding(torch.cat(inputs))

        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4, 8])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("vocab_size", [512, 32000, 64000, 128000])
def test_lm_head_logits_processor(dist_init, num_loras, device,
                                  vocab_size) -> None:

    torch.set_default_device(device)
    max_loras = 8
    lora_config = LoRAConfig(max_loras=max_loras,
                             max_lora_rank=8,
                             lora_dtype=torch.float16)

    def _pretest():
        linear = ParallelLMHead(vocab_size + lora_config.lora_extra_vocab_size,
                                1024,
                                vocab_size,
                                params_dtype=torch.float16)
        linear.weight.data = torch.rand_like(linear.weight.data)
        linear.weight.data[:, vocab_size:] = 0
        logits_processor = LogitsProcessor(
            vocab_size + lora_config.lora_extra_vocab_size, vocab_size)
        lora_logits_processor = LogitsProcessorWithLoRA(
            logits_processor, 1024, linear.weight.dtype, linear.weight.device)
        lora_logits_processor.create_lora_weights(max_loras, lora_config)

        return linear, logits_processor, lora_logits_processor

    for i in range(10):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)
        linear, logits_processor, lora_logits_processor = _pretest()

        # NOTE: all the generated loras share the same embeddings tensor.
        lora_dict, _ = populate_loras(
            id_to_index,
            layer=lora_logits_processor,
            layer_weights=linear.weight,
            generate_embeddings_tensor=1024,
        )
        embeddings_tensor = list(lora_dict.values())[0].embeddings_tensor
        embeddings_tensor_len = embeddings_tensor.shape[0]

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=8 * num_loras,  # * 3,
            input_size=(1, 1024),
            input_range=(0, 1),
            input_type=torch.float16,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)

        input_ = torch.rand(20, 1024)
        mapping_info = convert_mapping(
            lora_mapping,
            id_to_index,
            max_loras,
            vocab_size,
            lora_config.lora_extra_vocab_size,
        )
        lora_logits_processor.set_mapping(*mapping_info, )

        lora_result = lora_logits_processor._get_logits(
            hidden_states=torch.cat(inputs),
            embedding=linear.weight,
            embedding_bias=None)

        original_weight = linear.weight.clone()

        linear.weight[logits_processor.
                      org_vocab_size:logits_processor.org_vocab_size +
                      embeddings_tensor_len] = embeddings_tensor

        logits_processor.org_vocab_size = (vocab_size +
                                           lora_config.lora_extra_vocab_size)
        expected_results = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = logits_processor._get_logits(hidden_states=input_,
                                                  embedding=linear.weight,
                                                  embedding_bias=None)
            result[:, vocab_size + embeddings_tensor_len:] = float("-inf")
            result += input_ @ lora.lora_a @ lora.lora_b * lora.scaling
            expected_results.append(result)
        expected_result = torch.cat(expected_results)
        logits_processor.org_vocab_size = vocab_size

        # Check that resetting the lora weights succeeds

        for slot_idx in range(max_loras):
            lora_logits_processor.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=8 * num_loras * 3,
            input_size=(1, 1024),
            input_range=(0, 1),
            input_type=torch.float16,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)

        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras,
                                       vocab_size,
                                       lora_config.lora_extra_vocab_size)
        lora_logits_processor.set_mapping(*mapping_info, )

        lora_result = lora_logits_processor._get_logits(
            hidden_states=torch.cat(inputs),
            embedding=original_weight,
            embedding_bias=None)[:, :vocab_size]
        expected_result = logits_processor._get_logits(
            hidden_states=torch.cat(inputs),
            embedding=original_weight,
            embedding_bias=None)

        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4, 8])
@pytest.mark.parametrize("orientation", ["row", "column"])
@pytest.mark.parametrize("fully_shard", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_linear_parallel(dist_init, num_loras, orientation, fully_shard,
                         device) -> None:

    torch.set_default_device(device)
    max_loras = 8
    lora_config = LoRAConfig(max_loras=max_loras,
                             max_lora_rank=8,
                             fully_sharded_loras=fully_shard,
                             lora_dtype=torch.float16)

    def create_random_linear_parallel_layer():
        if orientation == "row":
            linear = RowParallelLinear(4096,
                                       4096,
                                       bias=False,
                                       params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = (RowParallelLinearWithLoRA(linear) if not fully_shard
                           else RowParallelLinearWithShardedLoRA(linear))
        else:
            linear = ColumnParallelLinear(4096,
                                          4096,
                                          bias=False,
                                          params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = (ColumnParallelLinearWithLoRA(linear)
                           if not fully_shard else
                           ColumnParallelLinearWithShardedLoRA(linear))
        lora_linear.create_lora_weights(max_loras, lora_config)

        return linear, lora_linear

    for i in range(10):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)
        linear, lora_linear = create_random_linear_parallel_layer()

        lora_dict, _ = populate_loras(
            id_to_index,
            layer=lora_linear,
            layer_weights=linear.weight,
        )

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)

        mapping_info = convert_mapping(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
            lora_config.lora_extra_vocab_size,
        )
        lora_linear.set_mapping(*mapping_info, )

        lora_result = lora_linear(torch.cat(inputs))[0]

        expected_results = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = linear(input_)[0]
            result += input_ @ lora.lora_a @ lora.lora_b * lora.scaling
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)

        # Check that resetting the lora weights succeeds

        for slot_idx in range(max_loras):
            lora_linear.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)

        mapping_info = convert_mapping(lora_mapping, id_to_index, max_loras,
                                       512, lora_config.lora_extra_vocab_size)
        lora_linear.set_mapping(*mapping_info, )

        lora_result = lora_linear(torch.cat(inputs))[0]
        expected_result = linear(torch.cat(inputs))[0]

        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4, 8])
@pytest.mark.parametrize("repeats", [1, 2, 3])
@pytest.mark.parametrize("fully_shard", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_column_parallel_packed(dist_init, num_loras, repeats, fully_shard,
                                device) -> None:

    torch.set_default_device(device)
    max_loras = 8
    lora_config = LoRAConfig(max_loras=max_loras,
                             max_lora_rank=8,
                             fully_sharded_loras=fully_shard,
                             lora_dtype=torch.float16)

    def create_column_parallel_packed_layer():
        if repeats == 2:
            linear = MergedColumnParallelLinear(4096, [4096] * repeats,
                                                bias=False,
                                                params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = (MergedColumnParallelLinearWithLoRA(linear)
                           if not fully_shard else
                           MergedColumnParallelLinearWithShardedLoRA(linear))
        elif repeats == 3:
            linear = QKVParallelLinear(4096,
                                       64,
                                       32,
                                       bias=False,
                                       params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = (MergedQKVParallelLinearWithLora(linear)
                           if not fully_shard else
                           MergedQKVParallelLinearWithShardedLora(linear))
        else:
            linear = QKVParallelLinear(4096,
                                       64,
                                       32,
                                       bias=False,
                                       params_dtype=torch.float16)
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = QKVParallelLinearWithLora(linear)

        @dataclass
        class FakeConfig:
            hidden_size = 4096
            num_key_value_heads = 32
            num_attention_heads = 32

        lora_linear.create_lora_weights(max_loras,
                                        lora_config,
                                        model_config=FakeConfig())

        return linear, lora_linear

    for i in range(10):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)

        linear, lora_linear = create_column_parallel_packed_layer()

        lora_dict, sublora_dict = populate_loras(
            id_to_index,
            layer=lora_linear,
            layer_weights=linear.weight,
            repeats=repeats,
        )

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)

        mapping_info = convert_mapping(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
            lora_config.lora_extra_vocab_size,
        )
        lora_linear.set_mapping(*mapping_info)

        lora_result = lora_linear(torch.cat(inputs))[0]

        expected_results = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            result = linear(input_)[0]
            subloras = sublora_dict[lora_id]
            for i, sublora in enumerate(subloras):
                result[:, sublora.lora_b.shape[1] * i:sublora.lora_b.shape[1] *
                       (i + 1)] += (input_ @ sublora.lora_a @ sublora.lora_b *
                                    sublora.scaling)
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)

        for slot_idx in range(max_loras):
            lora_linear.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping)

        mapping_info = convert_mapping(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
            lora_config.lora_extra_vocab_size,
        )
        lora_linear.set_mapping(*mapping_info)

        lora_result = lora_linear(torch.cat(inputs))[0]
        expected_result = linear(torch.cat(inputs))[0]

        rtol, atol = TOLERANCES[lora_result.dtype]
        assert torch.allclose(lora_result,
                              expected_result,
                              rtol=rtol,
                              atol=atol)
