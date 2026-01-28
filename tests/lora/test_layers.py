# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from copy import deepcopy
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import (
    BaseLayerWithLoRA,
    ColumnParallelLinearWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    LogitsProcessorWithLoRA,
    LoRAMapping,
    MergedColumnParallelLinearVariableSliceWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
    VocabParallelEmbeddingWithLoRA,
)
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

from .utils import DummyLoRAManager

TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.float32: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 2e-2),
}

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_cpu()),
    reason="Backend not supported",
)

DEVICES = (
    [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]
    if current_platform.is_cuda_alike()
    else ["cpu"]
)

# prefill stage(True) or decode stage(False)
STAGES = [True, False]

NUM_RANDOM_SEEDS = 2

VOCAB_PARALLEL_EMBEDDING_TEST_NUM_RANDOM_SEEDS = 2


@pytest.fixture(autouse=True)
def clean_cache_reset_device(reset_default_device):
    # Release any memory we might be holding on to. CI runs OOMs otherwise.
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT

    _LORA_B_PTR_DICT.clear()
    _LORA_A_PTR_DICT.clear()

    yield


@pytest.fixture(autouse=True)
def skip_cuda_with_stage_false(request):
    """
    On cuda-like platforms, we use the same kernels for prefill and decode
    stage, and 'stage' is generally ignored, so we only need to test once.
    """
    if current_platform.is_cuda_alike():
        try:
            if hasattr(request.node, "callspec") and hasattr(
                request.node.callspec, "params"
            ):
                params = request.node.callspec.params
                if "stage" in params and params["stage"] is False:
                    pytest.skip("Skip test when stage=False")
        except Exception:
            pass
    yield


def get_random_id_to_index(
    num_loras: int, num_slots: int, log: bool = True
) -> list[int | None]:
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
            "num_loras must be less than or equal to num_slots."
        )

    slots: list[int | None] = [None] * num_slots
    random_slot_selections = (torch.randperm(num_slots)[:num_loras]).tolist()
    for lora_id, slot_idx in enumerate(random_slot_selections, start=1):
        slots[slot_idx] = lora_id

    if log:
        print(f"Created lora_id_to_index mapping: {slots}.")

    return slots


def populate_loras(
    id_to_index: list[int | None],
    layer: BaseLayerWithLoRA,
    layer_weights: torch.Tensor,
    repeats: int = 1,
) -> tuple[dict[int, LoRALayerWeights], dict[int, list[LoRALayerWeights]]]:
    """This method populates the lora layers with lora weights.

    Args:
        id_to_index: a list of lora ids. The index of the lora id
            represents which memory slot the lora matrices are
            stored in. A None value indicates a free slot.
        layer: the LoRAlayer to populate.
        layer_weights: the PyTorch tensor containing the layer's
            weights.
        repeats: must only be set for column parallel packed
            layers. Indicates the number of loras to compose
            together to create a single lora layer.
    """

    # Dictionary that maps the lora ID to the
    # corresponding lora weights.
    lora_dict: dict[int, LoRALayerWeights] = dict()

    # Dictionary that maps the lora ID to the
    # corresponding subloras.
    sublora_dict: dict[int, list[LoRALayerWeights]] = dict()

    for slot_idx, lora_id in enumerate(id_to_index):
        if lora_id is not None:
            subloras: list[LoRALayerWeights] = []
            sublora_len = layer_weights.shape[0] // repeats
            for i in range(repeats):
                sublora = DummyLoRAManager(layer_weights.device).init_random_lora(
                    module_name=f"fake_{i}",
                    weight=layer_weights,
                )
                sublora.lora_b = sublora.lora_b[
                    (sublora_len * i) : (sublora_len * (i + 1)), :
                ]
                sublora.optimize()
                subloras.append(sublora)

            lora = PackedLoRALayerWeights.pack(subloras) if repeats > 1 else subloras[0]

            layer.set_lora(
                slot_idx,
                lora_a=lora.lora_a,
                lora_b=lora.lora_b,
            )

            lora_dict[lora_id] = lora
            sublora_dict[lora_id] = subloras

    return lora_dict, sublora_dict


def create_random_inputs(
    active_lora_ids: list[int],
    num_inputs: int,
    input_size: tuple[int, ...],
    input_range: tuple[float, float],
    input_type: torch.dtype = torch.int,
    device: torch.device = "cuda",
) -> tuple[list[torch.Tensor], list[int], list[int]]:
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

    inputs: list[torch.Tensor] = []
    index_mapping: list[int] = []
    prompt_mapping: list[int] = []

    for _ in range(num_inputs):
        if input_type == torch.int:
            inputs.append(
                torch.randint(
                    low=int(low), high=int(high), size=input_size, device=device
                )
            )
        else:
            inputs.append(
                torch.rand(size=input_size, dtype=input_type, device=device) * high
                + low
            )

        lora_id = random.choice(active_lora_ids)
        index_mapping += [lora_id] * input_size[0]
        prompt_mapping += [lora_id]

    return inputs, index_mapping, prompt_mapping


def check_punica_wrapper(punica_wrapper) -> bool:
    if current_platform.is_cuda_alike():
        from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

        return type(punica_wrapper) is PunicaWrapperGPU
    elif current_platform.is_cpu():
        from vllm.lora.punica_wrapper.punica_cpu import PunicaWrapperCPU

        return type(punica_wrapper) is PunicaWrapperCPU
    else:
        return False


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("vocab_size", [512, 32000, 64000, 128000])
@pytest.mark.parametrize("stage", STAGES)
def test_embeddings(
    default_vllm_config, dist_init, num_loras, device, vocab_size, stage
) -> None:
    # For multi-GPU testing of Triton kernel, we must explicitly set the CUDA
    # device, see: https://github.com/triton-lang/triton/issues/2925
    # Same below.
    if current_platform.is_cuda_alike():
        torch.cuda.set_device(device)

    torch.set_default_device(device)
    max_loras = 8
    lora_config = LoRAConfig(
        max_loras=max_loras, max_lora_rank=8, lora_dtype=torch.float16
    )
    punica_wrapper = get_punica_wrapper(8192, 256, device, lora_config=lora_config)
    assert check_punica_wrapper(punica_wrapper)

    def create_random_embedding_layer():
        embedding = VocabParallelEmbedding(vocab_size, 256)
        embedding.weight.data = torch.rand_like(embedding.weight.data)
        embedding.weight.data[vocab_size:, :] = 0
        lora_embedding = VocabParallelEmbeddingWithLoRA(embedding)
        lora_embedding.create_lora_weights(max_loras, lora_config)

        return embedding, lora_embedding

    for i in range(NUM_RANDOM_SEEDS):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)
        embedding, lora_embedding = create_random_embedding_layer()
        lora_embedding.set_mapping(punica_wrapper)
        lora_dict, _ = populate_loras(
            id_to_index,
            layer=lora_embedding,
            layer_weights=embedding.weight.T,
        )

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=num_loras * 3,
            input_size=(200,),
            input_range=(1, vocab_size),
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)
        punica_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            vocab_size,
        )

        lora_result = lora_embedding(torch.cat(inputs))

        expected_results: list[torch.Tensor] = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = embedding(input_)
            after_a = F.embedding(
                input_,
                lora.lora_a.T,
            )
            result += after_a @ lora.lora_b.T
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result, expected_result, rtol=rtol, atol=atol)

        # Check that resetting the lora weights succeeds

        for slot_idx in range(max_loras):
            lora_embedding.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=num_loras * 3,
            input_size=(200,),
            input_range=(1, vocab_size),
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)
        punica_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            vocab_size,
        )

        lora_result = lora_embedding(torch.cat(inputs))
        expected_result = embedding(torch.cat(inputs))

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result, expected_result, rtol=rtol, atol=atol)


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("vocab_size", [512, 32000, 64000, 256512])
@pytest.mark.parametrize("stage", STAGES)
def test_lm_head_logits_processor(
    default_vllm_config, dist_init, num_loras, device, vocab_size, stage
) -> None:
    if current_platform.is_cuda_alike():
        torch.cuda.set_device(device)

    torch.set_default_device(device)
    max_loras = 8
    lora_config = LoRAConfig(
        max_loras=max_loras, max_lora_rank=8, lora_dtype=torch.float16
    )
    punica_wrapper = get_punica_wrapper(8192, 256, device, lora_config=lora_config)
    assert check_punica_wrapper(punica_wrapper)

    def _pretest():
        linear = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=1024,
            params_dtype=torch.float16,
        )
        linear.weight.data = torch.rand_like(linear.weight.data)
        linear.weight.data[:, vocab_size:] = 0
        logits_processor = LogitsProcessor(vocab_size)
        lora_logits_processor = LogitsProcessorWithLoRA(
            logits_processor, 1024, linear.weight.dtype, linear.weight.device, None
        )
        lora_logits_processor.create_lora_weights(max_loras, lora_config)

        return linear, logits_processor, lora_logits_processor

    for i in range(NUM_RANDOM_SEEDS):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)
        linear, logits_processor, lora_logits_processor = _pretest()
        lora_logits_processor.set_mapping(punica_wrapper)

        lora_dict, _ = populate_loras(
            id_to_index,
            layer=lora_logits_processor,
            layer_weights=linear.weight,
        )

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=8 * num_loras,  # * 3,
            input_size=(1, 1024),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)
        punica_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            vocab_size,
        )
        input_ = torch.rand(20, 1024)

        lora_result = lora_logits_processor._get_logits(
            hidden_states=torch.cat(inputs), lm_head=linear, embedding_bias=None
        )

        original_lm_head = deepcopy(linear)

        expected_results: list[torch.Tensor] = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = logits_processor._get_logits(
                hidden_states=input_, lm_head=linear, embedding_bias=None
            )

            result += input_ @ lora.lora_a.T @ lora.lora_b.T * lora.scaling
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        # Check that resetting the lora weights succeeds

        for slot_idx in range(max_loras):
            lora_logits_processor.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=8 * num_loras * 3,
            input_size=(1, 1024),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)
        punica_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            vocab_size,
        )

        lora_result = lora_logits_processor._get_logits(
            hidden_states=torch.cat(inputs),
            lm_head=original_lm_head,
            embedding_bias=None,
        )[:, :vocab_size]
        expected_result = logits_processor._get_logits(
            hidden_states=torch.cat(inputs),
            lm_head=original_lm_head,
            embedding_bias=None,
        )

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result, expected_result, rtol=rtol, atol=atol)


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stage", STAGES)
def test_linear_replicated(
    default_vllm_config,
    dist_init,
    num_loras,
    device,
    stage,
) -> None:
    if current_platform.is_cuda_alike():
        torch.cuda.set_device(device)

    max_loras = 8
    torch.set_default_device(device)
    lora_config = LoRAConfig(
        max_loras=max_loras,
        max_lora_rank=8,
        lora_dtype=torch.float16,
    )
    punica_wrapper = get_punica_wrapper(8192, 256, device, lora_config=lora_config)
    assert check_punica_wrapper(punica_wrapper)

    def create_random_linear_replicated_layer():
        linear = ReplicatedLinear(4096, 4096, bias=False, params_dtype=torch.float16)
        linear.weight.data = torch.rand_like(linear.weight.data)
        lora_linear = ReplicatedLinearWithLoRA(linear)

        lora_linear.create_lora_weights(max_loras, lora_config)
        assert (
            lora_linear.n_slices
            == len(lora_linear.lora_a_stacked)
            == len(lora_linear.lora_b_stacked)
            == 1
        )
        return linear, lora_linear

    for i in range(NUM_RANDOM_SEEDS):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)
        linear, lora_linear = create_random_linear_replicated_layer()
        assert torch.equal(linear.weight, lora_linear.weight)
        lora_linear.set_mapping(punica_wrapper)
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
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)
        punica_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
        )

        lora_result = lora_linear(torch.cat(inputs))[0]

        expected_results: list[torch.Tensor] = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = linear(input_)[0]
            result += input_ @ lora.lora_a.T @ lora.lora_b.T * lora.scaling
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result, expected_result, rtol=rtol, atol=atol)

        # Check that resetting the lora weights succeeds

        for slot_idx in range(max_loras):
            lora_linear.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)

        punica_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
        )

        lora_result = lora_linear(torch.cat(inputs))[0]
        expected_result = linear(torch.cat(inputs))[0]

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result, expected_result, rtol=rtol, atol=atol)


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4])
@pytest.mark.parametrize("orientation", ["row", "column"])
@pytest.mark.parametrize("fully_shard", [True, False])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stage", STAGES)
def test_linear_parallel(
    default_vllm_config, dist_init, num_loras, orientation, fully_shard, device, stage
) -> None:
    if current_platform.is_cuda_alike():
        torch.cuda.set_device(device)

    max_loras = 8
    torch.set_default_device(device)
    lora_config = LoRAConfig(
        max_loras=max_loras,
        max_lora_rank=8,
        fully_sharded_loras=fully_shard,
        lora_dtype=torch.float16,
    )
    punica_wrapper = get_punica_wrapper(8192, 256, device, lora_config=lora_config)
    assert check_punica_wrapper(punica_wrapper)

    def create_random_linear_parallel_layer():
        if orientation == "row":
            linear = RowParallelLinear(
                4096, 4096, bias=False, params_dtype=torch.float16
            )
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = (
                RowParallelLinearWithLoRA(linear)
                if not fully_shard
                else RowParallelLinearWithShardedLoRA(linear)
            )
        else:
            linear = ColumnParallelLinear(
                4096, 4096, bias=False, params_dtype=torch.float16
            )
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = (
                ColumnParallelLinearWithLoRA(linear)
                if not fully_shard
                else ColumnParallelLinearWithShardedLoRA(linear)
            )
        lora_linear.create_lora_weights(max_loras, lora_config)
        assert (
            lora_linear.n_slices
            == len(lora_linear.lora_a_stacked)
            == len(lora_linear.lora_b_stacked)
            == 1
        )

        return linear, lora_linear

    for i in range(NUM_RANDOM_SEEDS):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)
        linear, lora_linear = create_random_linear_parallel_layer()
        assert torch.equal(linear.weight, lora_linear.weight)
        lora_linear.set_mapping(punica_wrapper)
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
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)
        punica_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
        )

        lora_result = lora_linear(torch.cat(inputs))[0]

        expected_results: list[torch.Tensor] = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            lora = lora_dict[lora_id]
            result = linear(input_)[0]
            result += input_ @ lora.lora_a.T @ lora.lora_b.T * lora.scaling
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result, expected_result, rtol=rtol, atol=atol)

        # Check that resetting the lora weights succeeds

        for slot_idx in range(max_loras):
            lora_linear.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)

        punica_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
        )

        lora_result = lora_linear(torch.cat(inputs))[0]
        expected_result = linear(torch.cat(inputs))[0]

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result, expected_result, rtol=rtol, atol=atol)


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4])
@pytest.mark.parametrize("repeats", [1, 2, 3])
@pytest.mark.parametrize("fully_shard", [True, False])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stage", STAGES)
def test_column_parallel_packed(
    default_vllm_config, dist_init, num_loras, repeats, fully_shard, device, stage
) -> None:
    if current_platform.is_cuda_alike():
        torch.cuda.set_device(device)

    max_loras = 8
    torch.set_default_device(device)
    lora_config = LoRAConfig(
        max_loras=max_loras,
        max_lora_rank=8,
        fully_sharded_loras=fully_shard,
        lora_dtype=torch.float16,
    )
    punica_wrapper = get_punica_wrapper(8192, 256, device, lora_config=lora_config)
    assert check_punica_wrapper(punica_wrapper)

    def create_column_parallel_packed_layer():
        if repeats == 2:
            linear = MergedColumnParallelLinear(
                4096, [4096] * repeats, bias=False, params_dtype=torch.float16
            )
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = (
                MergedColumnParallelLinearWithLoRA(linear)
                if not fully_shard
                else MergedColumnParallelLinearWithShardedLoRA(linear)
            )
        elif repeats == 3:
            linear = QKVParallelLinear(
                4096, 64, 32, bias=False, params_dtype=torch.float16
            )
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = (
                MergedQKVParallelLinearWithLoRA(linear)
                if not fully_shard
                else MergedQKVParallelLinearWithShardedLoRA(linear)
            )
        else:
            linear = QKVParallelLinear(
                4096, 64, 32, bias=False, params_dtype=torch.float16
            )
            linear.weight.data = torch.rand_like(linear.weight.data)
            lora_linear = (
                QKVParallelLinearWithLoRA(linear)
                if not fully_shard
                else QKVParallelLinearWithShardedLoRA(linear)
            )

        @dataclass
        class FakeConfig:
            hidden_size = 4096
            num_key_value_heads = 32
            num_attention_heads = 32

        n_slices = repeats
        lora_linear.create_lora_weights(
            max_loras, lora_config, model_config=FakeConfig()
        )
        assert (
            lora_linear.n_slices
            == len(lora_linear.lora_a_stacked)
            == len(lora_linear.lora_b_stacked)
            == n_slices
        )

        return linear, lora_linear

    for i in range(NUM_RANDOM_SEEDS):
        set_random_seed(i)

        id_to_index = get_random_id_to_index(num_loras, max_loras)

        linear, lora_linear = create_column_parallel_packed_layer()
        assert torch.equal(linear.weight, lora_linear.weight)
        lora_linear.set_mapping(punica_wrapper)
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
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)

        punica_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
        )

        lora_result = lora_linear(torch.cat(inputs))[0]

        expected_results: list[torch.Tensor] = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            result = linear(input_)[0]
            subloras = sublora_dict[lora_id]
            for i, sublora in enumerate(subloras):
                result[
                    :, sublora.lora_b.shape[0] * i : sublora.lora_b.shape[0] * (i + 1)
                ] += input_ @ sublora.lora_a.T @ sublora.lora_b.T * sublora.scaling
            expected_results.append(result)
        expected_result = torch.cat(expected_results)

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result, expected_result, rtol=rtol, atol=atol)

        for slot_idx in range(max_loras):
            lora_linear.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)

        punica_wrapper.update_metadata(
            lora_mapping,
            id_to_index,
            max_loras,
            512,
        )

        lora_result = lora_linear(torch.cat(inputs))[0]
        expected_result = linear(torch.cat(inputs))[0]

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result, expected_result, rtol=rtol, atol=atol)


@torch.inference_mode()
@pytest.mark.parametrize("num_loras", [1, 2, 4])
@pytest.mark.parametrize("num_slices", [3, 5])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stage", STAGES)
def test_merged_column_parallel_variable_slice(
    default_vllm_config, dist_init, num_loras, num_slices, device, stage
) -> None:
    if current_platform.is_cuda_alike():
        torch.cuda.set_device(device)

    max_loras = 8
    torch.set_default_device(device)
    lora_config = LoRAConfig(
        max_loras=max_loras, max_lora_rank=8, lora_dtype=torch.float16
    )
    punica_wrapper = get_punica_wrapper(8192, 256, device, lora_config=lora_config)

    # Set number of output slices
    output_sizes = [1024 + i * 256 for i in range(num_slices)]
    total_output = sum(output_sizes)

    def create_layer():
        # Create linear layer
        linear = MergedColumnParallelLinear(
            4096, output_sizes, bias=False, params_dtype=torch.float16
        )
        linear.weight.data = torch.rand_like(linear.weight.data)

        # Create linear layer with LoRA adapter
        lora_linear = MergedColumnParallelLinearVariableSliceWithLoRA(linear)
        lora_linear.create_lora_weights(max_loras, lora_config)
        return linear, lora_linear

    for i in range(NUM_RANDOM_SEEDS):
        set_random_seed(i)
        id_to_index = get_random_id_to_index(num_loras, max_loras)
        linear, lora_linear = create_layer()
        lora_linear.set_mapping(punica_wrapper)

        # Populate LoRA weights
        lora_dict, sublora_dict = {}, {}
        for slot_idx, lora_id in enumerate(id_to_index):
            if lora_id is not None:
                # Create random LoRA weights
                lora_a = torch.rand(8, 4096, dtype=torch.float16, device=device)
                lora_b = torch.rand(total_output, 8, dtype=torch.float16, device=device)
                lora_linear.set_lora(slot_idx, lora_a, lora_b)
                lora_dict[lora_id] = (lora_a, lora_b)

                # Split lora_b for expected computation
                sublora_dict[lora_id] = torch.split(lora_b, output_sizes, dim=0)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=list(lora_dict.keys()),
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)
        punica_wrapper.update_metadata(lora_mapping, id_to_index, max_loras, 512)

        # Compute LoRA result
        lora_result = lora_linear(torch.cat(inputs))[0]

        # Compute expected result
        expected_results = []
        for input_, lora_id in zip(inputs, prompt_mapping):
            result = linear(input_)[0]
            lora_a, _ = lora_dict[lora_id]
            offset = 0
            # Compute expected result for each sublora
            for lora_b_slice in sublora_dict[lora_id]:
                sz = lora_b_slice.shape[0]
                result[:, offset : offset + sz] += input_ @ lora_a.T @ lora_b_slice.T
                offset += sz
            expected_results.append(result)

        # Check that the LoRA result is close to the expected result
        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(
            lora_result, torch.cat(expected_results), rtol=rtol, atol=atol
        )

        # Reset LoRA weights and check results with zero LoRA weights
        for slot_idx in range(max_loras):
            lora_linear.reset_lora(slot_idx)

        inputs, index_mapping, prompt_mapping = create_random_inputs(
            active_lora_ids=[0],
            num_inputs=32 * num_loras,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
            device=device,
        )
        lora_mapping = LoRAMapping(index_mapping, prompt_mapping, is_prefill=stage)
        punica_wrapper.update_metadata(lora_mapping, id_to_index, max_loras, 512)

        # After resetting LoRA weights,
        # lora_linear should behave like the base linear layer
        lora_result = lora_linear(torch.cat(inputs))[0]
        expected_result = linear(torch.cat(inputs))[0]

        rtol, atol = TOLERANCES[lora_result.dtype]
        torch.testing.assert_close(lora_result, expected_result, rtol=rtol, atol=atol)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "seed", list(range(VOCAB_PARALLEL_EMBEDDING_TEST_NUM_RANDOM_SEEDS))
)
def test_vocab_parallel_embedding_indices(tp_size, seed, default_vllm_config):
    random.seed(seed)
    vocab_size = random.randint(4000, 64000)
    added_vocab_size = random.randint(0, 1024)
    org_vocab_size = vocab_size - added_vocab_size
    last_org_vocab_end_index = 0
    last_added_vocab_end_index = org_vocab_size
    computed_vocab_size = 0
    computed_org_vocab_size = 0
    computed_added_vocab_size = 0
    vocab_size_padded = -1

    all_org_tokens: list[int] = []
    all_added_tokens: list[int] = []
    token_ids: list[int] = []

    for tp_rank in range(tp_size):
        with (
            patch(
                "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
                return_value=tp_rank,
            ),
            patch(
                "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
                return_value=tp_size,
            ),
        ):
            vocab_embedding = VocabParallelEmbedding(
                vocab_size, 1, org_num_embeddings=org_vocab_size
            )
        vocab_size_padded = vocab_embedding.num_embeddings_padded
        shard_indices = vocab_embedding.shard_indices
        # Assert that the ranges are contiguous
        assert shard_indices.org_vocab_start_index == last_org_vocab_end_index
        assert shard_indices.added_vocab_start_index == last_added_vocab_end_index

        # Ensure that we are not exceeding the vocab size
        computed_vocab_size += shard_indices.num_elements_padded
        computed_org_vocab_size += shard_indices.num_org_elements
        computed_added_vocab_size += shard_indices.num_added_elements

        # Ensure that the ranges are not overlapping
        all_org_tokens.extend(
            range(
                shard_indices.org_vocab_start_index, shard_indices.org_vocab_end_index
            )
        )
        all_added_tokens.extend(
            range(
                shard_indices.added_vocab_start_index,
                shard_indices.added_vocab_end_index,
            )
        )

        token_ids.extend(
            range(
                shard_indices.org_vocab_start_index, shard_indices.org_vocab_end_index
            )
        )
        token_ids.extend(
            [-1]
            * (shard_indices.num_org_elements_padded - shard_indices.num_org_elements)
        )
        token_ids.extend(
            range(
                shard_indices.added_vocab_start_index,
                shard_indices.added_vocab_end_index,
            )
        )
        token_ids.extend(
            [-1]
            * (
                shard_indices.num_added_elements_padded
                - shard_indices.num_added_elements
            )
        )

        last_org_vocab_end_index = shard_indices.org_vocab_end_index
        last_added_vocab_end_index = shard_indices.added_vocab_end_index

    assert computed_vocab_size == vocab_size_padded
    assert computed_org_vocab_size == org_vocab_size
    assert computed_added_vocab_size == added_vocab_size

    # Ensure that the ranges are not overlapping
    assert len(all_org_tokens) == len(set(all_org_tokens))
    assert len(all_added_tokens) == len(set(all_added_tokens))
    assert not set(all_org_tokens).intersection(set(all_added_tokens))

    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
    reindex_mapping = vocab_embedding.get_sharded_to_full_mapping()
    assert reindex_mapping is not None or tp_size == 1
    if reindex_mapping is not None:
        reindexed_token_ids = token_ids_tensor[reindex_mapping]
        expected = torch.tensor(list(range(0, vocab_size)))
        assert reindexed_token_ids[:vocab_size].equal(expected)
        assert torch.all(reindexed_token_ids[vocab_size:] == -1)


def test_get_masked_input_and_mask():
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    # base tp 1 case, no padding
    modified_x, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=0,
        org_vocab_end_index=8,
        added_vocab_start_index=8,
        added_vocab_end_index=12,
        num_org_vocab_padding=0,
    )
    assert torch.equal(x, modified_x)

    # tp 2 case, no padding
    modified_x_rank_0, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=0,
        org_vocab_end_index=4,
        added_vocab_start_index=8,
        added_vocab_end_index=10,
        num_org_vocab_padding=0,
    )
    modified_x_rank_1, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=4,
        org_vocab_end_index=8,
        added_vocab_start_index=10,
        added_vocab_end_index=12,
        num_org_vocab_padding=0,
    )
    assert torch.equal(
        modified_x_rank_0, torch.tensor([0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 0, 0])
    )
    assert torch.equal(
        modified_x_rank_1, torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5])
    )

    # tp 4 case, no padding
    modified_x_rank_0, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=0,
        org_vocab_end_index=2,
        added_vocab_start_index=8,
        added_vocab_end_index=9,
        num_org_vocab_padding=0,
    )
    modified_x_rank_1, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=2,
        org_vocab_end_index=4,
        added_vocab_start_index=9,
        added_vocab_end_index=10,
        num_org_vocab_padding=0,
    )
    modified_x_rank_2, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=4,
        org_vocab_end_index=6,
        added_vocab_start_index=10,
        added_vocab_end_index=11,
        num_org_vocab_padding=0,
    )
    modified_x_rank_3, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=6,
        org_vocab_end_index=8,
        added_vocab_start_index=11,
        added_vocab_end_index=12,
        num_org_vocab_padding=0,
    )
    assert torch.equal(
        modified_x_rank_0, torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0])
    )
    assert torch.equal(
        modified_x_rank_1, torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0])
    )
    assert torch.equal(
        modified_x_rank_2, torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0])
    )
    assert torch.equal(
        modified_x_rank_3, torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2])
    )

    # base tp 1 case, with padding
    modified_x, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=0,
        org_vocab_end_index=8,
        added_vocab_start_index=8,
        added_vocab_end_index=12,
        num_org_vocab_padding=2,
    )
    assert torch.equal(
        modified_x, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13])
    )

    # tp 2 case, with padding
    modified_x_rank_0, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=0,
        org_vocab_end_index=4,
        added_vocab_start_index=8,
        added_vocab_end_index=10,
        num_org_vocab_padding=2,
    )
    modified_x_rank_1, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=4,
        org_vocab_end_index=8,
        added_vocab_start_index=10,
        added_vocab_end_index=12,
        num_org_vocab_padding=2,
    )
    assert torch.equal(
        modified_x_rank_0, torch.tensor([0, 1, 2, 3, 0, 0, 0, 0, 6, 7, 0, 0])
    )
    assert torch.equal(
        modified_x_rank_1, torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 6, 7])
    )

    # tp 4 case, with padding
    modified_x_rank_0, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=0,
        org_vocab_end_index=2,
        added_vocab_start_index=8,
        added_vocab_end_index=9,
        num_org_vocab_padding=2,
    )
    modified_x_rank_1, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=2,
        org_vocab_end_index=4,
        added_vocab_start_index=9,
        added_vocab_end_index=10,
        num_org_vocab_padding=2,
    )
    modified_x_rank_2, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=4,
        org_vocab_end_index=6,
        added_vocab_start_index=10,
        added_vocab_end_index=11,
        num_org_vocab_padding=2,
    )
    modified_x_rank_3, _ = get_masked_input_and_mask(
        x,
        org_vocab_start_index=6,
        org_vocab_end_index=8,
        added_vocab_start_index=11,
        added_vocab_end_index=12,
        num_org_vocab_padding=2,
    )
    assert torch.equal(
        modified_x_rank_0, torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0])
    )
    assert torch.equal(
        modified_x_rank_1, torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0])
    )
    assert torch.equal(
        modified_x_rank_2, torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0])
    )
    assert torch.equal(
        modified_x_rank_3, torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4])
    )


def test_variable_slice_lora_class_selection(default_vllm_config, dist_init):
    """Test that MergedColumnParallelLinearVariableSliceWithLoRA is selected
    only for nemotron-h style models (checkpoint has single weight but layer
    has 3+ output slices).

    This verifies that from_layer selects
    MergedColumnParallelLinearVariableSliceWithLoRA
    before ColumnParallelLinearWithLoRA for layers with 3+ output sizes, since
    ColumnParallelLinearWithLoRA's slice_lora_b assumes exactly 2 slices.
    """
    from vllm.lora.utils import from_layer

    lora_config = LoRAConfig(max_loras=8, max_lora_rank=8, lora_dtype=torch.float16)

    # Case 1: MergedColumnParallelLinear with 3+ output sizes and
    # packed_modules_list with 1 item (nemotron-h style)
    # -> MergedColumnParallelLinearVariableSliceWithLoRA should be selected
    layer_3_slices = MergedColumnParallelLinear(
        4096, [1024, 1280, 1536], bias=False, params_dtype=torch.float16
    )
    packed_modules_single = ["mlp"]

    assert MergedColumnParallelLinearVariableSliceWithLoRA.can_replace_layer(
        source_layer=layer_3_slices,
        lora_config=lora_config,
        packed_modules_list=packed_modules_single,
    ), "MergedColumnParallelLinearVariableSliceWithLoRA should handle 3+ slices"

    # ColumnParallelLinearWithLoRA should NOT match 3+ slices
    # (its slice_lora_b assumes exactly 2 slices)
    assert not ColumnParallelLinearWithLoRA.can_replace_layer(
        source_layer=layer_3_slices,
        lora_config=lora_config,
        packed_modules_list=packed_modules_single,
    ), (
        "ColumnParallelLinearWithLoRA should NOT handle 3+ slices "
        "(slice_lora_b assumes 2 slices)"
    )

    # Verify from_layer selects the correct class (Variable, not base)
    selected_layer = from_layer(
        layer_3_slices,
        max_loras=8,
        lora_config=lora_config,
        packed_modules_list=packed_modules_single,
    )
    assert isinstance(
        selected_layer, MergedColumnParallelLinearVariableSliceWithLoRA
    ), (
        f"from_layer should select MergedColumnParallelLinearVariableSliceWithLoRA "
        f"for 3+ slices, got {type(selected_layer).__name__}"
    )

    # Case 2: MergedColumnParallelLinear with 2 output sizes and
    # packed_modules_list with 1 item (standard gate_up style)
    # -> ColumnParallelLinearWithLoRA should be selected
    # -> MergedColumnParallelLinearVariableSliceWithLoRA should NOT match
    layer_2_slices = MergedColumnParallelLinear(
        4096, [2048, 2048], bias=False, params_dtype=torch.float16
    )

    assert ColumnParallelLinearWithLoRA.can_replace_layer(
        source_layer=layer_2_slices,
        lora_config=lora_config,
        packed_modules_list=packed_modules_single,
    ), "ColumnParallelLinearWithLoRA should handle 2 slices"

    assert not MergedColumnParallelLinearVariableSliceWithLoRA.can_replace_layer(
        source_layer=layer_2_slices,
        lora_config=lora_config,
        packed_modules_list=packed_modules_single,
    ), "MergedColumnParallelLinearVariableSliceWithLoRA should NOT handle 2 slices"

    # Verify from_layer selects ColumnParallelLinearWithLoRA for 2 slices
    selected_layer_2 = from_layer(
        layer_2_slices,
        max_loras=8,
        lora_config=lora_config,
        packed_modules_list=packed_modules_single,
    )
    assert isinstance(selected_layer_2, ColumnParallelLinearWithLoRA), (
        f"from_layer should select ColumnParallelLinearWithLoRA "
        f"for 2 slices, got {type(selected_layer_2).__name__}"
    )
    # But NOT the Variable subclass
    assert not isinstance(
        selected_layer_2, MergedColumnParallelLinearVariableSliceWithLoRA
    ), (
        "from_layer should NOT select "
        "MergedColumnParallelLinearVariableSliceWithLoRA for 2 slices"
    )

    # Case 3: MergedColumnParallelLinear with 3+ items in packed_modules_list
    # -> MergedColumnParallelLinearVariableSliceWithLoRA should be selected
    packed_modules_three = ["gate_proj", "up_proj", "down_proj"]

    assert MergedColumnParallelLinearVariableSliceWithLoRA.can_replace_layer(
        source_layer=layer_3_slices,
        lora_config=lora_config,
        packed_modules_list=packed_modules_three,
    ), "MergedColumnParallelLinearVariableSliceWithLoRA should handle 3+ packed modules"

    # Case 4: MergedColumnParallelLinear with 2 items in packed_modules_list
    # -> MergedColumnParallelLinearWithLoRA should handle this (not Variable)
    packed_modules_two = ["gate_proj", "up_proj"]

    assert not MergedColumnParallelLinearVariableSliceWithLoRA.can_replace_layer(
        source_layer=layer_2_slices,
        lora_config=lora_config,
        packed_modules_list=packed_modules_two,
    ), (
        "MergedColumnParallelLinearVariableSliceWithLoRA"
        " should NOT handle 2 packed modules"
    )

    assert MergedColumnParallelLinearWithLoRA.can_replace_layer(
        source_layer=layer_2_slices,
        lora_config=lora_config,
        packed_modules_list=packed_modules_two,
    ), "MergedColumnParallelLinearWithLoRA should handle 2 packed modules"

    # Verify from_layer selects MergedColumnParallelLinearWithLoRA for 2 packed modules
    selected_layer_merged = from_layer(
        layer_2_slices,
        max_loras=8,
        lora_config=lora_config,
        packed_modules_list=packed_modules_two,
    )
    assert isinstance(selected_layer_merged, MergedColumnParallelLinearWithLoRA), (
        f"from_layer should select MergedColumnParallelLinearWithLoRA "
        f"for 2 packed modules, got {type(selected_layer_merged).__name__}"
    )

    # Case 5: Plain ColumnParallelLinear (not merged) - common in many models
    # -> ColumnParallelLinearWithLoRA should be selected
    plain_column_parallel = ColumnParallelLinear(
        4096, 4096, bias=False, params_dtype=torch.float16
    )

    assert ColumnParallelLinearWithLoRA.can_replace_layer(
        source_layer=plain_column_parallel,
        lora_config=lora_config,
        packed_modules_list=packed_modules_single,
    ), "ColumnParallelLinearWithLoRA should handle plain ColumnParallelLinear"

    assert not MergedColumnParallelLinearVariableSliceWithLoRA.can_replace_layer(
        source_layer=plain_column_parallel,
        lora_config=lora_config,
        packed_modules_list=packed_modules_single,
    ), (
        "MergedColumnParallelLinearVariableSliceWithLoRA "
        "should NOT handle plain ColumnParallelLinear"
    )

    # Verify from_layer selects ColumnParallelLinearWithLoRA for plain layer
    selected_plain = from_layer(
        plain_column_parallel,
        max_loras=8,
        lora_config=lora_config,
        packed_modules_list=packed_modules_single,
    )
    assert isinstance(selected_plain, ColumnParallelLinearWithLoRA), (
        f"from_layer should select ColumnParallelLinearWithLoRA "
        f"for plain ColumnParallelLinear, got {type(selected_plain).__name__}"
    )

    # Case 6: MergedColumnParallelLinear with exactly 2 output sizes
    # and empty packed_modules_list
    # -> ColumnParallelLinearWithLoRA should NOT match (packed_modules_list != 1)
    # -> MergedColumnParallelLinearVariableSliceWithLoRA should NOT match (< 3 slices)
    assert not ColumnParallelLinearWithLoRA.can_replace_layer(
        source_layer=layer_2_slices,
        lora_config=lora_config,
        packed_modules_list=[],
    ), "ColumnParallelLinearWithLoRA should NOT handle empty packed_modules_list"

    assert not MergedColumnParallelLinearVariableSliceWithLoRA.can_replace_layer(
        source_layer=layer_2_slices,
        lora_config=lora_config,
        packed_modules_list=[],
    ), (
        "MergedColumnParallelLinearVariableSliceWithLoRA "
        "should NOT handle 2 slices even with empty packed_modules_list"
    )
