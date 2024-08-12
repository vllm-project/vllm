from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
import torch

from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.utils import set_random_seed

TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.float32: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 2e-2),
}
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]
# We will launch different triton kernels between the prefill and decode
# stages, so we need to verify this. prefill stage(True) or decode stage(False)
STAGES = [True, False]

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

def create_random_inputs(
    num_inputs: int,
    input_size: Tuple[int, ...],
    input_range: Tuple[float, float],
    input_type: torch.dtype = torch.int,
) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """Creates random inputs.

    Args:
        num_inputs: the number of inputs to create.
        input_size: the size of each individual input.
        input_range: the range of values to include in the input.
            input_range[0] <= possible input values < input_range[1]
        input_type: the type of values in the input.
    """

    low, high = input_range

    inputs: List[torch.Tensor] = []
    for _ in range(num_inputs):
        if input_type == torch.int:
            inputs.append(
                torch.randint(low=int(low), high=int(high), size=input_size))
        else:
            inputs.append(
                torch.rand(size=input_size, dtype=input_type) * high + low)

    return inputs

@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("stage", STAGES)
def test_q_cross_kv_parallel_linear(device, 
                                    stage,
                                    ) -> None:

    torch.set_default_device(device)

    def create_q_cross_kv_parallel_linear_layer():
        linear = QKVParallelLinear(4096,
                                    64,
                                    32,
                                    bias=False,
                                    params_dtype=torch.float16)
        linear.weight.data = torch.rand_like(linear.weight.data)
        return linear

    for i in range(10):
        set_random_seed(i)
        linear = create_q_cross_kv_parallel_linear_layer()
        inputs = create_random_inputs(
            num_inputs=32,
            input_size=(1, 4096),
            input_range=(0, 1),
            input_type=torch.float16,
        )

        result = linear(torch.cat(inputs))[0]

        # expected_results: List[torch.Tensor] = []
        # for input_ in inputs:
        #     result = linear(input_)[0]

        #         result[:, sublora.lora_b.shape[1] * i:sublora.lora_b.shape[1] *
        #                (i + 1)] += (input_ @ sublora.lora_a @ sublora.lora_b *
        #                             sublora.scaling)
        #     expected_results.append(result)
        # expected_result = torch.cat(expected_results)

        # rtol, atol = TOLERANCES[result.dtype]
        # assert torch.allclose(result,
        #                       expected_result,
        #                       rtol=rtol,
        #                       atol=atol)

        # for slot_idx in range(max_loras):
        #     lora_linear.reset_lora(slot_idx)

        # inputs, index_mapping, prompt_mapping = create_random_inputs(
        #     active_lora_ids=[0],
        #     num_inputs=32 * num_loras,
        #     input_size=(1, 4096),
        #     input_range=(0, 1),
        #     input_type=torch.float16,
        # )
        # lora_mapping = LoRAMapping(index_mapping,
        #                            prompt_mapping,
        #                            is_prefill=stage)

        # punica_wrapper.update_metadata(
        #     lora_mapping,
        #     id_to_index,
        #     max_loras,
        #     512,
        #     lora_config.lora_extra_vocab_size,
        # )
        # # lora_linear.set_mapping(*mapping_info)

        # result = lora_linear(torch.cat(inputs))[0]
        # expected_result = linear(torch.cat(inputs))[0]

        # rtol, atol = TOLERANCES[result.dtype]
        # assert torch.allclose(result,
        #                       expected_result,
        #                       rtol=rtol,
        #                       atol=atol)