# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.sequence import IntermediateTensors, get_intermediate_tensor_num_tokens


def test_sequence_intermediate_tensors_equal():
    class AnotherIntermediateTensors(IntermediateTensors):
        pass

    intermediate_tensors = IntermediateTensors({})
    another_intermediate_tensors = AnotherIntermediateTensors({})
    assert intermediate_tensors != another_intermediate_tensors

    empty_intermediate_tensors_1 = IntermediateTensors({})
    empty_intermediate_tensors_2 = IntermediateTensors({})
    assert empty_intermediate_tensors_1 == empty_intermediate_tensors_2

    different_key_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    difference_key_intermediate_tensors_2 = IntermediateTensors(
        {"2": torch.zeros([2, 4], dtype=torch.int32)}
    )
    assert different_key_intermediate_tensors_1 != difference_key_intermediate_tensors_2

    same_key_different_value_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    same_key_different_value_intermediate_tensors_2 = IntermediateTensors(
        {"1": torch.zeros([2, 5], dtype=torch.int32)}
    )
    assert (
        same_key_different_value_intermediate_tensors_1
        != same_key_different_value_intermediate_tensors_2
    )

    same_key_same_value_intermediate_tensors_1 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    same_key_same_value_intermediate_tensors_2 = IntermediateTensors(
        {"1": torch.zeros([2, 4], dtype=torch.int32)}
    )
    assert (
        same_key_same_value_intermediate_tensors_1
        == same_key_same_value_intermediate_tensors_2
    )


@pytest.mark.parametrize("num_tokens", [0, 1, 5])
def test_intermediate_token_count_accepts_attn_res_and_aux_outputs(num_tokens):
    tensors = IntermediateTensors(
        {
            "hidden_states": torch.empty(num_tokens, 8),
            "residual": torch.empty(num_tokens, 3, 8),
            "aux_hidden_states_0": torch.empty(num_tokens, 8),
        }
    )
    assert get_intermediate_tensor_num_tokens(tensors) == num_tokens


@pytest.mark.parametrize("shapes", [{}, {"hidden_states": (8, 4), "residual": (4, 4)}])
def test_intermediate_token_count_rejects_ambiguous_layout(shapes):
    tensors = IntermediateTensors(
        {k: torch.empty(shape) for k, shape in shapes.items()}
    )
    with pytest.raises(AssertionError, match="common token count"):
        get_intermediate_tensor_num_tokens(tensors)
