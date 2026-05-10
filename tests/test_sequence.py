# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.sequence import IntermediateTensors


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


def test_intermediate_tensors_consolidate_residual():
    hs = torch.ones(4, 8) * 5.0
    res = torch.ones(4, 8) * 3.0
    it = IntermediateTensors({"hidden_states": hs, "residual": res})
    it.consolidate_residual()

    # residual set to None after consolidation
    assert it.tensors["residual"] is None
    # hidden_states = hs + res
    assert torch.allclose(it.tensors["hidden_states"], hs + res)

    # slicing must not crash when residual is None
    sliced = it[:2]
    assert sliced.tensors["residual"] is None
    assert torch.allclose(sliced.tensors["hidden_states"], it.tensors["hidden_states"][:2])

    # empty_like must not crash
    empty = IntermediateTensors.empty_like(it)
    assert empty.tensors["residual"] is None
    assert empty.tensors["hidden_states"].shape == hs.shape

    # __eq__ must not crash with None values
    it2 = IntermediateTensors({"hidden_states": hs + res, "residual": None})
    assert it == it2

    # consolidate should be a no-op when residual is already None
    it.consolidate_residual()
    assert it.tensors["residual"] is None

    # consolidate should be a no-op when keys are missing
    it3 = IntermediateTensors({"hidden_states": hs})
    it3.consolidate_residual()
    assert torch.allclose(it3.tensors["hidden_states"], hs)

    # consolidate on empty tensor
    it4 = IntermediateTensors({"hidden_states": hs[:0], "residual": res[:0]})
    it4.consolidate_residual()
    assert it4.tensors["residual"] is not None  # empty tensor → no consolidation
