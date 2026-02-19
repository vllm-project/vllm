# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.utils.collection_utils import swap_dict_values


@pytest.mark.parametrize(
    "obj,key1,key2",
    [
        # Tests for both keys exist
        ({1: "a", 2: "b"}, 1, 2),
        # Tests for one key does not exist
        ({1: "a", 2: "b"}, 1, 3),
        # Tests for both keys do not exist
        ({1: "a", 2: "b"}, 3, 4),
    ],
)
def test_swap_dict_values(obj, key1, key2):
    original_obj = obj.copy()

    swap_dict_values(obj, key1, key2)

    if key1 in original_obj:
        assert obj[key2] == original_obj[key1]
    else:
        assert key2 not in obj
    if key2 in original_obj:
        assert obj[key1] == original_obj[key2]
    else:
        assert key1 not in obj
