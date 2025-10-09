# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.v1.outputs import (
    convert_to_token_id_list,
    convert_to_token_ids,
    get_token_count,
)


def test_convert_to_token_id_list():
    assert convert_to_token_id_list(1) == [1]
    assert convert_to_token_id_list(2) == [2]

    # Return the original list back without creating new lists
    vs = [3, 4]
    assert convert_to_token_id_list(vs) is vs
    vs = [5, 6, 7]
    assert convert_to_token_id_list(vs) is vs


def test_convert_to_token_ids():
    assert convert_to_token_ids([10]) == 10
    assert convert_to_token_ids([20]) == 20

    # Return the original list back without creating new lists
    vs = [30, 40]
    assert convert_to_token_ids(vs) is vs
    vs = [50, 60, 70]
    assert convert_to_token_ids(vs) is vs


def test_get_token_count():
    assert get_token_count(100) == 1
    assert get_token_count(200) == 1
    assert get_token_count([300]) == 1
    assert get_token_count([400, 500]) == 2
    assert get_token_count([600, 700, 800]) == 3
