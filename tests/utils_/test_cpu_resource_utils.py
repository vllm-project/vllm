# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils.cpu_resource_utils import parse_id_list


def test_parse_id_list_range_and_singles():
    assert parse_id_list("0-2,4,7-8") == [0, 1, 2, 4, 7, 8]


def test_parse_id_list_single_number():
    assert parse_id_list("5") == [5]


def test_parse_id_list_range():
    assert parse_id_list("1-3") == [1, 2, 3]


def test_parse_id_list_mixed():
    assert parse_id_list("0,2-4,6") == [0, 2, 3, 4, 6]


def test_parse_id_list_empty_string():
    assert parse_id_list("") == []


def test_parse_id_list_dedup_and_sort():
    assert parse_id_list("3,1,2,1,3") == [1, 2, 3]


def test_parse_id_list_overlapping_range_and_single():
    assert parse_id_list("1-3,2,4") == [1, 2, 3, 4]


def test_parse_id_list_single_range():
    assert parse_id_list("0-0") == [0]


def test_parse_id_list_large_numbers():
    assert parse_id_list("100-102,200") == [100, 101, 102, 200]
