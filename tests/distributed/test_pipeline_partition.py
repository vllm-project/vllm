import os

import pytest

from vllm.distributed import pp_partition_strategy
from vllm.distributed.pp_partition_strategy import (
    even_with_remainder_even, even_with_reminder_in_last,
    get_current_partition_strategy, get_partition_strategy_names,
    less_layer_in_first_last_partition)


def verify_indices(fn, num_hidden_layers, pp_size, expected_layers):
    expected_start, expected_end = 0, expected_layers[0]
    for pp_rank in range(pp_size):
        start, end = fn(num_hidden_layers, pp_rank, pp_size)
        assert start == expected_start, \
            f"{pp_rank=}, start: {start}, expected_start: {expected_start}"
        assert end == expected_end, \
            f"{pp_rank=}, end: {end}, expected_end: {expected_end}"
        if pp_rank == pp_size - 1:
            break
        expected_start = expected_end
        expected_end = expected_start + expected_layers[pp_rank + 1]


@pytest.mark.parametrize("name", get_partition_strategy_names())
def test_config_strategy(name):
    # None is the alias of the default strategy
    # so it is already covered.
    if name is None:
        return

    bak = os.environ.get("VLLM_PIPELINE_PARTITION_STRATEGY", None)
    os.environ["VLLM_PIPELINE_PARTITION_STRATEGY"] = name
    assert get_current_partition_strategy() == \
        getattr(pp_partition_strategy, name)
    if bak is not None:
        os.environ["VLLM_PIPELINE_PARTITION_STRATEGY"] = bak


def test_has_default():
    assert None in get_partition_strategy_names()


@pytest.mark.parametrize("inputs", [(7, 1, (7, )), (40, 4, (10, 10, 10, 10)),
                                    (39, 5, (7, 7, 7, 7, 11))])
def test_even_with_reminder_in_last(inputs):
    num_hidden_layers, pp_size, expected_layers = inputs
    verify_indices(even_with_reminder_in_last, num_hidden_layers, pp_size,
                   expected_layers)


@pytest.mark.parametrize("inputs", [(7, 1, (7, )), (40, 4, (10, 10, 10, 10)),
                                    (39, 5, (7, 8, 8, 8, 8))])
def test_even_with_remainder_even(inputs):
    num_hidden_layers, pp_size, expected_layers = inputs
    verify_indices(even_with_remainder_even, num_hidden_layers, pp_size,
                   expected_layers)


@pytest.mark.parametrize("inputs", [(7, 1, (7, )), (7, 2, (3, 4)),
                                    (40, 4, (9, 11, 11, 9)),
                                    (39, 5, (6, 8, 9, 9, 7)),
                                    (37, 5, (6, 8, 8, 8, 7)),
                                    (32, 4, (7, 9, 9, 7))])
def test_less_layer_in_first_last_partition(inputs):
    num_hidden_layers, pp_size, expected_layers = inputs
    verify_indices(less_layer_in_first_last_partition, num_hidden_layers,
                   pp_size, expected_layers)
