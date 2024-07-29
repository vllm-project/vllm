# Copyright 2024 The vLLM team.
"""Pipeline parallelism partition strategies."""
from typing import Tuple

import vllm.envs as envs

REGISTRY = {}


def register_partition_strategy(is_default=False):

    def register_partition_strategy_fn(fn):
        name = fn.__name__
        if name in REGISTRY:
            raise ValueError(
                f"Cannot register duplicate partition strategy ({name})")
        REGISTRY[name] = fn

        if is_default:
            if None in REGISTRY:
                raise ValueError(
                    "Cannot register duplicate default partition strategy")
            REGISTRY[None] = fn
        return fn

    return register_partition_strategy_fn


def get_partition_strategy(name):
    if name not in REGISTRY:
        raise ValueError(f"Unknown partition strategy ({name})")
    return REGISTRY[name]


def get_partition_strategy_names():
    return REGISTRY.keys()


def get_current_partition_strategy():
    assert envs.VLLM_PIPELINE_PARTITION_STRATEGY is not None
    return get_partition_strategy(envs.VLLM_PIPELINE_PARTITION_STRATEGY)


def get_default_partition_strategy():
    return get_partition_strategy(None)


def get_pp_indices(num_hidden_layers: int, pp_rank: int,
                   pp_size: int) -> Tuple[int, int]:
    """Get the start and end layer indices for a given partition rank."""
    return get_current_partition_strategy()(num_hidden_layers, pp_rank,
                                            pp_size)


@register_partition_strategy(is_default=True)
def even_with_reminder_in_last(num_hidden_layers: int, pp_rank: int,
                               pp_size: int) -> Tuple[int, int]:
    """Evenly distribute layers across partitions.
    If the number of layers is not divisible by the number of partitions,
    the last partition will have the remaining layers.
    """
    layers_per_partition = num_hidden_layers // pp_size
    start_layer = pp_rank * layers_per_partition
    end_layer = start_layer + layers_per_partition

    if pp_rank == pp_size - 1:
        end_layer = num_hidden_layers

    return (start_layer, end_layer)


@register_partition_strategy()
def even_with_remainder_even(num_hidden_layers: int, pp_rank: int,
                             pp_size: int) -> Tuple[int, int]:
    """Evenly distribute layers across partitions.
    If the number of layers is not divisible by the number of partitions,
    the remaining layers are evenly distributed across all partitions
    except for the first partitions.
    """
    layers_per_partition = num_hidden_layers // pp_size
    reminder_layers = num_hidden_layers % pp_size

    start_layer, end_layer = 0, 0
    ret = (0, 0)
    for idx in range(pp_rank + 1):
        end_layer = start_layer + layers_per_partition
        if idx > 0 and reminder_layers > 0:
            end_layer += 1
            reminder_layers -= 1
        ret = (start_layer, end_layer)
        start_layer = end_layer

    return ret


@register_partition_strategy()
def less_layer_in_first_last_partition(num_hidden_layers: int, pp_rank: int,
                                       pp_size: int) -> Tuple[int, int]:
    """Let the middle partitions take one more layer to reduce layers in
    the first and the last partitions. This is useful to balance the kv-cache
    memory usage across partitions and usually benefits the throughput.

    The reminder layers are evenly distributed across partitions except for
    the first partition.
    """
    layers_per_partition = num_hidden_layers // pp_size

    if pp_size <= 2:
        return even_with_remainder_even(num_hidden_layers, pp_rank, pp_size)

    # Compute the number of layers for each partition.
    layers = []
    for idx in range(pp_size):
        if idx == 0 or idx == pp_size - 1:
            # First and last partition take less layers.
            delta_layer = -((pp_size - 2) // 2)
        else:
            # Middle partitions take one more layer.
            delta_layer = 1
        layers.append(layers_per_partition + delta_layer)

    # Handle reminder layers caused by non-divisible number of layers
    # or odd number of partitions.
    reminder_layers = num_hidden_layers - sum(layers)
    for curr_idx in range(pp_size - 1, -1, -1):
        if reminder_layers <= 0:
            break
        layers[curr_idx] += 1
        reminder_layers -= 1

    start_layer = sum(layers[:pp_rank])
    end_layer = start_layer + layers[pp_rank]
    return start_layer, end_layer
