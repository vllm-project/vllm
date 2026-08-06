# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MultipleOf:
    base: int


KernelBlockSize = int | MultipleOf


def select_common_block_size_from_constraints(
    kv_manager_block_size: int,
    constraints: Sequence[Sequence[KernelBlockSize]],
) -> int:
    """Select the largest manager-block divisor supported by every constraint."""

    def is_supported(block_size: int) -> bool:
        return all(
            any(
                block_size == supported
                if isinstance(supported, int)
                else block_size % supported.base == 0
                for supported in sizes
            )
            for sizes in constraints
        )

    if not constraints or any(not sizes for sizes in constraints):
        raise ValueError("Kernel block-size constraints must not be empty.")

    if is_supported(kv_manager_block_size):
        return kv_manager_block_size

    exact_sizes = {
        supported
        for sizes in constraints
        for supported in sizes
        if isinstance(supported, int)
    }
    for block_size in sorted(exact_sizes, reverse=True):
        if kv_manager_block_size % block_size == 0 and is_supported(block_size):
            return block_size

    raise ValueError(f"No common block size for {kv_manager_block_size}.")
