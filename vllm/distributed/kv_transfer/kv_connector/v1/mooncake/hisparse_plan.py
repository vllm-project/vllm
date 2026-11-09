# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-position planning for same-TP sparse MLA imports into HiSparse."""


def pair_pages(
    source_ids: list[int],
    source_positions: list[int],
    destination_ids: list[int],
    destination_positions: list[int],
) -> tuple[list[int], list[int]]:
    """Resolve each requested logical page independently of group numbering."""
    if len(source_ids) != len(source_positions) or len(destination_ids) != len(
        destination_positions
    ):
        raise ValueError("Page positions do not match physical block IDs")
    for positions in (source_positions, destination_positions):
        if any(position < 0 for position in positions) or any(
            left >= right for left, right in zip(positions, positions[1:])
        ):
            raise ValueError("Page positions must be nonnegative and increasing")
    if any(block <= 0 for block in source_ids + destination_ids):
        raise ValueError("Null blocks cannot participate in a HiSparse import")
    source_by_position = dict(zip(source_positions, source_ids))
    try:
        selected = [source_by_position[position] for position in destination_positions]
    except KeyError as error:
        raise ValueError(
            "Producer does not cover the requested logical page"
        ) from error
    return selected, list(destination_ids)
