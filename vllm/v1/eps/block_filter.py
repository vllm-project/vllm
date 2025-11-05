# ABOUTME: Helpers to rewrite block tables according to EPS union masks.
# ABOUTME: Applies per-request visitation decisions for EigenPage Summaries.

from collections.abc import Collection

from vllm.v1.worker.block_table import BlockTable


def apply_union_mask(
    block_table: BlockTable,
    row_idx: int,
    visit_groups: Collection[int],
    group_blocks: int,
    *,
    sentinel: int = 0,
) -> tuple[int, int]:
    """Filter the block ids for a request row based on group visitation."""
    if group_blocks <= 0:
        raise ValueError("group_blocks must be positive")

    num_rows = block_table.block_table.np.shape[0]
    if row_idx < 0 or row_idx >= num_rows:
        raise IndexError("row_idx out of range")

    num_blocks = block_table.num_blocks_per_row[row_idx]
    if num_blocks == 0:
        return 0, 0

    row = block_table.block_table.np[row_idx]
    kept_blocks: list[int] = []

    visit_groups_set = set(visit_groups)

    for block_offset in range(num_blocks):
        group_id = block_offset // group_blocks
        if group_id in visit_groups_set:
            kept_blocks.append(int(row[block_offset]))

    new_count = len(kept_blocks)
    if new_count == num_blocks:
        return num_blocks, new_count

    row[:new_count] = kept_blocks
    row[new_count:num_blocks] = sentinel
    block_table.num_blocks_per_row[row_idx] = new_count
    return num_blocks, new_count
