import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Optional

class BlockTableJax:

    def __init__(
        self,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        device: Optional[jax.Device] = None,
        pin_memory = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        _block_table_init = jnp.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            dtype=jnp.int32,
        )
        self.block_table: jax.Array = jax.device_put(_block_table_init, self.device) \
            if self.device else _block_table_init

        self.block_table_cpu: np.ndarray = np.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            dtype=np.int32,
        )
        self.block_table_np: np.ndarray = self.block_table_cpu

        self.num_blocks_per_row: np.ndarray = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping_cpu: np.ndarray = np.zeros(
            self.max_num_batched_tokens,
            dtype=np.int64
        )
        self.slot_mapping_np: np.ndarray = self.slot_mapping_cpu

        _slot_mapping_init = jnp.zeros(
            self.max_num_batched_tokens,
            dtype=jnp.int64
        )
        self.slot_mapping: jax.Array = jax.device_put(_slot_mapping_init, self.device) \
            if self.device else _slot_mapping_init

    def append_row(
        self,
        block_ids: List[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return
        num_blocks = len(block_ids)
        current_num_blocks = self.num_blocks_per_row[row_idx]
        new_total_blocks = current_num_blocks + num_blocks

        if new_total_blocks > self.max_num_blocks_per_req:
            raise ValueError(
                f"Cannot append {num_blocks} blocks to row {row_idx}. "
                f"Exceeds max_num_blocks_per_req ({self.max_num_blocks_per_req}). "
                f"Current blocks: {current_num_blocks}, trying to add: {num_blocks}"
            )

        self.block_table_cpu[row_idx, current_num_blocks:new_total_blocks] = block_ids
        self.num_blocks_per_row[row_idx] = new_total_blocks


    def add_row(self, block_ids: List[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.block_table_cpu[row_idx].fill(0)
        self.append_row(block_ids, row_idx)

    def move_row(self, src_idx: int, tgt_idx: int) -> None:
        num_blocks = self.num_blocks_per_row[src_idx]
        self.block_table_cpu[tgt_idx, :num_blocks] = self.block_table_cpu[src_idx, :num_blocks]
        if num_blocks < self.max_num_blocks_per_req:
             self.block_table_cpu[tgt_idx, num_blocks:] = 0
        self.num_blocks_per_row[tgt_idx] = num_blocks


    def swap_row(self, src_idx: int, tgt_idx: int) -> None:
        self.num_blocks_per_row[src_idx], self.num_blocks_per_row[tgt_idx] = \
            self.num_blocks_per_row[tgt_idx], self.num_blocks_per_row[src_idx]

        src_row_copy = self.block_table_cpu[src_idx].copy()
        self.block_table_cpu[src_idx] = self.block_table_cpu[tgt_idx]
        self.block_table_cpu[tgt_idx] = src_row_copy


    def commit(self, num_reqs: int) -> None:
        if num_reqs > self.max_num_reqs:
            raise ValueError(f"num_reqs ({num_reqs}) cannot exceed max_num_reqs ({self.max_num_reqs})")

        cpu_data_slice = jnp.array(self.block_table_cpu[:num_reqs], dtype=jnp.int32)
        device_data_slice = jax.device_put(cpu_data_slice, self.device) if self.device else cpu_data_slice
        self.block_table = self.block_table.at[:num_reqs].set(device_data_slice)

        if num_reqs < self.max_num_reqs:
            remaining_shape = (self.max_num_reqs - num_reqs, self.max_num_blocks_per_req)
            zeros_for_remaining = jnp.zeros(remaining_shape, dtype=jnp.int32)
            device_zeros = jax.device_put(zeros_for_remaining, self.device) if self.device else zeros_for_remaining
            self.block_table = self.block_table.at[num_reqs:].set(device_zeros)


    def clear(self) -> None:
        _block_table_init = jnp.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=jnp.int32,
        )
        self.block_table = jax.device_put(_block_table_init, self.device) \
            if self.device else _block_table_init

        self.block_table_cpu.fill(0)
        self.num_blocks_per_row.fill(0)

        _slot_mapping_init = jnp.zeros(
            self.max_num_batched_tokens,
            dtype=jnp.int64
        )
        self.slot_mapping = jax.device_put(_slot_mapping_init, self.device) \
            if self.device else _slot_mapping_init
        self.slot_mapping_cpu.fill(0)


    def get_device_tensor(self) -> jax.Array:
        return self.block_table

    def get_cpu_tensor(self) -> np.ndarray:
        return self.block_table_cpu

    def get_numpy_array(self) -> np.ndarray:
        return self.block_table_np
