# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared VRAM bank, its staging rows, the pool tables and layer offsets."""

from __future__ import annotations

from vllm.model_executor.layers.fused_moe.expert_pool.copy import copy_rows
from vllm.model_executor.layers.fused_moe.expert_pool.tables import (
    TENSORS,
    allocate_global_tables,
    check_global_tables,
    read_control,
    resident_per_layer,
    set_control,
)


class GlobalPool:
    """The shared bank, its staging views, the tables, and the layer offsets."""

    def __init__(self, device, sources, slots_per_layer, staging):
        # `sources`: one layer's six tensors in the bank's final layout; only
        # shapes/dtypes are read here (rows are filled by the layers).
        import torch

        self.slots_per_layer = list(slots_per_layer)
        self.staging_slots = staging
        self.tables = allocate_global_tables(
            device, sources[TENSORS[0]].shape[0], self.slots_per_layer, staging
        )
        self.rows = self.tables.pool_rows + staging
        self.offsets = [0]
        for slots in self.slots_per_layer[:-1]:
            self.offsets.append(self.offsets[-1] + slots)
        self.bank = {
            name: torch.zeros(
                (self.rows, *source.shape[1:]), dtype=source.dtype, device=device
            )
            for name, source in sources.items()
        }
        self.staging = {
            name: tensor[self.tables.pool_rows :] for name, tensor in self.bank.items()
        }
        self.row_bytes = sum(t[0].numel() * t.element_size() for t in sources.values())
        self.staging_bytes = self.row_bytes * staging
        self.pool_bytes = self.row_bytes * self.tables.pool_rows

    def offset(self, layer):
        return self.offsets[layer]

    def host_swap(self, layer, old_expert, new_expert):
        """Gate-closed exchange for init verification: `new_expert` takes the
        row of resident `old_expert`, which falls back to its RAM row. The
        caller copies the bytes."""
        tables = self.tables
        if int(tables.gate[0]):
            raise RuntimeError("Host swaps are only allowed while the gate is closed")
        E = tables.num_experts
        old_key, new_key = layer * E + old_expert, layer * E + new_expert
        row = int(tables.hot_phys[old_key])
        if row < 0 or int(tables.hot_phys[new_key]) >= 0:
            raise AssertionError("Swap does not match the current pool placement")
        tables.hot_phys[old_key], tables.cold_phys[old_key] = -1, old_expert
        tables.hot_phys[new_key], tables.cold_phys[new_key] = row, -1
        tables.row_key[row] = new_key

    def snapshot(self):
        """Validate the pool on the host; one copy per stats report."""
        check_global_tables(self.tables)
        return resident_per_layer(self.tables)

    def apply_control(self, **values):
        """Validate then write the controls; the gate is not touched here."""
        return set_control(self.tables, **values)

    def control(self):
        return read_control(self.tables)


def verify_bank_rows(pool, sources, sample: int = 4) -> dict[str, int]:
    """Compare up to `sample` resident rows per layer against the host
    source, byte for byte (host readback; only at safe boundaries).

    `sources[layer]` holds that layer's six host tensors. Returns the
    counts checked/mismatched; raises on the first mismatch."""
    import torch

    tables = pool.tables
    E = tables.num_experts
    row_key = tables.row_key.tolist()
    checked = 0
    per_layer = [0] * tables.num_layers
    for row, key in enumerate(row_key):
        if key < 0:
            continue
        layer, expert = divmod(key, E)
        if per_layer[layer] >= sample:
            continue
        per_layer[layer] += 1
        for name in TENSORS:
            # reshape before the byte view: a per-expert global scale row is
            # a 0-dim tensor, which cannot be viewed as bytes directly.
            got = pool.bank[name][row].detach().cpu().reshape(-1).view(torch.uint8)
            want = sources[layer][name][expert].reshape(-1).view(torch.uint8)
            if not torch.equal(got, want):
                raise AssertionError(
                    f"bank row {row} ({name}) differs from layer {layer} "
                    f"expert {expert}"
                )
        checked += 1
    return {"rows_checked": checked, "rows_resident": sum(1 for k in row_key if k >= 0)}


def copy_in(source, bank, buffers):
    """Copy the planned host rows of this layer into the bank rows."""
    copy_rows(
        source, bank, buffers.gather_src, buffers.gather_dst, buffers.gather_count
    )
