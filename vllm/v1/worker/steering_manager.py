# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manages per-request steering vector state for the model runner.

Tracks registered steering configs, assigns table rows, handles
reference counting, and populates per-layer steering_table buffers
with the correct combined (global + per_request) vectors.
"""

from collections import defaultdict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class SteeringManager:
    """Per-request steering config manager.

    Maintains a mapping from config hashes to steering table rows,
    handles reference counting for shared configs, and writes
    combined vectors into each layer's steering_table buffer.

    Table layout:
        Row 0: zeros sentinel (prefill / no steering)
        Row 1: global-only steering vector
        Rows 2..max_steering_configs+1: global + per_request combined
    """

    def __init__(self, max_steering_configs: int):
        self.max_steering_configs = max_steering_configs
        # config_hash -> assigned table row index (2-based)
        self.config_to_row: dict[int, int] = {}
        # config_hash -> {layer_idx: tensor} (per-request vectors only,
        # not combined)
        self.config_vectors: dict[int, dict[int, torch.Tensor]] = {}
        # config_hash -> number of active requests using this config
        self.config_refcounts: dict[int, int] = defaultdict(int)
        # Available row indices (rows 2 through max_steering_configs + 1)
        # Reversed so pop() gives lowest
        self.free_rows: list[int] = list(range(max_steering_configs + 1, 1, -1))
        # Current global vectors per layer (set via global steering API)
        self.global_vectors: dict[int, torch.Tensor] = {}

    def register_config(
        self,
        config_hash: int,
        vectors: dict[int, list[float]],
    ) -> int:
        """Register a steering config, return its table row index.

        If the config_hash is already registered, increments refcount
        and returns the existing row. Otherwise assigns a new row.

        Raises RuntimeError if no free rows are available.
        """
        if config_hash in self.config_to_row:
            self.config_refcounts[config_hash] += 1
            return self.config_to_row[config_hash]

        if not self.free_rows:
            raise RuntimeError(
                f"No free steering table rows. max_steering_configs="
                f"{self.max_steering_configs}, active configs="
                f"{len(self.config_to_row)}"
            )

        row = self.free_rows.pop()
        self.config_to_row[config_hash] = row
        self.config_refcounts[config_hash] = 1
        # Store per-request vectors as tensors
        self.config_vectors[config_hash] = {
            layer_idx: torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
            for layer_idx, vec in vectors.items()
        }
        return row

    def release_config(self, config_hash: int) -> None:
        """Decrement refcount. Free the row when it reaches 0."""
        if config_hash not in self.config_to_row:
            return
        self.config_refcounts[config_hash] -= 1
        if self.config_refcounts[config_hash] <= 0:
            row = self.config_to_row.pop(config_hash)
            self.config_vectors.pop(config_hash, None)
            del self.config_refcounts[config_hash]
            self.free_rows.append(row)

    def get_row_for_config(self, config_hash: int) -> int:
        """Return table row for a config.

        Returns 1 (global-only) if hash is 0 or unregistered.
        """
        if config_hash == 0:
            return 1
        return self.config_to_row.get(config_hash, 1)

    def update_global_vectors(self, layer_idx: int, vector: torch.Tensor) -> None:
        """Update cached global vector for a layer."""
        self.global_vectors[layer_idx] = vector.clone()

    def clear_global_vectors(self) -> None:
        """Clear all cached global vectors."""
        self.global_vectors.clear()

    def populate_steering_tables(
        self, steerable_layers: dict[int, "torch.nn.Module"]
    ) -> None:
        """Write current state into each layer's steering_table buffer.

        Row 0 = zeros (always)
        Row 1 = global vector for this layer (or zeros if no global)
        Rows 2+ = global + per_request for each active config
        """
        for layer_idx, mod in steerable_layers.items():
            table = mod.steering_table
            # Row 0: zeros
            table[0].zero_()

            # Row 1: global vector
            global_vec = self.global_vectors.get(layer_idx)
            if global_vec is not None:
                table[1].copy_(global_vec.squeeze(0).to(table.dtype))
            else:
                table[1].zero_()

            # Rows 2+: global + per_request
            for config_hash, row in self.config_to_row.items():
                per_request_vecs = self.config_vectors.get(config_hash, {})
                per_request_vec = per_request_vecs.get(layer_idx)

                if global_vec is not None and per_request_vec is not None:
                    combined = global_vec.squeeze(0) + per_request_vec.squeeze(0)
                    table[row].copy_(combined.to(table.dtype))
                elif global_vec is not None:
                    table[row].copy_(global_vec.squeeze(0).to(table.dtype))
                elif per_request_vec is not None:
                    table[row].copy_(per_request_vec.squeeze(0).to(table.dtype))
                else:
                    table[row].zero_()

    @property
    def num_active_configs(self) -> int:
        """Number of currently active per-request steering configs."""
        return len(self.config_to_row)
