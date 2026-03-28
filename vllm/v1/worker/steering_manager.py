# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manages per-request steering vector state for the model runner.

Tracks registered steering configs, assigns table rows, handles
reference counting, and populates per-layer steering_table buffers
with the correct combined (global + per_request) vectors.

Supports multiple hook points per layer. Each hook point has its own
steering table buffer (e.g. ``steering_table_pre_attn``) and global
vector cache.
"""

from collections import defaultdict

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.steering import HOOK_POINT_TABLE_ATTR

logger = init_logger(__name__)


class SteeringManager:
    """Per-request steering config manager.

    Maintains a mapping from config hashes to steering table rows,
    handles reference counting for shared configs, and writes
    combined vectors into each layer's per-hook-point steering_table
    buffers.

    Table layout (per hook point):
        Row 0: zeros sentinel (prefill / no steering)
        Row 1: global-only steering vector
        Rows 2..max_steering_configs+1: global + per_request combined
    """

    def __init__(self, max_steering_configs: int):
        self.max_steering_configs = max_steering_configs
        # config_hash -> assigned table row index (2-based)
        self.config_to_row: dict[int, int] = {}
        # config_hash -> {hook_point_str: {layer_idx: tensor}}
        # (per-request vectors only, not combined)
        self.config_vectors: dict[
            int, dict[str, dict[int, torch.Tensor]]
        ] = {}
        # config_hash -> number of active requests using this config
        self.config_refcounts: dict[int, int] = defaultdict(int)
        # Available row indices (rows 2 through max_steering_configs + 1)
        # Reversed so pop() gives lowest
        self.free_rows: list[int] = list(
            range(max_steering_configs + 1, 1, -1)
        )
        # Current global vectors: {hook_point_str: {layer_idx: tensor}}
        self.global_vectors: dict[str, dict[int, torch.Tensor]] = {}

    def register_config(
        self,
        config_hash: int,
        vectors: dict[str, dict[int, list[float]]],
    ) -> int:
        """Register a steering config, return its table row index.

        Args:
            config_hash: Deterministic hash identifying the config.
            vectors: ``{hook_point_str: {layer_idx: [floats]}}``

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
        # Store per-request vectors as tensors, keyed by hook point
        stored: dict[str, dict[int, torch.Tensor]] = {}
        for hook_point, layer_vecs in vectors.items():
            stored[hook_point] = {
                layer_idx: torch.tensor(
                    vec, dtype=torch.float32
                ).unsqueeze(0)
                for layer_idx, vec in layer_vecs.items()
            }
        self.config_vectors[config_hash] = stored
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

    def update_global_vectors(
        self,
        hook_point: str,
        layer_idx: int,
        vector: torch.Tensor,
    ) -> None:
        """Update cached global vector for a hook point and layer."""
        if hook_point not in self.global_vectors:
            self.global_vectors[hook_point] = {}
        self.global_vectors[hook_point][layer_idx] = vector.clone()

    def clear_global_vectors(self) -> None:
        """Clear all cached global vectors across all hook points."""
        self.global_vectors.clear()

    def populate_steering_tables(
        self, steerable_layers: dict[int, "torch.nn.Module"]
    ) -> None:
        """Write current state into each layer's per-hook steering_table
        buffers.

        For each hook point that has a table buffer on a layer:
            Row 0 = zeros (always)
            Row 1 = global vector for this hook+layer (or zeros)
            Rows 2+ = global + per_request for each active config
        """
        for hook_point, table_attr in HOOK_POINT_TABLE_ATTR.items():
            hp_str = hook_point.value
            for layer_idx, mod in steerable_layers.items():
                if not hasattr(mod, table_attr):
                    continue
                table = getattr(mod, table_attr)

                # Row 0: zeros
                table[0].zero_()

                # Row 1: global vector
                global_vec = self.global_vectors.get(
                    hp_str, {}
                ).get(layer_idx)
                if global_vec is not None:
                    table[1].copy_(
                        global_vec.squeeze(0).to(table.dtype)
                    )
                else:
                    table[1].zero_()

                # Rows 2+: global + per_request
                for config_hash, row in self.config_to_row.items():
                    per_req = (
                        self.config_vectors
                        .get(config_hash, {})
                        .get(hp_str, {})
                        .get(layer_idx)
                    )

                    if global_vec is not None and per_req is not None:
                        combined = (
                            global_vec.squeeze(0) + per_req.squeeze(0)
                        )
                        table[row].copy_(combined.to(table.dtype))
                    elif global_vec is not None:
                        table[row].copy_(
                            global_vec.squeeze(0).to(table.dtype)
                        )
                    elif per_req is not None:
                        table[row].copy_(
                            per_req.squeeze(0).to(table.dtype)
                        )
                    else:
                        table[row].zero_()

    @property
    def num_active_configs(self) -> int:
        """Number of currently active per-request steering configs."""
        return len(self.config_to_row)
