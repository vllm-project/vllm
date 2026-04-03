# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manages per-request steering vector state for the model runner.

Tracks registered steering configs, assigns table rows, handles
reference counting, and populates per-layer steering_table buffers
with the correct combined (global + per_request) vectors.

Supports multiple hook points per layer. Each hook point has its own
steering table buffer (e.g. ``steering_table_pre_attn``) and global
vector cache.

Supports phase-aware (prefill vs decode) steering with separate
global effective vectors for each phase.
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
        Row 0: zeros sentinel (no steering)
        Row 1: global prefill effective (global_base + global_prefill)
        Row 2: global decode effective (global_base + global_decode)
        Rows 3..max_steering_configs+2: phase-appropriate global
            + per_request combined
    """

    def __init__(self, max_steering_configs: int):
        self.max_steering_configs = max_steering_configs
        # (config_hash, phase) -> assigned table row index (3-based)
        self.config_to_row: dict[tuple[int, str], int] = {}
        # (config_hash, phase) -> {hook_point_str: {layer_idx: tensor}}
        # (per-request vectors only, not combined)
        self.config_vectors: dict[
            tuple[int, str], dict[str, dict[int, torch.Tensor]]
        ] = {}
        # (config_hash, phase) -> number of active requests using this config
        self.config_refcounts: dict[tuple[int, str], int] = defaultdict(int)
        # Available row indices (rows 3 through max_steering_configs + 2)
        # Reversed so pop() gives lowest
        self.free_rows: list[int] = list(range(max_steering_configs + 2, 2, -1))

        # Global vectors split into three tiers:
        #   base:    both-phases vectors (from global API steering_vector_* buffers)
        #   prefill: prefill-specific global vectors
        #   decode:  decode-specific global vectors
        self.global_base_vectors: dict[str, dict[int, torch.Tensor]] = {}
        self.global_prefill_vectors: dict[str, dict[int, torch.Tensor]] = {}
        self.global_decode_vectors: dict[str, dict[int, torch.Tensor]] = {}

    def register_config(
        self,
        config_hash: int,
        vectors: dict[str, dict[int, list[float]]],
        phase: str = "prefill",
    ) -> int:
        """Register a steering config, return its table row index.

        Args:
            config_hash: Deterministic hash identifying the config.
            vectors: ``{hook_point_str: {layer_idx: [floats]}}``
            phase: ``"prefill"`` or ``"decode"``

        If the ``(config_hash, phase)`` pair is already registered,
        increments refcount and returns the existing row. Otherwise
        assigns a new row. The same ``config_hash`` with a different
        phase gets its own independent row.

        Raises RuntimeError if no free rows are available.
        """
        key = (config_hash, phase)
        if key in self.config_to_row:
            self.config_refcounts[key] += 1
            return self.config_to_row[key]

        if not self.free_rows:
            raise RuntimeError(
                f"No free steering table rows. max_steering_configs="
                f"{self.max_steering_configs}, active configs="
                f"{len(self.config_to_row)}"
            )

        row = self.free_rows.pop()
        self.config_to_row[key] = row
        self.config_refcounts[key] = 1
        # Store per-request vectors as tensors, keyed by hook point
        stored: dict[str, dict[int, torch.Tensor]] = {}
        for hook_point, layer_vecs in vectors.items():
            stored[hook_point] = {
                layer_idx: torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
                for layer_idx, vec in layer_vecs.items()
            }
        self.config_vectors[key] = stored
        return row

    def release_config(self, config_hash: int, phase: str) -> None:
        """Decrement refcount for ``(config_hash, phase)``.

        Free the row when it reaches 0.
        """
        key = (config_hash, phase)
        if key not in self.config_to_row:
            return
        self.config_refcounts[key] -= 1
        if self.config_refcounts[key] <= 0:
            row = self.config_to_row.pop(key)
            self.config_vectors.pop(key, None)
            del self.config_refcounts[key]
            self.free_rows.append(row)

    def get_row_for_config(self, config_hash: int, is_prefill: bool = False) -> int:
        """Return table row for a config.

        For hash == 0 (no per-request steering):
            is_prefill=True  -> row 1 (global prefill effective)
            is_prefill=False -> row 2 (global decode effective)

        For registered per-request configs:
            Returns the assigned row (3+), looked up by
            ``(config_hash, "prefill"/"decode")``.

        For unregistered nonzero hashes:
            Falls back to row 1 (prefill) or row 2 (decode).
        """
        if config_hash == 0:
            return 1 if is_prefill else 2
        phase = "prefill" if is_prefill else "decode"
        row = self.config_to_row.get((config_hash, phase))
        if row is not None:
            return row
        # Fallback for unregistered nonzero hash
        return 1 if is_prefill else 2

    def update_global_vectors(
        self,
        hook_point: str,
        layer_idx: int,
        vector: torch.Tensor,
        phase: str = "base",
    ) -> None:
        """Update cached global vector for a hook point and layer.

        Args:
            hook_point: Hook point string (e.g. ``"post_mlp_pre_ln"``).
            layer_idx: Layer index.
            vector: The global vector tensor.
            phase: ``"base"``, ``"prefill"``, or ``"decode"``.
        """
        target = self._global_dict_for_phase(phase)
        if hook_point not in target:
            target[hook_point] = {}
        target[hook_point][layer_idx] = vector.clone()

    def clear_global_vectors(self) -> None:
        """Clear all cached global vectors across all phases and hook points."""
        self.global_base_vectors.clear()
        self.global_prefill_vectors.clear()
        self.global_decode_vectors.clear()

    def _global_dict_for_phase(self, phase: str) -> dict[str, dict[int, torch.Tensor]]:
        """Return the global vector dict for the given phase."""
        if phase == "base":
            return self.global_base_vectors
        elif phase == "prefill":
            return self.global_prefill_vectors
        elif phase == "decode":
            return self.global_decode_vectors
        else:
            raise ValueError(
                f"Invalid global vector phase: {phase!r}. "
                f"Must be 'base', 'prefill', or 'decode'."
            )

    def _get_global_vec(
        self,
        hp_str: str,
        layer_idx: int,
        source: dict[str, dict[int, torch.Tensor]],
    ) -> torch.Tensor | None:
        """Look up a global vector, returning None if absent."""
        return source.get(hp_str, {}).get(layer_idx)

    def _add_vecs(
        self,
        *vecs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Additively combine non-None tensors. Returns None if all None."""
        result: torch.Tensor | None = None
        for v in vecs:
            if v is None:
                continue
            squeezed = v.squeeze(0)
            result = squeezed.clone() if result is None else result + squeezed
        return result

    def populate_steering_tables(
        self, steerable_layers: dict[int, "torch.nn.Module"]
    ) -> None:
        """Write current state into each layer's per-hook steering_table
        buffers.

        For each hook point that has a table buffer on a layer:
            Row 0 = zeros (always)
            Row 1 = global_base + global_prefill (or zeros)
            Row 2 = global_base + global_decode (or zeros)
            Rows 3+ = phase-appropriate global + per_request
        """
        for hook_point, table_attr in HOOK_POINT_TABLE_ATTR.items():
            hp_str = hook_point.value
            for layer_idx, mod in steerable_layers.items():
                if not hasattr(mod, table_attr):
                    continue
                table = getattr(mod, table_attr)

                # Row 0: zeros
                table[0].zero_()

                # Fetch global vectors for this hook/layer
                base_vec = self._get_global_vec(
                    hp_str, layer_idx, self.global_base_vectors
                )
                prefill_vec = self._get_global_vec(
                    hp_str, layer_idx, self.global_prefill_vectors
                )
                decode_vec = self._get_global_vec(
                    hp_str, layer_idx, self.global_decode_vectors
                )

                # Row 1: global prefill effective = base + prefill
                global_prefill = self._add_vecs(base_vec, prefill_vec)
                if global_prefill is not None:
                    table[1].copy_(global_prefill.to(table.dtype))
                else:
                    table[1].zero_()

                # Row 2: global decode effective = base + decode
                global_decode = self._add_vecs(base_vec, decode_vec)
                if global_decode is not None:
                    table[2].copy_(global_decode.to(table.dtype))
                else:
                    table[2].zero_()

                # Rows 3+: phase-appropriate global + per_request
                for (config_hash, phase), row in self.config_to_row.items():
                    per_req = (
                        self.config_vectors.get((config_hash, phase), {})
                        .get(hp_str, {})
                        .get(layer_idx)
                    )
                    if phase == "prefill":
                        phase_global = global_prefill
                    else:
                        phase_global = global_decode

                    if phase_global is not None and per_req is not None:
                        combined = phase_global + per_req.squeeze(0).to(
                            phase_global.device
                        )
                        table[row].copy_(combined.to(table.dtype))
                    elif phase_global is not None:
                        table[row].copy_(phase_global.to(table.dtype))
                    elif per_req is not None:
                        table[row].copy_(per_req.squeeze(0).to(table.dtype))
                    else:
                        table[row].zero_()

    @property
    def num_active_configs(self) -> int:
        """Number of currently active per-request steering configs."""
        return len(self.config_to_row)
