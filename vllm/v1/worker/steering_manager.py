# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request steering state manager (shared-nothing, deterministic replay).

Determinism contract
--------------------
Steering state is shared-nothing with deterministic replay. Every worker
executes identical ``set_steering_vectors`` / ``clear_steering_vectors``
calls (via ``collective_rpc``) and sees an identical ``SchedulerOutput``
stream, so every worker's ``SteeringManager`` derives identical
``config_to_row`` assignments, identical ``free_rows`` state, and an
identical ``steering_index`` tensor each step -- even though each worker
stores vectors only for layers it physically owns. No cross-rank
collectives are needed in the hot path.

Concrete implications:

* Row allocation is fully rank-local. ``register_config`` runs on every
  rank for every config, regardless of whether that rank owns an
  affected layer. This is a correctness requirement, not an optimization:
  row IDs flow through ``steering_index`` into the ``apply_steering``
  gather on every rank, so they MUST match.
* Global-vector broadcast lives at the API layer (one ``collective_rpc``),
  not at the worker layer. There is no NCCL all-reduce of steering tables.
* ``SamplingParams.prefill_steering_config_hash`` and
  ``decode_steering_config_hash`` are pure functions of the request
  payload, identical on every rank.

See ``docs/design/steering_runtime.md`` section "Distributed execution"
for the full mental model.

Class responsibilities (unchanged by the contract):

Tracks registered steering configs, assigns table rows, handles
reference counting, and populates per-layer steering_table buffers
with the correct combined (global + per_request) vectors. Supports
multiple hook points per layer; each hook point has its own steering
table buffer (e.g. ``steering_table_pre_attn``) and global vector cache.
Supports phase-aware (prefill vs decode) steering with separate global
effective vectors for each phase.
"""

from collections import defaultdict

import numpy as np
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

    def __init__(
        self,
        max_steering_configs: int,
        device: torch.device | None = None,
    ):
        self.max_steering_configs = max_steering_configs
        self.device = device
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
        #   base:    both-phases vectors (from global API)
        #   prefill: prefill-specific global vectors
        #   decode:  decode-specific global vectors
        self.global_base_vectors: dict[str, dict[int, torch.Tensor]] = {}
        self.global_prefill_vectors: dict[str, dict[int, torch.Tensor]] = {}
        self.global_decode_vectors: dict[str, dict[int, torch.Tensor]] = {}

        # When True, populate_steering_tables() needs to run to bring the
        # per-layer table buffers in sync with current state. Set by every
        # state mutator (register_config new-row path, release_config
        # refcount->0 path, update_global_vectors, clear_global_vectors);
        # cleared at the end of populate_steering_tables. Initialized True
        # so the first populate call always runs.
        self._tables_dirty: bool = True

        # Cached scratch tensors for populate_steering_tables. ``indices``
        # is the GPU int64 tensor of target row positions
        # ``[0, 1, 2, *config_rows]`` and ``zero_row`` is a hidden-size
        # fp32 zeros tensor used as the row-0 / no-vector fallback. Their
        # contents only depend on ``config_to_row`` (and the per-layer
        # table device/hidden_size, which is fixed). They DO NOT depend
        # on global-vector updates, so we cache them and only invalidate
        # in register_config / release_config (the two paths that mutate
        # ``config_to_row``).
        #
        # ``_indices_dirty`` is independent of ``_tables_dirty``: every
        # global-vector update sets ``_tables_dirty`` (forcing a populate)
        # but does NOT need to rebuild the scratch tensors.
        self._cached_indices: torch.Tensor | None = None
        self._cached_zero_row: torch.Tensor | None = None
        self._cached_ordered_configs: list[tuple[tuple[int, str], int]] | None = None
        self._indices_dirty: bool = True

    def register_config(
        self,
        config_hash: int,
        vectors: dict[str, dict[int, list[float]]],
        phase: str = "prefill",
        *,
        locally_owned_layers: frozenset[int] | None = None,
    ) -> int:
        """Register a steering config, return its table row index.

        Args:
            config_hash: Deterministic hash identifying the config.
            vectors: ``{hook_point_str: {layer_idx: [floats]}}``
            phase: ``"prefill"`` or ``"decode"``
            locally_owned_layers: If provided, only layers in this set
                have tensors materialized on this worker.  Layers
                outside the set are skipped at tensor-construction time
                but row allocation still proceeds, so row IDs remain
                identical across ranks (distributed-steering
                determinism contract).  When ``None`` (default), no
                filtering — all layers in ``vectors`` get tensors.

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
        # Store per-request vectors as tensors, keyed by hook point.
        # Under PP, each rank only owns a subset of decoder layers, so
        # materializing tensors for non-local layers is pure waste.
        # Row allocation above is unconditional — the filter only
        # affects what tensors get constructed, not which row is
        # assigned.
        # Per-layer vectors are batched into ONE stacked H2D copy per hook
        # point. Building each row as its own ``torch.tensor(list,
        # device=cuda)`` triggers a synchronous ``cudaMemcpy`` per layer,
        # which dominates the phase-transition cost when many configs are
        # registered at the start of a decode step. Stacking up front and
        # transferring once amortizes the sync to a single cost per hook.
        stored: dict[str, dict[int, torch.Tensor]] = {}
        for hook_point, layer_vecs in vectors.items():
            items = [
                (layer_idx, vec)
                for layer_idx, vec in layer_vecs.items()
                if locally_owned_layers is None or layer_idx in locally_owned_layers
            ]
            if not items:
                stored[hook_point] = {}
                continue
            layer_idxs = [layer_idx for layer_idx, _ in items]
            raw_vecs = [vec for _, vec in items]
            stacked = self._stack_vectors_to_device(raw_vecs)
            # ``stacked[i:i+1]`` is a (1, hidden) view, matching the
            # per-layer ``.unsqueeze(0)`` shape that ``_populate_one_table``
            # expects to ``.squeeze(0)``. No extra copy.
            stored[hook_point] = {
                layer_idx: stacked[i : i + 1] for i, layer_idx in enumerate(layer_idxs)
            }
        self.config_vectors[key] = stored
        # New row content needs to be written into the per-layer tables on
        # the next populate call. (Refcount-hit path doesn't set this flag
        # because the row's contents are already in the table.)
        self._tables_dirty = True
        # config_to_row changed; the cached indices/ordered_configs scratch
        # is now stale and must be rebuilt on the next populate.
        self._indices_dirty = True
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
            # The row is now stale (no one references it), but mark dirty so
            # if another config gets assigned to this row before the next
            # populate, the populate runs and overwrites the stale content.
            self._tables_dirty = True
            # config_to_row shrunk; cached indices scratch is stale.
            self._indices_dirty = True

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
        *,
        locally_owned_layers: frozenset[int] | None = None,
    ) -> None:
        """Update cached global vector for a hook point and layer.

        Args:
            hook_point: Hook point string (e.g. ``"post_mlp"``).
            layer_idx: Layer index.
            vector: The global vector tensor.
            phase: ``"base"``, ``"prefill"``, or ``"decode"``.
            locally_owned_layers: If provided and ``layer_idx`` is not
                in the set, this call is a no-op.  Defense-in-depth
                for the distributed-steering determinism contract:
                callers in the mixin already filter by locally-present
                layers, but self-defending the manager means its
                invariants do not depend on the caller.
        """
        if locally_owned_layers is not None and layer_idx not in locally_owned_layers:
            return
        target = self._global_dict_for_phase(phase)
        if hook_point not in target:
            target[hook_point] = {}
        target[hook_point][layer_idx] = vector.clone()
        # Global rows 1, 2 and all per-request rows depend on this state.
        self._tables_dirty = True

    def clear_global_vectors(self) -> None:
        """Clear all cached global vectors across all phases and hook points."""
        self.global_base_vectors.clear()
        self.global_prefill_vectors.clear()
        self.global_decode_vectors.clear()
        self._tables_dirty = True

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

    def _stack_vectors_to_device(self, vecs: list[list[float]]) -> torch.Tensor:
        """Stack a list of equal-length float vectors into a (N, hidden)
        tensor on ``self.device`` via a single batched H2D copy.

        For CUDA targets, the array is assembled in numpy, wrapped via
        ``torch.from_numpy``, pinned, and copied with ``non_blocking=True``
        — one ``cudaMemcpy`` regardless of ``len(vecs)``. Per-layer
        ``torch.tensor(list, device=cuda)`` would otherwise issue one
        synchronous transfer per row, which dominates the
        ``register_config`` cost when many configs enter decode at once.
        """
        try:
            arr = np.asarray(vecs, dtype=np.float32)
        except (ValueError, TypeError) as exc:  # ragged / non-numeric input
            raise ValueError(
                "register_config received steering vectors of inconsistent "
                "shape or non-numeric dtype; expected a list of equal-length "
                f"float vectors. Underlying error: {exc}"
            ) from exc
        if arr.ndim != 2:
            raise ValueError(
                "register_config expected a 2D stack of steering vectors "
                f"(N, hidden); got array with shape {arr.shape}."
            )
        cpu_t = torch.from_numpy(arr)
        if self.device is None:
            return cpu_t
        if self.device.type == "cuda":
            cpu_t = cpu_t.pin_memory()
            return cpu_t.to(self.device, non_blocking=True)
        return cpu_t.to(self.device)

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

        Optimizations vs. the naive per-(hook, layer) loop:

        1.  ``indices`` (GPU int64) and ``zero_row`` (GPU fp32) scratch
            tensors are cached on the manager and only rebuilt when
            ``config_to_row`` mutates (register/release). Global-vector
            updates do NOT invalidate them.
        2.  Row assembly across all active (hook, layer) tables produces
            a single ``(num_active_tables, num_rows, hidden)`` fp32
            tensor that is dtype-cast in one kernel launch, then written
            to each table via ``index_copy_``. This consolidates ~84
            independent ``stacked.to(dtype=...)`` casts on Gemma-3-4B
            (3 hooks * 28 layers) into one.
        """
        # Build a flat list of (table_buffer, hp_str, layer_idx) for every
        # (hook, layer) pair that actually has a table buffer registered.
        # Layers may register a SUBSET of hook tables (the
        # ``hasattr(mod, table_attr)`` check), so this drives the batched
        # scatter on the active set rather than assuming a dense layout.
        active_tables: list[tuple[torch.Tensor, str, int]] = []
        for hook_point, table_attr in HOOK_POINT_TABLE_ATTR.items():
            hp_str = hook_point.value
            for layer_idx, mod in steerable_layers.items():
                if not hasattr(mod, table_attr):
                    continue
                table = getattr(mod, table_attr)
                active_tables.append((table, hp_str, layer_idx))

        if not active_tables:
            self._tables_dirty = False
            return

        # Derive device and hidden_size from the first active table.
        # All tables registered through ``register_steering_buffers`` share
        # the same device and hidden_size by construction.
        first_table = active_tables[0][0]
        device = first_table.device
        hidden_size = first_table.shape[1]

        # Snapshot config_to_row ordering. This is ALWAYS needed for the
        # row-assembly loop below, but ``indices`` only needs rebuilding
        # when this ordering changed (register/release).
        if (
            self._indices_dirty
            or self._cached_indices is None
            or self._cached_zero_row is None
            or self._cached_ordered_configs is None
        ):
            new_ordered_configs: list[tuple[tuple[int, str], int]] = list(
                self.config_to_row.items()
            )
            target_indices_list = [0, 1, 2] + [row for _, row in new_ordered_configs]
            self._cached_indices = torch.tensor(
                target_indices_list, dtype=torch.long, device=device
            )
            self._cached_zero_row = torch.zeros(
                hidden_size, dtype=torch.float32, device=device
            )
            self._cached_ordered_configs = new_ordered_configs
            self._indices_dirty = False
        indices: torch.Tensor = self._cached_indices
        zero_row: torch.Tensor = self._cached_zero_row
        ordered_configs: list[tuple[tuple[int, str], int]] = (
            self._cached_ordered_configs
        )

        # Build all rows for all active tables in fp32. ``all_rows`` ends
        # up shape ``(num_active_tables, num_rows, hidden)``. We do ONE
        # ``.to(dtype=table.dtype)`` cast on the whole stack instead of
        # per-(hook, layer), then index_copy_ each layer's slice.
        num_rows = 3 + len(ordered_configs)
        per_table_rows: list[list[torch.Tensor]] = []
        for _table, hp_str, layer_idx in active_tables:
            base_vec = self._get_global_vec(hp_str, layer_idx, self.global_base_vectors)
            prefill_vec = self._get_global_vec(
                hp_str, layer_idx, self.global_prefill_vectors
            )
            decode_vec = self._get_global_vec(
                hp_str, layer_idx, self.global_decode_vectors
            )

            global_prefill = self._add_vecs(base_vec, prefill_vec)
            global_decode = self._add_vecs(base_vec, decode_vec)

            rows: list[torch.Tensor] = [zero_row]  # row 0: always zero
            rows.append(global_prefill if global_prefill is not None else zero_row)
            rows.append(global_decode if global_decode is not None else zero_row)

            for (config_hash, phase), _row_idx in ordered_configs:
                per_req = (
                    self.config_vectors.get((config_hash, phase), {})
                    .get(hp_str, {})
                    .get(layer_idx)
                )
                if phase == "prefill":
                    phase_global = global_prefill
                elif phase == "decode":
                    phase_global = global_decode
                else:
                    raise ValueError(
                        f"Invalid phase: {phase!r}. Must be 'prefill' or 'decode'."
                    )

                if phase_global is not None and per_req is not None:
                    # Per-request vectors are registered from raw Python lists
                    # and default to CPU; global vectors inherit the model's
                    # device. Align to the global's device before adding so a
                    # CPU/CUDA mix doesn't raise.
                    per_req_aligned = per_req.squeeze(0).to(phase_global.device)
                    row_content = phase_global + per_req_aligned
                elif phase_global is not None:
                    row_content = phase_global
                elif per_req is not None:
                    row_content = per_req.squeeze(0)
                else:
                    row_content = zero_row
                rows.append(row_content)

            assert len(rows) == num_rows
            per_table_rows.append(rows)

        # Stack all rows into one fp32 tensor of shape
        # ``(num_active_tables, num_rows, hidden)`` and split by dtype.
        # The vast majority of deployments use a single dtype across all
        # tables, so the dtype loop is one iteration in the common case.
        flat_rows: list[torch.Tensor] = [
            r for table_rows in per_table_rows for r in table_rows
        ]
        stacked_fp32 = torch.stack(flat_rows).reshape(
            len(active_tables), num_rows, hidden_size
        )

        # Group by dtype so we can do one cast per dtype.
        dtype_to_indices: dict[torch.dtype, list[int]] = defaultdict(list)
        for i, (table, _hp, _layer) in enumerate(active_tables):
            dtype_to_indices[table.dtype].append(i)

        for dtype, table_indices_in_active in dtype_to_indices.items():
            # One batched cast covering every table that uses this dtype.
            casted = stacked_fp32[table_indices_in_active].to(dtype=dtype)
            for casted_pos, active_pos in enumerate(table_indices_in_active):
                table = active_tables[active_pos][0]
                table.index_copy_(0, indices, casted[casted_pos])

        # All per-layer table buffers now reflect current state. Subsequent
        # calls can be skipped by the caller until a mutator sets dirty again.
        self._tables_dirty = False

    @property
    def num_active_configs(self) -> int:
        """Number of currently active per-request steering configs."""
        return len(self.config_to_row)
