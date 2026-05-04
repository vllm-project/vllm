# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define activation steering functionality mixin for model runners.
"""

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config.steering_types import (
    SteeringVectorSpec,
    merge_steering_specs,
    resolve_effective_vectors,
    scale_steering_spec,
)
from vllm.exceptions import SteeringVectorError
from vllm.logger import init_logger
from vllm.model_executor.layers.steering import (
    HOOK_POINT_TABLE_ATTR,
    SteeringHookPoint,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.steering_manager import SteeringManager


def _get_steering_ranks() -> tuple[int, int]:
    """Return ``(tp_rank, pp_rank)`` for the current worker.

    Used to tag steering RPC results so the router can detect TP
    divergence (a server-side invariant violation). Guarded so that
    tests / single-rank setups that haven't initialized the
    distributed groups still work.
    """
    try:
        from vllm.distributed.parallel_state import (
            get_pp_group,
            get_tp_group,
        )

        return (get_tp_group().rank_in_group, get_pp_group().rank_in_group)
    except Exception:
        return (0, 0)


if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)


# Defined as a mixin for GPUModelRunner
class SteeringModelRunnerMixin:
    """Consolidates all activation-steering state and logic on the model runner.

    Mirrors the ``LoRAModelRunnerMixin`` pattern: the mixin owns every
    piece of steering-related state and exposes the public API
    (``set_steering_vectors``, ``clear_steering_vectors``,
    ``list_steerable_layers``, ``get_steering_status``) that
    ``WorkerBase`` and its concrete subclasses delegate to via thin
    passthroughs.
    """

    # --- class-level attribute declarations --------------------------------
    # All steering state is initialised eagerly by ``_init_steering_state``
    # at the end of ``GPUModelRunner.load_model``.  The class-level
    # defaults below cover the pre-init window (e.g. unit tests that
    # construct the mixin without going through load_model) so plain
    # attribute access is safe without ``hasattr`` guards.
    _steering_manager: SteeringManager | None = None
    _steerable_layers_cache: dict[int, nn.Module] | None = None
    _pending_steering_transitions: list[
        tuple[str, int, dict[str, dict[int, list[float]]], str]
    ]
    _pending_steering_registrations: list[
        tuple[str, int, dict[str, dict[int, list[float]]], str]
    ]
    _req_steering_phase: dict[str, str]
    _steering_index_dirty: bool
    # Worker-side mirror of the API server's named steering module
    # registry.  Populated via ``register_steering_modules`` RPC during
    # API server bootstrap and on every /v1/steering/modules/{register,
    # unregister} call.  Per-process, per-worker — collective_rpc
    # guarantees identical state across TP × PP ranks.
    _steering_module_registry: dict[
        str,
        tuple[
            SteeringVectorSpec | None,
            SteeringVectorSpec | None,
            SteeringVectorSpec | None,
        ],
    ]
    # Set of layer indices physically owned by this worker.  Under PP,
    # this is a contiguous subset of ``[0, num_layers)``; under single-
    # worker and under TP (which replicates all layers per rank), it
    # equals the full model's layer set.  Threaded into
    # ``SteeringManager`` calls so non-local tensors are never
    # materialized on this rank.
    _locally_owned_layers: frozenset[int]
    # CPU scratch arrays used by ``_update_steering_buffers`` to build
    # the per-token row mapping in a single ``np.repeat`` + non-blocking
    # H2D copy, replacing the per-request slice-assign loop.  The
    # per-request scratches are sized to ``max_num_seqs``; the
    # row-per-token scratch is a pinned-memory torch tensor sized to
    # ``max_num_batched_tokens`` so the H2D copy can actually overlap
    # compute (``non_blocking=True`` on a non-pinned source silently
    # falls back to a synchronous copy).  ``None`` when steering is
    # inactive.
    _steering_rows_scratch: np.ndarray | None = None
    _steering_n_tokens_scratch: np.ndarray | None = None
    _steering_index_pinned: torch.Tensor | None = None

    # Attributes provided by the concrete model runner that mixes this
    # class in.  Declared here purely so static type checking can see
    # them — there is no runtime assignment.
    if TYPE_CHECKING:
        vllm_config: VllmConfig
        input_batch: InputBatch
        requests: dict[str, CachedRequestState]

        def get_model(self) -> nn.Module: ...

    # -----------------------------------------------------------------------
    # Eager initialisation
    # -----------------------------------------------------------------------

    def _init_steering_state(self) -> None:
        """Initialise steering state at the end of model load.

        Walks the loaded model for layers that registered steering
        buffers, captures the buffer device, and constructs the
        ``SteeringManager``.  Must be called exactly once — typically
        from ``GPUModelRunner.load_model`` after the model is fully
        loaded.

        When steering is disabled (no ``SteeringConfig``) or the model
        has no steerable layers, ``_steering_manager`` stays ``None``
        so per-step ``_update_steering_buffers`` and the public API
        methods can short-circuit cheaply.
        """
        steerable: dict = {}
        if hasattr(self, "get_model"):
            for mod in self.get_model().modules():
                if not hasattr(mod, "layer_idx"):
                    continue
                has_any_table = any(
                    hasattr(mod, attr) for attr in HOOK_POINT_TABLE_ATTR.values()
                )
                if has_any_table:
                    steerable[mod.layer_idx] = mod
        self._steerable_layers_cache = steerable
        self._locally_owned_layers = frozenset(steerable.keys())
        self._req_steering_phase = {}
        self._steering_index_dirty = False
        self._pending_steering_transitions = []
        self._pending_steering_registrations = []
        self._steering_module_registry = {}

        steering_config = getattr(self.vllm_config, "steering_config", None)
        if steering_config is None or not steerable:
            self._steering_manager = None
            return

        # Resolve device from the first steerable layer's table buffer
        # so per-request vectors are allocated on the same device,
        # avoiding CPU->GPU copies each step.
        table_device: torch.device | None = None
        for mod in steerable.values():
            for attr in HOOK_POINT_TABLE_ATTR.values():
                if hasattr(mod, attr):
                    table_device = getattr(mod, attr).device
                    break
            if table_device is not None:
                break

        self._steering_manager = SteeringManager(
            steering_config.max_steering_configs,
            device=table_device,
        )

        # Pre-allocate CPU scratch buffers for the vectorized
        # steering_index build in ``_update_steering_buffers``.  The
        # per-request numpy buffers hold one entry per request in the
        # batch (bounded by ``max_num_seqs``); the pinned torch tensor
        # holds the expanded per-token row array (bounded by
        # ``max_num_batched_tokens``) and is the source of the single
        # H2D copy each step.  Pinning lets ``non_blocking=True`` on
        # the copy actually overlap with model compute.
        scheduler_config = getattr(self.vllm_config, "scheduler_config", None)
        if scheduler_config is not None:
            max_tokens = int(scheduler_config.max_num_batched_tokens)
            max_seqs = int(getattr(scheduler_config, "max_num_seqs", max_tokens))
            self._steering_rows_scratch = np.zeros(max_seqs, dtype=np.int64)
            self._steering_n_tokens_scratch = np.zeros(max_seqs, dtype=np.int64)
            try:
                self._steering_index_pinned = torch.zeros(
                    max_tokens, dtype=torch.long, pin_memory=True
                )
            except RuntimeError:
                # Pinned memory unavailable (e.g. CPU-only test
                # environment); fall back to a regular CPU tensor.
                self._steering_index_pinned = torch.zeros(max_tokens, dtype=torch.long)

    # -----------------------------------------------------------------------
    # Steerable-layer discovery and vector-spec validation
    # -----------------------------------------------------------------------

    def _steerable_layers(self) -> dict:
        """Return ``{layer_idx: module}`` for layers with steering buffers.

        Works with any model runner that exposes ``get_model()``,
        including the V2 runner.  Result is cached after first
        successful discovery.

        A layer is considered steerable if it has ``layer_idx`` and at
        least one ``steering_table_*`` buffer for any hook point.
        """
        cache = self._steerable_layers_cache
        if cache is not None:
            return cache

        if not hasattr(self, "get_model"):
            return {}
        layers: dict = {}
        for mod in self.get_model().modules():
            if not hasattr(mod, "layer_idx"):
                continue
            has_any_table = any(
                hasattr(mod, attr) for attr in HOOK_POINT_TABLE_ATTR.values()
            )
            if has_any_table:
                layers[mod.layer_idx] = mod

        if layers:
            self._steerable_layers_cache = layers

        return layers

    def _validate_vectors_spec(
        self,
        vectors_data: dict[str, dict[int, list[float]]],
        steerable: dict,
    ) -> set[int]:
        """Validate hook-point / layer / vector combinations.

        Returns the set of valid layer indices on this worker.
        Raises ``SteeringVectorError`` on invalid hook points,
        mismatched sizes, or non-finite values.
        """
        valid_indices: set[int] = set()
        for hook_point_str, layer_vecs in vectors_data.items():
            try:
                hp_enum = SteeringHookPoint(hook_point_str)
            except ValueError as exc:
                raise SteeringVectorError(
                    f"Invalid hook point: {hook_point_str!r}"
                ) from exc
            table_attr = HOOK_POINT_TABLE_ATTR[hp_enum]

            for idx, vec_values in layer_vecs.items():
                if idx not in steerable:
                    continue
                mod = steerable[idx]
                if not hasattr(mod, table_attr):
                    raise SteeringVectorError(
                        f"Hook point {hook_point_str!r} not active on layer {idx}"
                    )
                buf = getattr(mod, table_attr)
                expected_size = buf.shape[1]
                if len(vec_values) != expected_size:
                    raise SteeringVectorError(
                        f"Layer {idx} ({hook_point_str}): expected "
                        f"vector of size {expected_size}, "
                        f"got {len(vec_values)}"
                    )
                if not all(math.isfinite(v) for v in vec_values):
                    raise SteeringVectorError(
                        f"Layer {idx} ({hook_point_str}): steering "
                        f"vector contains non-finite values "
                        f"(NaN or Infinity)"
                    )
                valid_indices.add(idx)
        return valid_indices

    def list_steerable_layers(self) -> dict[int, list[str]]:
        """Return steerable layers on this worker with their hook points.

        Returns ``{layer_idx: [hook_point_name, ...]}`` for every
        layer owned by this worker that has at least one steering
        table buffer registered. Hook-point names are sorted for
        determinism.
        """
        result: dict[int, list[str]] = {}
        for idx, mod in self._steerable_layers().items():
            hooks = sorted(
                hp.value
                for hp, attr in HOOK_POINT_TABLE_ATTR.items()
                if hasattr(mod, attr)
            )
            result[idx] = hooks
        return result

    def _notify_manager_vectors(
        self,
        vectors_data: dict[str, dict[int, list[float]]],
        steerable: dict,
        valid_indices: set[int],
        phase: str,
    ) -> None:
        """Notify SteeringManager of global vector changes for a given
        phase (``"base"``, ``"prefill"``, or ``"decode"``).

        Converts the raw ``list[float]`` values from *vectors_data*
        into tensors matching the layer buffer dtype/device, then passes
        them to the manager.  This avoids reading from shared buffers,
        which would silently use stale or overwritten data for
        phase-specific tiers.
        """
        mgr = self._steering_manager
        if mgr is None:
            return
        locally_owned = getattr(self, "_locally_owned_layers", None)
        for hook_point_str, layer_vecs in vectors_data.items():
            table_attr = HOOK_POINT_TABLE_ATTR[SteeringHookPoint(hook_point_str)]
            for idx, vec_values in layer_vecs.items():
                if idx not in valid_indices or idx not in steerable:
                    continue
                mod = steerable[idx]
                if hasattr(mod, table_attr):
                    buf = getattr(mod, table_attr)
                    t = torch.tensor(vec_values, dtype=buf.dtype, device=buf.device)
                    mgr.update_global_vectors(
                        hook_point_str,
                        idx,
                        t,
                        phase=phase,
                        locally_owned_layers=locally_owned,
                    )

    # -----------------------------------------------------------------------
    # Public steering API (mirrored by thin passthroughs on the worker)
    # -----------------------------------------------------------------------

    def set_steering_vectors(
        self,
        vectors: dict[str, dict[int, list[float]]] | None = None,
        prefill_vectors: dict[str, dict[int, list[float]]] | None = None,
        decode_vectors: dict[str, dict[int, list[float]]] | None = None,
        replace: bool = False,
        validate_only: bool = False,
    ) -> tuple[int, int, list[int]]:
        """Set activation steering vectors from plain Python data.

        Supports three-tier steering:

        - *vectors*: base vectors applied to both prefill and decode.
          Notified to SteeringManager with ``phase="base"``.
        - *prefill_vectors*: phase-specific vectors for prefill only.
          Notified to SteeringManager with ``phase="prefill"``.
        - *decode_vectors*: phase-specific vectors for decode only.
          Notified to SteeringManager with ``phase="decode"``.

        All vectors should already be in pre-scaled flat-list form
        (the API router normalizes co-located scales before calling
        this method).

        When *replace* is ``True``, all existing vectors across all
        tiers are cleared before applying.

        When *validate_only* is ``True``, vectors are validated
        without being applied.

        Returns:
            ``(tp_rank, pp_rank, sorted_valid_layers)``. The rank info
            lets the router detect TP-divergence (a server-side
            invariant violation — TP ranks within the same PP stage
            must own identical layer sets). The sorted layer list is
            the set of layer indices actually updated (or *would* be
            updated when *validate_only*) on this worker. The router
            unions these across workers.
        """
        tp_rank, pp_rank = _get_steering_ranks()
        steerable = self._steerable_layers()
        if not steerable:
            return (tp_rank, pp_rank, [])

        # Collect all tiers with data.
        all_tiers: list[tuple[str, dict[str, dict[int, list[float]]]]] = []
        if vectors:
            all_tiers.append(("base", vectors))
        if prefill_vectors:
            all_tiers.append(("prefill", prefill_vectors))
        if decode_vectors:
            all_tiers.append(("decode", decode_vectors))

        if not all_tiers:
            if replace:
                self.clear_steering_vectors()
            return (tp_rank, pp_rank, [])

        # Validate all tiers.
        valid_indices: set[int] = set()
        for _phase, tier_data in all_tiers:
            valid_indices.update(self._validate_vectors_spec(tier_data, steerable))

        if not valid_indices:
            return (tp_rank, pp_rank, [])

        if validate_only:
            return (tp_rank, pp_rank, sorted(valid_indices))

        # Clear if replacing.
        if replace:
            self.clear_steering_vectors()

        # Notify manager with base vectors.
        if vectors:
            self._notify_manager_vectors(vectors, steerable, valid_indices, "base")

        # Phase-specific vectors go only to the manager, not the shared
        # buffers — writing them would overwrite base values and cause
        # get_steering_status() to report the wrong tier.
        if prefill_vectors:
            self._notify_manager_vectors(
                prefill_vectors, steerable, valid_indices, "prefill"
            )

        if decode_vectors:
            self._notify_manager_vectors(
                decode_vectors, steerable, valid_indices, "decode"
            )

        return (tp_rank, pp_rank, sorted(valid_indices))

    def clear_steering_vectors(self) -> None:
        """Clear all tiers (base, prefill, decode) in the SteeringManager."""
        mgr = self._steering_manager
        if mgr is not None:
            mgr.clear_global_vectors()

    def get_steering_status(self) -> dict:
        """Return per-hook-point status for active layers.

        Returns ``{layer_idx: {hook_point: {"norm": float,
        "prefill_norm"?: float, "decode_norm"?: float}}}`` for
        layers/hook-points that have a non-zero steering vector.
        """
        result: dict = {}
        mgr = self._steering_manager
        if mgr is None:
            return result
        for phase_name, phase_dict in [
            ("base", mgr.global_base_vectors),
            ("prefill", mgr.global_prefill_vectors),
            ("decode", mgr.global_decode_vectors),
        ]:
            norm_key = "norm" if phase_name == "base" else f"{phase_name}_norm"
            for hp_str, layer_vecs in phase_dict.items():
                for layer_idx, vec in layer_vecs.items():
                    norm = vec.norm().item()
                    if norm > 0.0:
                        if layer_idx not in result:
                            result[layer_idx] = {}
                        if hp_str not in result[layer_idx]:
                            result[layer_idx][hp_str] = {}
                        result[layer_idx][hp_str][norm_key] = round(norm, 6)
        return result

    # -----------------------------------------------------------------------
    # Worker-side named steering module registry
    # -----------------------------------------------------------------------

    @staticmethod
    def _module_payload_to_specs(
        payload: dict,
    ) -> tuple[
        SteeringVectorSpec | None,
        SteeringVectorSpec | None,
        SteeringVectorSpec | None,
    ]:
        """Normalize a broadcast payload entry into three tier specs.

        Layer keys may arrive as strings (when the payload round-tripped
        through JSON) or ints (when it was constructed in-process).  We
        coerce to int here so subsequent comparisons against the worker's
        layer-owned set are consistent.
        """

        def _coerce(spec):
            if spec is None:
                return None
            coerced: SteeringVectorSpec = {}
            for hook, layer_dict in spec.items():
                converted: dict[int, object] = {}
                for layer_key, entry in layer_dict.items():
                    converted[int(layer_key)] = entry
                if converted:
                    coerced[hook] = converted  # type: ignore[assignment]
            return coerced or None

        return (
            _coerce(payload.get("vectors")),
            _coerce(payload.get("prefill_vectors")),
            _coerce(payload.get("decode_vectors")),
        )

    def register_steering_modules(
        self,
        modules: dict[str, dict],
        replace: bool = False,
    ) -> None:
        """Worker-side handler for the named-module broadcast.

        *modules* maps module name to a dict with optional ``vectors``,
        ``prefill_vectors`` and ``decode_vectors`` (the same shape that
        :class:`SteeringModuleRegistry.dump_for_broadcast` emits).  When
        *replace* is ``True`` the worker's registry is cleared before the
        new entries are stored — used during API-server startup to push
        the initial registry state.

        Mirrors the strict-capacity contract of the rest of the steering
        runtime: requests referencing a name that has not yet been
        broadcast raise loudly in :meth:`_resolve_request_steering`
        rather than silently falling back to inline-only behaviour.
        """
        if replace:
            self._steering_module_registry.clear()
        for name, payload in modules.items():
            if not isinstance(payload, dict):
                raise SteeringVectorError(
                    f"Steering module '{name}' broadcast payload is not a dict"
                )
            self._steering_module_registry[name] = self._module_payload_to_specs(
                payload
            )
        if modules:
            logger.debug(
                "Worker received %d steering module(s) (replace=%s)",
                len(modules),
                replace,
            )

    def unregister_steering_modules(self, names: list[str]) -> None:
        """Drop the listed names from the worker-side registry."""
        for name in names:
            self._steering_module_registry.pop(name, None)
        if names:
            logger.debug(
                "Worker unregistered %d steering module(s)",
                len(names),
            )

    def _resolve_request_steering(
        self,
        sp: SamplingParams,
        phase: str,
    ) -> dict[str, dict[int, list[float]]] | None:
        """Resolve the effective steering for a request in the given *phase*.

        Encapsulates the two cases:

        - **Inline-only** (``sp.steering_module_ref`` is ``None``):
          returns the existing ``effective_prefill_steering`` /
          ``effective_decode_steering`` cached property — bit-for-bit
          identical to today.
        - **Named module (+ optional inline overrides)**: looks up the
          named module in ``self._steering_module_registry``, applies
          the request's module-level scale uniformly via
          :func:`scale_steering_spec`, merges the result with any inline
          tier specs via :func:`merge_steering_specs`, then collapses to
          pre-scaled flat vectors via
          :func:`resolve_effective_vectors`.  The merge order matches the
          original server-side ``resolve_for_request`` so semantics are
          preserved.

        Raises :class:`RuntimeError` when the request references a name
        that is missing from the worker's registry.  This matches the
        strict-capacity contract elsewhere in the steering runtime —
        silent fall-through to inline-only would change the request
        payload after the scheduler has already committed to a hash.
        """
        if phase not in ("prefill", "decode"):
            raise ValueError(f"phase must be 'prefill' or 'decode', got {phase!r}")

        ref = sp.steering_module_ref
        if ref is None:
            return (
                sp.effective_prefill_steering
                if phase == "prefill"
                else sp.effective_decode_steering
            )

        name, scale = ref
        module_specs = self._steering_module_registry.get(name)
        if module_specs is None:
            available = sorted(self._steering_module_registry.keys())
            raise RuntimeError(
                f"Steering module '{name}' is not registered on this worker. "
                f"Available: {available or 'none'}.  This indicates the "
                "module-registry RPC has not been broadcast yet, or the "
                "module was unregistered after the request was scheduled."
            )

        base_spec, prefill_spec, decode_spec = module_specs
        scaled_base = scale_steering_spec(base_spec, scale)
        phase_module_spec = (
            scale_steering_spec(prefill_spec, scale)
            if phase == "prefill"
            else scale_steering_spec(decode_spec, scale)
        )
        inline_phase_spec = (
            sp.prefill_steering_vectors
            if phase == "prefill"
            else sp.decode_steering_vectors
        )

        merged_base = merge_steering_specs(scaled_base, sp.steering_vectors)
        merged_phase = merge_steering_specs(phase_module_spec, inline_phase_spec)
        return resolve_effective_vectors(merged_base, merged_phase)

    # -----------------------------------------------------------------------
    # Per-step buffer / index maintenance
    # -----------------------------------------------------------------------

    def _update_steering_buffers(self, scheduler_output: "SchedulerOutput") -> None:
        """Update per-layer steering tables and the shared steering index.

        Each step:
        1. Drain any deferred steering registrations
        2. Populate each layer's per-hook steering_table from the manager
        3. Build the steering_index mapping tokens to table rows

        The ``SteeringManager`` is constructed eagerly during model
        load by ``_init_steering_state``.  When steering is disabled
        or no steerable layers exist, the manager is ``None`` and this
        function short-circuits — model code (e.g. Gemma3) registers
        per-layer steering_table buffers unconditionally so the forward
        path stays branch-free.
        """
        if self._steering_manager is None or not self._steerable_layers_cache:
            return

        # Process deferred steering entries with a two-queue priority
        # model.  Transitions (prefill→decode) are drained first because
        # they represent in-flight requests that already consumed KV
        # cache.  New-request registrations are only attempted once the
        # transitions queue is empty.  Entries are dropped when the
        # originating request has finished or changed phase, preventing
        # row leaks.
        if self._pending_steering_transitions:
            still_transitions: list[
                tuple[str, int, dict[str, dict[int, list[float]]], str]
            ] = []
            for (
                d_req_id,
                d_hash,
                d_vecs,
                d_phase,
            ) in self._pending_steering_transitions:
                if d_req_id not in self.requests:
                    continue
                if self._req_steering_phase.get(d_req_id) != d_phase:
                    continue
                try:
                    self._steering_manager.register_config(
                        d_hash,
                        d_vecs,
                        phase=d_phase,
                        locally_owned_layers=self._locally_owned_layers,
                    )
                except RuntimeError:
                    still_transitions.append((d_req_id, d_hash, d_vecs, d_phase))
            self._pending_steering_transitions = still_transitions

        if (
            not self._pending_steering_transitions
            and self._pending_steering_registrations
        ):
            still_pending: list[
                tuple[str, int, dict[str, dict[int, list[float]]], str]
            ] = []
            for (
                d_req_id,
                d_hash,
                d_vecs,
                d_phase,
            ) in self._pending_steering_registrations:
                if d_req_id not in self.requests:
                    continue
                if self._req_steering_phase.get(d_req_id) != d_phase:
                    continue
                try:
                    self._steering_manager.register_config(
                        d_hash,
                        d_vecs,
                        phase=d_phase,
                        locally_owned_layers=self._locally_owned_layers,
                    )
                except RuntimeError:
                    still_pending.append((d_req_id, d_hash, d_vecs, d_phase))
            self._pending_steering_registrations = still_pending

        # Short-circuit when no steering state is actually active. The model
        # runner allocates per-layer steering buffers (zero-initialized) and
        # the forward path always calls apply_steering, but if no per-request
        # configs are registered and no global vectors have been set, every
        # gather hits the zero sentinel and adds nothing. There is nothing
        # to populate.
        #
        # Correctness: when we previously had active steering and now don't
        # (e.g., the last steered request just finished), the steering_index
        # may still contain non-zero row references from the previous step.
        # We must zero it before returning to ensure all gathers point to
        # row 0. We only do this on the transition; in the steady "nothing
        # ever active" case the index is already zero from initialization.
        if (
            not self._steering_manager.config_to_row
            and not self._steering_manager.global_base_vectors
            and not self._steering_manager.global_prefill_vectors
            and not self._steering_manager.global_decode_vectors
        ):
            if self._steering_index_dirty:
                any_layer = next(iter(self._steerable_layers_cache.values()))
                steering_index = cast(torch.Tensor, any_layer.steering_index)
                steering_index.zero_()
                self._steering_index_dirty = False
            return

        # 1. Populate steering tables — but only if state has changed since
        # the last populate. populate_steering_tables() clears the flag at
        # the end, and every state mutator (register_config new-row,
        # release_config refcount->0, update_global_vectors,
        # clear_global_vectors) sets it. In steady-state decode steps
        # where no config churn happens, this skips ~102 kernel launches
        # per step.
        if self._steering_manager._tables_dirty:
            self._steering_manager.populate_steering_tables(
                self._steerable_layers_cache
            )

        # 2. Build steering index
        # Get the shared steering_index buffer (all layers share one tensor)
        any_layer = next(iter(self._steerable_layers_cache.values()))
        steering_index = cast(torch.Tensor, any_layer.steering_index)

        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids

        # Vectorized build: walk requests once to record each request's
        # table row + token count into pre-allocated CPU int64 scratch
        # buffers, then expand the row-per-request array into a
        # row-per-token array via ``np.repeat`` and copy the whole
        # thing to the GPU in a single non-blocking H2D.  Replaces
        # ``num_reqs`` independent ``_set_item`` kernel launches per
        # step with one ``copy_``.
        rows_scratch = self._steering_rows_scratch
        n_tokens_scratch = self._steering_n_tokens_scratch
        index_pinned = self._steering_index_pinned
        assert rows_scratch is not None
        assert n_tokens_scratch is not None
        assert index_pinned is not None

        # Grow per-request scratches if the batch ever exceeds the
        # initial sizing.  This is defensive — ``max_num_seqs`` should
        # bound ``num_reqs`` — but cheap to handle.
        if rows_scratch.shape[0] < num_reqs:
            rows_scratch = np.zeros(num_reqs, dtype=np.int64)
            n_tokens_scratch = np.zeros(num_reqs, dtype=np.int64)
            self._steering_rows_scratch = rows_scratch
            self._steering_n_tokens_scratch = n_tokens_scratch

        active_count = 0
        for i in range(num_reqs):
            req_id = req_ids[i]
            n_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if n_tokens == 0:
                continue

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # Request not in batch yet (shouldn't happen but guard).
                # Row 0 is the no-steering sentinel.
                rows_scratch[active_count] = 0
                n_tokens_scratch[active_count] = n_tokens
                active_count += 1
                continue

            # Determine phase from num_computed vs num_prompt
            num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])
            num_prompt = int(self.input_batch.num_prompt_tokens[req_index])
            is_prefilling = num_computed < num_prompt

            if is_prefilling:
                # Prefill: use prefill steering hash
                prefill_hash = int(
                    self.input_batch.request_prefill_steering_hash[req_index]
                )
                row = self._steering_manager.get_row_for_config(
                    prefill_hash, is_prefill=True
                )
                rows_scratch[active_count] = row
                n_tokens_scratch[active_count] = n_tokens

                # Check if this request will transition to decode after
                # this step's tokens are processed. Must happen in this
                # same pass — the registration / refcount semantics are
                # externally observable.
                num_computed_after = num_computed + n_tokens
                if num_computed_after >= num_prompt:
                    self._handle_steering_transition(req_id, req_index, prefill_hash)
            else:
                # Decode: use decode steering hash
                decode_hash = int(
                    self.input_batch.request_decode_steering_hash[req_index]
                )
                row = self._steering_manager.get_row_for_config(
                    decode_hash, is_prefill=False
                )
                rows_scratch[active_count] = row
                n_tokens_scratch[active_count] = n_tokens

            active_count += 1

        # Single non-blocking H2D copy: expand per-request rows into
        # the per-token row array (written into the pre-allocated
        # pinned-memory scratch), then copy that prefix to the GPU
        # in one shot.
        if active_count > 0:
            expanded = np.repeat(
                rows_scratch[:active_count],
                n_tokens_scratch[:active_count],
            )
            n_expanded = int(expanded.shape[0])
            # Cap to the device buffer size; the scheduler enforces
            # this bound but cap defensively to avoid out-of-range
            # writes if upstream invariants ever drift.
            n_expanded = min(n_expanded, index_pinned.shape[0], steering_index.shape[0])
            # Stage in the pinned scratch so the copy is genuinely
            # asynchronous on CUDA devices.
            index_pinned[:n_expanded].copy_(torch.from_numpy(expanded[:n_expanded]))
            steering_index[:n_expanded].copy_(
                index_pinned[:n_expanded], non_blocking=True
            )
        else:
            n_expanded = 0

        # Zero out remaining positions so old tokens past the active
        # prefix read row 0 (the no-steering sentinel).
        if n_expanded < steering_index.shape[0]:
            steering_index[n_expanded:].zero_()

        # Mark the index as having non-zero row references this step. The
        # no-active-state short-circuit on a future step will zero the index
        # if needed when transitioning back to "nothing active".
        self._steering_index_dirty = True

    def _handle_steering_transition(
        self,
        req_id: str,
        req_index: int,
        prefill_hash: int,
    ) -> None:
        """Handle prefill->decode steering config transition.

        Called when a request will complete prefill after this step.
        Releases the prefill config and registers the decode config
        so it is ready for the next step's table population.

        If the steering table is at capacity, the decode registration
        is deferred to ``_pending_steering_registrations`` and retried
        on the next scheduler step.  The existing ``get_row_for_config``
        fallback (returns row 2 for unregistered decode hashes) provides
        graceful degradation during the deferral period.
        """
        mgr = self._steering_manager
        assert mgr is not None, (
            "_handle_steering_transition called without an initialised manager"
        )
        if prefill_hash != 0:
            mgr.release_config(prefill_hash, "prefill")

        decode_hash = int(self.input_batch.request_decode_steering_hash[req_index])
        if decode_hash != 0:
            req_state = self.requests.get(req_id)
            if req_state is not None and req_state.sampling_params is not None:
                sp = req_state.sampling_params
                effective_decode = self._resolve_request_steering(sp, "decode")
                if effective_decode:
                    try:
                        mgr.register_config(
                            decode_hash,
                            effective_decode,
                            phase="decode",
                            locally_owned_layers=self._locally_owned_layers,
                        )
                    except RuntimeError:
                        self._pending_steering_transitions.append(
                            (
                                req_id,
                                decode_hash,
                                effective_decode,
                                "decode",
                            )
                        )
                        logger.warning(
                            "Deferred decode steering config (hash=%d) "
                            "-- capacity full, will retry next step",
                            decode_hash,
                        )

        # Update phase tracking regardless of whether decode
        # registration succeeded or was deferred.
        self._req_steering_phase[req_id] = "decode"

    def _reset_steering_for_resumption(
        self,
        req_id: str,
        req_state: "CachedRequestState",
        new_num_computed_tokens: int,
    ) -> None:
        """Reset steering config registration when a request re-enters prefill.

        Called when a preempted request is resumed with num_computed_tokens
        reset. If the request had transitioned to decode before preemption,
        its decode config is still registered and its phase is stale.
        This helper releases the stale decode config and re-registers the
        prefill config (or defers it on capacity exhaustion).
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None:
            return
        prev_phase = self._req_steering_phase.get(req_id)
        if prev_phase != "decode":
            return
        if new_num_computed_tokens >= req_state.num_prompt_tokens:
            return  # still in decode, nothing to reset

        # Release the stale decode config.
        if req_state.decode_steering_config_hash != 0:
            mgr.release_config(req_state.decode_steering_config_hash, "decode")

        # Drop any stale deferred entries for this request.
        if self._pending_steering_transitions:
            self._pending_steering_transitions = [
                e for e in self._pending_steering_transitions if e[0] != req_id
            ]
        if self._pending_steering_registrations:
            self._pending_steering_registrations = [
                e for e in self._pending_steering_registrations if e[0] != req_id
            ]

        self._req_steering_phase[req_id] = "prefill"

        sp = req_state.sampling_params
        prefill_hash = req_state.prefill_steering_config_hash
        if prefill_hash == 0 or sp is None:
            return
        effective_prefill = self._resolve_request_steering(sp, "prefill")
        if not effective_prefill:
            return
        try:
            mgr.register_config(
                prefill_hash,
                effective_prefill,
                phase="prefill",
                locally_owned_layers=self._locally_owned_layers,
            )
        except RuntimeError:
            self._pending_steering_registrations.append(
                (req_id, prefill_hash, effective_prefill, "prefill")
            )
            logger.warning(
                "Deferred prefill steering config (hash=%d) on resumption "
                "-- capacity full, will retry next step",
                prefill_hash,
            )

    # -----------------------------------------------------------------------
    # Hooks called from _update_states() / _update_streaming_request()
    # -----------------------------------------------------------------------

    def _release_finished_steering_configs(
        self, finished_req_ids: "set[str] | list[str]"
    ) -> None:
        """Release the currently-active steering config for finished requests.

        Also drops deferred entries for those requests before
        ``self.requests`` is pruned, preventing row leaks.  Called
        before finished request state is popped so
        ``prefill_steering_config_hash`` /
        ``decode_steering_config_hash`` are still accessible.
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None:
            return

        for req_id in finished_req_ids:
            phase = self._req_steering_phase.pop(req_id, None)
            if phase is not None:
                req_state = self.requests.get(req_id)
                if req_state is not None:
                    if phase == "prefill":
                        h = req_state.prefill_steering_config_hash
                    else:
                        h = req_state.decode_steering_config_hash
                    if h != 0:
                        mgr.release_config(h, phase)

        # Also remove any deferred steering entries for finished
        # requests to prevent registering rows for dead requests.
        # (The retry loop also checks, but this eagerly drops
        # entries before self.requests is pruned below.)
        finished = set(finished_req_ids)
        if self._pending_steering_transitions:
            self._pending_steering_transitions = [
                entry
                for entry in self._pending_steering_transitions
                if entry[0] not in finished
            ]
        if self._pending_steering_registrations:
            self._pending_steering_registrations = [
                entry
                for entry in self._pending_steering_registrations
                if entry[0] not in finished
            ]

    def _register_initial_steering_config(
        self,
        req_id: str,
        new_req_data: "NewRequestData",
        req_state: "CachedRequestState",
    ) -> None:
        """Register the initial-phase steering config for a new request.

        Normally requests start in prefill, but a full prefix-cache hit
        (``num_computed >= num_prompt``) puts a request directly into
        decode.  Handles capacity-exhaustion deferral.
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None or new_req_data.sampling_params is None:
            return

        sp = new_req_data.sampling_params
        if new_req_data.num_computed_tokens >= req_state.num_prompt_tokens:
            # Already past prefill — register decode config.
            effective_decode = self._resolve_request_steering(sp, "decode")
            if new_req_data.decode_steering_config_hash != 0 and effective_decode:
                try:
                    mgr.register_config(
                        new_req_data.decode_steering_config_hash,
                        effective_decode,
                        phase="decode",
                        locally_owned_layers=self._locally_owned_layers,
                    )
                except RuntimeError:
                    self._pending_steering_registrations.append(
                        (
                            req_id,
                            new_req_data.decode_steering_config_hash,
                            effective_decode,
                            "decode",
                        )
                    )
                    logger.warning(
                        "Deferred decode steering config "
                        "(hash=%d) -- capacity full, "
                        "will retry next step",
                        new_req_data.decode_steering_config_hash,
                    )
            self._req_steering_phase[req_id] = "decode"
        else:
            # Normal: start in prefill; decode registered
            # on transition in _update_steering_buffers.
            effective_prefill = self._resolve_request_steering(sp, "prefill")
            if new_req_data.prefill_steering_config_hash != 0 and effective_prefill:
                try:
                    mgr.register_config(
                        new_req_data.prefill_steering_config_hash,
                        effective_prefill,
                        phase="prefill",
                        locally_owned_layers=self._locally_owned_layers,
                    )
                except RuntimeError:
                    self._pending_steering_registrations.append(
                        (
                            req_id,
                            new_req_data.prefill_steering_config_hash,
                            effective_prefill,
                            "prefill",
                        )
                    )
                    logger.warning(
                        "Deferred prefill steering config "
                        "(hash=%d) -- capacity full, "
                        "will retry next step",
                        new_req_data.prefill_steering_config_hash,
                    )
            self._req_steering_phase[req_id] = "prefill"

    def _refresh_streaming_steering(
        self,
        req_id: str,
        new_req_data: "NewRequestData",
        old_prefill_hash: int,
        old_decode_hash: int,
        new_prefill_hash: int,
        new_decode_hash: int,
    ) -> None:
        """Refresh steering state for a streaming re-added request.

        Streaming re-adds go back through prefill, so we must:
        1. Release the old config (whatever phase we were tracking)
        2. Purge stale deferred entries for this request
        3. Register the new prefill config
        4. Update phase tracking
        """
        mgr = getattr(self, "_steering_manager", None)
        if mgr is None:
            return

        # Release the old phase config.
        old_phase = self._req_steering_phase.get(req_id)
        if old_phase is not None:
            if old_phase == "prefill" and old_prefill_hash != 0:
                mgr.release_config(old_prefill_hash, "prefill")
            elif old_phase == "decode" and old_decode_hash != 0:
                mgr.release_config(old_decode_hash, "decode")

        # Purge stale deferred entries for this request.
        if self._pending_steering_transitions:
            self._pending_steering_transitions = [
                entry
                for entry in self._pending_steering_transitions
                if entry[0] != req_id
            ]
        if self._pending_steering_registrations:
            self._pending_steering_registrations = [
                entry
                for entry in self._pending_steering_registrations
                if entry[0] != req_id
            ]

        # Register new prefill config (streaming re-adds start
        # in prefill).
        sp = new_req_data.sampling_params
        effective_prefill = (
            self._resolve_request_steering(sp, "prefill") if sp is not None else None
        )
        if new_prefill_hash != 0 and sp is not None and effective_prefill:
            try:
                mgr.register_config(
                    new_prefill_hash,
                    effective_prefill,
                    phase="prefill",
                    locally_owned_layers=self._locally_owned_layers,
                )
            except RuntimeError:
                self._pending_steering_registrations.append(
                    (
                        req_id,
                        new_prefill_hash,
                        effective_prefill,
                        "prefill",
                    )
                )
                logger.warning(
                    "Deferred prefill steering config "
                    "(hash=%d) for streaming re-add -- "
                    "capacity full, will retry next step",
                    new_prefill_hash,
                )
            self._req_steering_phase[req_id] = "prefill"
        elif new_prefill_hash == 0 and new_decode_hash == 0:
            # No steering for this request anymore.
            self._req_steering_phase.pop(req_id, None)
        else:
            # Has hashes but no effective prefill vectors (e.g.,
            # decode-only steering).  Mark as prefill since the
            # request re-enters prefill; transition to decode
            # will handle decode registration.
            self._req_steering_phase[req_id] = "prefill"
