# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.exceptions import SteeringVectorError
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.steering import (
    HOOK_POINT_VECTOR_ATTR,
    SteeringHookPoint,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Initialize common worker components.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        from vllm.platforms import current_platform

        self.current_platform = current_platform

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> float:
        """Prepare model for execution through compilation/warmup.

        Returns:
            The accumulated compilation time in seconds.
        """
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())

    def _steerable_layers(self) -> dict:
        """Return ``{layer_idx: module}`` for layers with steering buffers.

        Works with any model runner that exposes ``get_model()``,
        including the V2 runner.  Result is cached after first
        successful discovery.

        A layer is considered steerable if it has ``layer_idx`` and at
        least one ``steering_vector_*`` buffer for any hook point.
        """
        cache = getattr(self, "_steerable_layers_cache", None)
        if cache is not None:
            return cache

        mr = self.model_runner
        if mr is None or not hasattr(mr, "get_model"):
            return {}
        layers: dict = {}
        for mod in mr.get_model().modules():
            if not hasattr(mod, "layer_idx"):
                continue
            has_any_vector = any(
                hasattr(mod, attr) for attr in HOOK_POINT_VECTOR_ATTR.values()
            )
            if has_any_vector:
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
            vec_attr = HOOK_POINT_VECTOR_ATTR[hp_enum]

            for idx, vec_values in layer_vecs.items():
                if idx not in steerable:
                    continue
                mod = steerable[idx]
                if not hasattr(mod, vec_attr):
                    raise SteeringVectorError(
                        f"Hook point {hook_point_str!r} not active on layer {idx}"
                    )
                buf = getattr(mod, vec_attr)
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

    def _apply_vectors_to_buffers(
        self,
        vectors_data: dict[str, dict[int, list[float]]],
        steerable: dict,
        valid_indices: set[int],
    ) -> None:
        """Copy validated vectors into layer steering buffers."""
        for hook_point_str, layer_vecs in vectors_data.items():
            vec_attr = HOOK_POINT_VECTOR_ATTR[SteeringHookPoint(hook_point_str)]
            for idx, vec_values in layer_vecs.items():
                if idx not in steerable or idx not in valid_indices:
                    continue
                mod = steerable[idx]
                if not hasattr(mod, vec_attr):
                    continue
                buf = getattr(mod, vec_attr)
                t = torch.tensor([vec_values], dtype=buf.dtype, device=buf.device)
                buf.copy_(t)

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

        When the manager has not been lazily initialized yet, the
        converted tensors are stored in
        ``self.model_runner._pending_steering_globals`` for replay
        during lazy init in ``_update_steering_buffers``.
        """
        if not hasattr(self, "model_runner") or self.model_runner is None:
            return
        mgr = getattr(self.model_runner, "_steering_manager", None)
        if mgr is None:
            # Manager not yet initialized -- capture current buffer
            # values for replay during lazy init.
            captured: dict[str, dict[int, torch.Tensor]] = {}
            for hook_point_str, layer_vecs in vectors_data.items():
                vec_attr = HOOK_POINT_VECTOR_ATTR[SteeringHookPoint(hook_point_str)]
                captured_layers: dict[int, torch.Tensor] = {}
                for idx, vec_values in layer_vecs.items():
                    if idx not in valid_indices or idx not in steerable:
                        continue
                    mod = steerable[idx]
                    if hasattr(mod, vec_attr):
                        buf = getattr(mod, vec_attr)
                        captured_layers[idx] = torch.tensor(
                            vec_values, dtype=buf.dtype, device=buf.device
                        )
                if captured_layers:
                    captured[hook_point_str] = captured_layers
            if captured:
                pending = getattr(self.model_runner, "_pending_steering_globals", None)
                if pending is None:
                    self.model_runner._pending_steering_globals = []
                    pending = self.model_runner._pending_steering_globals
                pending.append((captured, phase))
            return
        for hook_point_str, layer_vecs in vectors_data.items():
            vec_attr = HOOK_POINT_VECTOR_ATTR[SteeringHookPoint(hook_point_str)]
            for idx, vec_values in layer_vecs.items():
                if idx not in valid_indices or idx not in steerable:
                    continue
                mod = steerable[idx]
                if hasattr(mod, vec_attr):
                    buf = getattr(mod, vec_attr)
                    t = torch.tensor(vec_values, dtype=buf.dtype, device=buf.device)
                    mgr.update_global_vectors(hook_point_str, idx, t, phase=phase)

    def set_steering_vectors(
        self,
        vectors: dict[str, dict[int, list[float]]] | None = None,
        prefill_vectors: dict[str, dict[int, list[float]]] | None = None,
        decode_vectors: dict[str, dict[int, list[float]]] | None = None,
        replace: bool = False,
        validate_only: bool = False,
    ) -> list[int]:
        """Set activation steering vectors from plain Python data.

        Supports three-tier steering:

        - *vectors*: base vectors applied to both prefill and decode.
          These are stored in layer buffers (``steering_vector_*``).
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
            Sorted list of layer indices that were actually updated (or
            *would* be updated when *validate_only*) on this worker.
            The router unions these across workers.
        """
        steerable = self._steerable_layers()
        if not steerable:
            return []

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
            return []

        # Validate all tiers.
        valid_indices: set[int] = set()
        for _phase, tier_data in all_tiers:
            valid_indices.update(self._validate_vectors_spec(tier_data, steerable))

        if not valid_indices:
            return []

        if validate_only:
            return sorted(valid_indices)

        # Clear if replacing.
        if replace:
            self.clear_steering_vectors()

        # Apply base vectors to layer buffers and notify manager.
        if vectors:
            self._apply_vectors_to_buffers(vectors, steerable, valid_indices)
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

        return sorted(valid_indices)

    def clear_steering_vectors(self) -> None:
        """Zero all steering-vector buffers across all hook points and
        clear all tiers (base, prefill, decode) in the SteeringManager."""
        for mod in self._steerable_layers().values():
            for vec_attr in HOOK_POINT_VECTOR_ATTR.values():
                if hasattr(mod, vec_attr):
                    getattr(mod, vec_attr).zero_()

        # Notify SteeringManager
        if hasattr(self, "model_runner") and self.model_runner is not None:
            mgr = getattr(self.model_runner, "_steering_manager", None)
            if mgr is not None:
                mgr.clear_global_vectors()
            # Also clear any pending globals queued before manager init,
            # so they are not replayed on lazy initialization.
            if hasattr(self.model_runner, "_pending_steering_globals"):
                self.model_runner._pending_steering_globals = None

    def get_steering_status(self) -> dict:
        """Return per-hook-point status for active layers.

        Returns ``{layer_idx: {hook_point: {"norm": float,
        "prefill_norm"?: float, "decode_norm"?: float}}}`` for
        layers/hook-points that have a non-zero steering vector.

        Base norms come from layer buffers. Phase-specific norms
        (``prefill_norm``, ``decode_norm``) come from the
        SteeringManager's global phase vectors and are only present
        when those vectors have non-zero norms.
        """
        result: dict = {}
        # Base norms from layer buffers
        for idx, mod in self._steerable_layers().items():
            layer_info: dict[str, dict[str, float]] = {}
            for hp, vec_attr in HOOK_POINT_VECTOR_ATTR.items():
                if not hasattr(mod, vec_attr):
                    continue
                vec = getattr(mod, vec_attr)
                norm = vec.norm().item()
                if norm > 0.0:
                    layer_info[hp.value] = {"norm": round(norm, 6)}
            if layer_info:
                result[idx] = layer_info

        # Phase-specific norms from SteeringManager
        if hasattr(self, "model_runner") and self.model_runner is not None:
            mgr = getattr(self.model_runner, "_steering_manager", None)
            if mgr is not None:
                for phase_name, phase_dict in [
                    ("prefill", mgr.global_prefill_vectors),
                    ("decode", mgr.global_decode_vectors),
                ]:
                    for hp_str, layer_vecs in phase_dict.items():
                        for layer_idx, vec in layer_vecs.items():
                            norm = vec.norm().item()
                            if norm > 0.0:
                                if layer_idx not in result:
                                    result[layer_idx] = {}
                                if hp_str not in result[layer_idx]:
                                    result[layer_idx][hp_str] = {}
                                result[layer_idx][hp_str][f"{phase_name}_norm"] = round(
                                    norm, 6
                                )
            else:
                # Phase-specific norms from pending globals (before manager init)
                pending = getattr(self.model_runner, "_pending_steering_globals", None)
                if pending:
                    for captured_vectors, phase in pending:
                        if phase == "base":
                            continue  # base vectors are in layer buffers
                        phase_name = phase  # "prefill" or "decode"
                        for hp_str, layer_vecs in captured_vectors.items():
                            for layer_idx, vec in layer_vecs.items():
                                norm = vec.norm().item()
                                if norm > 0.0:
                                    if layer_idx not in result:
                                        result[layer_idx] = {}
                                    if hp_str not in result[layer_idx]:
                                        result[layer_idx][hp_str] = {}
                                    result[layer_idx][hp_str][f"{phase_name}_norm"] = (
                                        round(norm, 6)
                                    )
        return result

    def get_model_inspection(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        from vllm.model_inspection import format_model_inspection

        return format_model_inspection(self.get_model())

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None."""
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank = rpc_rank
        self.global_rank = self.rpc_rank if global_rank is None else global_rank

        # Initialized after init_worker is called
        self.worker: WorkerBase
        self.vllm_config: VllmConfig

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        envs = envs_list[self.rpc_rank]
        update_environment_variables(envs)

    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config = vllm_config

        vllm_config.enable_trace_function_call_for_thread()

        from vllm.plugins import load_general_plugins

        load_general_plugins()

        parallel_config = vllm_config.parallel_config
        if isinstance(parallel_config.worker_cls, str):
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(
                parallel_config.worker_cls
            )
        else:
            raise ValueError(
                "passing worker_cls is no longer supported. "
                "Please pass keep the class in a separate module "
                "and pass the qualified name of the class as a string."
            )

        if parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                parallel_config.worker_extension_cls
            )
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}."
                    )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            mm_config = vllm_config.model_config.multimodal_config
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

            self.mm_receiver_cache = None
        else:
            self.mm_receiver_cache = (
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config,
                    shared_worker_lock,
                )
            )

        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        self._apply_mm_cache(scheduler_output)

        return self.worker.execute_model(scheduler_output)

    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        self.worker.reset_mm_cache()
