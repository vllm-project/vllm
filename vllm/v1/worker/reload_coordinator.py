# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 model-runner owner for sealed weight-reload transactions."""

import time
from collections.abc import Iterable
from typing import Any, Protocol, cast

import torch

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.reload import (
    ReloadPlan,
    ReloadPlanBuilder,
    ReloadStatePolicy,
    ReloadTransaction,
    ReloadTransactionState,
    finalize_layerwise_reload,
    initialize_layerwise_reload,
)

logger = init_logger(__name__)


class ReloadCoordinatorOwner(Protocol):
    model_config: Any
    compilation_config: Any
    load_config: Any

    def get_model(self) -> torch.nn.Module: ...

    def reset_encoder_cache(self) -> None: ...

    def reset_mm_cache(self) -> None: ...


class ReloadCoordinator:
    """Own all refit plans and the single active reload transaction."""

    def __init__(self, owner: ReloadCoordinatorOwner) -> None:
        self.owner = owner
        self._plans: dict[int, ReloadPlan] = {}
        self._active: ReloadTransaction | None = None

    def plan_for(
        self,
        model: torch.nn.Module | None = None,
        model_config: Any | None = None,
    ) -> ReloadPlan:
        model = model or self.owner.get_model()
        key = id(model)
        if plan := self._plans.get(key):
            return plan

        builder = ReloadPlanBuilder.from_model(model, prefix="")
        config = model_config or self.owner.model_config
        if config.quantization is not None:
            builder.unsupported(
                f"quantization backend {config.quantization!r} has not declared "
                "a complete reload plan"
            )
        builder.state(
            "encoder_cache",
            policy=ReloadStatePolicy.INVALIDATE,
            action=self.owner.reset_encoder_cache,
        )
        builder.state(
            "multimodal_cache",
            policy=ReloadStatePolicy.INVALIDATE,
            action=self.owner.reset_mm_cache,
        )
        plan = builder.seal()
        self._plans[key] = plan
        return plan

    def begin(
        self,
        model: torch.nn.Module | None = None,
        model_config: Any | None = None,
        expected_inputs: Iterable[str] | None = None,
    ) -> ReloadTransaction:
        if self._active is not None:
            raise RuntimeError("A model reload transaction is already active")
        plan = self.plan_for(model, model_config)
        config = self.owner.compilation_config
        require_graph_safe = (
            config.mode == CompilationMode.STOCK_TORCH_COMPILE
            or config.cudagraph_mode != CUDAGraphMode.NONE
        )
        self._active = plan.begin(
            expected_inputs=expected_inputs,
            require_graph_safe=require_graph_safe,
        )
        return self._active

    def record_inputs(self, update_info: dict[str, Any]) -> None:
        transaction = self._require_active()
        for name in update_info.get("names", ()):
            transaction.record_input(
                name,
                allow_unknown=True,
                allow_duplicate=True,
            )

    def start_finalizing(self) -> None:
        self._require_active().start_finalizing()

    def prepare(self) -> None:
        self._require_active().prepare_or_raise()

    def commit(self) -> None:
        transaction = self._require_active()
        try:
            transaction.commit_prepared()
        finally:
            if transaction.state in (
                ReloadTransactionState.COMMITTED,
                ReloadTransactionState.FAILED,
            ):
                self._active = None

    def abort(self) -> None:
        if self._active is not None:
            self._active.abort()
            self._active = None

    def _clear_active(self) -> None:
        """Clear a transaction already committed through its direct API."""
        self._active = None

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """Reload the v2 model through the same sealed transaction as WTE."""
        if weights_iterator is None and not is_checkpoint_format:
            logger.warning(
                "Reloading from disk means that weights are in checkpoint format. "
                "Set `is_checkpoint_format=True` to avoid loading errors."
            )

        model = self.owner.get_model()
        weights_to_load = {name for name, _ in model.named_parameters()}
        started_at = time.perf_counter()

        if weights_iterator is None:
            model_loader = get_model_loader(self.owner.load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"Model reloading with `{self.owner.load_config.load_format}` "
                    "format is not supported"
                )
            if weights_path is not None:
                self.owner.model_config.model = weights_path
            weights_iterator = cast(
                Iterable[tuple[str, torch.Tensor]],
                model_loader.get_all_weights(self.owner.model_config, model),
            )

        expected_inputs = (
            weights_to_load if self.owner.model_config.quantization is None else None
        )
        transaction = self.begin(
            model,
            self.owner.model_config,
            expected_inputs=expected_inputs,
        )
        try:
            logger.info_once("Reloading weights inplace...")
            if is_checkpoint_format:
                initialize_layerwise_reload(model)
                loaded_weights = model.load_weights(weights_iterator)
                if expected_inputs is not None:
                    for name in loaded_weights:
                        transaction.record_input(name)
                transaction.start_finalizing()
                finalize_layerwise_reload(model, self.owner.model_config)
            else:
                loaded_weights = set()
                for name, loaded_weight in weights_iterator:
                    transaction.write(name, loaded_weight)
                    loaded_weights.add(name)

            transaction.commit_or_raise()
            self._clear_active()
        except BaseException:
            transaction.abort()
            self._clear_active()
            raise

        logger.info_once(
            "Reloading and processing weights took %.2f seconds",
            time.perf_counter() - started_at,
        )
        if self.owner.model_config.quantization is None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                logger.warning(
                    "Following weights were not loaded from checkpoint: %s",
                    weights_not_loaded,
                )

    def _require_active(self) -> ReloadTransaction:
        if self._active is None:
            raise RuntimeError("No model reload transaction is active")
        return self._active


__all__ = ["ReloadCoordinator", "ReloadCoordinatorOwner"]
