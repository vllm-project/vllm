# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Protocol

import torch
import torch.nn as nn

from vllm.distributed.eplb.eplb_state import EplbState
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import is_mixture_of_experts

logger = init_logger(__name__)


class _EPLBGPUModelRunnerProto(Protocol):
    parallel_config: Any
    device: torch.device
    load_config: Any
    speculator: Any | None
    speculative_config: Any | None
    model_config: Any
    eplb_state: EplbState | None
    eep_eplb_suppressed: bool

    def get_model(self) -> nn.Module: ...
    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None: ...


class EPLBGPUModelRunnerMixin:
    def _init_eplb_support(self: _EPLBGPUModelRunnerProto) -> None:
        self.eplb_state: EplbState | None = None
        self.eep_eplb_suppressed = False

    def _prepare_eplb_load(
        self: _EPLBGPUModelRunnerProto, load_dummy_weights: bool
    ) -> None:
        self.eplb_state = None
        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
        if load_dummy_weights:
            self.load_config.load_format = "dummy"

    def _maybe_register_speculator_for_eplb(
        self: _EPLBGPUModelRunnerProto, load_dummy_weights: bool
    ) -> bool:
        # if speculator is a moe model, add it to eplb
        if (
            self.speculator is None
            or not hasattr(self.speculator, "model")
            or not self.parallel_config.enable_eplb
            or load_dummy_weights
        ):
            return False

        draft_model = self.speculator.model
        if not is_mixture_of_experts(draft_model):
            return False

        assert not self.parallel_config.enable_elastic_ep, (
            "Elastic EP is not supported with draft model."
        )
        assert self.speculative_config is not None
        assert self.speculative_config.draft_model_config is not None
        assert self.eplb_state is not None
        self.eplb_state.add_model(
            draft_model,
            self.speculative_config.draft_model_config,
        )
        return True

    def _maybe_register_model_for_eplb(
        self: _EPLBGPUModelRunnerProto, load_dummy_weights: bool
    ) -> bool:
        if not self.parallel_config.enable_eplb or load_dummy_weights:
            return False

        model = self.get_model()
        if not is_mixture_of_experts(model):
            return False

        logger.info_once(
            "EPLB is enabled for model %s.", self.model_config.model, scope="local"
        )
        assert self.eplb_state is not None
        self.eplb_state.add_model(model, self.model_config)
        return True

    def _maybe_start_eplb_async_loop(
        self: _EPLBGPUModelRunnerProto, eplb_models_added: bool
    ) -> None:
        if (
            eplb_models_added
            and self.eplb_state is not None
            and self.eplb_state.is_async
        ):
            self.eplb_state.start_async_loop()

    def _maybe_step_eplb_for_dummy_run(
        self: _EPLBGPUModelRunnerProto, *, skip_eplb: bool, is_profile: bool
    ) -> None:
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

    def eplb_step(
        self: _EPLBGPUModelRunnerProto,
        is_dummy: bool = False,
        is_profile: bool = False,
    ) -> None:
        if not self.parallel_config.enable_eplb or self.eep_eplb_suppressed:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def setup_eplb_from_mapping(
        self: _EPLBGPUModelRunnerProto,
        expanded_physical_to_logical: torch.Tensor,
        old_num_physical_experts: int,
    ) -> None:
        model = self.get_model()
        assert is_mixture_of_experts(model)

        self.eplb_state = EplbState.from_mapping(
            model=model,
            model_config=self.model_config,
            device=self.device,
            parallel_config=self.parallel_config,
            expanded_physical_to_logical=expanded_physical_to_logical,
            num_valid_physical_experts=old_num_physical_experts,
        )
