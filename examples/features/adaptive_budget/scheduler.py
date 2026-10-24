# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An opt-in example of bounded token budgets using the scheduler extension API."""

import time

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    high_budget: int = Field(default=8192, ge=1)
    low_budget: int = Field(default=1024, ge=1)
    enter_prefills: int = Field(default=2, ge=1)
    exit_prefills: int = Field(default=0, ge=0)
    min_dwell_steps: int = Field(default=4, ge=0)
    guard_enabled: bool = True
    max_stall_ms: float = Field(default=250.0, gt=0, allow_inf_nan=False)
    guard_budget: int = Field(default=4096, ge=1)
    guard_interval_steps: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "BudgetConfig":
        if not self.low_budget <= self.guard_budget <= self.high_budget:
            raise ValueError("Require low_budget <= guard_budget <= high_budget")
        if self.exit_prefills >= self.enter_prefills:
            raise ValueError("Require exit_prefills < enter_prefills")
        return self


class BudgetController:
    """Select a total iteration budget without changing request order."""

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.limited = False
        self.step = 0
        self.last_switch = -config.min_dwell_steps
        self.last_guard = -config.guard_interval_steps

    def decide(self, prefills: int, decodes: int, stall_ms: float) -> int:
        cfg = self.config
        self.step += 1
        previous = self.limited
        if not prefills or not decodes:
            self.limited = False
        elif self.step - self.last_switch >= cfg.min_dwell_steps:
            if not self.limited and prefills >= cfg.enter_prefills:
                self.limited = True
            elif self.limited and prefills <= cfg.exit_prefills:
                self.limited = False
        if previous != self.limited:
            self.last_switch = self.step
        if not self.limited:
            return cfg.high_budget
        if (
            cfg.guard_enabled
            and stall_ms >= cfg.max_stall_ms
            and self.step - self.last_guard >= cfg.guard_interval_steps
        ):
            self.last_guard = self.step
            return cfg.guard_budget
        return cfg.low_budget


class AdaptiveBudgetScheduler(Scheduler):
    """Synchronous text-only example; the standard scheduler remains the default."""

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        sc = vllm_config.scheduler_config
        if sc.async_scheduling or not sc.enable_chunked_prefill:
            raise ValueError("Use synchronous scheduling with chunked prefill")
        if (
            sc.is_multimodal_model
            or sc.runner_type != "generate"
            or vllm_config.model_config.is_encoder_decoder
            or vllm_config.speculative_config is not None
            or vllm_config.lora_config is not None
            or vllm_config.kv_transfer_config is not None
            or vllm_config.ec_transfer_config is not None
            or vllm_config.parallel_config.world_size != 1
            or vllm_config.parallel_config.data_parallel_size != 1
        ):
            raise ValueError("This example requires single-GPU text generation")
        additional = vllm_config.additional_config
        if not isinstance(additional, dict):
            raise ValueError("additional_config must be a dictionary")
        self.budget_config = BudgetConfig.model_validate(
            additional.get("adaptive_budget", {})
        )
        capacity = sc.max_num_scheduled_tokens or sc.max_num_batched_tokens
        if self.budget_config.high_budget > capacity:
            raise ValueError("high_budget exceeds the scheduler's configured capacity")
        super().__init__(vllm_config, *args, **kwargs)
        self.budget_controller = BudgetController(self.budget_config)
        self.prefill_progress: dict[str, tuple[Request, int, float]] = {}

    def _observe(self, now: float) -> tuple[int, int, float]:
        prefills = decodes = 0
        oldest_stall_ms = 0.0
        live: set[str] = set()
        for req_id, req in self.requests.items():
            if req.status not in (
                RequestStatus.WAITING,
                RequestStatus.RUNNING,
                RequestStatus.PREEMPTED,
            ):
                continue
            # Replaying generated tokens after preemption is also prefill work.
            prefill_end = max(req.num_prompt_tokens, req.num_tokens - 1)
            if req.num_computed_tokens < prefill_end:
                prefills += 1
                live.add(req_id)
                old_req, old_count, last_progress = self.prefill_progress.get(
                    req_id, (req, req.num_computed_tokens, now)
                )
                if old_req is not req or req.num_computed_tokens > old_count:
                    last_progress = now
                self.prefill_progress[req_id] = (
                    req,
                    req.num_computed_tokens,
                    last_progress,
                )
                oldest_stall_ms = max(oldest_stall_ms, (now - last_progress) * 1000)
            elif req.status == RequestStatus.RUNNING:
                decodes += 1
        for req_id in self.prefill_progress.keys() - live:
            del self.prefill_progress[req_id]
        return prefills, decodes, oldest_stall_ms

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        budget = self.budget_controller.decide(*self._observe(time.monotonic()))
        capacity = self.max_num_scheduled_tokens
        self.max_num_scheduled_tokens = budget
        try:
            return super().schedule(*args, **kwargs)
        finally:
            self.max_num_scheduled_tokens = capacity

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            self.prefill_progress.clear()
