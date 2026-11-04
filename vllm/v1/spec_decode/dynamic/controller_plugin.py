# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attach the Dynamic SD controller to vLLM's scheduler.

Registered as a ``vllm.general_plugins`` entry point, so vLLM calls register()
in every process before it builds the scheduler. Nothing changes unless
VLLM_DSD_CONTROLLER is set; then two Scheduler methods are wrapped (no edits
to the scheduler's source):

- ``__init__`` builds the controller and swaps in ControllerLookup, so
  ``lookup[batch_size]`` asks the controller for K;
- ``make_spec_decoding_stats`` passes each request's drafted/accepted counts.

A missing hook point fails at start-up with the names of the missing methods.
"""

from __future__ import annotations

import functools
import logging
import os

from .controller import DynamicSDController, controller_from_env

logger = logging.getLogger(__name__)

ENV_SWITCH = "VLLM_DSD_CONTROLLER"
HOOK_METHODS = ("__init__", "make_spec_decoding_stats")


class ControllerLookup(list):
    """The Dynamic SD lookup vLLM indexes by batch size, answered by the controller.

    Keeps the original table as list contents, so anything else reading the
    lookup sees vLLM's values; only ``lookup[batch_size]`` goes to the
    controller.
    """

    def __init__(self, lookup: list[int], controller: DynamicSDController):
        super().__init__(lookup)
        self.controller = controller

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.controller.choose_k(index)
        return super().__getitem__(index)


def install(scheduler_cls: type) -> None:
    """Add the controller hooks to a scheduler class (idempotent)."""
    if getattr(scheduler_cls, "_dsd_controller_installed", False):
        return
    missing = [m for m in HOOK_METHODS if not hasattr(scheduler_cls, m)]
    if missing:
        raise RuntimeError(
            f"{ENV_SWITCH} is set but this vLLM's Scheduler has no {missing}"
        )

    orig_init = scheduler_cls.__init__  # type: ignore[misc]
    orig_stats = scheduler_cls.make_spec_decoding_stats  # type: ignore[attr-defined]

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.dynamic_sd_controller = None
        if not hasattr(self, "dynamic_sd_lookup"):
            raise RuntimeError(
                f"{ENV_SWITCH} is set but this vLLM's Scheduler has no "
                "dynamic_sd_lookup"
            )
        lookup = self.dynamic_sd_lookup
        if lookup is None:  # no Dynamic SD table in --speculative-config
            return
        controller = controller_from_env(list(lookup), self.num_spec_tokens)
        if controller is not None:
            logger.info(controller.describe())
            self.dynamic_sd_controller = controller
            self.dynamic_sd_lookup = ControllerLookup(lookup, controller)

    @functools.wraps(orig_stats)
    def make_spec_decoding_stats(
        self,
        spec_decoding_stats,
        num_draft_tokens,
        num_accepted_tokens,
        *args,
        **kwargs,
    ):
        controller = getattr(self, "dynamic_sd_controller", None)
        if controller is not None:
            controller.observe(num_draft_tokens, num_accepted_tokens)
        return orig_stats(
            self,
            spec_decoding_stats,
            num_draft_tokens,
            num_accepted_tokens,
            *args,
            **kwargs,
        )

    scheduler_cls.__init__ = __init__  # type: ignore[misc]
    scheduler_cls.make_spec_decoding_stats = make_spec_decoding_stats  # type: ignore[attr-defined]
    scheduler_cls._dsd_controller_installed = True  # type: ignore[attr-defined]


def register() -> None:
    """Entry point vLLM calls in each process; a no-op unless the controller
    is enabled."""
    if not os.environ.get(ENV_SWITCH):
        return
    from vllm.v1.core.sched.scheduler import Scheduler

    install(Scheduler)
