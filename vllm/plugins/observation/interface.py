# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.plugins import OBSERVATION_PLUGINS_GROUP, load_plugins_by_group
from vllm.utils.import_utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)


class ObservationAction(enum.IntEnum):
    CONTINUE = 0  # Normal execution.
    ABORT = 1  # Instantly terminate the request (FINISHED_ABORTED).
    PREEMPT = 2  # Evict the request back to the waiting queue (PREEMPTED).


@dataclass
class RequestContext:
    request_id: str
    is_prefill: bool
    chunk_idx: int
    num_cached_tokens: int
    batch_offset: int
    num_tokens: int


@dataclass
class ObservationResult:
    action: ObservationAction = ObservationAction.CONTINUE
    metadata: dict[str, Any] = field(default_factory=dict)


class ObservationPlugin:
    """Base class for observation plugins."""

    def __init__(self, vllm_config: VllmConfig | None = None) -> None:
        pass

    def get_observation_layers(self) -> list[int]:
        """Returns normalized layer indices (0 is first, -1 is last) to observe."""
        return []

    def observe_decode(self) -> bool:
        """
        Whether this plugin requires observing the decode phase
        (token-by-token generation).
        If False (default), the plugin will only observe the prefill phase,
        significantly improving performance as it avoids breaking CUDA graphs.
        Set to True only if per-token observation is strictly required.
        """
        return False

    def on_request_start(self, request_id: str, prompt: str | None = None) -> None:
        """
        Called when the engine officially accepts the request into the
        waiting queue, before any GPU execution occurs.
        Ideal for setting up trackers or validating text.
        """
        pass

    def on_step_batch(
        self,
        batch_hidden_states: dict[int, torch.Tensor],
        request_contexts: list[RequestContext],
    ) -> list[ObservationResult]:
        """
        Called repeatedly for every forward pass step on the GPU
        (Prefill, Chunk, or Decode).
        Returns a target Action (CONTINUE, ABORT) or async metadata for each request.
        """
        return [ObservationResult()] * len(request_contexts)

    def on_request_complete(self, request_id: str) -> None:
        """
        Called exactly once when a request is either fully generated, cleanly aborted,
        or preempted out of the system. Ideal for resource teardown and final telemetry.
        """
        pass

    def reload_config(self, config_data: dict[str, Any]) -> None:
        """
        Called by an external orchestrator (e.g., watching a ConfigMap)
        to update plugin parameters dynamically at runtime (e.g., threshold tuning).
        Implementations should use thread-safe locks when updating internal state.
        """
        pass


class PluginManager:
    """
    Manages the lifecycle and execution of multiple ObservationPlugins.
    Aggregates results and enforces execution order if needed.
    """

    def __init__(self, plugins: list[ObservationPlugin]):
        self.plugins = plugins

    def get_observation_layers(self) -> list[int]:
        layers = set()
        for p in self.plugins:
            layers.update(p.get_observation_layers())
        return sorted(list(layers))

    @property
    def observe_decode(self) -> bool:
        return any(p.observe_decode() for p in self.plugins)

    def on_request_start(self, request_id: str, prompt: str | None = None) -> None:
        for p in self.plugins:
            p.on_request_start(request_id, prompt)

    def on_step_batch(
        self,
        batch_hidden_states: dict[int, torch.Tensor],
        request_contexts: list[RequestContext],
    ) -> list[ObservationResult]:
        if not self.plugins:
            return [ObservationResult()] * len(request_contexts)

        # Initialize default results for the batch
        aggregated_results = [ObservationResult() for _ in range(len(request_contexts))]

        for plugin in self.plugins:
            # Get results from this specific plugin
            plugin_results = plugin.on_step_batch(batch_hidden_states, request_contexts)

            # Merge results: Highest severity action wins (ABORT > PREEMPT > CONTINUE)
            # Metadata dictionaries are merged. Keep in mind key collisions aren't
            # explicitly avoided here, so plugins should ideally use namespaced keys
            # (e.g. "safety_score").
            for i, result in enumerate(plugin_results):
                if result.action > aggregated_results[i].action:
                    aggregated_results[i].action = result.action
                aggregated_results[i].metadata.update(result.metadata)

        return aggregated_results

    def on_request_complete(self, request_id: str) -> None:
        for p in self.plugins:
            p.on_request_complete(request_id)


def load_observation_plugins(
    plugin_names: list[str] | None,
    vllm_config: VllmConfig,
) -> PluginManager:
    """
    Load observation plugins specified by names in the config.
    Returns a PluginManager containing the initialized plugins.
    """
    if not plugin_names:
        return PluginManager([])

    # Load all installed plugin in the group
    discovered_plugins = load_plugins_by_group(OBSERVATION_PLUGINS_GROUP)

    loadable_plugins = {}
    for name, func in discovered_plugins.items():
        try:
            assert callable(func)
            plugin_cls_qualname = func()
            if plugin_cls_qualname is not None:
                loadable_plugins[name] = plugin_cls_qualname
        except Exception:
            logger.warning("Failed to load plugin %s.", name, exc_info=True)

    loaded_plugins = []
    for requested_plugin in plugin_names:
        if requested_plugin not in loadable_plugins:
            logger.warning(
                "Requested observation plugin '%s' is not installed or "
                "failed to load. Available plugins: %s",
                requested_plugin,
                list(loadable_plugins.keys()),
            )
            continue

        try:
            plugin_cls = resolve_obj_by_qualname(loadable_plugins[requested_plugin])

            # Conditionally pass vllm_config if accepted by the plugin constructor
            init_signature = inspect.signature(plugin_cls.__init__)
            if "vllm_config" in init_signature.parameters:
                plugin_instance = plugin_cls(vllm_config=vllm_config)
            else:
                plugin_instance = plugin_cls()

            loaded_plugins.append(plugin_instance)
            logger.info("Successfully loaded observation plugin: %s", requested_plugin)
        except Exception:
            logger.exception(
                "Failed to initialize observation plugin %s",
                requested_plugin,
            )

    return PluginManager(loaded_plugins)
