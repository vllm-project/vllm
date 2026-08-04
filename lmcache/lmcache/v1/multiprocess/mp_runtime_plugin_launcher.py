# SPDX-License-Identifier: Apache-2.0

"""
Runtime plugin launcher for multiprocess (MP) mode.

Unlike the non-MP RuntimePluginLauncher which receives a single
LMCacheEngineConfig, the MP mode has multiple independent config
dataclasses (MPServerConfig, StorageManagerConfig,
ObservabilityConfig, etc.).  This launcher aggregates them all
into a single JSON blob so plugins get the full server config
via the LMCACHE_RUNTIME_PLUGIN_CONFIG environment variable.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import json

# First Party
from lmcache.logging import init_logger
from lmcache.v1.plugin.runtime_plugin_launcher import (
    RuntimePluginLauncher,
)
from lmcache.v1.utils.json_utils import make_json_safe, safe_asdict

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.config import (
        RuntimePluginConfig,
    )

logger = init_logger(__name__)


@dataclass
class _MPPluginConfig:
    """Thin wrapper that satisfies RuntimePluginLauncher's
    config duck-type contract (runtime_plugin_locations +
    to_json)."""

    runtime_plugin_locations: list[str]
    extra_config: dict[str, Any]
    configs_dict: dict[str, Any]

    def to_json(self) -> str:
        merged = dict(self.configs_dict)
        if self.extra_config:
            merged["runtime_plugin_extra_config"] = make_json_safe(self.extra_config)
        return json.dumps(merged)


class MPRuntimePluginLauncher:
    """Launch runtime plugins in MP mode with all server
    configs serialized into the environment.

    Usage::

        launcher = MPRuntimePluginLauncher(
            runtime_plugin_config=runtime_plugin_config,
            mp_config=mp_config,
            storage_manager_config=sm_config,
            obs_config=obs_config,
        )
        launcher.launch_plugins()
        # ... on shutdown ...
        launcher.stop_plugins()
    """

    def __init__(
        self,
        runtime_plugin_config: "RuntimePluginConfig",
        **configs: object,
    ) -> None:
        """Initialize the MP runtime plugin launcher.

        Aggregates arbitrary dataclass configs into a single
        JSON blob and delegates to RuntimePluginLauncher.

        Args:
            runtime_plugin_config: RuntimePluginConfig with
                locations and extra_config fields.
            **configs: Dataclass config objects to serialize
                and pass to plugins via environment variable.
        """
        # Build the aggregated JSON dict from all configs
        aggregated: dict = {}
        for name, cfg in configs.items():
            aggregated[name] = safe_asdict(cfg)

        wrapper = _MPPluginConfig(
            runtime_plugin_locations=runtime_plugin_config.locations,
            extra_config=getattr(runtime_plugin_config, "extra_config", {}),
            configs_dict=aggregated,
        )
        self._inner = RuntimePluginLauncher(
            config=wrapper,
            role=None,
            worker_count=1,
            worker_id=0,
        )
        logger.info("MPRuntimePluginLauncher initialized with %s", wrapper)

    def launch_plugins(self) -> None:
        """Launch all configured plugins."""
        self._inner.launch_plugins()

    def stop_plugins(self) -> None:
        """Terminate all plugin processes."""
        self._inner.stop_plugins()
