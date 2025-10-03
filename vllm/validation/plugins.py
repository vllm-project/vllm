# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)


class ModelType(int, enum.Enum):
    MODEL_TYPE_AI_MODEL = 1
    MODEL_TYPE_LORA = 2


class ModelValidationPlugin(ABC):
    """Base class for all model validation plugins"""

    @abstractmethod
    def model_validation_needed(self, model_type: ModelType, model_path: str) -> bool:
        """Have the plugin check whether it already validated the model
        at the given model_path."""
        return False

    @abstractmethod
    def validate_model(
        self, model_type: ModelType, model_path: str, model: str | None = None
    ) -> None:
        """Validate the model at the given model_path."""
        pass


@dataclass
class _ModelValidationPluginRegistry:
    plugins: dict[str, ModelValidationPlugin] = field(default_factory=dict)

    def register_plugin(self, plugin_name: str, plugin: ModelValidationPlugin):
        """Register a security plugin."""
        if plugin_name in self.plugins:
            logger.warning(
                "Model validation plugin %s is already registered, and will be "
                "overwritten by the new plugin %s.",
                plugin_name,
                plugin,
            )

        self.plugins[plugin_name] = plugin

    def model_validation_needed(self, model_type: ModelType, model_path: str) -> bool:
        """Check whether model validation was requested but was not done, yet.
        Returns False in case no model validation was requested or it is already
        done. Returns True if model validation was request but not done yet."""
        for plugin in self.plugins.values():
            if plugin.model_validation_needed(model_type, model_path):
                return True
        return False

    def validate_model(
        self, model_type: ModelType, model_path: str, model: str | None = None
    ) -> None:
        """Have all plugins validate the model at the given path. Any plugin
        that cannot validate it will throw an exception."""
        plugins = self.plugins.values()
        if plugins:
            for plugin in plugins:
                plugin.validate_model(model_type, model_path, model)
            logger.info("Successfully validated %s", model_path)


ModelValidationPluginRegistry = _ModelValidationPluginRegistry()
