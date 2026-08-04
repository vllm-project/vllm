# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating request telemetry reporters based on configuration.
"""

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional
import importlib

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.request_telemetry.base import RequestTelemetry


class RequestTelemetryFactory:
    """Factory for creating request telemetry reporters."""

    _registry: dict[str, Callable[[], type["RequestTelemetry"]]] = {}
    _instances: dict[str, "RequestTelemetry"] = {}

    @classmethod
    def register(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a telemetry reporter with lazy-loading module and class name.

        Args:
            name: The name to register the telemetry reporter under.
            module_path: The module path to import the class from.
            class_name: The class name to import from the module.

        Raises:
            ValueError: If a reporter with the same name is already registered.
        """
        if name in cls._registry:
            raise ValueError(f"Telemetry reporter '{name}' is already registered.")

        def loader() -> type["RequestTelemetry"]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create(
        cls,
        telemetry_type: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        use_singleton: bool = True,
    ) -> "RequestTelemetry":
        """Create a request telemetry reporter based on the specified type.

        Args:
            telemetry_type: The type of telemetry reporter to create.
                If None, defaults to "noop".
                Supported values: "noop", "fastapi".
            config: Optional dict of configuration options passed to the reporter.
                Each reporter defines its own expected keys.
            use_singleton: If True, returns a cached singleton instance.
                If False, creates a new instance each time.
                Note: Singleton is keyed by telemetry_type only.

        Returns:
            A RequestTelemetry instance.

        Raises:
            ValueError: If an unsupported telemetry type is specified.
        """
        if telemetry_type is None:
            telemetry_type = "noop"

        telemetry_type = telemetry_type.lower()

        if telemetry_type not in cls._registry:
            raise ValueError(
                f"Unsupported request telemetry type: {telemetry_type}. "
                f"Supported types: {list(cls._registry.keys())}"
            )

        if use_singleton and telemetry_type in cls._instances:
            return cls._instances[telemetry_type]

        if config is None:
            config = {}

        telemetry_cls = cls._registry[telemetry_type]()
        instance = telemetry_cls(config)

        if use_singleton:
            cls._instances[telemetry_type] = instance

        return instance

    @classmethod
    def get_registered_types(cls) -> list[str]:
        """Get a list of all registered telemetry types."""
        return list(cls._registry.keys())


# Register telemetry types here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current telemetry type.

RequestTelemetryFactory.register(
    "noop",
    "lmcache.integration.request_telemetry.noop",
    "NoOpRequestTelemetry",
)

RequestTelemetryFactory.register(
    "fastapi",
    "lmcache.integration.request_telemetry.fastapi",
    "FastAPIRequestTelemetry",
)
