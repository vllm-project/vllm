# SPDX-License-Identifier: Apache-2.0
"""Check mode registry implementation"""

# Standard
from typing import Callable, Dict, Optional
import importlib
import inspect
import os

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class CheckModeRegistry:
    """Registry for dynamically loaded check modes"""

    def __init__(self):
        self.modes: Dict[str, Callable] = {}
        self.loaded = False

    def register(self, name: str, func: Callable):
        """Register a check mode function"""
        if name in self.modes:
            raise ValueError(f"Check mode '{name}' already registered")
        self.modes[name] = func

    def load_modes(self):
        """Dynamically load all check mode modules"""
        if self.loaded:
            return

        # Get current package
        current_dir = os.path.dirname(__file__)

        # Find all modules with check_mode_ prefix
        for filename in os.listdir(current_dir):
            if filename.startswith("check_mode_") and filename.endswith(".py"):
                module_name = filename[:-3]  # Remove .py
                try:
                    module = importlib.import_module(
                        f".{module_name}", package=__package__
                    )
                    # Find and register mode functions
                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj) and hasattr(obj, "is_check_mode"):
                            self.register(obj.mode_name, obj)
                except ImportError as e:
                    logger.error(f"Failed to load check mode module {module_name}: {e}")

        self.loaded = True
        logger.info(f"Loaded {len(self.modes)} check modes")

    def get_mode(self, name: str) -> Optional[Callable]:
        """Get registered mode function. Returns None if the mode is not found."""
        if not self.loaded:
            self.load_modes()
        return self.modes.get(name)


def check_mode(name: str):
    """Decorator to mark functions as check modes"""

    def decorator(func):
        func.is_check_mode = True
        func.mode_name = name
        return func

    return decorator


# Global registry instance
registry = CheckModeRegistry()
