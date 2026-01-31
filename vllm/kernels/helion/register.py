# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Helion kernel registration with pre-tuned config selection.

This module leverages Helion's internal config selection infrastructure to use
pre-tuned configs instead of runtime autotuning.

How Helion Normally Works
-------------------------
For each kernel invocation, Helion:
1. Computes a cache key from input arguments
2. Looks up the key in its internal compilation cache
3. On cache miss, runs autotuning to find the best config
4. Compiles and caches the kernel with that config

How We Override It
------------------
We override two Helion hooks to use pre-tuned configs:

1. **key**: We provide a key function (derived from config_picker) that
   computes cache keys matching our pre-tuned config keys. This ensures Helion's
   internal cache uses keys that correspond to configs we've prepared.

2. **autotuner_fn**: We provide PresetConfigSearch which, instead of autotuning,
   simply returns the pre-tuned config for the computed key. On cache miss,
   Helion calls our autotuner which returns the author-prepared config.

Both hooks use the same config_picker logic to ensure the cache key computed
by key matches the config returned by the autotuner.

Key Classes
-----------
- HelionKernelWrapper: Wraps raw kernel + config_picker, creates configured ops
- ConfiguredHelionKernel: Platform-specific kernel registered as PyTorch custom op
- PresetConfigSearch: Custom autotuner that returns pre-tuned configs
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.library import Library

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import direct_register_custom_op

if not has_helion():
    raise ImportError(
        "register module requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
from helion.autotuner.base_search import BaseAutotuner
from helion.runtime.config import Config
from helion.runtime.settings import default_autotuner_fn

logger = init_logger(__name__)

vllm_helion_lib = Library("vllm_helion", "FRAGMENT")  # noqa


def validate_helion_settings(
    helion_settings: "helion.Settings | None", op_name: str
) -> None:
    """Validate that helion_settings doesn't contain conflicting options."""
    if helion_settings is None:
        return

    settings_dict = helion_settings.to_dict()

    if (
        "autotuner_fn" in settings_dict
        and settings_dict["autotuner_fn"] is not None
        and settings_dict["autotuner_fn"] is not default_autotuner_fn
    ):
        raise ValueError(
            f"HelionKernelWrapper for '{op_name}' uses a custom autotuner via "
            f"config picker. Remove 'autotuner_fn' from helion_settings and use "
            f"@{op_name}.register_config_picker instead."
        )

    # Warn if static_shapes is explicitly set to True since most vLLM ops need
    # dynamic shapes for variable batch sizes and sequence lengths
    if settings_dict.get("static_shapes") is True:
        logger.warning(
            "Kernel '%s' has static_shapes=True in helion_settings. "
            "Most vLLM ops require dynamic shapes for variable batch sizes "
            "and sequence lengths. Consider removing this setting.",
            op_name,
        )


class PresetConfigSearch(BaseAutotuner):
    """Custom autotuner that uses a preset config selector instead of autotuning."""

    def __init__(
        self,
        args: tuple[Any, ...],
        config_selector: Callable[[tuple[Any, ...]], Config],
    ):
        self.args = args
        self.config_selector = config_selector

    def autotune(self, *, skip_cache: bool = False) -> Config:
        return self.config_selector(self.args)


class ConfiguredHelionKernel:
    """A configured Helion kernel bound to a specific platform."""

    def __init__(
        self,
        op_name: str,
        config_picker: Callable[[tuple[Any, ...], list[str]], str | None],
        raw_kernel_func: Callable,
        helion_settings: "helion.Settings | None" = None,
    ):
        self.op_name = op_name
        self.config_picker = config_picker
        self.raw_kernel_func = raw_kernel_func
        self.helion_settings = helion_settings
        self._decorated_kernel = self._create_decorated_kernel()

    def __call__(self, *args, **kwargs):
        return self._decorated_kernel(*args, **kwargs)

    def _create_key_computer(self):
        """
        Create a key computer function derived from the config picker.

        The returned function receives kernel arguments unpacked (*args) to match
        Helion's key signature (called as self._key_fn(*args)).
        """
        if self.config_picker is None:
            raise RuntimeError(
                f"No config picker registered for kernel '{self.op_name}'. "
                f"Use @{self.op_name}.register_config_picker to register one."
            )

        def key_computer(*args):
            config_keys = list(self.configs.keys())
            selected_key = self.config_picker(args, config_keys)
            if selected_key:
                return selected_key
            return "default" if "default" in self.configs else None

        return key_computer

    def _create_config_selector(self, key_computer):
        def config_selector(args):
            # args is a tuple; key_computer expects unpacked args
            selected_config_key = key_computer(*args)

            if selected_config_key is None:
                raise ValueError(
                    f"Config picker returned None for kernel '{self.op_name}' "
                    f"with available config keys: {list(self.configs.keys())}"
                )

            if selected_config_key not in self.configs:
                raise ValueError(
                    f"Config picker returned invalid config key "
                    f"'{selected_config_key}' for kernel '{self.op_name}'. "
                    f"Available keys: {list(self.configs.keys())}"
                )

            return self.configs[selected_config_key]

        return config_selector

    def _load_platform_configs(self) -> None:
        from vllm.kernels.helion.config_manager import ConfigManager
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        self.platform = get_canonical_gpu_name()
        config_manager = ConfigManager.get_instance()
        self.configs = config_manager.get_platform_configs(self.op_name, self.platform)

        if not self.configs:
            raise ValueError(
                f"No configs available for kernel '{self.op_name}' "
                f"on platform '{self.platform}'"
            )

    def _create_decorated_kernel(self) -> Callable[..., Any]:
        self._load_platform_configs()

        key_computer = self._create_key_computer()
        config_selector = self._create_config_selector(key_computer)

        kernel_kwargs = {}
        if self.helion_settings:
            kernel_kwargs.update(self.helion_settings.to_dict())

        # Set static_shapes=False by default if user didn't explicitly set it to True
        # This is needed for dynamic batch sizes and sequence lengths in vLLM
        if kernel_kwargs.get("static_shapes") is not True:
            kernel_kwargs["static_shapes"] = False

        kernel_kwargs["autotuner_fn"] = lambda _, args: PresetConfigSearch(
            args, config_selector
        )
        kernel_kwargs["key"] = key_computer

        logger.debug(
            "Creating decorated kernel %s with custom autotuner on platform %s",
            self.op_name,
            self.platform,
        )
        return helion.kernel(**kernel_kwargs)(self.raw_kernel_func)


class HelionKernelWrapper:
    """Wrapper for Helion kernels that creates config-specific PyTorch custom ops."""

    def __init__(
        self,
        raw_kernel_func: Callable,
        op_name: str,
        fake_impl: Callable,
        helion_settings: "helion.Settings | None" = None,
    ):
        # Validate helion_settings doesn't conflict with our custom autotuner
        validate_helion_settings(helion_settings, op_name)

        self.raw_kernel_func = raw_kernel_func
        self.op_name = op_name
        self._fake_impl = fake_impl
        self.helion_settings = helion_settings
        self._config_picker: (
            Callable[[tuple[Any, ...], list[str]], str | None] | None
        ) = None

    def __call__(self, *args, **kwargs):
        configured_op = self.get_configured_op()
        return configured_op(*args, **kwargs)

    def register_config_picker(
        self, picker_func: Callable[[tuple[Any, ...], list[str]], str | None]
    ) -> Callable[[tuple[Any, ...], list[str]], str | None]:
        self._config_picker = picker_func
        return picker_func

    def get_configured_op(self) -> Any:
        assert self._config_picker is not None, (
            f"No config picker registered for kernel '{self.op_name}'. "
            f"Use @{self.op_name}.register_config_picker to register one."
        )

        if hasattr(torch.ops.vllm_helion, self.op_name):
            logger.debug("Op vllm_helion::%s already registered", self.op_name)
            return getattr(torch.ops.vllm_helion, self.op_name)

        configured_kernel = ConfiguredHelionKernel(
            op_name=self.op_name,
            config_picker=self._config_picker,
            raw_kernel_func=self.raw_kernel_func,
            helion_settings=self.helion_settings,
        )

        logger.info("Registering op: vllm_helion::%s", self.op_name)
        direct_register_custom_op(
            op_name=self.op_name,
            op_func=configured_kernel._decorated_kernel,  # Register decorated kernel
            # TODO(gmagogsfm): Implement automatic mutation/aliasing detection
            # for Helion kernels.
            mutates_args=None,
            fake_impl=self._fake_impl,
            target_lib=vllm_helion_lib,
        )
        return getattr(torch.ops.vllm_helion, self.op_name)
