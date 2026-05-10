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
- HelionKernelWrapper: Wraps raw kernel + config_picker, creates configured kernels
- ConfiguredHelionKernel: Platform-specific kernel with pre-tuned configs
- PresetConfigSearch: Custom autotuner that returns pre-tuned configs
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.library import Library

from vllm.kernels.helion.case_key import CaseKey
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

# TODO(gmagogsfm): Remove CustomOp fallback path (_get_or_register_custom_op,
# vllm_helion_lib, direct_register_custom_op) once vLLM requires PyTorch >= 2.11.
# FIXME(gmagogsfm): Re-enable HOP path once performance regression is fixed.
# _HOP_AVAILABLE = requires_torch_version("2.11")
_HOP_AVAILABLE = False

if _HOP_AVAILABLE:
    from helion._compat import supports_torch_compile_fusion
    from helion._compiler._dynamo.higher_order_ops import helion_kernel_side_table
    from helion._compiler._dynamo.variables import HelionKernelVariable
    from helion.runtime.kernel import Kernel
    from torch._dynamo.guards import GuardBuilder
    from torch._dynamo.variables.builder import VariableBuilder


logger = init_logger(__name__)

vllm_helion_lib = Library("vllm_helion", "FRAGMENT")  # noqa

ConfigPicker = Callable[[tuple[Any, ...], list[CaseKey]], CaseKey | None]


def validate_helion_settings(
    helion_settings: helion.Settings | None, op_name: str
) -> None:
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
            f"register_kernel(..., config_picker=...) instead."
        )

    if settings_dict.get("static_shapes") is True:
        logger.warning(
            "Kernel '%s' has static_shapes=True in helion_settings, "
            "which will be overridden to False. vLLM requires dynamic "
            "shapes for variable batch sizes and sequence lengths.",
            op_name,
        )


def create_helion_decorated_kernel(
    raw_kernel_func: Callable,
    helion_settings: helion.Settings | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    kernel_kwargs: dict[str, Any] = {}
    if helion_settings:
        kernel_kwargs.update(helion_settings.to_dict())

    # vLLM requires dynamic shapes for variable batch sizes and sequence lengths
    kernel_kwargs["static_shapes"] = False

    if extra_kwargs:
        kernel_kwargs.update(extra_kwargs)

    return helion.kernel(**kernel_kwargs)(raw_kernel_func)


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
        config_picker: ConfigPicker | None,
        raw_kernel_func: Callable,
        helion_settings: helion.Settings | None = None,
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
                f"A config_picker must be provided to register_kernel()."
            )

        picker = self.config_picker
        all_keys = list(self.configs.keys())
        default = CaseKey.default()
        has_default = default in self.configs

        def key_computer(*args):
            selected = picker(args, all_keys)
            if selected is not None:
                return str(selected)
            if has_default:
                return str(default)
            return None

        return key_computer

    def _create_config_selector(self, key_computer):
        str_to_key = {str(k): k for k in self.configs}

        def config_selector(args):
            selected_str = key_computer(*args)

            if selected_str is None:
                raise ValueError(
                    f"Config picker returned None for kernel "
                    f"'{self.op_name}' with available config keys: "
                    f"{list(self.configs.keys())}"
                )

            config_key = str_to_key.get(selected_str)
            if config_key is None:
                raise ValueError(
                    f"Config picker returned invalid config key "
                    f"'{selected_str}' for kernel "
                    f"'{self.op_name}'. "
                    f"Available keys: {list(self.configs.keys())}"
                )

            return self.configs[config_key]

        return config_selector

    def _load_platform_configs(self) -> None:
        from vllm.kernels.helion.config_manager import ConfigManager
        from vllm.kernels.helion.utils import get_canonical_gpu_name

        self.platform = get_canonical_gpu_name()
        config_manager = ConfigManager()
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

        extra_kwargs = {
            "autotuner_fn": lambda _, args: PresetConfigSearch(args, config_selector),
            "key": key_computer,
        }

        logger.debug(
            "Creating decorated kernel %s with custom autotuner on platform %s",
            self.op_name,
            self.platform,
        )
        return create_helion_decorated_kernel(
            self.raw_kernel_func, self.helion_settings, extra_kwargs
        )


class HelionKernelWrapper:
    """Wrapper for Helion kernels with pre-tuned config selection and HOP support."""

    def __init__(
        self,
        raw_kernel_func: Callable,
        op_name: str,
        fake_impl: Callable,
        config_picker: ConfigPicker,
        helion_settings: helion.Settings | None = None,
        input_generator: (Callable[[], dict[CaseKey, tuple[Any, ...]]] | None) = None,
    ):
        # Validate helion_settings doesn't conflict with our custom autotuner
        validate_helion_settings(helion_settings, op_name)

        self.raw_kernel_func = raw_kernel_func
        self.op_name = op_name
        self._fake_impl = fake_impl
        self.helion_settings = helion_settings
        self._config_picker = config_picker
        self._input_generator = input_generator
        self._configured_kernel: ConfiguredHelionKernel | None = None
        # TODO(@gmagogsfm): Remove this disable flag once integrated with vLLM IR,
        # which handles op enablement/disablement.
        self._disabled = False
        self._disabled_reason: str | None = None

        try:
            if not _HOP_AVAILABLE:
                self._get_or_register_custom_op()
            else:
                self.get_configured_op()
        except ValueError as e:
            self._disabled = True
            self._disabled_reason = str(e)
            logger.warning(
                "Helion kernel '%s' is disabled: %s",
                op_name,
                self._disabled_reason,
            )

    def __call__(self, *args, **kwargs):
        if self._disabled:
            raise RuntimeError(
                f"Helion kernel '{self.op_name}' is disabled: {self._disabled_reason}"
            )
        if not _HOP_AVAILABLE:
            op = getattr(torch.ops.vllm_helion, self.op_name)
            return op(*args, **kwargs)
        assert self._configured_kernel is not None, (
            f"Kernel '{self.op_name}' was not initialized. "
            "Please open an issue on GitHub."
        )

        # During Dynamo tracing, this call will be intercepted by our custom
        # HelionKernelWrapperVariable and handled via proper HOP emission.
        # During eager execution, call the kernel directly.
        return self._configured_kernel(*args, **kwargs)

    def get_inputs(self) -> dict[CaseKey, tuple[Any, ...]]:
        if self._input_generator is None:
            raise NotImplementedError(
                f"No input generator registered for kernel '{self.op_name}'. "
                f"Use register_kernel(..., input_generator=...) to register one."
            )
        return self._input_generator()

    def run_autotune(
        self,
        inputs: tuple[Any, ...],
        autotune_effort: str = "quick",
    ) -> Config:
        """Run autotuning for a single input configuration."""
        extra_kwargs = {
            "autotune_effort": autotune_effort,
            "autotune_ignore_errors": True,
        }
        autotune_kernel = create_helion_decorated_kernel(
            self.raw_kernel_func, self.helion_settings, extra_kwargs
        )
        return autotune_kernel.autotune(inputs)

    def get_configured_op(self) -> ConfiguredHelionKernel:
        if self._disabled:
            raise RuntimeError(
                f"Helion kernel '{self.op_name}' is disabled: {self._disabled_reason}"
            )
        if self._configured_kernel is None:
            self._configured_kernel = ConfiguredHelionKernel(
                op_name=self.op_name,
                config_picker=self._config_picker,
                raw_kernel_func=self.raw_kernel_func,
                helion_settings=self.helion_settings,
            )
        return self._configured_kernel

    def _get_or_register_custom_op(self) -> Any:
        if hasattr(torch.ops.vllm_helion, self.op_name):
            return getattr(torch.ops.vllm_helion, self.op_name)

        configured_kernel = self.get_configured_op()

        logger.info("Registering op: vllm_helion::%s", self.op_name)
        direct_register_custom_op(
            op_name=self.op_name,
            op_func=configured_kernel._decorated_kernel,
            mutates_args=None,
            fake_impl=self._fake_impl,
            target_lib=vllm_helion_lib,
        )
        return getattr(torch.ops.vllm_helion, self.op_name)


# Global registry for tracking all registered HelionKernelWrapper instances
_REGISTERED_KERNELS: dict[str, HelionKernelWrapper] = {}


def get_registered_kernels() -> dict[str, HelionKernelWrapper]:
    return _REGISTERED_KERNELS.copy()


def get_kernel_by_name(kernel_name: str) -> HelionKernelWrapper | None:
    return _REGISTERED_KERNELS.get(kernel_name)


def infer_fake_impl(
    kernel_func: Callable,
    helion_settings: helion.Settings | None = None,
) -> Callable:
    def helion_fake_kernel(*args, **kwargs):
        kernel_kwargs = {}
        if helion_settings:
            kernel_kwargs.update(helion_settings.to_dict())

        temp_decorated_kernel = helion.kernel(**kernel_kwargs)(kernel_func)

        # Bind with args to get config_spec, then get a valid default config
        bound = temp_decorated_kernel.bind(args)
        default_config = bound.config_spec.default_config()
        compiled_runner = bound.compile_config(default_config)

        return compiled_runner(*args, **kwargs, _launcher=lambda *a, **kw: None)

    return helion_fake_kernel


def register_kernel(
    op_name: str | None = None,
    *,
    config_picker: ConfigPicker,
    fake_impl: Callable | None = None,
    helion_settings: helion.Settings | None = None,
    input_generator: (Callable[[], dict[CaseKey, tuple[Any, ...]]] | None) = None,
) -> Callable[[Callable], HelionKernelWrapper]:
    """Register a Helion kernel with pre-tuned config selection.

    Args:
        config_picker: Required. Receives ``(args, config_keys)``
            where each config key is a ``dict[str, Any]`` mapping
            parameter names to values.  Return the best-matching
            dict, or ``None`` to fall back to the default config.

            Example::

                def pick_config(args, config_keys):
                    x = args[0]
                    best = min(config_keys, key=lambda k: abs(k["size"] - x.shape[0]))
                    return best

        input_generator: Optional. Returns ``dict[str, tuple]`` where
            each key is a serialized config key and each value is a
            tuple of arguments to pass to the kernel.

            Example::

                def generate_inputs():
                    return {
                        "4096": (torch.randn(4096, device="cuda"), 0.5),
                        "8192": (torch.randn(8192, device="cuda"), 0.5),
                    }
    """

    def decorator(kernel_func: Callable) -> HelionKernelWrapper:
        final_op_name = op_name if op_name else kernel_func.__name__

        if final_op_name in _REGISTERED_KERNELS:
            raise ValueError(
                f"Helion kernel '{final_op_name}' is already registered. "
                f"Use a different op_name or check for duplicate registrations."
            )

        final_fake_impl = fake_impl
        if final_fake_impl is None:
            final_fake_impl = infer_fake_impl(kernel_func, helion_settings)
            logger.debug(
                "Auto-generated fake_impl for Helion kernel '%s'",
                kernel_func.__name__,
            )

        kernel_wrapper = HelionKernelWrapper(
            raw_kernel_func=kernel_func,
            op_name=final_op_name,
            fake_impl=final_fake_impl,
            config_picker=config_picker,
            helion_settings=helion_settings,
            input_generator=input_generator,
        )

        _REGISTERED_KERNELS[final_op_name] = kernel_wrapper

        logger.info(
            "Registered Helion kernel '%s' as HelionKernelWrapper",
            kernel_func.__name__,
        )

        return kernel_wrapper

    return decorator


# Register HelionKernelWrapper with Dynamo's variable tracker system
if _HOP_AVAILABLE:

    def _register_vllm_helion_dynamo_variable():
        """Register HelionKernelWrapper with Dynamo's VariableBuilder.

        When Dynamo encounters a HelionKernelWrapper during tracing, this
        extracts the underlying Helion Kernel and delegates to Helion's own
        registered Kernel handler, which handles HOP emission, side table
        registration, and inductor lowering setup.
        """

        def wrap_helion_kernel_wrapper(
            builder: VariableBuilder, value: HelionKernelWrapper
        ):
            kernel = value.get_configured_op()._decorated_kernel
            if supports_torch_compile_fusion():
                helion_handler = VariableBuilder._type_dispatch()[Kernel]
                return helion_handler(builder, kernel)
            kernel_idx = helion_kernel_side_table.add_kernel(kernel)
            builder.install_guards(GuardBuilder.ID_MATCH)
            return HelionKernelVariable(kernel, kernel_idx, source=builder.source)

        dispatch = VariableBuilder._type_dispatch()
        dispatch[HelionKernelWrapper] = wrap_helion_kernel_wrapper

    # Register immediately when the module is imported
    _register_vllm_helion_dynamo_variable()
