# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Module for logging intermediate tensors during model execution.

This module provides functionality to capture and save intermediate tensors
(inputs and outputs) from PyTorch modules during forward passes.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.hooks import RemovableHandle

from vllm.config import IntermediateLoggingConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Global step counter
_CURRENT_STEP = 0

_CURRENT_STEP_MODULE_CALL_STEP: dict[str, int] = {}

IL_MODULE_NAME = "_il_module_name"
IL_MODULE_CALL_IDX = "_il_module_call_idx"

# Utility functions for intermediate logging


def should_log_step(config):
    """Check if the current step should be logged based on the step IDs.

    Args:
        config: The IntermediateLoggingConfig instance.

    Returns:
        True if the current step should be logged, False otherwise.
    """
    if not is_log_enabled(config):
        return False

    # If log_step_ids is empty, log all steps
    if not config.log_step_ids:
        return True

    # Otherwise, check if current step is in the set of step IDs to log
    return get_step() in config._step_id_set


def should_log_device(config, device_name):
    """Check if a device should be logged based on the device names.

    Args:
        config: The IntermediateLoggingConfig instance.
        device_name: The name of the device to check (e.g., 'cuda:0', 'cpu').

    Returns:
        True if the device should be logged, False otherwise.
        If device_names is empty, all devices are logged.
    """
    if not is_log_enabled(config):
        return False
    # If device_names is empty, log all devices
    if not config.device_names:
        return True

    # Otherwise, check if device_name is in the list of device names to log
    return device_name in config.device_names


def should_log_module(config, module_name, module: torch.nn.Module) -> bool:
    """Check if a module should be logged based on the name regex patterns.

    Args:
        config: The IntermediateLoggingConfig instance.
        module_name: The name of the module to check.

    Returns:
        True if the module should be logged, False otherwise.
        If no patterns are defined, all modules are logged.
        If patterns are defined, the module is logged if it matches ANY pattern.
    """
    if not is_log_enabled(config):
        return False
    # If no patterns are defined, log all modules
    if not config._compiled_module_calls:
        set_il_module_name(module, module_name)
        set_il_module_call_idx(module, -1)
        return True

    # Check if the module name matches any of the patterns
    for pattern, call_idx in config._compiled_module_calls.items():
        match = pattern.search(module_name)
        if match:
            logger.debug(
                "Module %s, %s matches pattern: '%s', call_idx=%s",
                module_name,
                module.__class__.__name__,
                pattern.pattern,
                call_idx,
            )
            set_il_module_name(module, module_name)
            set_il_module_call_idx(module, call_idx)
            return True
    return False


def is_log_enabled(config):
    if not config or not config.enabled:
        return False
    if torch.compiler.is_compiling():
        logger.debug("Not logging because torch.compile is in progress")
        return False
    return True


def get_il_module_name(module: torch.nn.Module) -> str:
    return getattr(module, IL_MODULE_NAME, module.__class__.__name__)


def get_il_module_call_idx(module: torch.nn.Module) -> int:
    return getattr(module, IL_MODULE_CALL_IDX, -1)


def set_il_module_name(module: torch.nn.Module, name: str) -> None:
    setattr(module, IL_MODULE_NAME, name)


def set_il_module_call_idx(module: torch.nn.Module, idx: int) -> None:
    setattr(module, IL_MODULE_CALL_IDX, idx)


_global_config: Optional[IntermediateLoggingConfig] = None


@contextmanager
def intermediate_logging(config: Optional[IntermediateLoggingConfig]):
    """
    Temporarily sets the global config for the duration of the context.
    :param config: Keyword arguments to set as global config
    """
    global _global_config
    old_config = _global_config
    try:
        _global_config = config
        yield
    finally:
        _global_config = old_config


def get_current_il_config():
    return _global_config


def save_tensors(tensor: Any, file_path: str) -> Any:
    """Utility function to dump tensor to a file.

    Args:
        tensor: The tensor to dump. Can be a torch.Tensor, a list/tuple of
               tensors, or a dictionary containing tensors.
        file_path: Base path where to save the tensor (without extension).
    """

    if isinstance(tensor, torch.Tensor):
        device_name = str(tensor.device)
        intermediate_log_config = get_current_il_config()
        if not should_log_device(intermediate_log_config, device_name):
            return tensor
        pt_path = f"{file_path}_{device_name.replace(':', '_')}.pt"
        try:
            torch.save(tensor, pt_path)
            logger.debug("Saved tensor of shape %s to %s", tensor.shape,
                         pt_path)
        except Exception as e:
            logger.warning("Failed to save tensor to %s: %s", pt_path, e)
        return tensor

    if isinstance(tensor, (list, tuple)):
        for i, item in enumerate(tensor):
            save_tensors(item, f"{file_path}_{i}")
        return tensor
    if isinstance(tensor, dict):
        for k, v in tensor.items():
            save_tensors(v, f"{file_path}_{k}")
        return tensor


def step_fwd(module: torch.nn.Module, inputs: tuple[Any, ...],
             outputs: Any) -> None:
    """Hook to increment the global step counter after a forward pass.

    Args:
        module: The PyTorch module being executed.
        inputs: The inputs to the module's forward function.
        outputs: The outputs from the module's forward function.
    """
    if get_current_il_config() is None:
        return
    # Increment the global step counter
    increment_step()
    global _CURRENT_STEP_MODULE_CALL_STEP
    _CURRENT_STEP_MODULE_CALL_STEP = {}


def _prepare_module_log_dir(
    intermediate_log_config: IntermediateLoggingConfig,
    module_name: str,
    is_pre_fwd: bool = False,
) -> Path:
    # Create a unique directory for this step if not
    dump_dir = Path(
        intermediate_log_config.output_run_dir) / f"step_{get_step()}"
    dump_dir.mkdir(exist_ok=True, parents=True)

    # Create module directory
    suffix = ""
    module_call_idx = get_current_step_module_call(module_name)
    if module_call_idx > 0:
        suffix = f"_{module_call_idx}"
    module_dir = dump_dir / (module_name + suffix)
    if is_pre_fwd:
        _log_module_call(intermediate_log_config, module_name + suffix)
    module_dir.mkdir(exist_ok=True, parents=True)
    logger.debug("Logging module %s inputs/outputs to %s", module_name,
                 module_dir)
    return module_dir


def _log_module_call(
    intermediate_log_config: IntermediateLoggingConfig,
    module_name: str,
) -> None:
    file = (Path(intermediate_log_config.output_run_dir) /
            f"step_{get_step()}" / "module_calls.txt")
    with open(file, "a") as f:
        f.write(f"{module_name}\n")


def update_current_step_module_call(module_name: str) -> None:
    logger.debug("Updating current step module call for %s", module_name)
    global _CURRENT_STEP_MODULE_CALL_STEP
    if module_name not in _CURRENT_STEP_MODULE_CALL_STEP:
        _CURRENT_STEP_MODULE_CALL_STEP[module_name] = 0
    else:
        _CURRENT_STEP_MODULE_CALL_STEP[module_name] += 1


def get_current_step_module_call(module_name: str) -> int:
    return _CURRENT_STEP_MODULE_CALL_STEP.get(module_name, 0)


def prepare_log_current_fwd(module,
                            is_pre_fwd: bool = False) -> Optional[Path]:
    intermediate_log_config = get_current_il_config()
    if intermediate_log_config is None or not intermediate_log_config.enabled:
        return None
    if not should_log_step(intermediate_log_config):
        return None

    module_name = get_il_module_name(module)
    log_call_idx = get_il_module_call_idx(module)
    current_call_idx = get_current_step_module_call(module_name)
    should_log = True
    if log_call_idx >= 0 and current_call_idx != log_call_idx:
        should_log = False

    log_dir = None
    if is_pre_fwd:
        update_current_step_module_call(module_name)
    if should_log:
        log_dir = _prepare_module_log_dir(intermediate_log_config,
                                          module_name,
                                          is_pre_fwd=is_pre_fwd)
    return log_dir


def log_pre_fwd_hook(module: torch.nn.Module,
                     inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Hook to capture module inputs before forward pass.

    Args:
        module: The PyTorch module being executed.
        inputs: The inputs to the module's forward function.

    Returns:
        The unchanged inputs.
    """
    if log_dir := prepare_log_current_fwd(module, is_pre_fwd=True):
        save_tensors(inputs, str(log_dir / "inputs"))
    return inputs


def log_post_fwd_hook(module: torch.nn.Module, inputs: tuple[Any, ...],
                      outputs: Any) -> None:
    """Hook to capture module outputs after forward pass.

    Args:
        module: The PyTorch module being executed.
        inputs: The inputs to the module's forward function.
        outputs: The outputs from the module's forward function.
    """
    if log_dir := prepare_log_current_fwd(module, is_pre_fwd=False):
        save_tensors(outputs, str(log_dir / "outputs"))
        intermediate_log_config = get_current_il_config()
        assert intermediate_log_config is not None, \
            "IL config should not be None"
        if intermediate_log_config.log_post_fwd_inputs:
            save_tensors(inputs, str(log_dir / "post_fwd_inputs"))


def get_step() -> int:
    """Get the current global step counter.

    Returns:
        The current global step counter.
    """
    return _CURRENT_STEP


def increment_step() -> int:
    """Increment the global step counter.

    Returns:
        The new step counter value.
    """
    global _CURRENT_STEP
    _CURRENT_STEP += 1
    return _CURRENT_STEP


def reset_step() -> None:
    """Reset the global step counter to zero."""
    global _CURRENT_STEP
    _CURRENT_STEP = 0


class IntermediatesLogger:
    """Class to manage logging of intermediate tensors during model
    execution."""

    def __init__(self, config: IntermediateLoggingConfig):
        self.config = config
        self.hooks: list[tuple[str, str, Optional[RemovableHandle],
                               Optional[RemovableHandle]]] = []
        logger.debug("Created IntermediatesLogger with config: %s", config)
        path = Path(config.output_run_dir)
        path.mkdir(exist_ok=True, parents=True)
        # Log configuration
        logger.info("Intermediates will be logged in %s",
                    config.output_run_dir)

    def register_hooks(self, model: torch.nn.Module) -> None:
        """Register hooks for the model.

        Args:
            model: The PyTorch model to register hooks for.
        """

        for name, module in model.named_modules():
            if name and should_log_module(self.config, name, module):
                pre_hook = module.register_forward_pre_hook(log_pre_fwd_hook)
                logger.debug("Registered pre_fwd hook for %s",
                             module.__class__.__name__)
                post_hook = module.register_forward_hook(log_post_fwd_hook)
                logger.debug("Registered post_fwd hook for %s",
                             module.__class__.__name__)
                self.hooks.append((name, module, pre_hook, post_hook))

        # Register a step counter hook for the root model
        step_hook = model.register_forward_hook(step_fwd)
        self.hooks.append(("", model, None, step_hook))
        logger.info("Registered hooks for %s modules", len(self.hooks))

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for _, _, pre_hook, post_hook in self.hooks:
            if pre_hook is not None:
                pre_hook.remove()
            if post_hook is not None:
                post_hook.remove()

        logger.info("Removed %s hooks", len(self.hooks))
        self.hooks = []


def register_intermediate_hooks(
        model: torch.nn.Module,
        config: Optional[IntermediateLoggingConfig] = None
) -> IntermediatesLogger:
    """Register hooks to log intermediate tensors for a model.

    Args:
        model: The PyTorch model to log intermediates for.
        config: Configuration for intermediate logging. If provided, this takes
               precedence over kwargs.

    Returns:
        An IntermediatesLogger instance that can be used to manage the hooks.
    """
    logger_instance = IntermediatesLogger(config)
    logger_instance.register_hooks(model)
    return logger_instance
