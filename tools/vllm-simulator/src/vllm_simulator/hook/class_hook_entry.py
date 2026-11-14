import builtins
from types import FunctionType
from typing import List, Union

from vllm_simulator.hook.base_hook import BaseHook, _register_hooks
from vllm_simulator.utils import get_logger

logger = get_logger()


CLASS_HOOKS: List[BaseHook] = []

_builtins_build_class_ = builtins.__build_class__


def _custom_build_class_(func, name: str, *bases, **kwargs):
    for hook in CLASS_HOOKS:
        if name != hook.HOOK_CLASS_NAME:
            continue
        module_name = None
        if isinstance(func, FunctionType):
            module_name = func.__globals__.get("__name__", "")
        if module_name != hook.HOOK_MODULE_NAME:
            continue
        logger.debug(f"Hooking Class: {hook.__name__} into {module_name}|{name}")
        target_class = _builtins_build_class_(func, name, *bases, **kwargs)
        try:
            hook.hook(target_class)
        except Exception as e:
            logger.warning(
                "Failed to hook class [%s]. Error type: %s, message: %r",
                name,
                type(e).__name__,
                e,
            )
        return target_class

    return _builtins_build_class_(func, name, *bases, **kwargs)


def install_class_hooks(hooks: Union[List[BaseHook], BaseHook]):
    _register_hooks(CLASS_HOOKS, hooks)
    builtins.__build_class__ = _custom_build_class_
