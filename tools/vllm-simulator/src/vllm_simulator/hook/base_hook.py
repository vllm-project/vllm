from typing import List, Optional, Union


class BaseHook:
    HOOK_CLASS_NAME: Optional[str] = None
    HOOK_MODULE_NAME: Optional[str] = None

    @classmethod
    def hook(cls, target) -> None:
        """
        Return a new target or simply modify the target reference.
        """
        raise NotImplementedError


def _register_hooks(HOOKS: List[BaseHook], hooks: Union[List[BaseHook], BaseHook]):
    if isinstance(hooks, list):
        for hook in hooks:
            if not issubclass(hook, BaseHook):
                raise TypeError("The hook should inherit from BaseHook.")
            HOOKS.append(hook)
    elif isinstance(hooks, type) and issubclass(hooks, BaseHook):
        HOOKS.append(hooks)
    else:
        raise TypeError(
            "The type of registered hook should be a list of BaseHook or a single BaseHook."
        )
