import re
from typing import Set, Tuple

import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.levels import CompilationLevel
from vllm.platforms import current_platform
from vllm.utils import is_cpu, is_hip, is_xpu


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        # By default, we assume that XPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        # By default, we assume that CPU ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs):
        # By default, we assume that TPU ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    def forward_gaudi(self, *args, **kwargs):
        # By default, we assume that Gaudi ops are compatible with the
        # PyTorch-native implementation.
        # NOTE(woosuk): This is a placeholder for future extensions.
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        # NOTE(woosuk): Here we assume that vLLM was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.

        if not self._enabled():
            return self.forward_native

        if is_hip():
            return self.forward_hip
        elif is_cpu():
            return self.forward_cpu
        elif current_platform.is_tpu():
            return self.forward_tpu
        elif is_xpu():
            return self.forward_xpu
        else:
            return self.forward_cuda

    @staticmethod
    def _get_enabled_ops() -> Tuple[bool, Set[str], Set[str]]:
        """
        Parse the VLLM_ENABLE_CUSTOM_OPS environment variable to determine
         which custom ops are enabled. By default, custom ops are enabled
         if VLLM_TORCH_COMPILE_LEVEL < CompilationLevel.INDUCTOR.
         Specifying 'all' or 'none' will override this default.

        :return: A tuple of (default_on, enabled_ops, disabled_ops)
        """

        # filter empty strings
        env_ops = list(
            filter(lambda x: len(x) > 0, envs.VLLM_ENABLE_CUSTOM_OPS))

        use_all, use_none = env_ops.count("all"), env_ops.count("none")
        assert use_all + use_none <= 1, \
            "Cannot specify both 'all' and 'none' in VLLM_ENABLE_CUSTOM_OPS"

        # On by default if VLLM_TORCH_COMPILE_LEVEL < CompilationLevel.INDUCTOR
        default_on = envs.VLLM_TORCH_COMPILE_LEVEL < CompilationLevel.INDUCTOR

        # override the default if 'all' or 'none' is specified
        default_on = default_on and not bool(use_none) or bool(use_all)
        enabled_ops, disabled_ops = set(), set()

        for op in env_ops:
            if op == "all" or op == "none":
                continue

            assert re.match(r"^-?[a-z0-9_]+$",
                            op), f"Invalid custom op name: '{op}'"

            if op.startswith("-"):
                assert op[1:] not in {"all", "none"}, \
                    f"Cannot disable all or none: '{op}'"
                disabled_ops.add(op[1:])
            else:
                enabled_ops.add(op)

        assert (len(enabled_ops & disabled_ops) == 0
                ), "Cannot enable and disable the same custom ops: " + str(
                    enabled_ops & disabled_ops)

        return default_on, enabled_ops, disabled_ops

    @classmethod
    def _init_enabled_ops(cls):
        cls.default_on, cls.enabled_ops, cls.disabled_ops = (
            cls._get_enabled_ops())

    def _enabled(self) -> bool:
        return ((CustomOp.default_on or self.name in CustomOp.enabled_ops)
                and self.name not in CustomOp.disabled_ops)


CustomOp._init_enabled_ops()
