# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import Any

import torch
from torch.library import Library, infer_schema

vllm_ir_lib = Library("vllm_ir", "FRAGMENT")


def register_op(
    f: Callable | None = None,
    *,
    name: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
) -> "IrOp | Callable[[Callable], IrOp]":
    def decorator(_f: Callable):
        op_name = _f.__name__ if name is None else name
        return IrOp(op_name, _f, tags)

    if f is not None:
        return decorator(f)

    return decorator


class IrOp:
    registry: dict[str, "IrOp"] = {}

    def __init__(
        self, name: str, native_impl: Callable, tags: tuple[torch.Tag, ...] = ()
    ):
        self.name = name
        self._impls: dict[str, object] = {}
        self._native_impl = native_impl
        self._schema_str = infer_schema(native_impl, mutates_args=[])

        # torch registration
        dispatch_key = "CPU"  # TODO
        vllm_ir_lib.define(self.name + self._schema_str, tags=tags)
        vllm_ir_lib.impl(self.name, self._inner_call, dispatch_key=dispatch_key)
        assert hasattr(torch.ops.vllm_ir, name)

        self._torch_op_call = getattr(torch.ops.vllm_ir, name).default

        assert name not in self.registry
        self.registry[name] = self

    def register_fake(self):
        """
        Register a fake impl for the torch custom op. If this method is not called,
        the native implementation is used directly for the fake implementation.
        """
        pass  # TODO(luka)

    def register_impl(self, provider: str):
        def decorator(f: Callable):
            print(f"Registering {provider} for op {self.name}")
            assert provider not in self._impls
            self._impls[provider] = f
            return

        return decorator

    #
    # def register_impl(provider: str, enable_if):
    #     """Register an implementation for this custom op"""
    #     def decorator(f):
    #         from vllm.ir import IrOpImpl
    #         return IrOpImpl(self, provider, f, enable_if)

    def _inner_call(self, *args, **kwargs) -> Any:
        """Direct call to torch op, could also skip the torch layer if eager?"""
        print(f"Direct call to {self.name} with args {args} with kwargs {kwargs}")
        return self._native_impl(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        return self._torch_op_call(*args, **kwargs)
