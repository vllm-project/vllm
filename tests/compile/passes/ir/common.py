import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest

from vllm.ir.op import IrOp, IrOpImpl


def _filter_supported_inputs(impl: IrOpImpl, inputs: Iterable[Iterable[Any]]):
    # filter only those inputs args that the impl supports
    supported_inputs = []
    for input in inputs:
        input = list(input)
        if impl.supports_all_args or impl.supports_args(*input):
            supported_inputs.append(input)
    return supported_inputs


@dataclass
class LoweringTestConfig:
    op: IrOp

    # inputs to pass to op impl tests, only valid inputs are
    # passed to corresponding op impl
    inputs: Iterable[Iterable[Any]] | None = None

    # set of indices to mark as unbacked when running relevant tests
    # pass argument name and indices within that argument to mark unbacked
    unbacked_idx: dict[str, Iterable[int]] | None = None

    config_registry: ClassVar[dict[str, "LoweringTestConfig"]] = {}

    def __post_init__(self):
        # test config cannot be defined twice
        assert self.op.name not in LoweringTestConfig.config_registry

        # ensure params exist in the function signature
        unbacked_idx = self.unbacked_idx or {}
        signature = inspect.signature(self.op.impls["native"].impl_fn)
        params = [name for name, _ in signature.parameters.items()]
        missing_args = set(unbacked_idx) - set(params)
        assert not missing_args, (
            "Following args are missing in the native function signature: "
            f"{', '.join(missing_args)}\n"
            f"Input params: {', '.join(unbacked_idx.keys())}\n"
            f"Native function params: {', '.join(params)}"
        )

        LoweringTestConfig.config_registry[self.op.name] = self

    @classmethod
    def get_test_inputs(cls):
        # get pytest parametrize params for the test_op_lowering test
        params = []
        for config in cls.config_registry.values():
            op = config.op
            for impl in op.impls.values():
                inputs = config.inputs or []
                inputs = _filter_supported_inputs(impl, inputs)
                params.append(
                    pytest.param(
                        op.name,
                        impl.provider,
                        inputs,
                        config.unbacked_idx,
                        marks=[
                            pytest.mark.skipif(
                                not inputs,
                                reason="No valid inputs for the implementation",
                            ),
                            pytest.mark.skipif(
                                not impl.supported,
                                reason="Implementation is not supported",
                            ),
                        ],
                    )
                )
        return params
