import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple

import pytest

from vllm.ir.op import IrOp, IrOpImpl


def _filter_supported_inputs(impl: IrOpImpl, inputs: list[list[Any]]):
    # filter only those inputs args that the impl supports
    supported_inputs = []
    for input in inputs:
        if impl.supports_all_args or impl.supports_args(*input):
            supported_inputs.append(input)
    return supported_inputs


@dataclass
class LoweringTestConfig:
    # vllm ir op to run tests on
    op: IrOp

    # example inputs to the op, inputs should contain atleast one valid input that
    # passes supports_all_args for all tested implementations
    inputs: list[list[Any]] = field(default_factory=list, repr=False)

    # set of args that have batched inputs, this input will be passed to torch.compile
    # with the first dimension as an unbacked int
    batched_args: list[str] = field(default_factory=list)

    config_registry: ClassVar[dict[str, "LoweringTestConfig"]] = {}

    def __post_init__(self):
        # test config cannot be defined twice
        assert self.op.name not in LoweringTestConfig.config_registry

        # ensure batched args exist in the function signature
        signature = inspect.signature(self.op.impls["native"].impl_fn)
        params = [name for name, _ in signature.parameters.items()]
        missing_args = set(self.batched_args) - set(params)
        assert not missing_args, (
            "Following args are missing in the native function signature: "
            f"{', '.join(missing_args)}\n"
            f"Input params: {', '.join(self.batched_args)}\n"
            f"Native function params: {', '.join(params)}"
        )

        LoweringTestConfig.config_registry[self.op.name] = self

    @classmethod
    def get_test_op_lowering_params(cls):
        # get pytest parametrize params for the test_op_lowering test
        params = []
        for config in cls.config_registry.values():
            op = config.op
            for impl in op.impls.values():
                inputs = _filter_supported_inputs(impl, config.inputs)
                params.append(
                    pytest.param(
                        op.name,
                        impl.provider,
                        inputs,
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

    @classmethod
    def get_test_batch_specialization_params(cls):
        """
        get pytest parametrize params for the test_batch_specialization test
        """
        params = []
        for config in cls.config_registry.values():
            op = config.op
            for impl in op.impls.values():
                inputs = _filter_supported_inputs(impl, config.inputs)
                params.append(
                    pytest.param(
                        op.name,
                        impl.provider,
                        inputs,
                        config.batched_args,
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


class ModelLoweringInfo(NamedTuple):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    hf_overrides: Callable[[int], dict] = lambda n: {"num_hidden_layers": n}
