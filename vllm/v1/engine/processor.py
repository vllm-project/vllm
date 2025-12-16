# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings


def __getattr__(name: str):
    if name == "Processor":
        from .input_processor import InputProcessor

        warnings.warn(
            "`vllm.v1.engine.processor.Processor` has been moved to "
            "`vllm.v1.engine.input_processor.InputProcessor`. "
            "The old name will be removed in v0.14.",
            DeprecationWarning,
            stacklevel=2,
        )

        return InputProcessor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
