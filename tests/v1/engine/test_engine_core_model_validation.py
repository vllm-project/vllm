# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor
from vllm.validation.plugins import (
    ModelType,
    ModelValidationPlugin,
    ModelValidationPluginRegistry,
)


class MyModelValidator(ModelValidationPlugin):
    def model_validation_needed(self, model_type: ModelType, model_path: str) -> bool:
        return True

    def validate_model(
        self, model_type: ModelType, model_path: str, model: str | None = None
    ) -> None:
        raise BaseException("Model did not validate")


def test_engine_core_model_validation(
    monkeypatch: pytest.MonkeyPatch, dummy_opt_unmodified_path
):
    my_model_validator = MyModelValidator()
    ModelValidationPluginRegistry.register_plugin("test", my_model_validator)

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        engine_args = EngineArgs(model=dummy_opt_unmodified_path)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        with (
            set_default_torch_num_threads(1),
            pytest.raises(BaseException, match="Model did not validate"),
        ):
            EngineCore(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=False,
            )
