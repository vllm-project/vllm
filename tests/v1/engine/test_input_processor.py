# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine.input_processor import InputProcessor


def test_process_inputs_validates_params_by_default():
    """_validate_params is called when skip_params_validation is False."""
    with patch.object(InputProcessor, "_validate_params") as mock_validate, \
         patch.object(InputProcessor, "_validate_lora"), \
         patch.object(InputProcessor, "__init__", return_value=None):
        processor = object.__new__(InputProcessor)
        processor.vllm_config = MagicMock()
        processor.vllm_config.parallel_config.data_parallel_size = 1
        processor.vllm_config.parallel_config.data_parallel_size_local = 1
        processor.vllm_config.parallel_config.local_engines_only = False

        sp = SamplingParams()
        tasks = ("generate",)

        # process_inputs will fail after the validation section due to missing
        # real infrastructure — that's fine; we only care about validate calls.
        with pytest.raises(Exception):
            processor.process_inputs(
                request_id="r1",
                prompt={"type": "token_ids", "prompt_token_ids": [1, 2, 3]},
                params=sp,
                supported_tasks=tasks,
                skip_params_validation=False,
            )

        mock_validate.assert_called_once_with(sp, tasks)


def test_process_inputs_skips_validation_when_flag_set():
    """_validate_params is NOT called when skip_params_validation is True."""
    with patch.object(InputProcessor, "_validate_params") as mock_validate, \
         patch.object(InputProcessor, "_validate_lora"), \
         patch.object(InputProcessor, "__init__", return_value=None):
        processor = object.__new__(InputProcessor)
        processor.vllm_config = MagicMock()
        processor.vllm_config.parallel_config.data_parallel_size = 1
        processor.vllm_config.parallel_config.data_parallel_size_local = 1
        processor.vllm_config.parallel_config.local_engines_only = False

        sp = SamplingParams()
        tasks = ("generate",)

        with pytest.raises(Exception):
            processor.process_inputs(
                request_id="r1",
                prompt={"type": "token_ids", "prompt_token_ids": [1, 2, 3]},
                params=sp,
                supported_tasks=tasks,
                skip_params_validation=True,
            )

        mock_validate.assert_not_called()
