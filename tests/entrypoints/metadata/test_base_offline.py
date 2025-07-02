# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest
from pydantic import ValidationError

from vllm.entrypoints.metadata.generate import GenerateBrief, GenerateMetadata

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

MODEL_NAME = "Qwen/Qwen3-0.6B"
task = "generate"
max_model_len = 1024
enable_prefix_caching = False

expected_brief = GenerateBrief(task=task,
                               served_model_name=MODEL_NAME,
                               max_model_len=max_model_len,
                               enable_prefix_caching=enable_prefix_caching)


def test_offline(vllm_runner):
    with vllm_runner(
            MODEL_NAME,
            max_model_len=max_model_len,
            task=task,
            enable_prefix_caching=enable_prefix_caching) as vllm_model:
        metadata: GenerateMetadata = vllm_model.model.metadata

        assert isinstance(metadata, GenerateMetadata)

        with pytest.raises(ValidationError):
            # should not be able to modify the metadata
            metadata.brief.task = "foo"
