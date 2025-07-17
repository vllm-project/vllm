# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest

from vllm.model_executor.model_loader.registry import ModelLoaderRegistry


@pytest.fixture(autouse=True)
def isolate_model_loader_registry():
    """Isolates changes to the ModelLoaderRegistry for each test."""
    original_loaders = copy.copy(ModelLoaderRegistry._model_loaders)
    yield
    ModelLoaderRegistry._model_loaders = original_loaders
