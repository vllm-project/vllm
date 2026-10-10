# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager


@pytest.fixture(autouse=True)
def _workspace_manager_for_compile_passes(cleanup_fixture):
    if not current_platform.is_rocm():
        yield
        return

    # Release workspace tensors before cleanup_fixture flushes the allocator.
    reset_workspace_manager()
    if torch.accelerator.is_available():
        init_workspace_manager(torch.device(0))
    yield
    reset_workspace_manager()
