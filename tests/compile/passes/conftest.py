# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager


@pytest.fixture(autouse=True)
def _workspace_manager_for_compile_passes():
    # Reset before each test so stale GPU allocations do not leak across tests
    # at high concurrency (max-in-flight >= 32 on DPX cluster), then init for
    # passes such as MLA RoPE fusion that require WorkspaceManager.
    reset_workspace_manager()
    if torch.cuda.is_available():
        init_workspace_manager(torch.device("cuda"))
    yield
    reset_workspace_manager()
