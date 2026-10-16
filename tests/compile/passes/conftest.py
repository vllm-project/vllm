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


@pytest.fixture(scope="module")
def rocm_aiter_fusion():
    """The ROCm AITER fusion pass module, skipped where AITER cannot run.

    Its patterns name AITER custom ops, which are registered only where AITER is
    usable, so the module is imported lazily.
    """
    from vllm._aiter_ops import is_aiter_found_and_supported

    if not is_aiter_found_and_supported():
        pytest.skip("ROCm with AITER only")
    # Registers torch.ops.vllm.gemm_with_dynamic_quant, which the toy graphs call.
    import vllm.model_executor.kernels.linear.mxfp4.aiter  # noqa: F401
    from vllm.compilation.passes.fusion import rocm_aiter_fusion as fusion_module

    return fusion_module
