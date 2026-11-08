# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 MTP drafter config view on the NVIDIA path."""

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

from ..registry import HF_EXAMPLE_MODELS

pytestmark = pytest.mark.skipif(
    current_platform.is_rocm(), reason="ROCm uses the AMD Step5 MTP module"
)


@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
def test_mtp_config_view(pipeline_parallel_size: int):
    """The drafter's config view combines already-initialized configs. Building
    it must neither validate the draft architecture against the target's
    deployment (PP > 1 raised NotImplementedError) nor re-resolve the target's
    shared configs (the CUDA graph mode was downgraded)."""
    from vllm.models.step5.nvidia.mtp import _get_mtp_vllm_config

    info = HF_EXAMPLE_MODELS.get_hf_info("Step5ForConditionalGeneration")
    info.check_available_online(on_fail="skip")
    vllm_config = EngineArgs(
        model=info.default,
        max_model_len=4096,
        pipeline_parallel_size=pipeline_parallel_size,
        distributed_executor_backend="mp",
        load_format="dummy",
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    ).create_engine_config()
    assert vllm_config.speculative_config is not None
    draft_model_config = vllm_config.speculative_config.draft_model_config
    compilation_config = vllm_config.compilation_config
    cudagraph_mode = compilation_config.cudagraph_mode
    use_breakable_cudagraph = compilation_config.use_breakable_cudagraph

    view = _get_mtp_vllm_config(vllm_config)

    assert view.model_config.hf_text_config is draft_model_config.hf_text_config
    assert view.parallel_config is vllm_config.parallel_config
    assert not draft_model_config.is_submodel_config
    assert compilation_config.cudagraph_mode == cudagraph_mode
    assert compilation_config.use_breakable_cudagraph == use_breakable_cudagraph
