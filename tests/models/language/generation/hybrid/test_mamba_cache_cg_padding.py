# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.language.generation._hybrid_models import (
    HYBRID_MODELS,
    SSM_MODELS,
    _set_conv_state_layout,
)
from vllm.config import CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

pytestmark = pytest.mark.hybrid_model


@pytest.mark.parametrize("model", [SSM_MODELS[0], HYBRID_MODELS[0]])
@pytest.mark.parametrize("max_tokens", [20])
@pytest.mark.parametrize("conv_state_layout", ["SD", "DS"])
def test_mamba_cache_cg_padding(
    vllm_runner,
    example_prompts,
    monkeypatch,
    model: str,
    max_tokens: int,
    conv_state_layout: str,
) -> None:
    """
    This test is for verifying that mamba cache is padded to CG captured
    batch size. If it's not, a torch RuntimeError will be raised because
    tensor dimensions aren't compatible.
    """
    _set_conv_state_layout(monkeypatch, conv_state_layout)

    vllm_config = EngineArgs(model=model, trust_remote_code=True).create_engine_config()
    cudagraph_dispatcher = CudagraphDispatcher(vllm_config)
    cudagraph_dispatcher.initialize_cudagraph_keys(
        vllm_config.compilation_config.cudagraph_mode
    )
    if cudagraph_dispatcher.cudagraph_mode == CUDAGraphMode.NONE:
        pytest.skip("CUDA/XPU graph is disabled.Please enable it to run this test. ")
    while (
        len(example_prompts)
        == cudagraph_dispatcher.dispatch(len(example_prompts))[1].num_tokens
    ):
        example_prompts.append(example_prompts[0])

    try:
        with vllm_runner(model, enable_chunked_prefill=True) as vllm_model:
            vllm_model.generate_greedy(example_prompts, max_tokens)
    except RuntimeError:
        pytest.fail(
            "Couldn't run batch size which is not equal to a Cuda Graph "
            "captured batch size. "
            "Could be related to mamba cache not padded correctly"
        )
