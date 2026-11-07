# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AMD Qwen3.5 model (eager + breakable CUDA graphs, fused norms) matches
HF greedy logprobs: at TP1 on the Triton GDN path, and at TP2 on the AITER GDN
path with the AITER all-reduce fused into the norms."""

import pytest
from transformers import AutoModelForImageTextToText

from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AMD Qwen3.5 path requires ROCm"
)

MODEL = "Qwen/Qwen3.5-0.8B"


@pytest.mark.parametrize("tp_size,use_aiter", [(1, "0"), (2, "1")])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    use_aiter: str,
) -> None:
    if not current_platform.is_device_capability(95):
        pytest.skip("The AMD Qwen3.5 model is only used on gfx950")
    if current_platform.device_count() < tp_size:
        pytest.skip(f"Need >= {tp_size} GPUs")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", use_aiter)
    max_tokens, num_logprobs = 32, 5

    with hf_runner(MODEL, auto_cls=AutoModelForImageTextToText) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs
        )

    with vllm_runner(
        MODEL,
        tensor_parallel_size=tp_size,
        max_model_len=2048,
        max_num_seqs=4,
        enable_chunked_prefill=True,
        gpu_memory_utilization=0.3,
        language_model_only=True,
    ) as vllm_model:
        config = vllm_model.llm.llm_engine.vllm_config.compilation_config
        assert config.use_breakable_cudagraph
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
