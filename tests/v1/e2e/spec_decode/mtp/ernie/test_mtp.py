# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.utils import large_gpu_mark, single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig
from vllm.platforms import current_platform

from ...utils import assert_request_outputs_match, compute_acceptance_len

MODEL = "baidu/ERNIE-4.5-21B-A3B-PT"
PROMPTS = [
    "The capital of France is",
    "Water freezes at a temperature of",
    "The sequence 1, 2, 3, 4 continues with",
    "Artificial intelligence is",
    "The largest planet in the solar system is",
    "太阳从东方升起，西方",
    "北京是中国的",
    "从前有一座山，山上",
]


@single_gpu_only
@large_gpu_mark(min_gb=48)
def test_ernie_mtp_matches_target_and_accepts_drafts(vllm_runner):
    """ERNIE MTP must preserve greedy output and contribute accepted tokens."""
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
        ignore_eos=True,
        seed=0,
    )
    attention_backend = "TRITON_ATTN" if current_platform.is_rocm() else "auto"
    runner_kwargs = {
        "attention_backend": attention_backend,
        "block_size": None,
        "compilation_config": CompilationConfig(),
        "enable_chunked_prefill": None,
        "max_model_len": 512,
        "max_num_seqs": 1,
    }

    with vllm_runner(MODEL, **runner_kwargs) as ref_runner:
        ref_outputs = ref_runner.llm.generate(PROMPTS, sampling_params)

    with vllm_runner(
        MODEL,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        disable_log_stats=False,
        **runner_kwargs,
    ) as spec_runner:
        spec_outputs = spec_runner.llm.generate(PROMPTS, sampling_params)
        acceptance_len = compute_acceptance_len(spec_runner.llm.get_metrics())

    assert_request_outputs_match(
        ref_outputs,
        spec_outputs,
        required_matches=int(0.8 * len(ref_outputs)) + 1,
        context="ERNIE MTP",
    )
    assert acceptance_len > 1.0, f"ERNIE MTP acceptance length: {acceptance_len:.3f}"
