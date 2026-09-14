# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import pytest

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from tests.utils import single_gpu_only
from vllm.config import CompilationConfig

from ...utils import compute_acceptance_len
from ..utils import run_acceptance_length_eval


@dataclass(frozen=True)
class DFlashCorrectnessConfig:
    model: str
    draft_model: str
    expected_accuracy: float
    expected_acceptance_len: float
    num_speculative_tokens: int = 16
    max_model_len: int = 4096
    max_num_seqs: int = 128
    num_questions: int = 1319
    use_chat_completions: bool = False
    enforce_eager: bool = False
    disable_flashinfer_sampler: bool = False


QWEN3_DFLASH = DFlashCorrectnessConfig(
    model="Qwen/Qwen3-8B",
    draft_model="z-lab/Qwen3-8B-DFlash-b16",
    expected_accuracy=0.8,
    expected_acceptance_len=3.5,
)

LAGUNA_DFLASH_NVFP4 = DFlashCorrectnessConfig(
    model="poolside/Laguna-XS-2.1-NVFP4",
    draft_model="poolside/Laguna-XS-2.1-DFlash-NVFP4",
    expected_accuracy=0.7,  # Standard GSM8K sanity floor.
    expected_acceptance_len=3.55 * 0.9,
    num_speculative_tokens=15,
    max_model_len=8192,
    max_num_seqs=32,
    num_questions=200,
    use_chat_completions=True,
    enforce_eager=True,
    disable_flashinfer_sampler=True,
)


@pytest.mark.parametrize("use_mrv2", [False, True])
def test_dflash_reference_acceptance_lengths(
    monkeypatch: pytest.MonkeyPatch,
    use_mrv2: bool,
    vllm_runner,
):
    run_acceptance_length_eval(
        monkeypatch,
        vllm_runner,
        spec_config={
            "model": QWEN3_DFLASH.model,
            "trust_remote_code": True,
            "speculative_config": {
                "method": "dflash",
                "model": QWEN3_DFLASH.draft_model,
                "num_speculative_tokens": QWEN3_DFLASH.num_speculative_tokens,
                "max_model_len": 32768,
            },
            "max_model_len": 32768,
            "max_num_seqs": 128,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": False,
            "disable_log_stats": False,
        },
        # Table 1 in https://arxiv.org/pdf/2602.06036.
        expected_acceptance_lengths={
            "mt-bench": 4.24,
            "humaneval": 6.50,
            "gsm8k": 6.54 * 0.975,
        },
        chat_template_kwargs={"enable_thinking": False},
        use_mrv2=use_mrv2,
    )


@single_gpu_only
@pytest.mark.parametrize(
    ("config", "use_mrv2"),
    [
        pytest.param(
            QWEN3_DFLASH,
            False,
            id="qwen3-mrv1",
        ),
        pytest.param(
            QWEN3_DFLASH,
            True,
            id="qwen3-mrv2",
        ),
        pytest.param(
            LAGUNA_DFLASH_NVFP4,
            True,
            id="laguna-nvfp4-mrv2",
        ),
    ],
)
def test_dflash_correctness(
    monkeypatch: pytest.MonkeyPatch,
    config: DFlashCorrectnessConfig,
    use_mrv2: bool,
    vllm_runner,
):
    """Guard GSM8K accuracy and batched acceptance length for DFlash models."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_mrv2 else "0")
    if config.disable_flashinfer_sampler:
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    with vllm_runner(
        config.model,
        block_size=None,
        trust_remote_code=True,
        speculative_config={
            "method": "dflash",
            "model": config.draft_model,
            "num_speculative_tokens": config.num_speculative_tokens,
            "max_model_len": config.max_model_len,
        },
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization=0.85,
        enforce_eager=config.enforce_eager,
        disable_log_stats=False,
        enable_chunked_prefill=None,
        compilation_config=CompilationConfig(),
    ) as spec_runner:
        spec_llm = spec_runner.llm
        results = evaluate_gsm8k_offline(
            spec_llm,
            num_questions=config.num_questions,
            use_chat_completions=config.use_chat_completions,
        )
        accuracy = results["accuracy"]
        acceptance_len = compute_acceptance_len(spec_llm.get_metrics())
        context = (
            f"DFlash target={config.model}, draft={config.draft_model}, MRV2={use_mrv2}"
        )
        print(
            f"{context}: GSM8K accuracy={accuracy:.3f}, "
            f"acceptance_len={acceptance_len:.2f}"
        )

        assert accuracy >= config.expected_accuracy, (
            f"{context}: GSM8K accuracy {accuracy:.3f} is below "
            f"{config.expected_accuracy:.3f}; acceptance_len={acceptance_len:.3f}"
        )
        assert acceptance_len >= config.expected_acceptance_len, (
            f"{context}: acceptance_len {acceptance_len:.3f} is below "
            f"{config.expected_acceptance_len:.3f}; accuracy={accuracy:.3f}"
        )
