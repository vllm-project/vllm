# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.models.utils import check_logprobs_close
from tests.utils import single_gpu_only
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.distributed import cleanup_dist_env_and_memory

from .._correctness import check_mtp_correctness


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "expected_accuracy_threshold"],
    [
        (
            ("mtp", "Qwen/Qwen3.5-0.8B-Base", 1, None),
            False,
            0.20,
        ),  # hybrid + MTP, ref: ~34%-35%
    ],
    ids=["qwen3_5-hybrid"],
)
@single_gpu_only
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int, str | None],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    vllm_runner,
):
    check_mtp_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        vllm_runner,
    )


@single_gpu_only
def test_qwen3_5_mtp_prefix_cache_reuses_last_safe_block(
    monkeypatch: pytest.MonkeyPatch,
):
    """Offloading uses the successor-proven MTP boundary without accuracy loss."""
    aligned_page_size = 544
    prompt_len = 2 * aligned_page_size + 1
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=32,
        logprobs=5,
    )

    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")
        llm = LLM(
            model="Qwen/Qwen3.5-0.8B-Base",
            trust_remote_code=True,
            enable_prefix_caching=True,
            max_model_len=2048,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 1,
                "max_model_len": 2048,
            },
            kv_transfer_config={
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": 512 << 20,
                },
            },
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        assert llm.llm_engine.vllm_config.cache_config.prefix_match_unit is None

        tokenizer = llm.get_tokenizer()
        source = (
            "The following context is intentionally repeated to exercise prefix "
            "caching. " * 300
        ) + "Explain why deterministic inference should be reproducible."
        prompt_token_ids = tokenizer.encode(source)[-prompt_len:]
        assert len(prompt_token_ids) == prompt_len
        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

        cold_output = llm.generate([prompt], sampling_params)[0]
        warm_output = llm.generate([prompt], sampling_params)[0]

        assert cold_output.num_cached_tokens == 0
        max_safe_hit = 2 * aligned_page_size
        # An unconditional one-page EAGLE drop would report only 544 tokens.
        assert warm_output.num_cached_tokens == max_safe_hit

        cold_completion = cold_output.outputs[0]
        warm_completion = warm_output.outputs[0]
        assert cold_completion.token_ids == warm_completion.token_ids
        check_logprobs_close(
            outputs_0_lst=[
                (
                    list(cold_completion.token_ids),
                    cold_completion.text,
                    cold_completion.logprobs,
                )
            ],
            outputs_1_lst=[
                (
                    list(warm_completion.token_ids),
                    warm_completion.text,
                    warm_completion.logprobs,
                )
            ],
            name_0="cold_mtp",
            name_1="warm_mtp",
            always_check_logprobs=True,
        )

        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()
