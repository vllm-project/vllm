# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 PCP + PP generation on four NVIDIA SM90+ GPUs."""

import pytest


def _pcp_pipeline_config(config):
    # Preserve the 40-layer sharing topology with smaller expert/Engram weights.
    for key, value in {
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "engram_num_embeddings": [32768] * len(config.engram_layer_ids),
        "engram_vocab_size": 1000,
    }.items():
        setattr(config, key, value)
    return config


@pytest.mark.distributed(num_gpus=4)
def test_deepseek_v41_pcp_pipeline_generation(vllm_runner, monkeypatch):
    """PP preserves PCP generation across chunked prefill and stage-entry Engram."""
    from tests.models.utils import check_logprobs_close
    from vllm import SamplingParams

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    prompts = [
        {"prompt_token_ids": [1000 + i % 100 for i in range(length)]}
        for length in (7, 607, 129)
    ]
    outputs = []
    for pp_size in (1, 2):
        # The explicit partition applies only to the PP2 run.
        if pp_size == 1:
            monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
        else:
            monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "14,26")
        with vllm_runner(
            "deepseek-ai/DeepSeek-V4.1-Flash",
            hf_overrides=_pcp_pipeline_config,
            load_format="dummy",
            language_model_only=True,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            prefill_context_parallel_size=2,
            pipeline_parallel_size=pp_size,
            distributed_executor_backend="mp",
            attention_config={"backend": "FLASHMLA_SPARSE_DSV41"},
            enable_prefix_caching=False,
            swa_bounded_replay=False,
            async_scheduling=False,
            enforce_eager=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=256,
            max_num_seqs=4,
            max_model_len=1024,
            block_size=64,
        ) as runner:
            result = runner.llm.generate(
                prompts,
                SamplingParams(
                    temperature=0, max_tokens=8, ignore_eos=True, logprobs=5
                ),
            )
            assert [len(item.outputs[0].token_ids) for item in result] == [8] * 3
            outputs.append(
                [
                    (item.outputs[0].token_ids, "", item.outputs[0].logprobs)
                    for item in result
                ]
            )
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="PCP2",
        name_1="PCP2+PP2",
        always_check_logprobs=True,
    )
