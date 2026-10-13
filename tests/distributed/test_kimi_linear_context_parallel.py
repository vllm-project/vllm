# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from pathlib import Path

import pytest

from tests.conftest import VllmRunner
from tests.utils import create_new_process_for_each_test
from vllm import SamplingParams, TokensPrompt
from vllm.config import CUDAGraphMode
from vllm.transformers_utils.configs.kimi_k3 import KimiK3Config

MODEL = "moonshotai/Kimi-Linear-48B-A3B-Instruct"


def _make_tiny_overrides() -> dict:
    linear_attn_config = {
        "full_attn_layers": [4],
        "head_dim": 128,
        "kda_layers": [1, 2, 3],
        "num_heads": 8,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
    }

    return {
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "vocab_size": 256,
        "hidden_size": 256,
        "head_dim": 32,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "num_experts": None,
        "num_experts_per_token": None,
        "num_shared_experts": 0,
        "q_lora_rank": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "linear_attn_config": linear_attn_config,
        "max_position_embeddings": 512,
        "model_max_length": 512,
    }


def _run_tiny_model(
    vllm_runner: type[VllmRunner], dcp_size: int
) -> list[tuple[list[int], list[float]]]:
    with vllm_runner(
        model_name=MODEL,
        skip_tokenizer_init=True,
        load_format="dummy",
        hf_overrides=_make_tiny_overrides(),
        tensor_parallel_size=2,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.25,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    ) as runner:
        lengths = [1, 32, 33, 129]
        prompts = [
            TokensPrompt(
                prompt_token_ids=[
                    3 + ((request_idx + token_idx) % 251) for token_idx in range(length)
                ]
            )
            for request_idx, length in enumerate(lengths)
        ]
        outputs = runner.llm.generate(
            prompts,
            SamplingParams(
                temperature=0,
                max_tokens=8,
                seed=0,
                ignore_eos=True,
                logprobs=20,
            ),
            use_tqdm=False,
        )

    results = []
    for request_output in outputs:
        completion = request_output.outputs[0]
        token_ids = list(completion.token_ids)
        assert completion.logprobs is not None
        selected_logprobs = [
            step_logprobs[token_id].logprob
            for token_id, step_logprobs in zip(token_ids, completion.logprobs)
        ]
        results.append((token_ids, selected_logprobs))
    return results


@create_new_process_for_each_test()
@pytest.mark.distributed(num_gpus=2)
def test_kimi_linear_dcp_tiny(
    vllm_runner: type[VllmRunner],
    num_gpus_available: int,
) -> None:
    if num_gpus_available < 2:
        pytest.skip("Need at least 2 GPUs")

    baseline = _run_tiny_model(vllm_runner, dcp_size=1)
    dcp = _run_tiny_model(vllm_runner, dcp_size=2)

    assert [tokens for tokens, _ in dcp] == [tokens for tokens, _ in baseline]
    logprob_drifts = []
    for (_, baseline_logprobs), (_, dcp_logprobs) in zip(baseline, dcp):
        for baseline_logprob, dcp_logprob in zip(baseline_logprobs, dcp_logprobs):
            assert math.isfinite(baseline_logprob)
            assert math.isfinite(dcp_logprob)
            logprob_drifts.append(abs(baseline_logprob - dcp_logprob))

    assert max(logprob_drifts) <= 1e-2


def _make_tiny_k3_config(model_dir: Path) -> str:
    config = KimiK3Config(
        text_config=_make_tiny_overrides(),
        vision_config={
            "vt_num_attention_heads": 2,
            "vt_num_hidden_layers": 1,
            "vt_hidden_size": 32,
            "vt_intermediate_size": 64,
            "qkv_hidden_size": 48,
        },
        architectures=["KimiK3ForConditionalGeneration"],
    )
    config.save_pretrained(model_dir)
    return str(model_dir)


def _run_k3_partial_prefix_reuse(
    vllm_runner: type[VllmRunner], model_name: str, dcp_size: int
) -> tuple[list[int], list[float], int]:
    block_size = 256
    common_prefix = [3 + (token_idx % 251) for token_idx in range(block_size)]
    prime_prompt = TokensPrompt(prompt_token_ids=common_prefix + [17])
    replay_prompt = TokensPrompt(prompt_token_ids=common_prefix + [18, 19])
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=8,
        seed=0,
        ignore_eos=True,
        logprobs=20,
    )

    with vllm_runner(
        model_name=model_name,
        skip_tokenizer_init=True,
        load_format="dummy",
        language_model_only=True,
        tensor_parallel_size=2,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        seed=0,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.85,
        block_size=block_size,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
    ) as runner:
        assert (
            runner.llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
            == CUDAGraphMode.FULL_AND_PIECEWISE
        )
        runner.llm.generate(
            [prime_prompt],
            sampling_params,
            use_tqdm=False,
        )
        replay_output = runner.llm.generate(
            [replay_prompt],
            sampling_params,
            use_tqdm=False,
        )[0]

    completion = replay_output.outputs[0]
    token_ids = list(completion.token_ids)
    assert len(token_ids) == sampling_params.max_tokens
    assert completion.logprobs is not None
    assert len(completion.logprobs) == sampling_params.max_tokens
    selected_logprobs = [
        step_logprobs[token_id].logprob
        for token_id, step_logprobs in zip(token_ids, completion.logprobs)
    ]
    return token_ids, selected_logprobs, replay_output.num_cached_tokens


@create_new_process_for_each_test()
@pytest.mark.distributed(num_gpus=2)
def test_kimi_k3_dcp_partial_prefix_reuse(
    vllm_runner: type[VllmRunner],
    num_gpus_available: int,
    tmp_path: Path,
) -> None:
    if num_gpus_available < 2:
        pytest.skip("Need at least 2 GPUs")

    model_name = _make_tiny_k3_config(tmp_path / "tiny-kimi-k3")
    baseline_tokens, baseline_logprobs, baseline_cached = _run_k3_partial_prefix_reuse(
        vllm_runner, model_name, dcp_size=1
    )
    dcp_tokens, dcp_logprobs, dcp_cached = _run_k3_partial_prefix_reuse(
        vllm_runner, model_name, dcp_size=2
    )

    assert baseline_cached == 256
    assert dcp_cached == 256
    assert dcp_tokens == baseline_tokens
    assert len(dcp_logprobs) == len(baseline_logprobs) == 8
    logprob_drifts = []
    for baseline_logprob, dcp_logprob in zip(baseline_logprobs, dcp_logprobs):
        assert math.isfinite(baseline_logprob)
        assert math.isfinite(dcp_logprob)
        logprob_drifts.append(abs(baseline_logprob - dcp_logprob))
    assert max(logprob_drifts) <= 1e-2
