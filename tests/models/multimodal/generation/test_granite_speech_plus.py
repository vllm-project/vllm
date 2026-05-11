# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import pytest
from transformers import AutoModelForSpeechSeq2Seq

from vllm.logprobs import SampleLogprobs

from ....conftest import AudioTestAssets, HfRunner, PromptAudioInput, VllmRunner
from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

HF_AUDIO_PROMPT = (
    "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
    "<|end_of_text|>\n"
    "<|start_of_role|>user<|end_of_role|>"
    "<|audio|>can you transcribe the speech into a written format?"
    "<|end_of_text|>\n"
    "<|start_of_role|>assistant<|end_of_role|>"
)

MODEL_NAME = "ibm-granite/granite-speech-4.1-2b-plus"


def vllm_to_hf_output(
    vllm_output: tuple[list[int], str, SampleLogprobs | None],
) -> tuple[list[int], str, SampleLogprobs | None]:
    output_ids, output_str, out_logprobs = vllm_output
    return output_ids, output_str + "<|end_of_text|>", out_logprobs


def run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    inputs: Sequence[tuple[list[str], PromptAudioInput]],
    model: str,
    *,
    max_model_len: int,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    tensor_parallel_size: int,
    distributed_executor_backend: str | None = None,
):
    with vllm_runner(
        model,
        runner="generate",
        max_model_len=max_model_len,
        max_num_seqs=1,
        dtype=dtype,
        limit_mm_per_prompt={"audio": 1},
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=True,
    ) as vllm_model:
        vllm_outputs_per_case = [
            vllm_model.generate_greedy_logprobs(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                audios=audios,
            )
            for prompts, audios in inputs
        ]

    with hf_runner(model, dtype=dtype, auto_cls=AutoModelForSpeechSeq2Seq) as hf_model:
        hf_processor = hf_model.processor
        eos_token_id = hf_processor.tokenizer.eos_token_id
        hf_outputs_per_case = [
            hf_model.generate_greedy_logprobs_limit(
                prompts,
                max_tokens,
                num_logprobs=num_logprobs,
                audios=[audios],
                eos_token_id=eos_token_id,
            )
            for prompts, audios in inputs
        ]

    for hf_outputs, vllm_outputs in zip(hf_outputs_per_case, vllm_outputs_per_case):
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[vllm_to_hf_output(out) for out in vllm_outputs],
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_model_len", [2048])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [10])
def test_models(
    hf_runner,
    vllm_runner,
    audio_assets: AudioTestAssets,
    dtype: str,
    max_model_len: int,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    audio, sr = audio_assets[0].audio_and_sample_rate
    assert sr == 16000

    run_test(
        hf_runner,
        vllm_runner,
        [([HF_AUDIO_PROMPT], [audio])],
        MODEL_NAME,
        dtype=dtype,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
    )
