"""Compare the outputs of HF and vLLM for Whisper models using greedy sampling.

Run `pytest tests/models/encoder_decoder/audio/test_whisper.py`.
"""
from typing import Optional

import pytest

from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset

from ....utils import fork_new_process_for_each_test, multi_gpu_test

PROMPTS = [
    {
        "prompt":
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {
            "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
        },
    },
    {  # Test explicit encoder/decoder prompt
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {
                "audio": AudioAsset("winning_call").audio_and_sample_rate,
            },
        },
        "decoder_prompt":
        "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    }
]

EXPECTED = {
    "openai/whisper-medium": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its fleece was quite as"
        " slow, and everywhere that Mary went the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez swung on the line"
        " down the left field line for Obeyshev. Here comes Joy. Here is"
        " Jorgen at third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh"
        " my."
    ],
    "openai/whisper-large-v3": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its feet were quite as"
        " slow, and everywhere that Mary went, the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez. Swung on the line."
        " Now the left field line for a base hit. Here comes Joy. Here is"
        " Junior to third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh,"
        " my."
    ]
}


def run_test(
    model: str,
    *,
    enforce_eager: bool,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
) -> None:
    prompt_list = PROMPTS * 10
    expected_list = EXPECTED[model] * 10

    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
        enforce_eager=enforce_eager,
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=200,
    )

    outputs = llm.generate(prompt_list, sampling_params)

    for output, expected in zip(outputs, expected_list):
        print(output.outputs[0].text)
        assert output.outputs[0].text == expected


@fork_new_process_for_each_test
@pytest.mark.parametrize("model",
                         ["openai/whisper-medium", "openai/whisper-large-v3"])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_models(model, enforce_eager) -> None:
    run_test(model, enforce_eager=enforce_eager, tensor_parallel_size=1)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model", ["openai/whisper-large-v3"])
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("distributed_executor_backend", ["ray", "mp"])
def test_models_distributed(model, enforce_eager,
                            distributed_executor_backend) -> None:
    run_test(model,
             enforce_eager=enforce_eager,
             tensor_parallel_size=2,
             distributed_executor_backend=distributed_executor_backend)
