import pytest

import vllm
from tests.utils import multi_gpu_test
from vllm.sampling_params import SamplingParams

MODELS = ["Snowflake/Llama-3.1-SwiftKV-8B-Instruct-FP8"]
CONVERSATIONS = [
    [{
        "role": "user",
        "content": "Hello!"
    }],
    [{
        "role": "user",
        "content": "Who is the president of the United States?"
    }],
    [{
        "role": "user",
        "content": "What is the capital of France?"
    }],
    [{
        "role": "user",
        "content": "What is the future of AI?"
    }],
]
EXPECTED_OUTPUTS = [
    "Hello! How can I assist you today?",
    "As of my cut-off knowledge in December 2023, the President of the United "
    "States is Joe",
    "The capital of France is Paris.",
    "The future of AI is vast and rapidly evolving, with numerous potential "
    "developments and applications on the horizon.",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("enforce_eager", [True, False])
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@multi_gpu_test(num_gpus=2)
def test_model(model, enforce_eager, tensor_parallel_size) -> None:
    llm = vllm.LLM(
        model,
        enforce_eager=enforce_eager,
        enable_chunked_prefill=True,
        tensor_parallel_size=tensor_parallel_size,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

    for idx, conversation in enumerate(CONVERSATIONS):
        outputs = llm.chat(
            conversation,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        assert outputs[0].outputs[0].text == EXPECTED_OUTPUTS[idx]
