# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm import LLM, EngineArgs, LLMEngine, SamplingParams
from vllm.control_vectors.request import ControlVectorRequest

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
cv_path_happy = "yuu-biz/qwen-cv-example/happy_vector_qwen.gguf"
cv_path_spanish = "yuu-biz/qwen-cv-example/english_spanish_vector_qwen.gguf"


@pytest.fixture
def requests():
    prompt_text = "Write a story about a dog:"  # noqa: E501

    return [
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("spanish", 1, cv_path_spanish, 2.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("spanish", 2, cv_path_spanish, 1.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            None,
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("spanish", 3, cv_path_spanish, -1.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("spanish", 4, cv_path_spanish, -2.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("happy", 5, cv_path_happy, 2.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("happy", 6, cv_path_happy, 1.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            None,
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("happy", 7, cv_path_happy, -1.0),
        ),
        (
            prompt_text,
            SamplingParams(temperature=0.0,
                           max_tokens=100,
                           stop=["[/assistant]"]),
            ControlVectorRequest("happy", 8, cv_path_happy, -2.0),
        ),
    ]


def do_sample(engine, prompts):
    request_id = 0
    results = set()
    list_results = []
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params, cv_request = prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               control_vector_request=cv_request)
            request_id += 1

        request_outputs = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                results.add(request_output.outputs[0].text)
    list_results = [{
        "request_id": request_output.request_id,
        "generation": request_output.outputs[0].text,
    } for request_output in request_outputs]
    return results, list_results


def test_cv_adapter(requests):
    engine_args = EngineArgs(
        model=MODEL_PATH,
        enable_control_vector=True,
        max_control_vectors=10,
        max_num_seqs=20,
        gpu_memory_utilization=0.4,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    result, list_result = do_sample(engine, requests)
    assert len(result) == 9


def test_offline_inferance(requests):
    llm = LLM(
        model=MODEL_PATH,
        enable_control_vector=True,
        max_control_vectors=10,
        max_num_seqs=20,
        gpu_memory_utilization=0.4,
    )

    results = []
    for request in requests:
        prompt, sampling_param, cv_request = request
        result = llm.generate(prompt,
                              sampling_param,
                              control_vector_request=cv_request)
        results.append({
            "prompt":
            prompt,
            "generation":
            result[0].outputs[0].text,
            "cv_name":
            cv_request.control_vector_name if cv_request else None,
            "scale":
            cv_request.scale_factor if cv_request else None
        })
    assert len(results) == 10


if __name__ == "__main__":
    test_cv_adapter(requests())
