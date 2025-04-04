# SPDX-License-Identifier: Apache-2.0
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.control_vectors.request import ControlVectorRequest

MODEL_PATH = "mistralai/Mistral-7B-v0.1"
cv_path = "raywanb/mistral-cv-example/mistral-7b-v0.1-control-vector.gguf"


def do_sample(engine):

    prompt_text = "Finish the sentence. Life is good because:"  # noqa: E501

    # first prompt with a control vector and second without.
    prompts = [(prompt_text,
                SamplingParams(temperature=0.0,
                               max_tokens=100,
                               stop=["[/assistant]"]),
                ControlVectorRequest("chaotic", 1, cv_path, 1.0)),
               (prompt_text,
                SamplingParams(temperature=0.0,
                               max_tokens=100,
                               stop=["[/assistant]"]), None)]

    request_id = 0
    results = set()
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
    return results


def test_cv_adapter():
    engine_args = EngineArgs(model=MODEL_PATH, enable_control_vector=True)
    engine = LLMEngine.from_engine_args(engine_args)
    result = do_sample(engine)
    assert len(result) == 2
