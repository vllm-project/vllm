from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.prompt_adapter.request import PromptAdapterRequest

MODEL_PATH = "bigscience/bloomz-560m"
pa_path = 'stevhliu/bloomz-560m_PROMPT_TUNING_CAUSAL_LM'
pa_path2 = 'swapnilbp/angry_tweet_ptune'


def do_sample(engine):

    prompts = [
        ("Tweet text: I have complaints! Label: ",
         SamplingParams(temperature=0.0, max_tokens=3, stop_token_ids=[3]),
         PromptAdapterRequest("hate_speech", 1, pa_path2, 8)),
        ("Tweet text: I have no problems Label: ",
         SamplingParams(temperature=0.0, max_tokens=3, stop_token_ids=[3]),
         PromptAdapterRequest("hate_speech2", 2, pa_path2, 8)),
        ("Tweet text: I have complaints! Label: ",
         SamplingParams(temperature=0.0, max_tokens=3), None),
        ("Tweet text: I have no problems Label: ",
         SamplingParams(temperature=0.0, max_tokens=3, stop_token_ids=[3]),
         PromptAdapterRequest("complain", 3, pa_path, 8)),
    ]

    request_id = 0
    results = set()
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params, pa_request = prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               prompt_adapter_request=pa_request)
            request_id += 1

        request_outputs = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                results.add(request_output.outputs[0].text)
    return results


def test_multi_prompt_adapters():
    engine_args = EngineArgs(model=MODEL_PATH,
                             max_prompt_adapters=3,
                             enable_prompt_adapter=True,
                             max_prompt_adapter_token=8)
    engine = LLMEngine.from_engine_args(engine_args)
    expected_output = {
        ' quot;I', 'hate speech', 'no complaint', 'not hate speech'
    }
    assert do_sample(engine) == expected_output
