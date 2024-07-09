from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest

MODEL_PATH = "meta-llama/Llama-2-7b-hf"
pa_path = snapshot_download(repo_id="swapnilbp/llama_tweet_ptune")
lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")


def do_sample(engine):

    prompt_text = "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]"  # noqa: E501

    # first prompt with a prompt adapter and second without adapter
    prompts = [
        (prompt_text,
         SamplingParams(temperature=0.0, max_tokens=100,
                        stop=["[/assistant]"]),
         PromptAdapterRequest("hate_speech", 1, pa_path,
                              8), LoRARequest("sql_test", 1, lora_path)),
        (prompt_text,
         SamplingParams(temperature=0.0, max_tokens=100,
                        stop=["[/assistant]"]), None,
         LoRARequest("sql_test", 1, lora_path)),
    ]

    request_id = 0
    results = set()
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params, pa_request, lora_request = prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               prompt_adapter_request=pa_request,
                               lora_request=lora_request)
            request_id += 1

        request_outputs = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                results.add(request_output.outputs[0].text)
    return results


def test_lora_prompt_adapter():
    engine_args = EngineArgs(model=MODEL_PATH,
                             enable_prompt_adapter=True,
                             enable_lora=True,
                             max_num_seqs=60,
                             max_prompt_adapter_token=8)
    engine = LLMEngine.from_engine_args(engine_args)
    result = do_sample(engine)

    expected_output = {
        "  SELECT icao FROM table_name_74 WHERE airport = 'lilongwe international airport' "  # noqa: E501
    }
    assert result == expected_output
