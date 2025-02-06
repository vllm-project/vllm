# SPDX-License-Identifier: Apache-2.0

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.lora.request import LoRARequest

LoRA_PATH = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test")


def do_sample(engine):

    prompt_text = "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]"  # noqa: E501

    prompts = [
        (prompt_text,
         SamplingParams(temperature=0.0, max_tokens=100,
                        stop=["[/assistant]"]),
         LoRARequest("sql_test", 1, LoRA_PATH)),
    ]

    request_id = 0
    results = set()
    while prompts or engine.has_unfinished_requests():
        if prompts:
            prompt, sampling_params, lora_request = prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                results.add(request_output.outputs[0].text)
    return results


def test_lora_batch_expansion():
    MODEL_PATH = "meta-llama/Llama-2-7b-hf"
    SPEC_MODEL = "JackFram/llama-68m"

    engine_args = EngineArgs(model=MODEL_PATH,
                             enable_lora=True,
                             enforce_eager=True,
                             disable_log_stats=True,
                             max_num_seqs=60)
    engine = LLMEngine.from_engine_args(engine_args)
    result = do_sample(engine)

    del engine

    # run speculative decoding with batch expansion.
    engine_args = EngineArgs(model=MODEL_PATH,
                             speculative_model=SPEC_MODEL,
                             num_speculative_tokens=3,
                             enable_lora=True,
                             enforce_eager=True,
                             disable_log_stats=True,
                             disable_logprobs_during_spec_decoding=True,
                             speculative_disable_mqa_scorer=True,
                             max_num_seqs=60)
    engine = LLMEngine.from_engine_args(engine_args)
    spec_result = do_sample(engine)

    assert result == spec_result


def test_lora_mqa_scorer():
    MODEL_PATH = "meta-llama/Llama-2-7b-hf"
    SPEC_MODEL = "meta-llama/Llama-2-7b-hf"

    engine_args = EngineArgs(model=MODEL_PATH,
                             enable_lora=True,
                             enforce_eager=True,
                             disable_log_stats=True,
                             max_num_seqs=60)
    engine = LLMEngine.from_engine_args(engine_args)
    result = do_sample(engine)

    del engine

    # run speculative decoding with mqa scorer.
    engine_args = EngineArgs(model=MODEL_PATH,
                             speculative_model=SPEC_MODEL,
                             num_speculative_tokens=3,
                             enable_lora=True,
                             enforce_eager=True,
                             disable_log_stats=True,
                             disable_logprobs_during_spec_decoding=True,
                             speculative_disable_mqa_scorer=False,
                             gpu_memory_utilization=0.8,
                             max_num_seqs=60)
    engine = LLMEngine.from_engine_args(engine_args)
    spec_result = do_sample(engine)

    assert result == spec_result
