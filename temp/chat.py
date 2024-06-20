import json
import math
from typing import List, Tuple
import os
from transformers import AutoTokenizer
from huggingface_hub import login
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput


# login(token=os.environ.get("HF_TOKEN"))

MAX_GEN_TOKENS = 500
beam_search_params = SamplingParams(
    temperature=0,
    min_tokens=100,
    max_tokens=150,
    use_beam_search=True,
    n=3
)


def get_prompt(model, file_path="./prompts.json", magic_word=True) -> List[Tuple[str, SamplingParams]]:
    with open(file_path, "r") as f:
        prompts = json.load(f)
    
    prompt = prompts[model]
    if magic_word:
        prompt = (
            "Remember: my favorite color is mint green. "
            "Here is a Harry Potter excerpt: " + prompt + 
            " First, summarize this excerpt. Then, print my favorite color AFTER the summary."
        )
    return [(prompt, SamplingParams(min_tokens=100, max_tokens=500, temperature=0.5)) for _ in range(4)]


def get_long_prompt(file_path="./paxos-paper.txt", count=1) -> Tuple[str, SamplingParams]:
    # this file is 4060 tokens
    with open(file_path, "r") as f:
        prompt = f.read()

    prompt = "Remember: the magic word is apple. " + prompt + " Then, print the magic word given earlier."
    return [(prompt, SamplingParams(
                temperature=1,
                min_tokens=100,
                max_tokens=MAX_GEN_TOKENS,
            )) for _ in range(count)]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]],
                     tokenizer):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                # print("\nPROMPT:")
                # print(request_output.prompt)

                out = request_output.outputs[0]
                num_tokens = len(out.token_ids)
                cum_logprob = out.cumulative_logprob
                avg_logprob = cum_logprob / num_tokens
                print(f"\nOUTPUT: ({num_tokens} tokens)")
                print(out.text, "\n")
                print("Output stats:", cum_logprob, avg_logprob, out.finish_reason, f"isnan={math.isnan(cum_logprob)}")


def main():
    model = "meta-llama/Llama-2-13b-chat-hf"
    model = "mistralai/Mixtral-8x7B-Instruct-v0.1" # TODO
    model = "tiiuae/falcon-7b-instruct"
    model = "bigscience/bloom-7b1"
    model = "mistralai/Mistral-7B-Instruct-v0.2" # llama under the hood
    model = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = "mosaicml/mpt-7b-chat"
    model = "lmsys/vicuna-7b-v1.5"
    args = EngineArgs(
        model=model,
        enforce_eager=True,
        block_size=16,
        dtype="bfloat16",
        use_attention_sinks=True
    )

    engine = LLMEngine.from_engine_args(args)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # prompts = get_prompt(model, magic_word=True)
    prompts = get_long_prompt()
    process_requests(engine, prompts, tokenizer)

    # model = engine.model_executor.driver_worker.model_runner.model
    # for module_name, module in model.named_modules(remove_duplicate=False):
    #     print(module_name)
        # parts = module_name.split(".")
        # if len(parts) != 5: continue
        # if parts[-1] == "attn":
        #     print(module_name)
        #     print(dir(module))


if __name__ == "__main__":
    main()