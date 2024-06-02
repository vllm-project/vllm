import json
from typing import List, Tuple
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from transformers import AutoTokenizer
from huggingface_hub import login


login(token='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')
MAX_GEN_TOKENS = 1000


def get_prompt(model, file_path="./prompts.json") -> List[Tuple[str, SamplingParams]]:
    with open(file_path, "r") as f:
        prompts = json.load(f)
    
    prompt = prompts[model]
    return [(prompt, SamplingParams(max_tokens=MAX_GEN_TOKENS))]


def get_long_prompt(file_path="./paxos_paper.txt", count=1) -> Tuple[str, SamplingParams]:
    # this file is 4060 tokens
    with open(file_path, "r") as f:
        prompt = f.read()

    return [(prompt, SamplingParams(
                temperature=1,
                min_tokens=100,
                max_tokens=MAX_GEN_TOKENS
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

                text = request_output.outputs[0].text
                num_tokens = len(tokenizer.tokenize(text))
                print(f"\nOUTPUT: ({num_tokens} tokens)")
                print(text, "\n")
                print(request_output.outputs)


def main():
    model = "meta-llama/Llama-2-13b-chat-hf"
    model = "lmsys/vicuna-7b-v1.5"
    model = "mistralai/Mixtral-8x7B-Instruct-v0.1" # TODO
    model = "mistralai/Mistral-7B-Instruct-v0.2" # llama under the hood
    model = "tiiuae/falcon-7b-instruct" # alibi is garbage
    model = "bigscience/bloom-7b1"
    args = EngineArgs(
        model=model,
        enforce_eager=True,
        block_size=16,
        use_attention_sinks=True
    )

    engine = LLMEngine.from_engine_args(args)
    print("max model len", engine.model_config.max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = get_prompt(model)
    # prompts = get_long_prompt()
    process_requests(engine, prompts, tokenizer)


if __name__ == "__main__":
    main()