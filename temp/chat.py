import json
from typing import List, Tuple
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from transformers import AutoTokenizer
from huggingface_hub import login


login(token='see Notes')
MAX_GEN_TOKENS = 200


def get_chat_prompts(file_path="./mt_bench.jsonl") -> List[Tuple[str, SamplingParams]]:
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    
    prompts = []
    for sample in list_data_dict:
        prompts += sample["turns"]
    
    return [(prompt, SamplingParams(max_tokens=MAX_GEN_TOKENS)) for prompt in prompts]


def get_long_prompt(file_path="./paxos_paper.txt") -> Tuple[str, SamplingParams]:
    # this file is 4060 tokens
    with open(file_path, "r") as f:
        prompt = f.read()

    return [(prompt, SamplingParams(max_tokens=MAX_GEN_TOKENS))]


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
    # context length 4096
    # model = "lmsys/vicuna-7b-v1.5"
    model = "meta-llama/Llama-2-13b-chat-hf"
    args = EngineArgs(
        model=model,
        enforce_eager=True,
        max_model_len=4096,
        block_size=16
    )

    engine = LLMEngine.from_engine_args(args)
    print("max model len", engine.scheduler_config.max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # prompts = get_chat_prompts()
    prompts = get_long_prompt()
    process_requests(engine, prompts, tokenizer)


if __name__ == "__main__":
    main()