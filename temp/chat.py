import json
from typing import List, Tuple
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from transformers import AutoTokenizer


N = 2048


def get_prompts(file_path="./mt_bench.jsonl") -> List[Tuple[str, SamplingParams]]:
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    
    prompts = []
    for sample in list_data_dict:
        prompts += sample["turns"]
    
    return [(prompt, SamplingParams(max_tokens=N)) for prompt in prompts]


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
                print("\nPROMPT:")
                print(request_output.prompt)

                text = request_output.outputs[0].text
                num_tokens = len(tokenizer.tokenize(text))
                print(f"\nOUTPUT: ({num_tokens} tokens)")
                print(text)


def main():
    # context length 2048
    model = "facebook/opt-125m"
    args = EngineArgs(model=model, max_model_len=N)

    engine = LLMEngine.from_engine_args(args)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompts = get_prompts()
    process_requests(engine, prompts, tokenizer)


if __name__ == "__main__":
    main()