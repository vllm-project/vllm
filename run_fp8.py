import argparse

from transformers import AutoTokenizer

from vllm import LLM

choices = [
    "llama-static", "mistral-static", "mistral-dynamic", "mixtral-static",
    "opt-static", "tinyllama-fp16", "qwen-fp16"
]

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=choices)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.type == "llama-static":
        model_name = "nm-testing/Meta-Llama-3-8B-Instruct-FP8"
    elif args.type == "mistral-static":
        model_name = "nm-testing/mistral-fp8-static"
    elif args.type == "mistral-dynamic":
        model_name = "nm-testing/mistral-fp8-dynamic"
    elif args.type == "opt-static":
        model_name = "nm-testing/opt-125m-fp8-static"
    elif args.type == 'tinyllama-fp16':
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    elif args.type == 'qwen-fp16':
        model_name = "Qwen/CodeQwen1.5-7B-Chat"
    else:
        raise ValueError(f"--type should be in {choices}")

    model = LLM(model_name,
                enforce_eager=True,
                max_model_len=1024,
                quantization="fp8")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": "What is open source software?"
            }],
            tokenize=False,
            add_generation_prompt=True)
    else:
        prompt = "The best thing about"
    print(f"----- Prompt: {prompt}")

    outputs = model.generate(prompt)
    generation = outputs[0].outputs[0].text
    print(f"----- Generation: {generation}")
