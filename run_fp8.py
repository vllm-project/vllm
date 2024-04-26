import argparse

from transformers import AutoTokenizer

from vllm import LLM

choices = ["llama-static", "mistral-static", "mistral-dynamic", "mixtral-static"]

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
    elif args.type == 'mixtral-static':
        model_name = "nm-testing/Mixtral-8x7B-Instruct-v0.1-FP8"
    else:
        raise ValueError(f"--type should be in {choices}")

    model = LLM(model_name,
                enforce_eager=True,
                max_model_len=1024)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = tokenizer.apply_chat_template([{
        "role": "user",
        "content": "What is your name"
    }], tokenize=False, add_generation_prompt=True)
    print(f"----- Prompt: {prompt}")

    outputs = model.generate(prompt)
    print(outputs)
    generation = outputs[0].outputs[0].text
    print(f"----- Generation: {generation}")
