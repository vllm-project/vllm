import argparse

from transformers import AutoTokenizer

from vllm import LLM

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["static", "dynamic"])

if __name__ == "__main__":
    args = parser.parse_args()

    if args.type == "static":
        model_name = "nm-testing/mistral-fp8-static"
    elif args.type == "dynamic":
        model_name = "nm-testing/mistral-fp8-dynamic"
    else:
        raise ValueError("--type should be `static` or `dynamic`")

    # tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer_name = model_name

    model = LLM(model_name,
                tokenizer=tokenizer_name,
                enforce_eager=True,
                max_model_len=1024)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    prompt = tokenizer.apply_chat_template([{
        "role": "user",
        "content": "What is your name"
    }], tokenize=False, add_generation_prompt=True)
    print(f"----- Prompt: {prompt}")

    outputs = model.generate(prompt)
    generation = outputs[0].outputs[0].text
    print(f"----- Generation: {generation}")
