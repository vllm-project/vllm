from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)

model_path = "/mnt/vast/home/patrick/mistral_nemo"
# model_path = "mistralai/Mistral-7B-Instruct-v0.3" 
for model in [model_path]:  # or "mistralai/Mistral-7B-Instruct-v0.3"
    llm = LLM(model=model, tokenizer_mode="mistral", max_model_len=8192, load_format="consolidated")
    # llm = LLM(model=model, tokenizer_mode="mistral", max_model_len=8192)
    # llm = LLM(model=model, tokenizer_mode="mistral", max_model_len=8192)
    # llm = LLM(model=model, max_model_len=8192, tokenizer_mode="mistral")

    outputs = llm.generate(prompts, sampling_params)

    print(outputs)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")#!/usr/bin/env python3
