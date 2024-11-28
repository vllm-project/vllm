from huggingface_hub import hf_hub_download

from vllm import LLM, SamplingParams


def run_gguf_inference(model_path):
    PROMPT_TEMPLATE = "<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"  # noqa: E501
    system_message = "You are a friendly chatbot who always responds in the style of a pirate."  # noqa: E501
    # Sample prompts.
    prompts = [
        "How many helicopters can a human eat in one sitting?",
        "What's the future of AI?",
    ]
    prompts = [
        PROMPT_TEMPLATE.format(system_message=system_message, prompt=prompt)
        for prompt in prompts
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=128)

    # Create an LLM.
    llm = LLM(model=model_path,
              tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              gpu_memory_utilization=0.95)

    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    filename = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
    model = hf_hub_download(repo_id, filename=filename)
    run_gguf_inference(model)
