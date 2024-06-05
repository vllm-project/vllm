from huggingface_hub import hf_hub_download

from vllm import LLM, SamplingParams


def run_gguf_inference(model_path):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Create an LLM.
    llm = LLM(model=model_path,
              tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              load_format="gguf",
              quantization="gguf")

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
