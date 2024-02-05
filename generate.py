from vllm import LLM, SamplingParams
import torch

prompts = [
            "Hello, my name is",
                "The president of the United States is",
                    "The capital of France is",
                        "The future of AI is",
                        ]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


llm = LLM(model="mistralai/Mistral-7B-v0.1", enforce_eager=True, dtype=torch.float16)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

