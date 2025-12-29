# from vllm import LLM
# from vllm.sampling_params import SamplingParams

# if __name__ == "__main__":
#     # Initialize LLM
#     llm = LLM(model="google/gemma-3-270m-it")

#     # Create sampling params with activation extraction
#     # Note: gemma-3-270m has fewer layers, so use smaller indices
#     sampling_params = SamplingParams(
#         temperature=0.8,
#         top_p=0.95,
#         max_tokens=50,
#         # extract_activations=True,
#         # activation_layers=[0, 5, 10],  # Extract from layers 0, 5, and 10
#     )

#     # Generate with activation extraction
#     prompts = ["Hello, how are you?"]
#     outputs = llm.generate(prompts, sampling_params)

#     # Access activations
#     for output in outputs:
#         for completion in output.outputs:
#             if completion.activations:
#                 print(f"Available activation layers: {completion.activations.keys()}")
#                 for layer_idx, activation in completion.activations.items():
#                     print(f"Layer {layer_idx} activation shape: {activation.shape}")


from vllm import LLM

if __name__ == "__main__":
    llm = LLM(model="google/gemma-3-270m-it")
    outputs = llm.generate(["Hello"])
    print(outputs[0].outputs[0].text)