import argparse

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Use model from www.modelscope.cn
# FROM_MODELSCOPE=True python examples/offline_inference.py \
# --model="damo/nlp_gpt2_text-generation_english-base" \
# --revision="v1.0.0" --from_modelscope
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using the LLM class directly")
    parser.add_argument("--model",
                        type=str,
                        default="facebook/opt-125m",
                        help="name or path of the huggingface model to use")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="the specific model version to use. It can be a branch "
        "name, a tag name, or a commit id. If unspecified, will use "
        "the default version.")
    args = parser.parse_args()
    # Create an LLM.
    llm = LLM(model=args.model, revision=args.revision)
    # Generate texts from the prompts. The output is a list
    # of RequestOutput objects that contain the prompt,
    # generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
