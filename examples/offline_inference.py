import os

os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
os.environ["VLLM_RAY_DISABLE_LOG_TO_DRIVER"] = "1"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

if __name__ == "__main__":

    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

    model_name = "opensourcerelease/DeepSeek-R1-bf16"
    model_name = "deepseek-ai/DeepSeek-R1"
    model_name = "/data/models/DeepSeek-R1"
    # model_name = "/data/models/DeepSeek-R1-bf16"
    # model_name = "/data/models/DeepSeek-R1-bf16-small/"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    # model_name = "meta-llama/Llama-3.2-1B"

    # Create an LLM.
    llm = LLM(model=model_name,
            device="hpu",
            dtype="bfloat16",
            #   load_format="dummy",
            tensor_parallel_size=8,
            trust_remote_code=True,
            max_model_len=1024)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
