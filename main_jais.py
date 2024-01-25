# coding=utf-8
# !/usr/bin/python3.8
"""
__synopsis__    : Added support for Jais into vLLM.
__description__ : This script contains code to run Jais model using vLLM.
__project__     : 
__author__      : Samujjwal Ghosh <samujjwal.ghosh@core42.ai>, Samta Kamboj <samta.kamboj@core42.ai>
__version__     : "0.1"
__date__        : "25 Jan, 2024"
"""

from vllm import LLM, SamplingParams

def load_model_vllm(model_path, dtype="float16", tensor_parallel_size=1,):
    print(f'Loading model from path: [{model_path}]')
    llm = LLM(model=model_path, trust_remote_code=True, 
            dtype=dtype, 
            tensor_parallel_size=tensor_parallel_size,
            #   enforce_eager=True,
            #   gpu_memory_utilization=0.95,
            swap_space=16,
            # block_size=32,
            )
    
    return llm

def main(n_gpus=1, model_path='core42/jais-30b-chat-v1'):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The future of AI is",
    ]    

    # load the model
    llm = load_model_vllm(model_path, tensor_parallel_size=n_gpus)

    # set the params for generations
    sampling_params = SamplingParams(
        n=1, 
        temperature=0.7, 
        top_p=0.7, 
        max_tokens=200, 
        frequency_penalty=0.2, 
        presence_penalty=0.2,
    )

    # generate the outputs
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, \t\t Generated text: {generated_text!r}")


if __name__ == "__main__":
    """
    NOTE:
    1. Tested only with Jais `13b` and `30b` models
    2. Works only with vLLM "0.2.1-post1" tag
    3. `13b` can only be used on a single GPU due to non-divisibility of FF layer dim
    4. `30b` can only be used either on a single GPU or two GPUs due to non-divisibility of FF layer dim
    5. Need to modify the config.json file to add extra attributes
    """
    main(1, model_path = 'core42/jais-30b-chat-v1')