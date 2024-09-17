from typing import List
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tt_metal.models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration

from vllm import LLM, SamplingParams
from vllm import ModelRegistry


def main():
    # Generation args
    ignore_eos = True
    max_num_seqs = 32  # max sequences in a single batch
    
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 16  # 64 prompts
    sampling_params = [
        SamplingParams(max_tokens=32, ignore_eos=ignore_eos),
        SamplingParams(max_tokens=32, ignore_eos=ignore_eos),
        SamplingParams(max_tokens=32, ignore_eos=ignore_eos),
        SamplingParams(max_tokens=32, ignore_eos=ignore_eos),
    ] * 16

    # Create an LLM.
    ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaModelForGeneration)
    llm = LLM(model="meta-llama/Meta-Llama-3.1-70B", block_size=64, max_num_seqs=max_num_seqs)

    # TODO: Double check how many different seq lengths need to be compiled for decode
    print("Starting compile run")
    generate_tokens(llm, prompts[:max_num_seqs], sampling_params[:max_num_seqs], print_output=False)
    print("Finished compile run")

    print("Starting inference run")
    generate_tokens(llm, prompts, sampling_params)
    print("Finished inference run")
    

def generate_tokens(llm : LLM, prompts, sampling_params : List[SamplingParams], print_output=True):
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if print_output:
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            

if __name__ == "__main__":
    main()
