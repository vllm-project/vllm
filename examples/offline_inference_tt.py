from typing import List
import os
import sys
import json
import argparse

from vllm import LLM, SamplingParams
from vllm import ModelRegistry

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tt_metal.models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration


def run_inference(prompts_json, default_max_tokens=128, max_seqs_in_batch=32, num_repeat_prompts=2, measure_perf=False):
    # Generation args
    ignore_eos = True if measure_perf else False
    
    # Load prompts from a JSON file
    with open(prompts_json, 'r') as file:
        prompts = json.load(file)
    assert isinstance(prompts, list), "Prompts must be a list of strings"
    if num_repeat_prompts is not None:
        prompts = prompts * num_repeat_prompts
    print("Number of prompts:", len(prompts))
    sampling_params = SamplingParams(max_tokens=default_max_tokens, ignore_eos=ignore_eos)

    # Create an LLM.
    ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaModelForGeneration)
    llm = LLM(model="meta-llama/Meta-Llama-3.1-70B", block_size=64, max_num_seqs=max_seqs_in_batch, max_model_len=4096)

    if measure_perf:
        # TODO: Double check how many different seq lengths need to be compiled for decode
        print("Starting compile run")
        generate_tokens(llm, prompts[:max_seqs_in_batch], sampling_params[:max_seqs_in_batch], print_output=False)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_json", type=str, default="tt_metal/prompts.json", help="Path to JSON file containing prompts")
    args = parser.parse_args()

    run_inference(args.prompts_json)
