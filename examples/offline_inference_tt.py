from typing import List
import os
import sys
import json
import argparse
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm import ModelRegistry

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tt_metal.models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration


def run_inference(
    prompts_json,
    default_max_tokens=128,
    max_seqs_in_batch=32,
    num_repeat_prompts=2,
    measure_perf=False,
    perf_prompt_len=None,
    greedy_sampling=False,  # Option to use greedy decoding instead of top-k/p
):
    # Generation args
    ignore_eos = True if measure_perf else False
    
    if greedy_sampling:
        sampling_params = SamplingParams(max_tokens=default_max_tokens, ignore_eos=ignore_eos, temperature=0.0)
    else:
        sampling_params = SamplingParams(max_tokens=default_max_tokens, ignore_eos=ignore_eos, top_k=10, top_p=0.9, temperature=1.0)

    # Create an LLM.
    ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaModelForGeneration)
    llm = LLM(model="meta-llama/Meta-Llama-3.1-70B", block_size=64, max_num_seqs=max_seqs_in_batch, max_model_len=4096, disable_log_stats=False)

    if not measure_perf:
        # Load prompts from a JSON file
        with open(prompts_json, 'r') as file:
            prompts = json.load(file)
        assert isinstance(prompts, list), "Prompts must be a list of strings"
        if num_repeat_prompts is not None:
            prompts = prompts * num_repeat_prompts
        print("Number of prompts:", len(prompts))
        
        generate_tokens(llm, prompts, sampling_params, print_output=True)
    else:
        print("Note: Ignoring prompts for performance measurement")
        run_inference_perf(llm, sampling_params, max_seqs_in_batch, input_prompt_len=perf_prompt_len)


def run_inference_perf(
    llm : LLM,
    sampling_params,
    max_seqs_in_batch,
    prompts=None,
    input_prompt_len=None  # Used to generate dummy prompts if prompts is None
):
    assert llm.llm_engine.log_stats, "disable_log_stats=False is required for llm to use stat loggers"
    if prompts is not None:
        print("Measuring performance with given prompts")
        prompts = prompts[:max_seqs_in_batch]  # Only run a single batch for performance measurement
    else:
        assert input_prompt_len is not None, "input_prompt_len is required to generate dummy prompts"
        print("Measuring performance with dummy prompts of length", input_prompt_len)
        prompt_token_ids = [[0]*input_prompt_len]*max_seqs_in_batch  # dummy prompts
    sampling_params = sampling_params[:max_seqs_in_batch] if isinstance(sampling_params, list) else sampling_params
    
    # Set an arbitrary max_tokens to simulate generating multiple tokens consecutively
    sampling_params.max_tokens = 33  # 1 prefill output token + 32 decode output tokens
    
    # Compile run
    print("Starting compile run")
    generate_tokens(llm, prompts, sampling_params, prompt_token_ids, print_output=False)
    print("Finished compile run")
    llm.llm_engine.stat_loggers['global'].reset()  # Reset stats before inference run

    # Inference runs
    print("Starting inference runs")
    N_warmup = 1
    N_inference = 5
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:  # Reset stats after warmup
            llm.llm_engine.stat_loggers['global'].reset()
        generate_tokens(llm, prompts, sampling_params, prompt_token_ids, print_output=False)
    print("Finished inference runs")

    # Collect stats
    ttft = llm.llm_engine.stat_loggers['global'].time_to_first_token.avg
    tpot = llm.llm_engine.stat_loggers['global'].time_per_output_token.avg
    print(f"Average time to first token (batch): {ttft} s")
    print(f"Average decode throughput: {1/tpot} t/s/u")
    

def generate_tokens(llm : LLM, prompts, sampling_params, prompt_token_ids=None, print_output=True):
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params, prompt_token_ids)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if print_output:
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_json", type=str, default="tt_metal/prompts.json", help="Path to JSON file containing prompts")
    parser.add_argument("--measure_perf", action="store_true", help="Measure performance")
    parser.add_argument("--perf_prompt_len", type=int, default=128, help="Length of dummy prompts for performance measurement")
    parser.add_argument("--greedy_sampling", action="store_true", help="Use greedy decoding instead of top-k/p")
    args = parser.parse_args()

    run_inference(
        args.prompts_json,
        measure_perf=args.measure_perf,
        perf_prompt_len=args.perf_prompt_len,
        greedy_sampling=args.greedy_sampling
    )
