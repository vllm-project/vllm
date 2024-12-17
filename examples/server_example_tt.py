import argparse
import os
import sys
import runpy

from vllm import ModelRegistry

# Import and register models from tt-metal
from models.demos.t3000.llama2_70b.tt.generator_vllm import TtLlamaForCausalLM
from models.demos.llama3.tt.generator_vllm import TtMllamaForConditionalGeneration
ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaForCausalLM)
ModelRegistry.register_model("TTMllamaForConditionalGeneration", TtMllamaForConditionalGeneration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_modal", action="store_true", help="Run multi-modal inference with Llama3.2-11b")
    args = parser.parse_args()
    
    if args.multi_modal:
        model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        if os.environ.get("MESH_DEVICE") is None:
            os.environ["MESH_DEVICE"] = "N300"
        else:
            assert os.environ["MESH_DEVICE"] in ["N300", "T3K_LINE"], "Invalid MESH_DEVICE for multi-modal inference"
        sys.argv.remove("--multi_modal")  # remove the flag for the API server
    else:
        model = "meta-llama/Meta-Llama-3.1-70B"
        os.environ["MESH_DEVICE"] = "T3K_RING"
    
    sys.argv.extend([
        "--model", model,
        "--block_size", "64",
        "--max_num_seqs", "32",
        "--max_model_len", "131072",
        "--max_num_batched_tokens", "131072",
        "--num_scheduler_steps", "10",
    ])
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()