from vllm import LLM, SamplingParams

import argparse
import os

model_path = "/data/models/DeepSeek-R1/"
# model_path = "deepseek-ai/DeepSeek-V2-Lite"

# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=model_path, help="The model path.")
#parser.add_argument("--model", type=str, default="/data/models/DeepSeek-R1/", help="The model path.")
parser.add_argument("--tokenizer", type=str, default=model_path, help="The model path.")
#parser.add_argument("--model", type=str, default="/data/models/DeepSeek-R1-bf16-small/", help="The model path.")
#parser.add_argument("--tokenizer", type=str, default="opensourcerelease/DeepSeek-R1-bf16", help="The model path.")
parser.add_argument("--tp_size", type=int, default=8, help="The number of threads.")
args = parser.parse_args()

os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
# os.environ["VLLM_RAY_DISABLE_LOG_TO_DRIVER"] = "1"
# os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
os.environ["VLLM_MOE_N_SLICE"] = "8"
os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"

if __name__ == "__main__":

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, max_tokens=50)
    model = args.model
    if args.tp_size == 1:
        llm = LLM(
            model=model, 
            tokenizer=args.tokenizer,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=1024,
        )
    else:
        llm = LLM(
            model=model, 
            tokenizer=args.tokenizer,
            tensor_parallel_size=args.tp_size,
            distributed_executor_backend='mp',
            trust_remote_code=True,
            max_model_len=1024,
            dtype="bfloat16",
        )

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")