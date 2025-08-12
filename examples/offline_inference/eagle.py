# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="./examples/data/gsm8k.jsonl",
    help="downloaded from the eagle repo " \
    "https://github.com/SafeAILab/EAGLE/blob/main/eagle/data/"
)
parser.add_argument("--max_num_seqs", type=int, default=8)
parser.add_argument("--num_prompts", type=int, default=80)
parser.add_argument("--num_spec_tokens", type=int, default=2)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--draft_tp", type=int, default=1)
parser.add_argument("--enforce_eager", action='store_true')
parser.add_argument("--enable_chunked_prefill", action='store_true')
parser.add_argument("--max_num_batched_tokens", type=int, default=2048)
parser.add_argument("--temp", type=float, default=0)

args = parser.parse_args()

print(args)

model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
eagle_dir = "abhigoyal/EAGLE-LLaMA3-Instruct-8B-vllm"

max_model_len = 2048

tokenizer = AutoTokenizer.from_pretrained(model_dir)

if os.path.exists(args.dataset):
    prompts = []
    num_prompts = args.num_prompts
    with open(args.dataset) as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["turns"][0])
else:
    prompts = ["The future of AI is", "The president of the United States is"]

prompts = prompts[:args.num_prompts]
num_prompts = len(prompts)

prompt_ids = [
    tokenizer.apply_chat_template([{
        "role": "user",
        "content": prompt
    }],
                                  add_generation_prompt=True)
    for prompt in prompts
]

llm = LLM(
    model=model_dir,
    trust_remote_code=True,
    tensor_parallel_size=args.tp,
    enable_chunked_prefill=args.enable_chunked_prefill,
    max_num_batched_tokens=args.max_num_batched_tokens,
    enforce_eager=args.enforce_eager,
    max_model_len=max_model_len,
    max_num_seqs=args.max_num_seqs,
    gpu_memory_utilization=0.8,
    speculative_model=eagle_dir,
    num_speculative_tokens=args.num_spec_tokens,
    speculative_draft_tensor_parallel_size=args.draft_tp,
    speculative_max_model_len=max_model_len,
    disable_log_stats=False,
)

sampling_params = SamplingParams(temperature=args.temp, max_tokens=256)

outputs = llm.generate(prompt_token_ids=prompt_ids,
                       sampling_params=sampling_params)

# calculate the average number of accepted tokens per forward pass, +1 is
# to account for the token from the target model that's always going to be
# accepted
acceptance_counts = [0] * (args.num_spec_tokens + 1)
for output in outputs:
    for step, count in enumerate(output.metrics.spec_token_acceptance_counts):
        acceptance_counts[step] += count

print(f"mean acceptance length: \
    {sum(acceptance_counts) / acceptance_counts[0]:.2f}")
