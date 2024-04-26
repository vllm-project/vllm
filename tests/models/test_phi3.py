import os
import sys

from vllm import LLM, SamplingParams

os.environ["NCCL_DEBUG"] = "WARN"

tp = int(sys.argv[1]) if len(sys.argv) >= 2 else 1

long_prompt = ""

# your prompts path
file_path = ""
with open(file_path) as f:
    long_prompt = f.read().strip()

prompts = [long_prompt]

# your model path
model_path = ""

sampling_params = SamplingParams(temperature=0)
llm = LLM(
    model=model_path,
    tokenizer=model_path,
    enforce_eager=False,
    trust_remote_code=True,
    block_size=16,
    tensor_parallel_size=tp,
)

outputs = llm.generate(prompts, sampling_params)

print("1st run:")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"result:\n {generated_text}")

# -----
# exit()

prompts = [
    "The president of Microsoft is " * 300,
    "Wikipedia\n" * 10 + "Wikipedia is a",
]

outputs2 = llm.generate(prompts, sampling_params)

print(">>>> 2nd run")
for output in outputs2:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"result:\n {generated_text}")
