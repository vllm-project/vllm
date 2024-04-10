import os
import sys
os.environ['NCCL_DEBUG']='WARN'

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import (get_tokenizer)
import json

prompts = ['who is the founder of Microsoft ' * 10]

# model_path="/data/users/yunanzhang/hf/checkpoints/TLG4.7.3/iter_0078678_hf/"
model_path='/mnt/std-cache/users/xihlin/checkpoints/tlgv4.7-phase2/tlgv4'

sampling_params = SamplingParams(temperature=0)
llm = LLM(model=model_path, tokenizer=model_path, enforce_eager=True, trust_remote_code=True, block_size=32)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"result:\n {generated_text}")