import os
import sys
os.environ['NCCL_DEBUG']='WARN'

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import (get_tokenizer)
import json

prompts = ['who is the founder of Microsoft']

model_path="/data/users/yunanzhang/hf/checkpoints/TLG4.7.3/iter_0078678_hf/"
sampling_params = SamplingParams(temperature=0)
llm = LLM(model=model_path,tokenizer='TLGv4Tokenizer',enforce_eager=True)
outputs = llm.generate(prompts,sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"result:\n {generated_text}")