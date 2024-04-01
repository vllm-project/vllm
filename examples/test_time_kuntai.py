import torch
from tqdm import tqdm

import pdb
import os
from vllm import LLM, SamplingParams

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# *400 means 4000 tokens as input, the prompt is exactly 10 toks
long_prompt = ["You are an expert school principal in JCL library"] * 400
long_prompt = [' '.join(long_prompt)]


#llm = LLM(model="TheBloke/Llama-2-70b-Chat-AWQ", 
#          quantization="AWQ",tensor_parallel_size=2,
#          enforce_eager=True)
llm = LLM(model="lmsys/longchat-7b-16k")
tokenizer = llm.llm_engine.tokenizer
prompt_len = len(tokenizer.encode(long_prompt[0]))
print(f"Prompt Length: {prompt_len}")
#pdb.set_trace()

sampling_params = SamplingParams(temperature=0, max_tokens=1)


# Create an LLM.

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = llm.generate(long_prompt, sampling_params)
end.record()
torch.cuda.synchronize()

temp_time = start.elapsed_time(end)
print(temp_time)
pdb.set_trace()
    
print("end")
        
