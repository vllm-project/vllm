from vllm import LLM, SamplingParams
import pdb
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES']= '1'
# Sample prompts.
long_prompt = ["You are an expert school principal in JCL library"] * 400
prompts = [' '.join(long_prompt)]
#prompts = [
#    "Hello, my name is Jiayi Yao",
#]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1)

# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", gpu_memory_utilization=0.8)
#llm = LLM(model="lmsys/longchat-7b-16k")
#llm = LLM(model="gpt2")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.


def _pre_make_partial_bias(device,
                           max_seq_len=4096, 
                           num_heads=32,
                           dtype=torch.float16):
    padded_len = (max_seq_len + 7) // 8 * 8
    attn_mask = torch.triu(torch.ones(padded_len,
                                      padded_len,
                                      dtype=dtype,
                                      device=device),
                           diagonal=1)
    #FIXME(Jiayi): The first 1 (bsz) is a hack
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(1, 1, padded_len, padded_len) #FIXME(Jiayi): Now only focus on bsz=1
    attn_mask = attn_mask.expand(-1,num_heads,-1,-1)
    
    attn_mask_padded = torch.empty(
        1,
        num_heads,
        padded_len,
        padded_len,
        device=device,
        dtype=dtype,
    ).copy_(attn_mask)
    #attn_mask_padded = LowerTriangularMaskWithTensorBias(attn_mask_padded)
    return attn_mask_padded

device = llm.llm_engine.model_executor.driver_worker.model_runner.device
pre_mask = _pre_make_partial_bias(device=device)

#FIXME(Jiayi): Please align `recomp_ratio` and `recomp_ratios` automatically
llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata = {
    "check_layers":[1],
    "check": True,
    "recomp_ratios":[1.0],
    "recomp_ratio":1.0,
    "load_indices":[],
    "recomp_indices":[],
    "original_slot_mapping":None,
    "our_slot_mapping":None,
    "our_slot_mapping_for_check":None,
    "kv_cache_dtype": None,
    "attn_bias": None,
    "imp_token_indices": None,
    "org_seq_len": -1,
    "pre_mask":pre_mask}

#pdb.set_trace()

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

outputs = llm.generate(prompts, sampling_params)

end.record()
torch.cuda.synchronize()
temp_time = start.elapsed_time(end)
print(temp_time)

pdb.set_trace()


print("finished")