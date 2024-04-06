from vllm import LLM, SamplingParams
import pdb
import os
import torch

#os.environ['CUDA_VISIBLE_DEVICES']= '0'
# Sample prompts.
long_prompt = ["You are an expert school principal in JCL library"] * 380
prompts = [' '.join(long_prompt)]
#prompts = [
#    "Hello, my name is Jiayi Yao",
#]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1)

# Create an LLM.
#llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", gpu_memory_utilization=0.8)
llm = LLM(model="TheBloke/Llama-2-70b-Chat-AWQ", 
          quantization="AWQ",tensor_parallel_size=2,
          enforce_eager=True,
          gpu_memory_utilization=0.6)

#llm = LLM(model="lmsys/longchat-7b-16k")

print("test_single.py: 11111111111")

def _pre_make_partial_bias(device,
                           max_seq_len=4096, 
                           num_kv_heads=4,
                           num_queries_per_kv=8,
                           dtype=torch.float16):
    padded_len = (max_seq_len + 7) // 8 * 8
    attn_mask = torch.triu(torch.ones(padded_len,
                                      padded_len,
                                      dtype=dtype,
                                      device=device),
                           diagonal=1)
    #FIXME(Jiayi): The first 1 (bsz) is a hack
    attn_mask = (attn_mask * torch.finfo(dtype).min).view(1, 1, 1, padded_len, padded_len) #FIXME(Jiayi): Now only focus on bsz=1
    attn_mask = attn_mask.expand(-1,num_kv_heads,num_queries_per_kv,-1,-1)
    
    attn_mask_padded = torch.empty(
        1,
        num_kv_heads,
        num_queries_per_kv,
        padded_len,
        padded_len,
        device=device,
        dtype=dtype,
    ).copy_(attn_mask)
    #attn_mask_padded = LowerTriangularMaskWithTensorBias(attn_mask_padded)
    return attn_mask_padded

#device = llm.llm_engine.model_executor.driver_worker.model_runner.device
#pre_mask = _pre_make_partial_bias(device=device)
'''
if torch.distributed.get_rank()==0:
    import pdb
    pdb.set_trace()
torch.distributed.barrier()
group = 
meta_dicts = [{},{}]
allgather()
'''

#print(llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata)
#FIXME(Jiayi): Please align `recomp_ratio` and `recomp_ratios` automatically
cache_fuse_metadata = {
    "check_layers":[1],
    "check": True,
    "recomp_ratios":[0.15],
    "recomp_ratio":0.15,
    "load_indices":[],
    "recomp_indices":[],
    "original_slot_mapping":None,
    "our_slot_mapping":None,
    "our_slot_mapping_for_check":None,
    "kv_cache_dtype": None,
    "attn_bias": None,
    "imp_token_indices": [],
    "org_seq_len": -1,
    "pre_mask":None}

#print(llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata)

#pdb.set_trace()



outputs = llm.generate(prompts, sampling_params,cache_fuse_metadata=None)

outputs = llm.generate(prompts, sampling_params,cache_fuse_metadata=cache_fuse_metadata)
outputs = llm.generate(prompts, sampling_params,cache_fuse_metadata=cache_fuse_metadata)
outputs = llm.generate(prompts, sampling_params,cache_fuse_metadata=cache_fuse_metadata)
outputs = llm.generate(prompts, sampling_params,cache_fuse_metadata=cache_fuse_metadata)


print("finished")