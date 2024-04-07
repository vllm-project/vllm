from vllm import LLM, SamplingParams
import pdb
import os
import torch
from vllm.metis.kv_manager import kv_manager

os.environ['CUDA_VISIBLE_DEVICES']= '0'
# Sample prompts.
long_prompt = ["You are an expert school principal in JCL library"] * 3800
prompts = [' '.join(long_prompt)]
#prompts = [
#    "Hello, my name is Jiayi Yao",
#]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1)

# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", gpu_memory_utilization=0.99)
#llm = LLM(model="TheBloke/Llama-2-70b-Chat-AWQ", 
#          quantization="AWQ",tensor_parallel_size=2,
#          enforce_eager=True)


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

x = torch.rand((2000)).cuda()
xx = torch.topk(x, k=300).indices
x = torch.rand((1000)).cuda()
xx = torch.topk(x, k=400).indices

device = llm.llm_engine.model_executor.driver_worker.model_runner.device
pre_mask = _pre_make_partial_bias(device=device)

#FIXME(Jiayi): Please align `recomp_ratio` and `recomp_ratios` automatically
llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_fuse_metadata = {
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
    "imp_token_indices": None,
    "org_seq_len": -1,
    "pre_mask":pre_mask}

#FIXME Add the metadata dynamically
our_loader = kv_manager("/local/hanchen/", [])
llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_load_metadata = {
   "loader" :  our_loader,
   "hash": "kv_temp"
}


#pdb.set_trace()

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

outputs = llm.generate(prompts, sampling_params)

llm.llm_engine.model_executor.driver_worker.model_runner.model.model.cache_load_metadata = {
   "loader" :  our_loader,
   "hash": "kv_temp3"
}

start.record()

outputs = llm.generate(prompts, sampling_params)
outputs = llm.generate(prompts, sampling_params)
outputs = llm.generate(prompts, sampling_params)
outputs = llm.generate(prompts, sampling_params)
end.record()
temp_time = start.elapsed_time(end)
pdb.set_trace()
print("time spent is: ", temp_time)
torch.cuda.synchronize()


pdb.set_trace()


print("finished")