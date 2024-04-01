from vllm import LLM, SamplingParams
import pdb

# Sample prompts.
long_prompt = ["You are an expert school principal in JCL library"] * 400
prompts = [' '.join(long_prompt)]
#prompts = [
#    "Hello, my name is Jiayi Yao",
#]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=1)

# Create an LLM.
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
pdb.set_trace()
#llm = LLM(model="lmsys/longchat-7b-16k")
#llm = LLM(model="gpt2")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.


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
    "org_seq_len": None}

#pdb.set_trace()

outputs = llm.generate(prompts, sampling_params)
pdb.set_trace()

print("finished")