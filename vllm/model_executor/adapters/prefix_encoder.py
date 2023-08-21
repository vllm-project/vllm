# coding=utf-8
import os
import torch
from safetensors.torch import load_file as safe_load_file
from peft import (
    PeftType,
    PeftConfig,
    PromptLearningConfig,
    mapping,
    utils,
)
from vllm import cache_ops
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)

def set_prefix_encoder(adpater_model_and_path):
    global prefix_encoder
    try:
        if prefix_encoder is not None:
            pass
    except Exception as e:
        if isinstance(adpater_model_and_path,str):
            prefix_encoder = PrefixEncoder(adpater_model_and_path)
        elif isinstance(adpater_model_and_path, PrefixEncoder):
            prefix_encoder = adpater_model_and_path

def get_prefix_encoder():
    try:
        return prefix_encoder
    except Exception as e:
        return None
    
def get_prefix_tuning_encoder():
    try:
        if prefix_encoder.prefix_tunning:
            return prefix_encoder
        else:
            return None
    except Exception as e:
        return None
    
    

class PrefixEncoder(object):
    def __init__(self, adpater_model_and_path = None) -> None:
        if adpater_model_and_path and os.path.isfile(os.path.join(adpater_model_and_path, "adapter_config.json")):
            self.init_with_peft_format(adpater_model_and_path)
        self.kv_cache = None
    
    def custom_init(self, prompt_embedding,                 
                        num_virtual_tokens, num_layers,
                        num_attention_heads, prefix_tunning):
        self.prompt_embedding = prompt_embedding.half().cuda()
        self.num_virtual_tokens = num_virtual_tokens
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.prefix_tunning = prefix_tunning

    def init_with_peft_format(self,peft_model_id):
        self.peft_config = mapping.PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    peft_model_id,
                )
            ].from_pretrained(
                peft_model_id,
            )
        assert isinstance(self.peft_config, PromptLearningConfig)
        self.num_virtual_tokens = self.peft_config.num_virtual_tokens
        self.num_layers = self.peft_config.num_layers
        self.num_attention_heads = self.peft_config.num_attention_heads
        self.prefix_tunning = self.peft_config.peft_type == PeftType.PREFIX_TUNING
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.exists(os.path.join(peft_model_id, utils.SAFETENSORS_WEIGHTS_NAME)):
            filename = os.path.join(peft_model_id, utils.SAFETENSORS_WEIGHTS_NAME)
            use_safetensors = True
        elif os.path.exists(os.path.join(peft_model_id, utils.WEIGHTS_NAME)):
            filename = os.path.join(peft_model_id, utils.WEIGHTS_NAME)
            use_safetensors = False
        if use_safetensors:
            adapters_weights = safe_load_file(filename, device=torch_device)
        else:
            adapters_weights = torch.load(filename, map_location=torch.device(torch_device))
        self.prompt_embedding = adapters_weights["prompt_embeddings"].half()
        
    @property
    def num_virtual_token_blocks(self):
        return (self.num_virtual_tokens + self.block_size -1) // self.block_size
    
    def write_prompt_embedding_into_kvcache(self,gpu_allocator, gpu_cache, block_size):
        assert self.prefix_tunning,"only prefix embedding need write_prompt_embedding_into_kvcache."
        self.block_size = block_size
        prompt_embedding_table = []
        for _ in range(self.num_virtual_tokens_blocks):
            block = gpu_allocator.allocate()
            prompt_embedding_table.append(block)
                
        slot_mapping = []
        for i in range(self.num_virtual_tokens):
            block_number = prompt_embedding_table[i // self.block_size].block_number
            block_offset = i % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append(slot)
        slot_mapping = torch.cuda.IntTensor(slot_mapping)
        
        for layer_index,(key_cache, value_cache) in enumerate(gpu_cache):
            cache_k, cache_v = self.kv_cache[layer_index][:,0],self.kv_cache[layer_index][:,1]
            cache_ops.reshape_and_cache(
                cache_k,
                cache_v,
                key_cache,
                value_cache,
                slot_mapping,
            )
        return prompt_embedding_table

    
    def cat_prompt_embedding_with_input_embedding(self, inputs_embeds, prompt_lens):
        if not self.prefix_tunning:
            new_inputs_embeds = torch.empty(inputs_embeds.shape[0]+self.num_virtual_tokens*len(prompt_lens), *inputs_embeds.shape[1:])
            start,end = 0,0
            new_start,new_end = 0,0
            for prompt_len in prompt_lens:
                end += prompt_len
                new_end += prompt_len + self.num_virtual_tokens
                index = torch.range(new_start,new_start+ self.num_virtual_tokens-1, 
                                    device = inputs_embeds.device, dtype=torch.long)
            
                #get_tensor_model_parallel_rank 取对应的num_heads
                new_inputs_embeds.index_copy_(0,index,self.prompt_embedding)
                index = torch.range(new_start + self.num_virtual_tokens, new_end -1, 
                                    device = inputs_embeds.device, dtype=torch.long)
                new_inputs_embeds.index_copy_(0,index ,inputs_embeds[start:end])
                start += prompt_len
                new_start = start + self.num_virtual_tokens
            return new_inputs_embeds
        
    def cat_prompt_with_key_value(self, layer_index, prompt_lens, key, value):
        if self.prefix_tunning:
            if self.kv_cache is None:
                self.prompt_embedding = self.prompt_embedding.view(self.num_virtual_tokens, 2*self.num_layers, self.num_attention_heads, -1)
                self.kv_cache = self.prompt_embedding.split(2,dim=1)
            new_key = torch.empty(size=(key.shape[0]+self.num_virtual_tokens * len(prompt_lens),
                                        *key.shape[1:]),
                                dtype = key.dtype,
                                device = key.device)
            new_value = torch.empty(size=(value.shape[0]+self.num_virtual_tokens * len(prompt_lens),
                                        *value.shape[1:]),
                                dtype = value.dtype,
                                device = value.device)
            start,end = 0,0
            new_start,new_end = 0,0
            cache_k, cache_v = self.kv_cache[layer_index][:,0],self.kv_cache[layer_index][:,1]
            for prompt_len in prompt_lens:
                end += prompt_len
                new_end += prompt_len + self.num_virtual_tokens
                index = torch.range(new_start,new_start+ self.num_virtual_tokens-1, 
                                    device = key.device, dtype=torch.long)
                new_start += self.num_virtual_tokens
                #get_tensor_model_parallel_rank 取对应的num_heads
                new_key.index_copy_(0,index,cache_k)
                new_value.index_copy_(0,index,cache_v)
                index = torch.range(new_start, new_end -1, 
                                    device = key.device, dtype=torch.long)
                new_key.index_copy_(0,index ,key[start:end])
                new_value.index_copy_(0,index,value[start:end])
                
                start += prompt_len
                new_start += prompt_len
            return new_key,new_value