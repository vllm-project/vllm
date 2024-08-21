import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
llm = LLM(model='/home/zhn/xtts', gpu_memory_utilization=0.5, enforce_eager=True, add_bos_token=False, dtype=torch.float32)
prompts = [
    {
        "prompt": "[zh-cn]ni3hao3",
    }
]
sampling_params = SamplingParams(temperature=1, detokenize=False, stop_token_ids=[21803], max_tokens=2048, top_k=1)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.prompt)
    token_ids = output.outputs[0].token_ids
    for token_id in token_ids:
        print([x - 21178 for x in token_id])
# tokenizer = AutoTokenizer.from_pretrained('/home/zhn/xtts', add_bos_token=False)
# id = tokenizer.encode('[zh-cn]ni3hao3')

# torch.random.manual_seed(999)
# gpt = torch.load('/home/zhn/xtts/llama.pt')
# tts = torch.load('/home/zhn/xtts/checkpoint-902000.pt')

# llama = tts['model']['llama']
# layer_count = 24
# for i in range(layer_count):
#     name_qkv_0 = f'layers.{i}.attention.wqkv.weight'
#     name_qkv_1 = f'gpt.layers.{i}.self_attn.qkv_proj.weight'
#     llama[name_qkv_1] = llama.pop(name_qkv_0)
    
#     name_o_0 = f'layers.{i}.attention.wo.weight'
#     name_o_1 = f'gpt.layers.{i}.self_attn.o_proj.weight'
#     llama[name_o_1] = llama.pop(name_o_0)
    
#     name_gate_0 = f'layers.{i}.feed_forward.w1.weight'
#     name_gate_1 = f'gpt.layers.{i}.mlp.gate_proj.weight'
#     llama[name_gate_1] = llama.pop(name_gate_0)
    
#     name_up_0 = f'layers.{i}.feed_forward.w3.weight'
#     name_up_1 = f'gpt.layers.{i}.mlp.up_proj.weight'
#     llama[name_up_1] = llama.pop(name_up_0)
    
#     name_down_0 = f'layers.{i}.feed_forward.w2.weight'
#     name_down_1 = f'gpt.layers.{i}.mlp.down_proj.weight'
#     llama[name_down_1] = llama.pop(name_down_0)
    
#     name_ffn_norm_0 = f'layers.{i}.ffn_norm.weight'
#     name_ffn_norm_1 = f'gpt.layers.{i}.input_layernorm.weight'
#     llama[name_ffn_norm_1] = llama.pop(name_ffn_norm_0)
    
#     name_attn_norm_0 = f'layers.{i}.attention_norm.weight'
#     name_attn_norm_1 = f'gpt.layers.{i}.post_attention_layernorm.weight'
#     llama[name_attn_norm_1] = llama.pop(name_attn_norm_0)
    
# name_norm_0 = f'norm.weight'
# name_norm_1 = f'gpt.norm.weight'
# llama[name_norm_1] = llama.pop(name_norm_0)

# text_emb = llama['text_embeddings.weight']
# code_emb_0 = llama['code_embeddings.weight'][0:1026, :]
# code_emb_1 = llama['code_embeddings.weight'][1026:2052, :]
# all_0 = torch.cat([text_emb, code_emb_0], dim=0)
# all_1 = torch.cat([torch.zeros_like(text_emb), code_emb_1], dim=0)
# llama['emb_all.0.weight'] = all_0
# llama['emb_all.1.weight'] = all_1
# llama.pop('text_embeddings.weight')
# llama.pop('code_embeddings.weight')

# output = llama['output.weight']
# lm_head = output[7002:, :]
# lm_head_0 = lm_head[0:1026, :]
# lm_head_1 = lm_head[1026:2052, :]
# llama['lm_head.0.weight'] = lm_head_0
# llama['lm_head.1.weight'] = lm_head_1
# llama.pop('output.weight')

# torch.save(llama, '/home/zhn/xtts/llama.pt')
