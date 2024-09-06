from vllm import LLM, SamplingParams
import torch
torch.random.manual_seed(999)
# tts1 = torch.load('/home/zhn/ttslm_dev/GPT_merged_emb_nonorm.pt')
# tts2 = torch.load('/home/zhn/fishtts/checkpoint-1400000.bak')

# llama = tts2['model']['llama']

# llama.pop('freqs_cis')
# llama.pop('causal_mask')

# llama['emb_text.weight'] = llama['text_embeddings.weight']
# llama.pop('text_embeddings.weight')

# llama['emb_code.0.weight'] = llama['code_embeddings.weight'][0:1026]
# llama['emb_code.1.weight'] = llama['code_embeddings.weight'][1026:]
# llama.pop('code_embeddings.weight')

# layer = 24
# dim = 1536
# for i in range(layer):
#     qkv_name = f'layers.{i}.attention.wqkv.weight'
#     q = llama[qkv_name][0:dim]
#     k = llama[qkv_name][dim:2*dim]
#     v = llama[qkv_name][2*dim:]
#     llama[f'gpt.layers.{i}.self_attn.q_proj.weight'] = q
#     llama[f'gpt.layers.{i}.self_attn.k_proj.weight'] = k
#     llama[f'gpt.layers.{i}.self_attn.v_proj.weight'] = v
#     llama.pop(qkv_name)
    
#     wo_name = f'layers.{i}.attention.wo.weight'
#     wo = llama[wo_name]
#     llama[f'gpt.layers.{i}.self_attn.o_proj.weight'] = wo
#     llama.pop(wo_name)
    
#     gate_proj_name = f'layers.{i}.feed_forward.w1.weight'
#     w_gate = llama[gate_proj_name]
#     llama[f'gpt.layers.{i}.mlp.gate_proj.weight'] = w_gate
#     llama.pop(gate_proj_name)
    
#     gate_up_proj_name = f'layers.{i}.feed_forward.w3.weight'
#     w_gate_up = llama[gate_up_proj_name]
#     llama[f'gpt.layers.{i}.mlp.up_proj.weight'] = w_gate_up
#     llama.pop(gate_up_proj_name)
    
#     gate_down_proj_name = f'layers.{i}.feed_forward.w2.weight'
#     w_gate_down = llama[gate_down_proj_name]
#     llama[f'gpt.layers.{i}.mlp.down_proj.weight'] = w_gate_down
#     llama.pop(gate_down_proj_name)

#     attn_norm_name = f'layers.{i}.attention_norm.weight'
#     w_attn_norm = llama[attn_norm_name]
#     llama[f'gpt.layers.{i}.input_layernorm.weight'] = w_attn_norm
#     llama.pop(attn_norm_name)

#     ffn_norm_name = f'layers.{i}.ffn_norm.weight'
#     w_ffn_norm = llama[ffn_norm_name]
#     llama[f'gpt.layers.{i}.post_attention_layernorm.weight'] = w_ffn_norm
#     llama.pop(ffn_norm_name)


# norm_name = 'norm.weight'
# w_norm = llama[norm_name]
# llama['gpt.norm.weight'] = w_norm
# llama.pop(norm_name)

# output_name = 'output.weight'
# w_output = llama[output_name]
# llama['lm_head.0.weight'] = w_output[7002:7002+1026]
# llama['lm_head.1.weight'] = w_output[7002+1026:7002+1026*2]
# llama.pop(output_name)

# torch.save(llama, '/home/zhn/fishtts/llama.pt')

llm = LLM(model='/home/zhn/fishtts', gpu_memory_utilization=0.5, dtype=torch.float32)
prompts = [
    {
        "prompt_token_ids": [7001, 5023,   16,   62, 4550, 4557, 4790, 4963,    7, 4676, 4697,   17,
         4549, 2719, 4546,    7,  435,   20, 4499,   37, 1164, 4561, 4637,  828,
          566, 4496,    7,  120,   14, 4695,   32, 4765, 4594, 4648, 4513, 4692,
           37, 1164, 4555,  100, 4544, 4680,    7,   38, 4706,   36,  566, 4498,
         4717,   30, 1164, 4596,    7, 4597, 4858,  475,   20, 4496,   37, 1164,
         4499,    7,  132, 4604,   17, 4610,   17, 4650, 4603,   14, 4596, 4938,
         4513,    0, 0]
    }
]

sampling_params = SamplingParams(temperature=1, detokenize=False, stop_token_ids=[1025], max_tokens=2048, top_k=1, repetition_penalty=1.5, repetition_window=16)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.prompt)
    token_ids = output.outputs[0].token_ids
    for token_id in token_ids:
        print([x - 0 for x in token_id])
