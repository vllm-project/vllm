from vllm import LLM, SamplingParams
from tokenizers import Tokenizer
import pypinyin
import torch
torch.random.manual_seed(999)
# tts1 = torch.load('/home/zhn/ttslm_dev/GPT_merged_emb_nonorm.pt')
# tts2 = torch.load('/home/zhn/fishtts/checkpoint-1400000.bak')

# layer = 24
# dim = 1536
# num_audio_tokens = 1026
# num_text_tokens = 7002
# llama = tts2['model']['llama']

# llama.pop('freqs_cis')
# llama.pop('causal_mask')

# text_emb = llama['text_embeddings.weight']
# for i in range(100):
#     text_emb = torch.cat([text_emb, torch.zeros((1,dim), device=text_emb.device)], 0)
# llama['emb_text.weight'] = text_emb
# llama.pop('text_embeddings.weight')

# llama['emb_code.0.weight'] = llama['code_embeddings.weight'][0:num_audio_tokens]
# llama['emb_code.1.weight'] = llama['code_embeddings.weight'][num_audio_tokens-2:num_audio_tokens - 2 + num_audio_tokens]
# llama.pop('code_embeddings.weight')

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
# llama['lm_head.0.weight'] = w_output[num_text_tokens:num_text_tokens+num_audio_tokens]
# llama['lm_head.1.weight'] = w_output[num_text_tokens+num_audio_tokens:num_text_tokens+num_audio_tokens*2]
# llama.pop(output_name)

# torch.save(llama, '/home/zhn/fishtts/llama.pt')

texts = [
    '城市霓虹,夜幕低垂,梦想之光,闪烁不已。心向未来,勇往直前,在星空下,奋斗的旋律。',
    '在这个数字的世界里,你是我的唯一,爱情如同网络连接,无论距离多遥远。我们的心相互链接,在虚拟的空间中漫游,每条信息都是爱的表达,每个瞬间都是甜蜜的时刻。爱情不再是纸上文字,而是数码世界的交流,在屏幕上,我们相拥相视,你是我的电子爱情。']
llm_inputs = []
tokenizer = Tokenizer.from_file('/home/zhn/fishtts/vocab.json')
for text in texts:
    pinyin = "".join([p[0] for p in pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)])
    txt = f"[zh-cn]{pinyin}"
    txt = txt.replace(" ", "[SPACE]")
    token_ids = tokenizer.encode(txt).ids
    token_ids.insert(0, 7001)
    token_ids.append(0)
    token_ids.append(7003)
    llm_inputs.append(token_ids)

llm = LLM(model='/home/zhn/fishtts', gpu_memory_utilization=0.5, dtype=torch.float32, skip_tokenizer_init=True, enforce_eager=True)
prompts = [
    {"prompt_token_ids": llm_input} for llm_input in llm_inputs
]

sampling_params = SamplingParams(temperature=1, detokenize=False, stop_token_ids=[1025], max_tokens=2048, top_k=1, repetition_penalty=1.5, repetition_window=16)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.prompt)
    token_ids = output.outputs[0].token_ids
    for token_id in token_ids:
        print([x - 0 for x in token_id])
    print(len(token_ids))
