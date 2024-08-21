from vllm import LLM, SamplingParams
import torch
torch.random.manual_seed(999)
# tts = torch.load('/home/zhn/ttslm/GPT_merged_emb_nonorm.pt')

# text_emb_count = tts['emb_text.weight'].shape[0]
# audio_emb_count = tts['emb_code.0.weight'].shape[0]
# model_dim = tts['emb_text.weight'].shape[1]

# # append audio embeddings to text embeddings
# # all_0 = text_emb + audio_emb_0
# all_0 = torch.cat([tts['emb_text.weight'], tts['emb_code.0.weight']], dim=0)

# # all_1 = zero + audio_emb_1
# all_1 = torch.cat([torch.zeros(text_emb_count, model_dim), tts['emb_code.1.weight']], dim=0)

# # all_2 = zero + audio_emb_2
# all_2 = torch.cat([torch.zeros(text_emb_count, model_dim), tts['emb_code.2.weight']], dim=0)

# # all_3 = zero + audio_emb_3
# all_3 = torch.cat([torch.zeros(text_emb_count, model_dim), tts['emb_code.3.weight']], dim=0)

# # remove text emb and audio emb in the model
# tts.pop('emb_text.weight')
# tts.pop('emb_code.0.weight')
# tts.pop('emb_code.1.weight')
# tts.pop('emb_code.2.weight')
# tts.pop('emb_code.3.weight')

# # add new embeddings to the model
# tts['emb_all.0.weight'] = all_0
# tts['emb_all.1.weight'] = all_1
# tts['emb_all.2.weight'] = all_2
# tts['emb_all.3.weight'] = all_3

# for i in range(4):
#     original0 = tts[f'head_code.{i}.parametrizations.weight.original0']
#     original1 = tts[f'head_code.{i}.parametrizations.weight.original1']
#     # get the normalized weights based on the original 0 and 1
#     weight_norm0 = torch._weight_norm(original1, original0, dim=0)
#     tts.pop(f'head_code.{i}.parametrizations.weight.original0')
#     tts.pop(f'head_code.{i}.parametrizations.weight.original1')
#     tts[f'lm_head.{i}.weight'] = weight_norm0

# # save the model
# torch.save(tts, '/home/zhn/ttslm/GPT_merged_emb_nonorm.pt')

llm = LLM(model='/home/zhn/ttslm', gpu_memory_utilization=0.5, dtype=torch.float32, enforce_eager=True)
prompts = [
    {
        "prompt": "[Stts][empty_spk][speed_5]Your text one[Ptts]",
        "multi_modal_data": {"audio": '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'},
    },
    {
        "prompt": "[Stts][empty_spk][speed_5]Anther long string[Ptts]",
        "multi_modal_data": {"audio": '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'},
    }
]

for i in range(12):
    prompts.append(prompts[0])
    prompts.append(prompts[1])

sampling_params = SamplingParams(temperature=1, detokenize=False, stop_token_ids=[21803], max_tokens=2048, top_k=1)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.prompt)
    token_ids = output.outputs[0].token_ids
    for token_id in token_ids:
        print([x - 21178 for x in token_id])