from vllm import LLM, SamplingParams
import torch

# tts = torch.load('/home/largeniu/ttslm/GPT.pt')

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

# # save the model
# torch.save(tts, '/home/largeniu/ttslm/GPT_merged_emb.pt')

tokenizer = torch.load('/home/largeniu/g/ChatTTS/asset/tokenizer.pt')
llm = LLM(model='/home/largeniu/ttslm', skip_tokenizer_init=True)
llm.set_tokenizer(tokenizer)
prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")