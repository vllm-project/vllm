import asyncio
import time
from tokenizers import Tokenizer
import pypinyin
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

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

engine_args = AsyncEngineArgs(model='/home/zhn/fishtts', gpu_memory_utilization=0.5, dtype=torch.float32, skip_tokenizer_init=True, enforce_eager=True)
model = AsyncLLMEngine.from_engine_args(engine_args)
sampling_params = SamplingParams(temperature=1, detokenize=False, stop_token_ids=[1025], max_tokens=2048, top_k=1, repetition_penalty=1.5, repetition_window=16)
prompts = [
    {"prompt_token_ids": llm_input} for llm_input in llm_inputs
]

async def generate_streaming(prompt, id):
    results_generator = model.generate(prompt, sampling_params, request_id=id)
    count=0
    tokens = []
    async for request_output in results_generator:
        token_ids = request_output.outputs[0].token_ids
        print(f'{id}  {[x - 0 for x in token_ids[-1]]}')
        tokens.append([x - 0 for x in token_ids[-1]])
        count+=1
    
    print(id)
    for token in tokens:
        print(token)

async def generate():
    tasks = []
    for i in range(2):
        t = generate_streaming(prompts[i%2], i)
        tasks.append(t)
    await asyncio.gather(*tasks)

asyncio.run(generate())
