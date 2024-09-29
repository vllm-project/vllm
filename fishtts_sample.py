import argparse
import asyncio
import threading
import time
from typing import List

import numpy as np
import onnx
from vllm import LLM, SamplingParams
from tokenizers import Tokenizer
import pypinyin
import torch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import onnxruntime
import soundfile as sf
import queue

torch.random.manual_seed(999)

def convert_model():
    tts2 = torch.load('/data/fishtts/checkpoint-1400000.bak')

    layer = 24
    dim = 1536
    num_audio_tokens = 1026
    num_text_tokens = 7002
    llama = tts2['model']['llama']

    llama.pop('freqs_cis')
    llama.pop('causal_mask')

    text_emb = llama['text_embeddings.weight']
    for i in range(100):
        text_emb = torch.cat([text_emb, torch.zeros((1,dim), device=text_emb.device)], 0)
    llama['emb_text.weight'] = text_emb
    llama.pop('text_embeddings.weight')

    llama['emb_code.0.weight'] = llama['code_embeddings.weight'][0:num_audio_tokens]
    llama['emb_code.1.weight'] = llama['code_embeddings.weight'][num_audio_tokens-2:num_audio_tokens - 2 + num_audio_tokens]
    llama.pop('code_embeddings.weight')

    for i in range(layer):
        qkv_name = f'layers.{i}.attention.wqkv.weight'
        q = llama[qkv_name][0:dim]
        k = llama[qkv_name][dim:2*dim]
        v = llama[qkv_name][2*dim:]
        llama[f'gpt.layers.{i}.self_attn.q_proj.weight'] = q
        llama[f'gpt.layers.{i}.self_attn.k_proj.weight'] = k
        llama[f'gpt.layers.{i}.self_attn.v_proj.weight'] = v
        llama.pop(qkv_name)
        
        wo_name = f'layers.{i}.attention.wo.weight'
        wo = llama[wo_name]
        llama[f'gpt.layers.{i}.self_attn.o_proj.weight'] = wo
        llama.pop(wo_name)
        
        gate_proj_name = f'layers.{i}.feed_forward.w1.weight'
        w_gate = llama[gate_proj_name]
        llama[f'gpt.layers.{i}.mlp.gate_proj.weight'] = w_gate
        llama.pop(gate_proj_name)
        
        gate_up_proj_name = f'layers.{i}.feed_forward.w3.weight'
        w_gate_up = llama[gate_up_proj_name]
        llama[f'gpt.layers.{i}.mlp.up_proj.weight'] = w_gate_up
        llama.pop(gate_up_proj_name)
        
        gate_down_proj_name = f'layers.{i}.feed_forward.w2.weight'
        w_gate_down = llama[gate_down_proj_name]
        llama[f'gpt.layers.{i}.mlp.down_proj.weight'] = w_gate_down
        llama.pop(gate_down_proj_name)

        attn_norm_name = f'layers.{i}.attention_norm.weight'
        w_attn_norm = llama[attn_norm_name]
        llama[f'gpt.layers.{i}.input_layernorm.weight'] = w_attn_norm
        llama.pop(attn_norm_name)

        ffn_norm_name = f'layers.{i}.ffn_norm.weight'
        w_ffn_norm = llama[ffn_norm_name]
        llama[f'gpt.layers.{i}.post_attention_layernorm.weight'] = w_ffn_norm
        llama.pop(ffn_norm_name)


    norm_name = 'norm.weight'
    w_norm = llama[norm_name]
    llama['gpt.norm.weight'] = w_norm
    llama.pop(norm_name)

    output_name = 'output.weight'
    w_output = llama[output_name]
    llama['lm_head.0.weight'] = w_output[num_text_tokens:num_text_tokens+num_audio_tokens]
    llama['lm_head.1.weight'] = w_output[num_text_tokens+num_audio_tokens:num_text_tokens+num_audio_tokens*2]
    llama.pop(output_name)

    torch.save(llama, '/data/fishtts/llama.pt')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

texts = [
    '城市霓虹,夜幕低垂,梦想之光,闪烁不已。心向未来,勇往直前,在星空下,奋斗的旋律。',
    '在这个数字的世界里,你是我的唯一,爱情如同网络连接,无论距离多遥远。我们的心相互链接,在虚拟的空间中漫游,每条信息都是爱的表达,每个瞬间都是甜蜜的时刻。爱情不再是纸上文字,而是数码世界的交流,在屏幕上,我们相拥相视,你是我的电子爱情。',
    '探索清新世界的钥匙在此!用海洋微风洗衣粉,让您的衣物充满清晨海边的清新气息。我们的高效配方深层清洁衣物纤维去除顽固污渍的同时,带来持久的清香。不只是清洗更是衣物的焕新旅程。',
    '从现在开始,让我们的多功能厨师机器人成为你厨房里的得力助手。它可以搅拌,切碎,烹饪,烘焙,满足你所有烹饪需求。创新美食,只需轻松一按。',
    '打造完美家居生活,只需一款智能净化器。它能有效过滤空气中的污染物,释放负离子,让你每天呼吸到的都是最纯净的空气,为家人的健康护航。',
    '我刚看完《流浪地球》,这部电影真的很打动我。它不仅仅展示了科幻世界中的宏大景象,更通过对人类团结和奉献精神的刻画,让我对未来充满了思考。影片中的视觉效果和细腻的情感描写,让我觉得这是一部值得反复琢磨的作品。如果你喜欢有深度的科幻电影,这部绝对不会让你失望。',
    '下个月我计划去日本体验当地的文化。我特别期待去京都的古寺庙,想感受一下传统的日式建筑和庭园。东京的市场也让我兴奋,我打算品尝各种地道的小吃。此外,我计划学习一些基本的日语,这样能更好地融入当地生活,提升旅行的整体体验。你们有没有什么建议或者特别推荐的地方？',
    '在保持健康方面,我尝试了一些新的饮食习惯。现在我更多地选择新鲜的蔬菜和水果,减少了糖分和加工食品的摄入。我发现这种饮食方式不仅改善了我的体重,还提升了整体的能量水平。此外,保持充足的水分摄入也是关键,它有助于身体的代谢和排毒。你们有什么其他的健康饮食建议吗？',
    '为了提高学习效率,我采取了一些新方法。例如,我将复杂的学习任务拆分成小的目标,每完成一个小目标就能获得成就感。此外,我还使用了番茄工作法,设定25分钟专注学习,然后休息5分钟,这样可以有效避免疲劳。通过这些方法,我发现自己在学习过程中更加专注和高效。',
    '有一本书《思考,快与慢》给我留下了深刻的印象。这本书由丹尼尔·卡尼曼撰写,详细探讨了人类思维的两种模式——快速直观和缓慢理性。通过丰富的实证研究,作者揭示了我们在日常决策中的思维偏差。这本书不仅在理论上很有趣,对实际生活中的决策也提供了很多有益的启示。',
    '提升工作效率需要良好的时间管理。我发现将任务分解成小步骤,并逐步完成,能让工作变得更有条理。同时,使用待办事项列表和设置提醒也能帮助我保持高效。此外,我还注意到合理的休息和调整对工作效率至关重要。这样不仅提高了我的工作质量,也让我保持了良好的工作状态。',
    '探索不同的音乐风格是我最近的兴趣之一。我特别喜欢电子音乐,尤其是那些融合了传统乐器的作品。这种音乐风格不仅提供了新的听觉体验,也让我对音乐的表现形式有了更深的理解。我发现,了解和欣赏不同风格的音乐,能够丰富我的音乐视野和审美体验。',
    '照顾宠物需要全面的关注和细心的呵护。我了解到,定期带狗狗散步有助于它们的身体健康,同时提供丰富的玩具和定期的健康检查也很重要。此外,保持良好的饮食习惯对宠物的整体健康有很大影响。照顾宠物的过程中,了解它们的需求并给予关爱,能让它们生活得更加愉快和健康。',
    '处理社交媒体信息过载,是我近期面临的一个问题。为了避免被海量的信息分散注意力,我开始设置每天查看社交媒体的时间限制,同时选择关注一些高质量的内容。此外,我还定期清理不感兴趣的账号,这样能够保持信息的有效性和对内容的专注。你们有什么管理社交媒体的好方法吗？',
    '每个人都可以在日常生活中采取一些简单的环保行动。我开始减少一次性塑料的使用,进行垃圾分类,并尽量节约能源。这些小措施虽然看似微不足道,但积累起来对环境的保护却能产生积极影响。我相信,关注环保不仅是为了现在的生活,也为未来的子孙着想。你们在环保方面有什么实用的建议吗？',
    '她给我们发了一张照片,呃,在一个满是山、山珍海味婚礼上她拿了一个巨小的饭盒在吃,反正就一个特别清淡的啊,减脂营备的餐,然后她呢当时在群里发这个是,嗯,为了求表扬,哈哈哈!',
    '我这周末过得我觉得,我真的以为是真正意义上的休息,但是没想到周一的时候去上班,呃的时候,我,我还是感觉,呃,很奇怪,就是很提不起精神的感觉哎!',
    '嗯,我刚刚就在想,你说创造一个环境其实,今年,呃,我去采访一位,呃,很有就是,阅历的同龄人的时候,她劝我做一件事情就是找一个心理咨询师,呃,去聊一聊。'
    ]
llm_inputs = []
tokenizer = Tokenizer.from_file('/data/fishtts/vocab.json')
for text in texts:
    pinyin = "".join([p[0] for p in pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)])
    txt = f"[zh-cn]{pinyin}"
    txt = txt.replace(" ", "[SPACE]")
    token_ids = tokenizer.encode(txt).ids
    token_ids.insert(0, 7001)
    token_ids.append(0)
    token_ids.append(7003)
    llm_inputs.append(token_ids)

chunk_size=20
frame_shift=1200
hidden_size = 1536
speaker_embedding = torch.zeros((1, 192, 1), dtype=torch.float32).to('cuda')
sampling_params = SamplingParams(temperature=1, detokenize=False, stop_token_ids=[1025], ignore_eos=True, max_tokens=2048, top_k=1, repetition_penalty=1.5, repetition_window=16)
prompts = [
    {"prompt_token_ids": llm_input} for llm_input in llm_inputs
]

class Metrics:
    def __init__(self):
        self.time_start = 0
        self.time_end = 0
        self.time_first_byte = 0
        self.token_times = []
        
    def calc_non_streaming(self):
        total_time = self.time_end - self.time_start
        audio_time = len(self.token_times) * 50 / 1000
        rtf = total_time / audio_time
        latent_time = self.token_times[-1] - self.time_start
        first_byte_time = self.time_first_byte - self.time_start
        print(f'latent time: {latent_time}, first byte time: {first_byte_time}, total time: {total_time}, audio time: {audio_time}, rtf: {rtf}')

    def calc_streaming(self):
        total_time = self.time_end - self.time_start
        audio_time = len(self.token_times) * 50 / 1000
        rtf = total_time / audio_time
        first_chunk_time = self.token_times[chunk_size - 1] - self.time_start
        first_byte_time = self.time_first_byte - self.time_start
        print(f'first chunk time: {first_chunk_time}, first byte time: {first_byte_time}, total time: {total_time}, audio time: {audio_time}, rtf: {rtf}')

def generate_chunk_audio(latent):
    # pad to chunk_size
    latent_len = latent.size(1)
    if latent_len < chunk_size:
        latent = torch.cat([latent, torch.zeros((1, chunk_size - latent_len % chunk_size, hidden_size), dtype=torch.float32).to('cuda')], 1)
    onnxruntime_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), (latent, speaker_embedding))}
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
    onnxruntime_outputs = onnxruntime_outputs[0][0][0]
    if latent_len < chunk_size:
        return onnxruntime_outputs[:latent_len * frame_shift]
    return onnxruntime_outputs

def save_audio(total_audio, path):
    total_audio = np.concatenate(total_audio, axis=0)
    sf.write(path, total_audio, 24000)

def run():
    llm = LLM(model='/data/fishtts', gpu_memory_utilization=0.7, dtype=torch.float32, skip_tokenizer_init=True)
    for i in range(len(prompts)):
        metrics = Metrics()
        metrics.time_start = time.perf_counter()
        outputs = llm.generate(prompts[i], sampling_params)
        for output in outputs:
            token_ids = output.outputs[0].token_ids
            output_len = len(token_ids)
            cur_time = time.perf_counter()
            metrics.token_times.extend([cur_time] * output_len)
            print(f'{i}:  {output_len}')

            time_latent = time.perf_counter()
            total_audio = []
            chunk_num = output_len // chunk_size
            for j in range(chunk_num):
                latent = torch.stack(output.outputs[0].hidden_states, 0).unsqueeze(0).to('cuda')[:,j*chunk_size:(j+1)*chunk_size]
                onnxruntime_outputs = generate_chunk_audio(latent)
                metrics.time_first_byte = time.perf_counter()
                total_audio.append(onnxruntime_outputs)
            
            if output_len % chunk_size != 0:
                latent = torch.stack(output.outputs[0].hidden_states, 0).unsqueeze(0).to('cuda')[:,chunk_num*chunk_size:]
                onnxruntime_outputs = generate_chunk_audio(latent)
                total_audio.append(onnxruntime_outputs)

            save_audio(total_audio, f'hh_{i}.wav')
            print(f'save audio {i}')

        metrics.time_end = time.perf_counter()
        metrics.calc_non_streaming()

async def generate_audio_streaming(latent_queue: asyncio.Queue, id, metrics: Metrics):
    latent_buffer = []
    audio_data_buffer = []
    while True:
        latent = await latent_queue.get()
        if latent is None:
            break
        latent_buffer.append(latent)
        if len(latent_buffer) == chunk_size:
            latent = torch.stack(latent_buffer, 0).unsqueeze(0).to('cuda')
            audio_data_buffer.append(generate_chunk_audio(latent))
            latent_buffer = []

            if metrics.time_first_byte == 0:
                metrics.time_first_byte = time.perf_counter()

    if len(latent_buffer) > 0:
        latent = torch.stack(latent_buffer, 0).unsqueeze(0).to('cuda')
        audio_data_buffer.append(generate_chunk_audio(latent))
        
    save_audio(audio_data_buffer, f'hh_{id}.wav')
    print(f'save audio {id}')

async def generate_token_streaming(engine: AsyncLLMEngine, prompt, id, latent_queue: asyncio.Queue, metrics: Metrics):
    results_generator = engine.generate(prompt, sampling_params, request_id=id)
    tokens = []
    async for request_output in results_generator:
        metrics.token_times.append(time.perf_counter())
        token_ids = request_output.outputs[0].token_ids[-1]
        latent = request_output.outputs[0].hidden_states[-1]
        tokens.append(token_ids)
        latent_queue.put_nowait(latent)

    latent_queue.put_nowait(None)
    print(f'{id}:  {len(tokens)}')

async def get_request(
    input_requests,
    request_rate: float,
):
    requests = iter(input_requests)
    for request in requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

async def generate_streaming(engine, prompt, request_id) -> Metrics:
    metrics = Metrics()
    metrics.time_start = time.perf_counter()
    
    latent_queue = asyncio.Queue()
    vllm_task = asyncio.create_task(generate_token_streaming(engine, prompt, request_id, latent_queue, metrics))
    generator_task = asyncio.create_task(generate_audio_streaming(latent_queue, request_id, metrics))
    await vllm_task
    await generator_task
    metrics.time_end = time.perf_counter()
    return metrics

async def run_streaming(request_rate):
    engine_args = AsyncEngineArgs(model='/data/fishtts', gpu_memory_utilization=0.7, dtype=torch.float32, skip_tokenizer_init=True)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    if request_rate < 0:
        for i in range(len(prompts)):
            me = await generate_streaming(engine, prompts[i], i)
            me.calc_streaming()
    else:
        tasks: List[asyncio.Task] = []
        request_id = 0
        me = await generate_streaming(engine, prompts[0], 0)
        async for prompt in get_request(prompts, request_rate):
            tasks.append(asyncio.create_task(generate_streaming(engine, prompt, request_id)))
            request_id += 1
        
        metrics_list: List[Metrics] = await asyncio.gather(*tasks)
        for metrics in metrics_list:
            metrics.calc_streaming()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--request-rate",
                        type=float,
                        default=-1,
                        help="request rate per second")
    parser.add_argument("--chunk-size",
                        type=int,
                        default=20,
                        help="audio chunk size")

    args = parser.parse_args()
    
    if args.chunk_size:
        chunk_size = args.chunk_size
    
    ort_session = onnxruntime.InferenceSession('/data/fishtts/genertor.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
    warmup_input = {k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), (torch.zeros(1, chunk_size, hidden_size).to('cuda'), speaker_embedding))}
    warmup_outputs = ort_session.run(None, warmup_input)

    if not args.streaming:
        run()
    else:
        asyncio.run(run_streaming(args.request_rate))