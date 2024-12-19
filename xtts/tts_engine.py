import asyncio
import time
from typing import Any, Union

import numpy
import onnxruntime
import pypinyin
from tokenizers import Tokenizer
import torch
from transformers import LlamaConfig

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.llm import LLM

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from generator import AudioGenerator
from metrics import TtsMetrics
from preceiver_resampler import PreceiverResampler
from utils import *
from model_setting import ModelSetting

class XTtsEngine:
    def __init__(self, model_setting: ModelSetting):
        self.model_setting : ModelSetting = model_setting
        self.tokenizer = None
        self.llm_engine : Union[AsyncLLMEngine, LLM]= None
        self.audio_generator: AudioGenerator = None
        self.preceiever_sampler = None
        self.sampling_params = SamplingParams(temperature=1, detokenize=False, stop_token_ids=[1025], ignore_eos=True, max_tokens=2048, top_k=1, repetition_penalty=1.5, repetition_window=16)
        self.post_init()
    
    def post_init(self):
        
        # initialize tokenizer
        logger.info('Loading tokenizer...')
        self.tokenizer = Tokenizer.from_file('/home/zhn/fishtts/vocab.json')
        logger.info('Tokenizer loaded.')

        # initialize LLM
        logger.info('Loading LLM...')
        if self.model_setting.streaming:
            logger.info('Using AsyncLLMEngine...')
            engine_args = AsyncEngineArgs(model=self.model_setting.model_dir,
                                         gpu_memory_utilization=self.model_setting.gpu_memory_utilization,
                                         dtype=self.model_setting.dtype,
                                         skip_tokenizer_init=True)
            if self.model_setting.support_lora:
                engine_args.enable_lora = True
                engine_args.max_lora_rank = 128
            self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        else:
            logger.info('Using LLM...')
            if self.model_setting.support_lora:
                self.llm_engine = LLM(self.model_setting.model_dir,
                                      gpu_memory_utilization=self.model_setting.gpu_memory_utilization, 
                                      dtype=self.model_setting.dtype,
                                      skip_tokenizer_init=True, enable_lora=True, max_lora_rank=128)
            else:
                self.llm_engine = LLM(self.model_setting.model_dir,
                                      gpu_memory_utilization=self.model_setting.gpu_memory_utilization, 
                                      dtype=self.model_setting.dtype,
                                      skip_tokenizer_init=True)
        
        logger.info('LLM loaded.')

        # initialize audio generator
        logger.info('Loading audio generator...')
        self.audio_generator = AudioGenerator(self.model_setting)
        logger.info('Audio generator loaded.')
        
        self.preceiever_sampler = PreceiverResampler(self.model_setting)
        logger.info('Preceiver resampler loaded.')
    
    def warm_up(self, lora_path: str = None):
        lora_request = None
        if lora_path:
            lora_request=LoRARequest("lora", 1, lora_local_path=lora_path)
        else:
            lora_request = None
        if self.model_setting.streaming:
            prompts = self.text_to_prompts("你好")
            asyncio.run(self.generate_streaming(self.llm_engine, prompts[0], self.sampling_params, 'warmup', '.', lora_request=lora_request))

    def text_to_prompts(self, texts: Union[str, List[str]]) -> List[dict[str, Any]]:
        if isinstance(texts, str):
            texts = [texts]
        llm_inputs = []
        for text in texts:
            text_split = mix_sentence_spliter(text)
            token_ids_merge: List[int] = []
            for idx, sub_text in enumerate(text_split):
                locale = 'zh' if re.search(r'[\u4e00-\u9fa50-9]', sub_text) else 'en'
                txt = text_normalizer({'text': sub_text, 'locale': locale}, idx == len(text_split) - 1)
                if locale == 'zh':
                    txt = "".join([p[0] for p in pypinyin.pinyin(txt, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)])
                
                locale = "zh-cn" if locale == "zh" else locale
                txt = f"[{locale}]{txt}"
                txt = txt.replace(" ", "[SPACE]")
                token_ids: List[int] = self.tokenizer.encode(txt).ids
                token_ids_merge.extend(token_ids if idx == 0 else token_ids[1:])

            token_ids_merge.insert(0, 7001)
            token_ids_merge.append(0)
            token_ids_merge.append(7003)
            
            # append reference audio embeddings
            token_ids_merge = [7004] + [7005]* 15 + token_ids_merge
            llm_inputs.append(token_ids_merge)
        prompts = [
            {"prompt_token_ids": llm_input, "multi_modal_data":{ "audio": self.preceiever_sampler.get_reference_audio(len(llm_input) - 16 - 3) } } for llm_input in llm_inputs
        ]
        return prompts

    def synthesize(self, texts: List[str], output_dir: str, lora_path: str = None,
                   top_k: int = 1, top_p: float = 1, temperature: float = 1.0):
        if isinstance(texts, str):
            texts = [texts]
        logger.info(f'Synthesizing {len(texts)} texts...')
        prompts = self.text_to_prompts(texts)
        lora_request = None
        if lora_path:
            lora_request=LoRARequest("lora", 1, lora_local_path=lora_path)
        for i,p in enumerate(prompts):
            logger.debug(f'Processing text {i+1}/{len(prompts)}...')
            metrics = TtsMetrics(chunk_size=self.model_setting.chunk_size)
            metrics.time_start = time.perf_counter()
            sampling_params = SamplingParams(detokenize=False, stop_token_ids=[1025], ignore_eos=True, max_tokens=2048,
                                             repetition_penalty=1.5, repetition_window=16, 
                                             top_k=top_k, top_p=top_p, temperature=temperature)
            outputs = self.llm_engine.generate(p, sampling_params, lora_request=lora_request)
            output = outputs[0].outputs[0]
            token_ids = output.token_ids
            token_ids = token_ids[:-1]
            output_len = len(token_ids)
            cur_time = time.perf_counter()
            metrics.token_times.extend([cur_time] * output_len)
            logger.info(f'{output_len} tokens generated:')
            logger.debug(token_ids)
            latent = torch.stack(output.hidden_states, 0).unsqueeze(0).to('cuda')
            # the last token is EOS, should be excluded
            latent = latent[:, :output_len, :]
            total_audio = self.audio_generator.generate_audio(latent, metrics)
            save_audio(total_audio, f'{output_dir}/{i:03d}.wav', self.model_setting.cut_tail)
            logger.debug(f'Audio generated')
            metrics.time_end = time.perf_counter()
            metrics.calc_non_streaming()

    async def generate_token_streaming(self, engine: AsyncLLMEngine,
                                       prompt: dict[str, Any], lora_request: LoRARequest, id: str, sampling_params: SamplingParams,
                                       latent_queue: asyncio.Queue, metrics: TtsMetrics):
        results_generator = engine.generate(prompt, sampling_params, request_id=id, lora_request=lora_request)
        tokens = []
        logger.debug(f'Generating tokens for {id}...')
        async for request_output in results_generator:
            metrics.token_times.append(time.perf_counter())
            token_ids = request_output.outputs[0].token_ids[-1]
            latent = request_output.outputs[0].hidden_states[-1]
            tokens.append(token_ids)
            latent_queue.put_nowait(latent)
        # the last token is EOS, should be excluded
        tokens = tokens[:-1]
        logger.info(f'Tokens generated for {id}, {len(tokens)} tokens generated.')
        latent_queue.put_nowait(None)


    async def generate_audio_streaming(self, latent_queue: asyncio.Queue, id: str, metrics: TtsMetrics, output_dir: str):
        latent_buffer = []
        audio_data_buffer = []
        chunk_id = 0
        padding = self.model_setting.chunk_padding
        while True:
            latent = await latent_queue.get()
            if latent is None:
                break
            latent_buffer.append(latent)

            if chunk_id == 0 and len(latent_buffer) == self.model_setting.first_chunk_size:
                latent = torch.stack(latent_buffer, 0).unsqueeze(0).to('cuda')
                trim_end = True if self.model_setting.overlap_window > 0 else False
                audio_data_buffer.append(self.audio_generator.generate_chunk_audio(latent, metrics, padding=False, trim_begin=False, trim_end=trim_end))
                logger.debug(f'Chunk audio generated for promot {id} chunk {chunk_id}...')
                chunk_id += 1
                if trim_end:
                    latent_buffer = latent_buffer[-self.model_setting.overlap_window*2:]
                else:
                    latent_buffer = []

            elif len(latent_buffer) == self.model_setting.chunk_size:
                latent = torch.stack(latent_buffer, 0).unsqueeze(0).to('cuda')
                trim_begin = trim_end = True if self.model_setting.overlap_window > 0 else False
                audio_data_buffer.append(self.audio_generator.generate_chunk_audio(latent, metrics, padding=False, trim_begin=trim_begin, trim_end=trim_end))
                logger.debug(f'Chunk audio generated for promot {id} chunk {chunk_id}...')
                chunk_id += 1
                if trim_end:
                    latent_buffer = latent_buffer[-self.model_setting.overlap_window*2:]
                else:
                    latent_buffer = []

        if len(latent_buffer) > 0:
            latent = torch.stack(latent_buffer, 0).unsqueeze(0).to('cuda')
            # the last token is EOS, should be excluded
            latent = latent[:, :-1, :]
            trim_begin = True if self.model_setting.overlap_window > 0 else False
            audio_data_buffer.append(self.audio_generator.generate_chunk_audio(latent, metrics, padding=padding, trim_begin=trim_begin, trim_end=False))
            logger.debug(f'Chunk audio generated for promot {id} chunk {chunk_id}...')
            
        save_audio(audio_data_buffer, f'{output_dir}/{id}.wav', self.model_setting.cut_tail)
        logger.debug(f'Audio generated for prompt {id}.')

    async def generate_streaming(self, engine: AsyncLLMEngine,
                                 prompt: dict[str, Any],
                                 sampling_params: SamplingParams,
                                 request_id: str,
                                 output_dir: str,
                                 lora_request: LoRARequest = None) -> TtsMetrics:
        metrics = TtsMetrics(chunk_size=self.model_setting.chunk_size, first_chunk_size=self.model_setting.first_chunk_size)
        metrics.time_start = time.perf_counter()
        
        latent_queue = asyncio.Queue()
        vllm_task = asyncio.create_task(self.generate_token_streaming(engine, prompt, lora_request, request_id, sampling_params, latent_queue, metrics))
        generator_task = asyncio.create_task(self.generate_audio_streaming(latent_queue, request_id, metrics, output_dir))
        await vllm_task
        await generator_task
        metrics.time_end = time.perf_counter()
        return metrics

    async def synthesize_async(self, texts: List[str], output_dir: str, lora_path: str, request_rate: float = -1,
                               top_k: int = 1, top_p: float = 1, temperature: float = 1.0):
        if isinstance(texts, str):
            texts = [texts]
        logger.info(f'Synthesizing {len(texts)} texts streaming...')
        sampling_params = SamplingParams(detokenize=False, stop_token_ids=[1025], ignore_eos=True, max_tokens=2048,
                                            repetition_penalty=1.5, repetition_window=16, 
                                            top_k=top_k, top_p=top_p, temperature=temperature)
        prompts = self.text_to_prompts(texts)
        lora_request = None
        if lora_path:
            lora_request=LoRARequest("lora", 1, lora_local_path=lora_path)
        if request_rate < 0:
            for i in range(len(prompts)):
                me = await self.generate_streaming(self.llm_engine, prompts[i], sampling_params, f'{i:03d}', output_dir, lora_request=lora_request)
                me.calc_streaming()
        else:
            tasks: List[asyncio.Task] = []
            request_id = 0
            async for prompt in get_request(prompts, request_rate):
                tasks.append(asyncio.create_task(self.generate_streaming(self.llm_engine, prompt, sampling_params, f'{request_id:03d}', output_dir, lora_request=lora_request)))
                request_id += 1
            
            metrics_list: List[TtsMetrics] = await asyncio.gather(*tasks)
            for metrics in metrics_list:
                metrics.calc_streaming()