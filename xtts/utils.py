import asyncio
import logging
from typing import List

import numpy as np
import soundfile as sf
import torch
import re

logger = logging.getLogger(__name__)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def save_audio(total_audio: List[np.ndarray], path: str, cut_tail: int = 0):
    total_audio = np.concatenate(total_audio, axis=0)
    if cut_tail > 0:
        total_audio = total_audio[:-cut_tail * 24]
    sf.write(path, total_audio, 24000)

async def get_request(input_requests, request_rate: float):
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

def convert_model(path_torch: str, path_vllm: str):
    tts2 = torch.load(path_torch)

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

    # 0-1023: audio1, 1024 bos 1026 eos
    # 1027-2050: audio2, 2051 bos 2053 eos
    
    # 0-1023: audio1, 1024-2047: audio2
    # bos1 2048, eos1 2049
    # bos2 2050, eos2 2051
    llama['emb_code.0.weight'] = llama['code_embeddings.weight'][0:num_audio_tokens].clone()
    llama['emb_code.1.weight'] = llama['code_embeddings.weight'][num_audio_tokens-2:num_audio_tokens - 2 + num_audio_tokens].clone()
    llama['emb_code.0.weight'][1024]=llama['code_embeddings.weight'][2048]
    llama['emb_code.1.weight'][1024]=llama['code_embeddings.weight'][2050]
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

    torch.save(llama, path_vllm)

def convert_model_lora(path_torch: str, path_vllm: str):
    lora = torch.load(path_torch)
    
    layer = 24
    dim = 1536
    for i in range(layer):
        qkv_name_A = f'layers.{i}.attention.wqkv.lora_A'
        q_A = lora[qkv_name_A]
        k_A = lora[qkv_name_A]
        v_A = lora[qkv_name_A]
        lora[f'base_model.model.gpt.layers.{i}.self_attn.q_proj.lora_A.weight'] = q_A
        lora[f'base_model.model.gpt.layers.{i}.self_attn.k_proj.lora_A.weight'] = k_A
        lora[f'base_model.model.gpt.layers.{i}.self_attn.v_proj.lora_A.weight'] = v_A
        lora.pop(qkv_name_A)

        qkv_name_B = f'layers.{i}.attention.wqkv.lora_B'
        q_B = lora[qkv_name_B][0:dim]
        k_B = lora[qkv_name_B][dim:2*dim]
        v_B = lora[qkv_name_B][2*dim:]
        lora[f'base_model.model.gpt.layers.{i}.self_attn.q_proj.lora_B.weight'] = q_B
        lora[f'base_model.model.gpt.layers.{i}.self_attn.k_proj.lora_B.weight'] = k_B
        lora[f'base_model.model.gpt.layers.{i}.self_attn.v_proj.lora_B.weight'] = v_B
        lora.pop(qkv_name_B)

        wo_name_A = f'layers.{i}.attention.wo.lora_A'
        wo_A = lora[wo_name_A]
        lora[f'base_model.model.gpt.layers.{i}.self_attn.o_proj.lora_A.weight'] = wo_A
        lora.pop(wo_name_A)
        
        wo_name_B = f'layers.{i}.attention.wo.lora_B'
        wo_B = lora[wo_name_B]
        lora[f'base_model.model.gpt.layers.{i}.self_attn.o_proj.lora_B.weight'] = wo_B
        lora.pop(wo_name_B)
        
        gate_proj_name_A = f'layers.{i}.feed_forward.w1.lora_A'
        w_gate_A = lora[gate_proj_name_A]
        lora[f'base_model.model.gpt.layers.{i}.mlp.gate_proj.lora_A.weight'] = w_gate_A
        lora.pop(gate_proj_name_A)
        
        gate_proj_name_B = f'layers.{i}.feed_forward.w1.lora_B'
        w_gate_B = lora[gate_proj_name_B]
        lora[f'base_model.model.gpt.layers.{i}.mlp.gate_proj.lora_B.weight'] = w_gate_B
        lora.pop(gate_proj_name_B)
        
        gate_up_proj_name_A = f'layers.{i}.feed_forward.w3.lora_A'
        w_gate_up_A = lora[gate_up_proj_name_A]
        lora[f'base_model.model.gpt.layers.{i}.mlp.up_proj.lora_A.weight'] = w_gate_up_A
        lora.pop(gate_up_proj_name_A)
        
        gate_up_proj_name_B = f'layers.{i}.feed_forward.w3.lora_B'
        w_gate_up_B = lora[gate_up_proj_name_B]
        lora[f'base_model.model.gpt.layers.{i}.mlp.up_proj.lora_B.weight'] = w_gate_up_B
        lora.pop(gate_up_proj_name_B)
        
        gate_down_proj_name_A = f'layers.{i}.feed_forward.w2.lora_A'
        w_gate_down_A = lora[gate_down_proj_name_A]
        lora[f'base_model.model.gpt.layers.{i}.mlp.down_proj.lora_A.weight'] = w_gate_down_A
        lora.pop(gate_down_proj_name_A)
        
        gate_down_proj_name_B = f'layers.{i}.feed_forward.w2.lora_B'
        w_gate_down_B = lora[gate_down_proj_name_B]
        lora[f'base_model.model.gpt.layers.{i}.mlp.down_proj.lora_B.weight'] = w_gate_down_B
        lora.pop(gate_down_proj_name_B)

    num_audio_tokens = 1026
    num_text_tokens = 7002
    output_name_A = 'output.lora_A'
    w_output_A = lora[output_name_A]
    lora['base_model.model.lm_head.0.lora_A.weight'] = w_output_A
    lora['base_model.model.lm_head.1.lora_A.weight'] = w_output_A
    lora.pop(output_name_A)
    
    output_name_B = 'output.lora_B'
    w_output_B = lora[output_name_B]
    lora['base_model.model.lm_head.0.lora_B.weight'] = w_output_B[num_text_tokens:num_text_tokens+num_audio_tokens]
    lora['base_model.model.lm_head.1.lora_B.weight'] = w_output_B[num_text_tokens+num_audio_tokens:num_text_tokens+num_audio_tokens*2]
    lora.pop(output_name_B)
    
    # convert the model to fp16
    for k, v in lora.items():
        if isinstance(v, torch.Tensor):
            lora[k] = v.half()
    
    torch.save(lora, path_vllm)

def mix_sentence_spliter(text):
    segments_with_punctuation = re.findall(r'[\u4e00-\u9fff0-9]+|[a-zA-Z\s\'-]+|[,.!?，。！？"“”：:;；—（）(){}]', text)
    combined_segments = []
    for i, segment in enumerate(segments_with_punctuation):
        if i > 0 and re.match(r'[,.!?，。！？"“”：:;；—（）(){}]', segment):
            combined_segments[-1] += segment
        else:
            if len(combined_segments) > 0 and re.match(r'[\u4e00-\u9fff0-9]', combined_segments[-1]) and re.match(
                    r'[\u4e00-\u9fff0-9]', segment):
                combined_segments[-1] += segment
            elif len(combined_segments) > 0 and re.match(r'[a-zA-Z\s\'-]', combined_segments[-1]) and re.match(
                    r'[a-zA-Z\s\'-]', segment):
                combined_segments[-1] += segment
            else:
                combined_segments.append(segment)

    out_combined_segments = []
    for segment in combined_segments:
        if segment.strip():
            if out_combined_segments and out_combined_segments[-1] in '.,!?，。！？"“”：:;；—（）(){}':
                out_combined_segments[-1] += segment.strip()
            else:
                out_combined_segments.append(segment.strip())

    return out_combined_segments

def text_normalizer(x, is_lastsegment=True):
    x['before_norm_text'] = x['text']
    if x['locale'] == "zh":
        x['text'] = x['text'].replace(" ", "")
        x['text'] = x['text'].replace('.', '。').replace('！', '!').replace('？', '?').replace('，', ',').replace('：', ':')
        x['text'] = x['text'].replace('“', '"').replace('”', '"').replace('‘', '"').replace('’', '"')
        if is_lastsegment:
            if len(x['text']) > 0 and x['text'][-1] == '"':
                x['text'] = x['text'][:-1]

            if len(x['text']) > 0 and x['text'][-1] == ',':
                x['text'] = x['text'][:-1] + '。'

            x['text'] = x['text'].replace('—', '。')
            if len(x['text']) > 0 and x['text'][-1] not in ['。', '!', '?']:
                x['text'] += '。'

        if re.search('[a-zA-Z]', x['before_norm_text']):
            x['text'] = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9。，,!?《》、:"\' ]', '', x['text'])
            x['text'] = re.sub(r'(?<=[\u4e00-\u9fa5。，,!?《》、:"\'])\s+(?=[\u4e00-\u9fa5。，,!?《》、:"\'])', '', x['text'])
            x['text'] = re.sub(r'(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z])', '', x['text'])
            x['text'] = re.sub(r'(?<=[a-zA-Z])\s+(?=[\u4e00-\u9fa5])', '', x['text'])
        else:
            x['text'] = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9。，,!?《》、:"\']', '', x['text'])

        x['text'] = re.sub(r'([。，,!?])\1+', r'\1', x['text'])
    else:
        x['text'] = re.sub(r'[^\w.,!?\'" ]', '', x['text'])

        if is_lastsegment:
            if len(x['text']) > 0 and x['text'][-1] == ',':
                x['text'] = x['text'][:-1] + '.'
            if len(x['text']) > 0 and x['text'][-1] not in ['.', '!', '?', '"', '\'']:
                x['text'] += '.'

        x['text'] = re.sub(r'([,!?])\1+', r'\1', x['text'])
        x['text'] = re.sub(r'\.{2,}', '...', x['text'])
        x['text'] = re.sub(r'\s+([.,!?"\'])', r'\1', x['text'])
        x['text'] = re.sub(r'([.,!?"\'])\s+', r'\1', x['text'])

    x['text'] = re.sub(r"\s+", ' ', x['text'].lower()).strip()
    return x['text']


# trtexec --onnx=/home/zhn/fishtts/genertor.onnx --saveEngine=/home/zhn/fishtts/genertor.trt --memPoolSize=workspace:10000 --minShapes=input:1x1x1536,speaker_embedding:1x192x1 --maxShapes=input:1x512x1536,speaker_embedding:1x192x1 --optShapes=input:1x20x1536,speaker_embedding:1x192x1 --device=0