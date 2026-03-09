#!/usr/bin/env python3
"""Debug script to check Kimi-Audio with pipeline parallel."""

import torch
from vllm import LLM

# Test without PP first
print("=" * 80)
print("Testing Kimi-Audio WITHOUT pipeline parallel (TP=2)")
print("=" * 80)

try:
    llm_no_pp = LLM(
        model="/data1/moonshotai/Kimi-Audio-7B-Instruct",
        trust_remote_code=True,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        enforce_eager=True,
    )
    
    from transformers import AutoProcessor
    import soundfile as sf
    
    # Load audio
    audio, sr = sf.read("/root/learning/Kimi-Audio/test_audios/asr_example.wav")
    
    # Create prompt
    prompt = "<|im_kimia_user_msg_start|>请撰写这段语音：<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|><|im_msg_end|><|im_kimia_assistant_msg_start|>"
    
    outputs = llm_no_pp.generate(
        [{"prompt": prompt, "multi_modal_data": {"audio": (audio, sr)}}],
        max_tokens=96,
        temperature=0,
    )
    
    print(f"NO PP Output: {outputs[0].outputs[0].text}")
    
except Exception as e:
    print(f"NO PP Error: {e}")

print("\n" + "=" * 80)
print("Testing Kimi-Audio WITH pipeline parallel (TP=1, PP=2)")
print("=" * 80)

try:
    llm_pp = LLM(
        model="/data1/moonshotai/Kimi-Audio-7B-Instruct",
        trust_remote_code=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        enforce_eager=True,
    )
    
    outputs = llm_pp.generate(
        [{"prompt": prompt, "multi_modal_data": {"audio": (audio, sr)}}],
        max_tokens=96,
        temperature=0,
    )
    
    print(f"WITH PP Output: {outputs[0].outputs[0].text}")
    
except Exception as e:
    print(f"WITH PP Error: {e}")

print("\nDone!")
