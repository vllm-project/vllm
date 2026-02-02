# Kimi-Audio upstream ASR golden reference

## Goal
Verify upstream (non-vLLM) Kimi-Audio produces the expected Chinese transcript for `test_audios/asr_example.wav`.

## Environment
- Host: gpu010
- CUDA_VISIBLE_DEVICES=1 (use a mostly-free GPU; avoid the GPU used by vLLM server)
- Python: `/root/learning/vllm/.venv/bin/python`
- Model path: `/data1/moonshotai/Kimi-Audio-7B-Instruct`

## Command
Run from `/root/learning/vllm/Kimi-Audio`:

```bash
CUDA_VISIBLE_DEVICES=1 /root/learning/vllm/.venv/bin/python - <<'PY'
from kimia_infer.api.kimia import KimiAudio

model = KimiAudio(
    model_path="/data1/moonshotai/Kimi-Audio-7B-Instruct",
    load_detokenizer=False,
)

sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

messages_asr = [
    {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
    {"role": "user", "message_type": "audio", "content": "test_audios/asr_example.wav"},
]

_, text_output = model.generate(messages_asr, **sampling_params, output_type="text")
print("ASR_OUTPUT=", text_output)
PY
```

## Expected output

```
ASR_OUTPUT= 这并不是告别，这是一个篇章的结束，也是新篇章的开始。
```
