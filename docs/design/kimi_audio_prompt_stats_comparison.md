# Kimi-Audio prompt stats comparison (online vs prompt manager)

## Setup
- Server flags:
  - `--enable-mm-embeds --skip-mm-profiling --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 1`
  - `CUDA_VISIBLE_DEVICES=2 VLLM_WORKER_MULTIPROC_METHOD=spawn`
- Request:
  - `POST /v1/audio/transcriptions`
  - file: `/root/workspace/Kimi-Audio/test_audios/asr_example.wav`
  - prompt: `Please transcribe the following audio:`

## Online prompt stats (vLLM log)
```
[Kimi-Audio] prompt_stats request_id=transcribe-a5c7d57250c7f457_0 item=0 \
  audio_len=143 audio_hash=884bc5e0 text_len=143 text_hash=77a41f0d \
  mask_true=130 whisper_shape=(143, 5120) placeholder_len=143
```

## Offline prompt manager stats
Command:
```
PYTHONPATH=/root/workspace/Kimi-Audio:$PYTHONPATH \
  /root/learning/vllm/.venv/bin/python - <<'PY'
# see logs in progress.txt for full script
PY
```
Output:
```
offline_prompt_stats audio_len=143 audio_hash=d80d1734 text_len=143 text_hash=77a41f0d \
  mask_true=130 whisper_shape=(1, 130, 5120) placeholder_len=143
```

## Mismatch summary
- `text_input_ids`: **match** (len=143, hash=77a41f0d)
- `audio_input_ids`: **mismatch** (hash 884bc5e0 online vs d80d1734 offline)
- `is_continuous_mask`: **match** (mask_true=130)
- `whisper_input_features`: **mismatch**
  - online shape: `(143, 5120)`
  - offline shape: `(1, 130, 5120)`
- `placeholder_len`: **match** (143)
