# Kimi-Audio Pipeline Parallel Fix (CORRECTED)

## Problem

Kimi-Audio produced **different transcription outputs** with pipeline parallel (PP):
- **Without PP**: `"这并不是告别这是一个篇章的结束也是新篇章的开始"`
- **With PP**: `"这不，我刚刚在街上看到一个很有趣的现象。"`

## Root Cause

Kimi-Audio's `embed_input_ids` used **in-place tensor slice assignment**:

```python
# BROKEN: In-place slice assignment
inputs_embeds[pos:end_pos, :] = fused_embeds
```

This conflicts with vLLM v1 engine's buffer management for pipeline parallel.

## Solution: Use `masked_scatter_` (PP-Safe)

The fix preserves Kimi-Audio's fusion formula `(text + audio) * √2` but uses `masked_scatter_` instead of slice assignment.

### File: `/root/learning/vllm/vllm/model_executor/models/kimi_audio.py`

#### `embed_input_ids` Method

**BEFORE (Broken with PP):**
```python
def embed_input_ids(self, input_ids, multimodal_embeddings, ...):
    inputs_embeds = self.language_model.model.embed_tokens(input_ids)
    if multimodal_embeddings is None:
        return inputs_embeds
    
    audio_embeds = multimodal_embeddings[0]
    scale_factor = 2**0.5
    audio_mask = input_ids == KIMIA_TEXT_BLANK
    
    if audio_mask.any():
        pos = audio_mask.nonzero()[0][0] + 1
        end_pos = pos + audio_embeds.shape[0]
        text_embeds = inputs_embeds[pos:end_pos, :]
        fused_embeds = (text_embeds + audio_embeds) * scale_factor
        inputs_embeds[pos:end_pos, :] = fused_embeds  # ❌ In-place!
    
    return inputs_embeds
```

**AFTER (Fixed for PP):**
```python
def embed_input_ids(self, input_ids, multimodal_embeddings, is_multimodal, ...):
    inputs_embeds = self._embed_text_input_ids(
        input_ids,
        self.language_model.model.embed_input_ids,
        is_multimodal=is_multimodal,
    )

    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds

    audio_embeds = multimodal_embeddings[0]
    scale_factor = 2**0.5

    # Use masked_scatter_ for PP-safe fusion
    # Kimi-Audio formula: (text + audio) * √2
    if is_multimodal is not None and is_multimodal.any():
        text_at_audio = inputs_embeds[is_multimodal]
        fused = (text_at_audio + audio_embeds) * scale_factor
        inputs_embeds = inputs_embeds.masked_scatter(is_multimodal.unsqueeze(-1), fused)
    else:
        # Fallback: find audio positions from input_ids
        audio_mask = input_ids == KIMIA_TEXT_BLANK
        if audio_mask.any():
            text_at_audio = inputs_embeds[audio_mask]
            fused = (text_at_audio + audio_embeds) * scale_factor
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask.unsqueeze(-1), fused)

    return inputs_embeds
```

## Key Differences from Qwen2-Audio/Qwen3-ASR

| Model | Fusion Formula | Method |
|-------|---------------|--------|
| Qwen2-Audio | `inputs_embeds[audio_pos] = audio_embeds` | Replace |
| Qwen3-ASR | `inputs_embeds[audio_pos] = audio_embeds` | Replace |
| **Kimi-Audio** | `(inputs_embeds[audio_pos] + audio_embeds) * √2` | **Add + Scale** |

Kimi-Audio has a **unique fusion formula** that adds text and audio embeddings with √2 scaling, unlike other models that just replace.

## Why `masked_scatter_` Works

1. **PP-Safe**: Designed for distributed tensor operations
2. **No buffer aliasing**: Doesn't conflict with v1 engine's pre-allocated buffers
3. **Preserves semantics**: Still computes `(text + audio) * √2`
4. **Uses `is_multimodal` mask**: Aligned with vLLM's multimodal position tracking

## Testing

```bash
# Restart server
pkill -f "vllm serve"

# Test with PP
vllm serve /data1/moonshotai/Kimi-Audio-7B-Instruct \
  --trust-remote-code --port 8090 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --chat-template template_kimi_audio.jinja

# Test transcription
curl -sS http://localhost:8090/v1/audio/transcriptions \
 -F file=@/root/learning/Kimi-Audio/test_audios/asr_example.wav \
 -F model=/data1/moonshotai/Kimi-Audio-7B-Instruct \
 -F language=zh -F temperature=0 \
 -F prompt="请撰写这段语音：" -F max_completion_tokens=96
```

**Expected Output:**
```json
{"text":"这并不是告别这是一个篇章的结束也是新篇章的开始",...}
```

## Notes

- The fix preserves Kimi-Audio's unique `(text + audio) * √2` fusion
- Uses `masked_scatter_` which is the standard approach for PP-safe tensor operations
- Should work with TP=1,PP=2 and TP=2,PP=2 configurations
