# Qwen2.5-Omni Offline Inference Examples

This folder provides several example scripts on how to inference Qwen2.5-Omni offline.

## Thinker Only

```bash
# Audio + image + video
python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
    -q mixed_modalities

# Read vision and audio inputs from a single video file
python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
    -q use_audio_in_video

# Multiple audios
python examples/generate/multimodal/qwen2_5_omni/only_thinker.py \
    -q multi_audios
```

This script will run the thinker part of Qwen2.5-Omni, and generate text response.

You can also test Qwen2.5-Omni on a single modality:

```bash
# Process audio inputs
python examples/generate/multimodal/audio_language_offline.py \
    --model-type qwen2_5_omni

# Process image inputs
python examples/generate/multimodal/vision_language_offline.py \
    --modality image \
    --model-type qwen2_5_omni

# Process video inputs
python examples/generate/multimodal/vision_language_offline.py \
    --modality video \
    --model-type qwen2_5_omni
```
