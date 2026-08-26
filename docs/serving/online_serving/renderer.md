# Renderer APIs

Our renderer API is designed to disaggregate the render phase(preprocessing) and enable a token-in / token-out API server.

- GPU-less deployment of frontend: Allow preprocessing (tokenization, MM input processing) and postprocessing (detokenization, tool call parsing, reasoning parsing) to run without GPU.
- Disaggregated tokenization: Support use cases such as llm-d, Dynamo, and custom frontends that need to leverage vLLM's preprocessing logic without running the full inference engine.
- Tokens-in / tokens-out engine: Make the engine a pure token-in / token-out service, decoupled from request preprocessing.

## API Reference

- [Completions Render API](renderer.md) (`/v1/completions/render`)
    - Render completion requests
- [Chat Completions Render API](renderer.md) (`/v1/chat/completions/render`)
    - Render chat completions

## Shipping source image bytes instead of tensors

By default a multimodal render response carries the processed tensors inline: `features.kwargs_data` holds a base64-encoded `pixel_values` (and friends) per item, which is typically hundreds of KB to several MB per image. Setting `skip_pixel_values: true` on a render request returns the **original encoded image bytes** in `features.raw_images` instead, and leaves `features.kwargs_data` null. The two fields are mutually exclusive.

Everything else is unchanged: the renderer still runs the HF processor, because `mm_placeholders` lengths are derived from the processed tensor shapes. What the flag saves is serialization and wire bytes, not frontend CPU.

```bash
curl http://localhost:8000/v1/chat/completions/render -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3-VL-2B-Instruct",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    {"type": "text", "text": "What is in this image?"}
  ]}],
  "skip_pixel_values": true
}'
```

The response can be posted to `/inference/v1/generate` unchanged. That tier reloads each image from its bytes and re-runs the processor, consulting the multimodal cache first, so repeated images cost nothing.

Constraints:

- **Images only.** Video hashes fold in frame-sampling metadata, and audio and prompt embeds never retain source bytes. If any item in the request lacks them, the whole response falls back to `kwargs_data` and logs a warning.
- **Both tiers must run the same model and the same `--media-io-kwargs`.** The generate tier recomputes `mm_hashes` from the bytes and rejects the request with a 400 if they disagree with the render tier's.
- **`mm_processor_kwargs` are not carried over** to `/inference/v1/generate`; a render request that used them is caught by the same hash check.

For the post processing counterpart that turns generated token IDs back into OpenAI compatible responses, see the [Derenderer APIs](derenderer.md).
