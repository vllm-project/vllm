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

For the post processing counterpart that turns generated token IDs back into OpenAI compatible responses, see the [Derenderer APIs](derenderer.md).

## Multimodal Render Features

Multimodal render responses include a `features` object with per-modality
hashes, placeholder ranges, and serialized processor data. When the model
exposes placeholder-metadata or `keep_on_cpu` fields (for example
`image_grid_thw`), the response also includes `mm_metadata`. Each
`mm_metadata` entry is a base64-encoded `MultiModalKwargsItem` containing
only those fields, not encoder inputs such as `pixel_values`.

The arrays in `mm_hashes`, `mm_placeholders`, `kwargs_data`, and
`mm_metadata` use the same per-modality item order. Downstream workers
should split those fields:

- Encode requests keep `kwargs_data`.
- Prefill requests may omit `kwargs_data` and send `mm_metadata` only when
  `ec_transfer_params` is also set, so embeddings are loaded by the EC
  connector. Omitting `kwargs_data` without `ec_transfer_params` is rejected.
- Legacy clients that ignore `mm_metadata` and keep sending `kwargs_data`
  continue to work.
