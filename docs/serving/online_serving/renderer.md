# Renderer APIs

Our renderer API is designed to disaggregate the render phase(preprocessing) and enable a token-in / token-out API server.

- GPU-less deployment of frontend: Allow preprocessing (tokenization, MM input processing) and postprocessing (detokenization, tool call parsing, reasoning parsing) to run without GPU.
- Disaggregated tokenization: Support use cases such as llm-d, Dynamo, and custom frontends that need to leverage vLLM's preprocessing logic without running the full inference engine.
- Tokens-in / tokens-out engine: Make the engine a pure token-in / token-out service, decoupled from request preprocessing.

The dedicated `vllm launch render` server always exposes the `/render` and
`/derender` endpoints when `VLLM_ENABLE_SCALE_OUT_ENDPOINTS` is unset or set to
`1`. An explicit value of `0` conflicts with the renderer command and is
rejected at startup.

Scale-out endpoints, including `/render`, `/derender`, and
`/inference/v1/generate`, are disabled by default on a standard inference
server. To expose them with `vllm serve`, opt in explicitly:

```bash
VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1 vllm serve <model>
```

## API Reference

- [Completions Render API](renderer.md) (`/v1/completions/render`)
    - Render completion requests
- [Chat Completions Render API](renderer.md) (`/v1/chat/completions/render`)
    - Render chat completions

For the post processing counterpart that turns generated token IDs back into OpenAI compatible responses, see the [Derenderer APIs](derenderer.md).
