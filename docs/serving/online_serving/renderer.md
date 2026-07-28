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
- [Responses Render API](renderer.md) (`/v1/responses/render`)
    - Render a self-contained Responses request

The Responses render endpoint uses the same prompt construction as
`/v1/responses` and returns one token-in `GenerateRequest`. It is stateless:
inline history is supported, but `previous_response_id` is not. Callers must
resolve stored response state and include the resulting history in the request
before rendering.

```bash
curl http://localhost:8000/v1/responses/render \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "input": "Explain prefix caching in one sentence.",
        "max_output_tokens": 32
    }'
```

For the post processing counterpart that turns generated token IDs back into OpenAI compatible responses, see the [Derenderer APIs](derenderer.md).
