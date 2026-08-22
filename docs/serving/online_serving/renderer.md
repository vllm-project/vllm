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

## Standalone Docker image

The Rust renderer can run as a standalone Docker image without Python, PyTorch,
CUDA, model weights, or vLLM kernels:

```bash
docker buildx bake -f docker/docker-bake.hcl rust-renderer

docker run --rm -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm:rust-renderer \
    render Qwen/Qwen3-32B --host 0.0.0.0 --max-model-len 32768
```

The container entrypoint is `vllm-rs`, so additional `render` flags can be
passed after the image name.
