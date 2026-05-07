# vLLM render API

The vLLM render API is designed to disaggregate the render phase(preprocessing) and enable a token-in / token-out API server.

- GPU-less deployment of frontend: Allow preprocessing (tokenization, MM input processing) and postprocessing (detokenization, tool call parsing, reasoning parsing) to run without GPU.
- Disaggregated tokenization: Support use cases such as llm-d, Dynamo, and custom frontends that need to leverage vLLM's preprocessing logic without running the full inference engine.
- Tokens-in / tokens-out engine: Make the engine a pure token-in / token-out service, decoupled from request preprocessing.

## Render API

- Completions render API (`/v1/completions/render`)
   - Render completion requests
- Chat Completions render API (`/v1/chat/completions/render`)
   - Render Chat Completions

## Tokenizer API

Our Tokenizer API is a simple wrapper over [HuggingFace-style tokenizers](https://huggingface.co/docs/transformers/en/main_classes/tokenizer).
It consists of two endpoints:

- `/tokenize` corresponds to calling `tokenizer.encode()`.
- `/detokenize` corresponds to calling `tokenizer.decode()`.