# vllm launch render

Launch a GPU-less rendering server that runs the full request preprocessing
pipeline — chat template rendering, tokenization, and tool/reasoning parsing —
without any GPU or inference engine.

`vllm launch render` accepts the same arguments as [`vllm serve`](./serve.md).

## Arguments

--8<-- "docs/generated/argparse/serve.inc.md"
