# Model Bash

This directory contains scripts for running various performance analyses in vLLM to facilitate profiling.

We use Justfiles for our workflow:

```bash
uv pip install rust-just
```


## Generating Decode Traces

- launch vllm

```bash
TP_SIZE=1
just launch {{TP_SIZE}}
```



