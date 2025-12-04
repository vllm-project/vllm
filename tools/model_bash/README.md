# Model Bash

This directory contains scripts for running various performance analyses in vLLM to facilitate profiling.

We use Justfiles for our workflow:

```bash
uv pip install rust-just
```


## Generating Decode Traces

- launch vllm

```bash
export MODEL=openai/gpt-oss-20b
export TP_SIZE=1

just launch $MODEL $TP_SIZE
```

- generate trace for the batch size you want (4, 8, 16, ...):

```bash
export MODEL=openai/gpt-oss-20b
export BATCH_SIZE=16

just trace_decode $MODEL $BATCH_SIZE
```

- generate sweep of performance at various concurrencies:

```bash
export MODEL=openai/gpt-oss-20b
export INPUT_LEN=1000
export OUTPUT_LEN=1000

just sweep $MODEL $INPUT_LEN $OUTPUT_LEN
```



