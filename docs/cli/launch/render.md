# vllm launch render

## Overview

`vllm launch render` starts a GPU-less rendering server for preprocessing and
postprocessing only.

```bash
vllm launch render meta-llama/Llama-3.2-1B-Instruct --port 8100
```

This command reuses the standard serving parser, so model, frontend,
networking, and related CLI options follow the same conventions as
[`vllm serve`](../serve.md).

## JSON CLI Arguments

--8<-- "docs/cli/json_tip.inc.md"

## Arguments

--8<-- "docs/generated/argparse/launch_render.inc.md"
