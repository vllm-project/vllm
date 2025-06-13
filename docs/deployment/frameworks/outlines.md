---
title: Outlines
---
[](){ #deployment-outlines }

[Outlines](https://github.com/dottxt-ai/outlines) is a Python library that allows you to use Large Language Model in a simple and robust way (with structured generation). It is built by .txt, and is already used in production by many companies.

It allows you to deploy a large language model (LLM) server with vLLM as the backend, which exposes OpenAI-compatible endpoints.

Outlines supports models available via vLLM's offline batched inference interface.

## Integration

- Setup vLLM and outlines environment (suggest Linux platform).

```
pip install "outlines[vllm]"
```

- Use `vLLM` in `Outlines`.

```
from vllm.sampling_params import SamplingParams
from outlines import models, generate

# Load the model
model = models.vllm("microsoft/Phi-3-mini-4k-instruct")

# Generate text
generator = generate.text(model)
params = SamplingParams(n=2, frequency_penalty=1., min_tokens=2)
answer = generator("Hi", sampling_params=params)
print(answer)
```

For details, see the tutorial [Using vLLM in Outlines](https://dottxt-ai.github.io/outlines/latest/reference/models/vllm/).

## Use OpenAI and compatible API

- Setup vLLM and outlines environment (suggest Linux platform).

```
pip install vllm
pip install "outlines[openai]"
```

- Start the vLLM server with the supported chat completion model, e.g.

```console
vllm serve qwen/Qwen1.5-0.5B-Chat
```

- Set the environment variables:

```
export OPENAI_API_KEY="empty"
OPENAI_BASE_URL=http://{your-vllm-server-host}:{your-vllm-server-port}/v1
```

- Call the `vLLM` API with `Outlines`.

```
import os
from outlines import models, generate

model = models.openai(
    "qwen/Qwen1.5-0.5B-Chat",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)
generator = generate.text(model)
answer = generator("hi")
print(answer)
```

For details, see the tutorial [OpenAI and compatible APIs](https://dottxt-ai.github.io/outlines/latest/reference/models/openai/).
