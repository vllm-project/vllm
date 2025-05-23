---
title: dstack
---
[](){ #deployment-dstack }

<p align="center">
    <img src="https://i.ibb.co/71kx6hW/vllm-dstack.png" alt="vLLM_plus_dstack"/>
</p>

vLLM can be run on a cloud based GPU machine with [dstack](https://dstack.ai/), an open-source framework for running LLMs on any cloud. This tutorial assumes that you have already configured credentials, gateway, and GPU quotas on your cloud environment.

To install dstack client, run:

```console
pip install "dstack[all]
dstack server
```

Next, to configure your dstack project, run:

```console
mkdir -p vllm-dstack
cd vllm-dstack
dstack init
```

Next, to provision a VM instance with LLM of your choice (`NousResearch/Llama-2-7b-chat-hf` for this example), create the following `serve.dstack.yml` file for the dstack `Service`:

```yaml
type: service

python: "3.11"
env:
    - MODEL=NousResearch/Llama-2-7b-chat-hf
port: 8000
resources:
    gpu: 24GB
commands:
    - pip install vllm
    - vllm serve $MODEL --port 8000
model:
    format: openai
    type: chat
    name: NousResearch/Llama-2-7b-chat-hf
```

Then, run the following CLI for provisioning:

```console
$ dstack run . -f serve.dstack.yml

⠸ Getting run plan...
 Configuration  serve.dstack.yml
 Project        deep-diver-main
 User           deep-diver
 Min resources  2..xCPU, 8GB.., 1xGPU (24GB)
 Max price      -
 Max duration   -
 Spot policy    auto
 Retry policy   no

 #  BACKEND  REGION       INSTANCE       RESOURCES                               SPOT  PRICE
 1  gcp   us-central1  g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
 2  gcp   us-east1     g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
 3  gcp   us-west1     g2-standard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (disk)  yes   $0.223804
    ...
 Shown 3 of 193 offers, $5.876 max

Continue? [y/n]: y
⠙ Submitting run...
⠏ Launching spicy-treefrog-1 (pulling)
spicy-treefrog-1 provisioning completed (running)
Service is published at ...
```

After the provisioning, you can interact with the model by using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://gateway.<gateway domain>",
    api_key="<YOUR-DSTACK-SERVER-ACCESS-TOKEN>"
)

completion = client.chat.completions.create(
    model="NousResearch/Llama-2-7b-chat-hf",
    messages=[
        {
            "role": "user",
            "content": "Compose a poem that explains the concept of recursion in programming.",
        }
    ]
)

print(completion.choices[0].message.content)
```

!!! note
    dstack automatically handles authentication on the gateway using dstack's tokens. Meanwhile, if you don't want to configure a gateway, you can provision dstack `Task` instead of `Service`. The `Task` is for development purpose only. If you want to know more about hands-on materials how to serve vLLM using dstack, check out [this repository](https://github.com/dstackai/dstack-examples/tree/main/deployment/vllm)
