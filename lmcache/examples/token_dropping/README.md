# Token Dropping Examples using LMCache SDK

Long prompts create large KV caches that eat up GPU memory and limit how many
requests fit in a batch. Smaller batch means lower decode throughput. To 
improve decode throughput, we then need to stuff more requests in a batch.
*Token dropping*, analogous to its name, select tokens to drop (by half in
the two examples) to shrink each request's KV cache and improve decode
throughput by 1.5-1.7x. The example also demonstrates that the generation
accuracy is unaffected, even improved, by a good token dropping algorithm 
(SnapKV was chosen for this demonstration).

The two examples use the LMCache SDK to do this: the SDK **retrieves** a 
request's cached tensors, **modifies** them, and **stores** them back for vLLM
to decode from. Users only need to supply the token dropping function, and the
SDK's batch and stream APIs does the job in an offline manner.

There is also an example meant to be run in Google Colab's GPU T4 which uses
smaller model and 

## Examples

| Notebook | Strategy | Needs query tensor? |
| --- | --- | --- |
| [random_token_dropping.ipynb](./random_token_dropping.ipynb) | Drops a random subset of past tokens. Uses the KV cache only. | No |
| [snapkv_token_dropping.ipynb](./snapkv_token_dropping.ipynb) | SnapKV: keeps the first and last window, as well as the tokens the recent-window queries attend to most. Needs each request's query tensor to score importance. | Yes |

[snapkv_colab.ipynb](./snapkv_colab.ipynb) is a demonstration done in Google
Colab Notebook, which can also be accessed here:
[Google Colab SnapKV SDK Example](https://colab.research.google.com/drive/1JtyqhRIqmACDoQ7PKra1QszsRvE2wA7I?usp=sharing).

## Prerequisites

* **GPU.** To see token dropping raise the decode batch size, tune
  `--gpu-memory-utilization` together with the number of requests. These
  examples were run on a single RTX 6000 PRO.
* **LMCache**. The SDK can use either shared memory transport or pickle 
  transport. This example uses shared memory. To transfer the query tensors,
  pass `--enable transfer_query` when starting LMCache.
* **vLLM** with below patch.

### vLLM patch to expose intermediate tensor

Many token dropping algorithms need the query tensor to rank the importance of
tokens. SnapKV is one of it. vLLM does not expose the intermediate tensors to 
the KV connector by default. A 10-line change adds it:

```sh
cd /path/to/vllm
git apply /path/to/LMCache/examples/token_dropping/vllm-export-intermediate-tensors.diff
```

When booting up vLLM, activate the code path by adding 
`lmcache.mp.transfer_intermediate_tensors` to the connector config.
By default, the QRingBuffer, a temporary staging buffer for containing
query tensor, has the capacity to hold the query tensor of 2 forward
passes. However, it can also be configured via `"lmcache.mp.q.ring_depth":2`.

```json
--kv-transfer-config '{
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lmcache.mp.port": 6555,
        "lmcache.mp.transfer_intermediate_tensors": true,
        "lmcache.mp.q.ring_depth":2
    }
}'
```

The random-dropping example, being the simplest example that demonstrates 
decode throughput improvement, does **not** need this patch or flag. It only
works with the KV cache.

This patch has been tested on vLLM versions 0.23.0 until 0.25.1.

## Dataset

The notebooks load the 
[`raniayu/token-dropping-demo`](https://huggingface.co/datasets/raniayu/token-dropping-demo)
dataset. The samples included in this dataset are taken from
[LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) by
choosing 30 examples whose prompt is closest to 10240 tokens (Qwen3-8B tokenizer).

The Google Colab notebook loads a shorter dataset, adjusted to GPU T4's capacity, 
[`raniayu/token-dropping-demo-short`](https://huggingface.co/datasets/raniayu/token-dropping-demo-short).
This short dataset is taken from 
[ehovy/race](https://huggingface.co/datasets/ehovy/race) 
by choosing 10 examples whose prompt length is closest to 1024 tokens and
having unique prefix contexts.
