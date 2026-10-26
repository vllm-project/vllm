# Remote draft placement (experimental)

Remote placement separates a target-conditioned speculator from the target
process. The target remains responsible for verification and sampling. Placement
is configured independently of the speculative method; native local execution
remains the default.

The standalone GPU runtime provides same-host target-conditioned execution with
independent draft KV. It supports greedy, eager MRV2 execution, target TP1 or TP2
and draft TP1. The protocol test adapter is for development only.

## Configuration and lifecycle

`speculative_config.remote_draft` takes a single `endpoint`, a `transport`
(`auto`, `inline` or `cuda_ipc`) and a `failure_policy` (`target_only` or `error`). `auto`
selects inline. Use a Unix socket or loopback TCP endpoint on the same host.
The protocol does not authenticate peers; non-loopback server binding requires
explicit `--allow-remote-connections`.

The default request budget is 100 ms and the initial connection budget is
30,000 ms. A request timeout under `target_only` permanently disables remote
speculation for affected requests, returning zero valid drafts while the target
continues. `error` surfaces the failure. Reconnection may admit new identities;
it does not rebuild the old requests' draft KV. Generation and session fencing
prevent late replies from reactivating retired state.

Protocol v2.2 assigns one server execution deadline to each pipelined prefill
group. Waiting behind an earlier chunk consumes that group's remaining budget.
With older peers, the client waits for each prefill ACK before sending the next
chunk, using the same round deadline. Already executing GPU work is not cancelled.
When buffered final prompt chunks require multiple draft forwards, they also
share the proposal deadline; expired rows do not start another forward.

Feature schema and frame byte budgets are checked before tensor serialization.
Connections, partial frames, pending bytes and adapter waiters have server-side
bounds. A shutdown grace timeout does not authorize freeing GPU storage whose
completion is unknown; the process must be restarted in that state.

This placement does not provide predictive caching, cross-node execution or
N:M routing. Moving a speculator to another GPU does not by itself overlap its
computation with target verification or guarantee higher throughput.

## Standalone GPU example

For MiMo-7B-RL MTP K=1, start the draft process on its assigned GPU:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 .venv/bin/python -m vllm.v1.spec_decode.remote.server \
  --adapter gpu --endpoint unix:///tmp/vllm-draft.sock --device cuda:1 \
  --target-model XiaomiMiMo/MiMo-7B-RL --draft-model XiaomiMiMo/MiMo-7B-RL \
  --method mtp --num-speculative-tokens 1 --dtype bfloat16 \
  --max-batch-size 8 --max-model-len 4096 --max-feature-tokens 512 \
  --draft-kv-cache-gib 1 --block-size 16 --attention-backend FLASH_ATTN \
  --target-attention-backend FLASH_ATTN
```

Use immutable target/draft revisions for reproducible evaluations. The server
loads and warms its actual prefill/decode path before listening. Required shared
weights are validated without loading the target transformer stack.

Configure the target on GPU 0 with the matching model, revisions and limits:

```python
from vllm import LLM

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-RL",
    max_model_len=4096,
    max_num_seqs=8,
    max_num_batched_tokens=512,
    enable_prefix_caching=False,
    async_scheduling=False,
    enforce_eager=True,
    attention_config={"backend": "FLASH_ATTN"},
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
        "draft_sample_method": "greedy",
        "attention_backend": "FLASH_ATTN",
        "remote_draft": {
            "endpoint": "unix:///tmp/vllm-draft.sock",
            "transport": "inline",
            "request_timeout_ms": 5000,
            "failure_policy": "error",
        },
    },
)
```

Set `VLLM_USE_V2_MODEL_RUNNER=1` for the target too. Both processes must agree on
target/draft device assignments and checkpoint/schema identities. The example's
5000 ms budget is explicit; it does not establish that the 100 ms default is
suitable for long prefills. EAGLE3 currently accepts GPT-OSS and Qwen3 targets;
see [own-KV MTP contracts](remote_mtp.md) for the experimental MTP families.
Prefix caching, input streaming, async scheduling, multimodal input, LoRA and
`draft_lm_head_quantization` are rejected for this backend. Native local speculation retains its own support.

## CUDA IPC feature transport

On a single NVIDIA host, `transport="cuda_ipc"` explicitly selects bounded
CUDA IPC arenas for target features. `auto` continues to select inline.
The receiver copies features into independently owned GPU storage before
acknowledging a lease; this is not zero-copy. Both processes must see the
source GPU, and the actual device UUIDs and peer-copy support are validated.

Producer arenas are reserved before target KV profiling. A timeout, disconnect
or cancel does not prove a device copy has finished: uncertain storage is
quarantined, further admission is fenced, and the process requires restart.
Ordinary request failures preserve the selected target-only or error policy.

For a historical two-GPU correctness, throughput and memory comparison, see
[validation snapshot](remote_placement_validation.md). Its measurements belong
to the recorded source revision and do not validate subsequent commits.
