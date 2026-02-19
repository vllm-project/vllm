# Attention Instrumentation Guide

vLLM's attention instrumentation lets you extract per-layer attention scores
from the OpenAI-compatible API during inference.

## Quick Start

### 1. Start the server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-4b-it \
  --enforce-eager \
  --enable-attention-instrumentation \
  --attention-instrumentation-layers 2,18,27 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching
```

> `--no-enable-chunked-prefill` and `--no-enable-prefix-caching` are required
> to capture attention for **all** prompt tokens. Without them vLLM may only
> buffer the last prefill chunk or skip cached tokens entirely.

### 2. Request attention scores

```python
import json
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8000/v1")

# with_raw_response prevents the SDK from stripping unknown fields
raw = client.chat.completions.with_raw_response.create(
    model="gemma-3-4b-it",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"attn_capture": 1, "attn_capture_layers": "2,18,27"},
)
response = json.loads(raw.content)
```

**cURL:**
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-3-4b-it",
       "messages":[{"role":"user","content":"Hello"}],
       "attn_capture":1, "attn_capture_layers":"2,18,27"}'
```

### 3. Parse and analyze

```python
from attention_instrumentation import extract_attention_from_response, AttentionAnalyzer

attn_by_layer = extract_attention_from_response(response)
# attn_by_layer[18]["scores"]  →  np.ndarray [T, num_heads, T]
#                                             (query × head × key)

layer    = attn_by_layer[18]
analyzer = AttentionAnalyzer(layer["scores"], layer["token_meta"])

# Head-averaged top-5 keys for the last query token
for idx, weight, tok_type in analyzer.top_attended_tokens(
        token_idx=-1, top_k=5, avg_heads=True):
    print(f"  key {idx} ({tok_type}): {weight:.3f}")

# Single-head attention vector
attn_vec = analyzer.attention_for_token(token_idx=-1, head_idx=3)

# Cross-modal attention (vision → language fraction)
cross = analyzer.cross_modal_attention(avg_heads=True)
print(f"Vision→Language: {cross:.2%}")
```

## Built-in Examples

Run the included demos directly:

```bash
# Needle-in-a-haystack: recall a value by key from a long list
python attention_instrumentation.py --example needle

# Codename retrieval: find an agent's codename from a structured log
python attention_instrumentation.py --example codename

# Override layers
python attention_instrumentation.py --example needle --layers 0,9,18
```

Each example prints per-layer attention in compact form:

```
── L18 (T=138 H=8 prompt=131) ──
  avg  <bos>(0)=0.24  :(132)=0.17  alpha(131)=0.08◀  ...
  h7   alpha(131)=0.23◀  ↵(127)=0.13  :(132)=0.12
```

`◀` marks the needle/target token. `avg` is head-averaged; `h0`–`h7` are per-head.

## Response format

`attn_capture_data` is a list of per-layer objects:

```json
{
  "attn_capture_data": [
    {
      "layer_idx": 18,
      "data": "<base64(gzip(float16 array))>",
      "shape": [138, 8, 138],
      "token_meta": {
        "prompt_len": 131,
        "total_len": 138,
        "vision_ranges": [],
        "lang_ranges": [{"start": 0, "end": 131}]
      }
    }
  ]
}
```

`shape` is `[T, num_heads, T]` — query × head × key. Scores are
gzip-compressed, base64-encoded `float16` arrays.

## Server parameters

| Flag | Description |
|------|-------------|
| `--enable-attention-instrumentation` | Enable the feature |
| `--attention-instrumentation-layers LAYERS` | `"all"`, `"2,18,27"`, or `"18"` |
| `--no-enable-chunked-prefill` | Required for full prompt coverage |
| `--no-enable-prefix-caching` | Required for full prompt coverage |

## Client parameters

| Field | Values |
|-------|--------|
| `attn_capture` | `0` (off) / `1` (on) |
| `attn_capture_layers` | Overrides server-side layer list |

## Notes

- Scores are softmax probabilities (0–1) with causal mask applied
- Supports multimodal inputs (text + images) via `vision_ranges` / `lang_ranges`
- Overhead is proportional to the number of captured layers
